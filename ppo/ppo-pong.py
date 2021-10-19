# hide tensorflow infos, warnings, or errors
import os

from tensorflow.python.keras.utils.np_utils import normalize
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore',category=UserWarning)

import tensorflow as tf

# hide more tensorflow warnings
tf.get_logger().setLevel('ERROR')
# Disable eager execution for performance
tf.compat.v1.disable_eager_execution()

import gym
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import ray
ray.init()

# Reduce warnings
gym.logger.set_level(40)

gamma = 0.99
lmbda = 0.95
epsilon = 0.2
entropy_coef = 0.01
num_players = 8
num_steps = 2048 // num_players
num_episodes = 1000
epochs = 4
batch_size = 64
actor_learning_rate = 3e-4
critic_learning_rate = actor_learning_rate / 2
save_every = 300 # seconds

observation_shape = (80,80,2)
nr_actions = 6

log_folder = Path('..')/'ppo-pong'
if not log_folder.exists(): log_folder.mkdir()
    
actor_file = str(log_folder/'actor.h5')
critic_file = str(log_folder/'critic.h5')


## Utils

def discount(values, dones, gamma):
    discounted_values = np.zeros_like(values, np.float32)
    next_value = np.zeros_like(values[0], np.float32)
    for i in reversed(range(len(values))):
        next_value = values[i] + (1-dones[i]) * gamma * next_value
        discounted_values[i] = next_value
    return discounted_values

def get_new_log_file(log_folder, prefix):
    if not log_folder.exists(): log_folder.mkdir()
    indices = [int(log_file.name[len(prefix):-4]) for log_file in log_folder.glob(f'{prefix}*.log')]
    maxIndex = max(indices) if indices else 0
    return log_folder / f'{prefix}{maxIndex+1}.log'


# ## Models

def build_actor_model():
    observation = Input(observation_shape, name='observation')
    h = Conv2D(32, 5, strides=3, padding='valid', activation='relu', kernel_initializer='he_uniform', name='conv1')(observation)
    h = Conv2D(32, 3, strides=2, padding='valid', activation='relu', kernel_initializer='he_uniform', name='conv2')(h)
    h = Flatten(name='features')(h)
    h = Dense(256, activation='relu', kernel_initializer='he_uniform', name='dense1')(h)
    h = Dense(128, activation='relu', kernel_initializer='he_uniform', name='dense2')(h)
    action_probs = Dense(nr_actions, activation='softmax', name='action_probs')(h)
    return Model(observation, action_probs, name='actor')


def build_actor_train_model(actor, lr):
    observation = Input(observation_shape, name='observation')
    action_probs = actor(observation)
    
    old_action_probs = Input(nr_actions, name='old_action_probs')
    advantage = Input(1, name='advantage')
    
    def loss(y_true, y_pred):
        selected_action = y_true
        action_probs = y_pred

        # ppo
        old_action_prob = K.sum(old_action_probs * selected_action, axis=-1, keepdims=True)
        action_prob = K.sum(action_probs * selected_action, axis=-1, keepdims=True)
        ratio = action_prob / (old_action_prob + 1e-9)
        loss1 = ratio * advantage
        loss2 = K.clip(ratio, 1-epsilon, 1+epsilon) * advantage
        ppo_loss = -K.minimum(loss1, loss2)
        
        # entropy
        entropy = -K.sum(action_probs * K.log(action_probs + 1e-9), axis=-1, keepdims=True)
        entropy_loss = -entropy_coef * entropy

        return ppo_loss + entropy_loss
    
    model = Model([observation, old_action_probs, advantage], action_probs, name='actor_train')
    model.compile(loss=loss, optimizer=Adam(lr), experimental_run_tf_function=False)
    return model


def build_critic_model():
    observation = Input(observation_shape, name='observation')
    h = Conv2D(16, 5, strides=3, padding='valid', activation='relu', kernel_initializer='he_uniform', name='conv1')(observation)
    h = Conv2D(32, 3, strides=2, padding='valid', activation='relu', kernel_initializer='he_uniform', name='conv2')(h)
    h = Flatten(name='features')(h)
    h = Dense(256, activation='relu', kernel_initializer='he_uniform', name='dense1')(h)
    h = Dense(128, activation='relu', kernel_initializer='he_uniform', name='dense2')(h)
    state_value = Dense(1, activation='linear', name='state_value')(h)
    return Model(observation, state_value, name='critic')

def build_critic_train_model(critic, lr):
    critic.compile(loss='mse', optimizer=Adam(lr))
    return critic

@ray.remote
class Logger:
    def __init__(self, prefix, num_episodes):
        self.num_episodes = num_episodes
        filename = get_new_log_file(log_folder, prefix)
        print(f'Logging to {filename}')
        self.file = open(filename, 'w')
        self.fps = 0
        self.average_reward = None
        self.count = 0
        self.progress = tqdm(total=num_episodes)

    def close(self):
        self.progress.close()
        self.progress = None

        self.file.close()
        self.file = None
    
    def set_fps(self, fps):
        self.fps = fps
        self._update_postfix()

    def log(self, time, length, total_reward):
        self.file.write(f'{time.isoformat()},{length},{total_reward}\n')
        self.file.flush()

        self.average_reward = 0.9 * self.average_reward + 0.1 * total_reward if self.average_reward else total_reward
        self.count += 1
        self.progress.update(1)
        self._update_postfix()
        
    def _update_postfix(self):
        average_reward = f'{self.average_reward:.1f}' if self.average_reward else 'None'
        self.progress.set_postfix_str(f'fps={self.fps:.1f}, rew={average_reward}')

    def is_done(self):
        return self.count >= self.num_episodes

@ray.remote
class Player:
    def __init__(self, logger):
        self.logger = logger
        self.env = gym.make('PongDeterministic-v4')
        self._reset()

    def _reset(self):
        state = self.env.reset()
        # start with 15 noops to skip initial idle time
        for _ in range(15):
            state, _, _, _ = self.env.step(0)
        self.state = self._process(state)
        self.prev_state = self.state
        self.length = 0
        self.total_reward = 0

    def get_observation(self):
        observation = np.dstack([self.prev_state, self.state])
        return observation

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        self.length += 1
        self.total_reward += reward

        # auto-reset
        if done:
            self.logger.log.remote(datetime.now(), self.length, self.total_reward)

            self._reset()
        else:
            self.prev_state = self.state
            self.state = self._process(state)

        observation = np.dstack([self.prev_state, self.state])
        return observation, reward, done

    def _process(self, obs):
        # crop to 160x160
        obs = obs[34:194]
        # downsample to 80x80
        obs = obs[::2,::2]
        # take red channel only
        obs = obs[:,:,0]
        # change background to 0 and everything else to 1
        obs[obs == 144] = 0
        obs[obs == 109] = 0
        obs[obs != 0] = 1
        # flatten
        return obs.astype(np.float32)


# ## Training

def run():
    prev_time = datetime.now()

    # create models
    actor = build_actor_model()
    actor.summary()
    actor_train = build_actor_train_model(actor, actor_learning_rate)
    critic = build_critic_model()
    critic.summary()
    critic_train = build_critic_train_model(critic, critic_learning_rate)

    # load previous weights if available
    if Path(actor_file).exists():
        print(f'Loading previous weights for actor from {actor_file}')
        actor.load_weights(actor_file, by_name=True)
    if Path(critic_file).exists():
        print(f'Loading previous weights for critic from {critic_file}')
        critic.load_weights(critic_file, by_name=True)
    last_save = datetime.now()

    # initialize players
    logger = Logger.remote(f'log_', num_episodes)
    players = [Player.remote(logger) for _ in range(num_players)]
    observations = np.array(ray.get([p.get_observation.remote() for p in players]))

    while not ray.get(logger.is_done.remote()):
        # get batch
        batch_observations = np.empty((num_steps,num_players,)+observation_shape)
        batch_values = np.empty((num_steps+1,num_players,))
        batch_action_probs = np.empty((num_steps,num_players,nr_actions))
        batch_actions = np.empty((num_steps,num_players,nr_actions))
        batch_rewards = np.empty((num_steps,num_players,))
        batch_dones = np.empty((num_steps,num_players,))

        for step in range(num_steps):
            action_probs = actor.predict_on_batch(observations)
            values = critic.predict_on_batch(observations).squeeze(-1)
            actions = [np.random.choice(nr_actions, p=p) for p in action_probs]
            
            results = ray.get([p.step.remote(actions[i]) for i,p in enumerate(players)])
            next_observations = np.array([o for o,r,d in results])
            rewards = [r for o,r,d in results]
            dones = [d for o,r,d in results]
            
            batch_observations[step] = observations
            batch_values[step] = values
            batch_action_probs[step] = action_probs
            batch_actions[step] = to_categorical(actions, num_classes=nr_actions)
            batch_rewards[step] = rewards
            batch_dones[step] = dones

            observations = next_observations

        batch_values[num_steps] = critic.predict_on_batch(observations).squeeze(-1)

        # generalized advantage estimation (GAE)
        td_targets = batch_rewards + (1-batch_dones) * gamma * batch_values[1:]
        td_errors = td_targets - batch_values[:-1]
        advantages = discount(td_errors, batch_dones, lmbda * gamma)

        # discounted rewards
        target_values = advantages + batch_values[:-1]

        # flatten
        def squish(a):
            return a.reshape((a.shape[0]*a.shape[1],)+a.shape[2:])
        train_observations = squish(batch_observations)
        action_probs = squish(batch_action_probs)
        advantages = squish(advantages)
        actions = squish(batch_actions)
        target_values = squish(target_values)
        
        # train
        actor_train.fit([train_observations, action_probs, advantages], actions, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=0)
        critic_train.fit(train_observations, target_values, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=0)

        # save periodically
        now = datetime.now()
        if (now - last_save).total_seconds() > save_every:
            actor.save(actor_file)
            critic.save(critic_file)
            last_save = now

        # update fps
        now = datetime.now()
        delta = (now - prev_time).total_seconds()
        prev_time = now
        fps = (num_steps * num_players) / delta
        logger.set_fps.remote(fps)

    actor.save(actor_file)
    critic.save(critic_file)
    last_save = now

if __name__ == '__main__':
    run()
