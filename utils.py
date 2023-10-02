import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import gymnasium
from tqdm import trange

class ExperienceBuffer(object):
    def __init__(self, capacity, observation_shape, action_shape):
        self.capacity = capacity
        self.states = np.empty((capacity,) + observation_shape, dtype=np.float32)
        self.actions = np.empty((capacity,) + action_shape, dtype=np.int64)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_states = np.empty((capacity,) + observation_shape, dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
        self.writePos = 0
        self.count = 0

    def is_filled(self, n):
        return self.count >= n

    def append(self, state, action, reward, next_state, done):
        self.states[self.writePos] = state
        self.actions[self.writePos] = action
        self.rewards[self.writePos] = reward
        self.next_states[self.writePos] = next_state
        self.dones[self.writePos] = done

        self.writePos = (self.writePos + 1) % self.capacity
        self.count += 1
        if self.count > self.capacity: self.count = self.capacity

    def sample(self, n):
        indices = np.random.choice(self.count, n)
        return self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], self.dones[indices]
    
    def get_all(self):
        return self.states[:self.count], self.actions[:self.count], self.rewards[:self.count], self.next_states[:self.count], self.dones[:self.count]
    
    def reset(self):
        self.writePos = 0
        self.count = 0



current_display = None

def init_virtual_display_if_needed():
    global current_display
    if current_display is None:
        try:
            from pyvirtualdisplay import Display
            current_display = Display(visible=False, size=(400, 300)).start()
        except:
            print('Virtual display not used')

def create_frame(env):
    init_virtual_display_if_needed()

    # Setup display for first frame
    fig, ax = plt.subplots(figsize=(8,6))
    ax.axis('off')
    img = ax.imshow(env.render())
    plt.show()
    return (fig, img, env)

def update_frame(frame):
    fig, img, env = frame
    # Update displayed frame
    img.set_data(env.render())
    clear_output(wait=True)
    display(fig)



def run_episode(env : gymnasium.Env, policy, on_step = None):
    observations = []
    actions = []
    rewards = []

    done, length, score = False, 0, 0

    observation, _ = env.reset()

    while not done:
        action = policy(observation)
        observations.append(observation)
        actions.append(action)

        next_observation, reward, done, trunc, _ = env.step(action)
        rewards.append(reward)

        length += 1
        score += reward

        if trunc: done = True

        if on_step: on_step(observation, action, next_observation, reward, done, length, score)

        observation = next_observation

    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    rewards = np.array(rewards, dtype=np.float32)

    return length, score, observations, actions, rewards


def run_environment(env : gymnasium.Env, n_episodes: int, policy, on_step = None, on_episode_end = None):
    avg_factor = 0.99
    avg_length = None
    avg_score = None

    E = trange(n_episodes)
    for episode in E:
        length, score, observations, actions, rewards = run_episode(env, policy, on_step)

        avg_length = (avg_factor * avg_length + (1-avg_factor) * length) if avg_length else length
        avg_score = (avg_factor * avg_score + (1-avg_factor) * score) if avg_score else score
        E.set_postfix(avg_length=f'{avg_length:.1f}', avg_score=f'{avg_score:.1f}')
        
        if on_episode_end: on_episode_end(episode, observations, actions, rewards, length, score)


def evaluate(env: gymnasium.Env, policy):
    env.reset()
    frame = create_frame(env)

    def on_step(observation, action, next_observation, reward, done, length, score):
        update_frame(frame)

    length, score, _, _, _ = run_episode(env, policy, on_step)
    
    print(f'Episode length: {length}, return: {score}')
    return length, score


def epsilon_greedy_policy(observation, env, greedy_policy, epsilon):
    if np.random.uniform() < epsilon:
        return env.action_space.sample()
    else:
        return greedy_policy(observation)


def discount(values, gamma):
    discounted_values = []
    discounted_value = 0
    for value in reversed(values):
        discounted_value = value + gamma * discounted_value
        discounted_values.insert(0, discounted_value)
    discounted_values = np.array(discounted_values, dtype=np.float32)
    return np.expand_dims(discounted_values, axis=-1)
        
