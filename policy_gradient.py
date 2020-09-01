import gym
import numpy as np

class PongEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PongEnv, self).__init__()

        self.env = gym.make('PongNoFrameskip-v4')

        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(80,80,2), dtype=np.float32)

        self.memory = np.zeros((80,80,2), dtype=np.float32)


    def reset(self):
        observation = self.env.reset()
        self.memory = np.roll(self.memory, 1, axis=2)
        self.memory[:,:,0] = self._process(observation)
        
        # 59 NoOp actions at the start will make sure the game is really started.
        for _ in range(59):
            observation, reward, done, info = self.env.step(0)
            self.memory = np.roll(self.memory, 1, axis=2)
            self.memory[:,:,0] = self._process(observation)

        return self.memory


    def step(self, action):
        # Repeat action 4 times
        total_reward = 0
        for _ in range(4):
            observation, reward, done, info = self.env.step(action)
            total_reward += reward

        self.memory = np.roll(self.memory, 1, axis=2)
        self.memory[:,:,0] = self._process(observation)
        
        return self.memory, total_reward, done, info


    def render(self, mode='human'):
        return self.env.render(mode)


    def close (self):
        self.env.close()


    def _process(self, obs):
        # crop to 160x160
        # downsample to 80x80
        # take red channel only
        obs = obs[34:194:2,::2,0]
        # change background to 0 and everything else to 1
        obs[obs == 144] = 0
        obs[obs == 109] = 0
        obs[obs != 0] = 1
        return obs.astype(np.float32)