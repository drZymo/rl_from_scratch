import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

class ExperienceBuffer(object):
    def __init__(self, capacity, observation_shape, action_shape):
        self.capacity = capacity
        self.states = np.empty((capacity,) + observation_shape)
        self.actions = np.empty((capacity,) + action_shape)
        self.rewards = np.empty((capacity, 1))
        self.next_states = np.empty((capacity,) + observation_shape)
        self.dones = np.empty((capacity, 1))
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


def create_frame(env):
    # Setup display for first frame
    fig, ax = plt.subplots(figsize=(8,6))
    ax.axis('off')
    img = ax.imshow(env.render('rgb_array'))
    plt.show()
    return (fig, img, env)

def update_frame(frame):
    fig, img, env = frame
    # Update displayed frame
    img.set_data(env.render('rgb_array'))
    clear_output(wait=True)
    display(fig)

def init_display():
    try:
        from pyvirtualdisplay import Display
        disp = Display(visible=False, size=(400, 300)).start()
    except:
        print('Virtual display not used')
