import matplotlib.pyplot as plt

_WIDTH = 7
_HEIGHT = 6
_WIND_PROB = 0.2

class GridWorldMDP(object):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    ACTIONS = [UP, RIGHT, DOWN, LEFT]
    REWARDS = [-100, -1, 0, 1]
    STATES = list(range(_HEIGHT*_WIDTH))

    def _is_terminal(self, x, y):
        return self._is_end(x, y) or self._is_dead(x, y)
               
    def _is_dead(self, x, y):
        return x == 0 or y == 0 or x == (_WIDTH - 1) or y == (_HEIGHT - 1) or \
            (y == (_HEIGHT - 2) and x > 1 and x < (_WIDTH - 2))

    def _is_end(self, x, y):
        return y == (_HEIGHT - 2) and x == (_WIDTH - 2)
    
    def p(self, state, action, next_state, reward):
        curr_x, curr_y = state % _WIDTH, state // _WIDTH
        next_x, next_y = next_state % _WIDTH, next_state // _WIDTH
        
        probs = [0] * (_HEIGHT * _WIDTH)
        rewards = [0] * (_HEIGHT * _WIDTH)
    
        curr_terminal = self._is_terminal(curr_x, curr_y)
        
        if curr_terminal:
            probs[state] = 1
            rewards[state] = 0
        else:
            target_x, target_y = curr_x, curr_y
            if action == self.UP:
                target_y = (target_y - 1) if target_y > 0 else 0
            elif action == self.DOWN:
                target_y = (target_y + 1) if target_y < (_HEIGHT - 1) else (_HEIGHT - 1)
            elif action == self.LEFT:
                target_x = (target_x - 1) if target_x > 0 else 0
            elif action == self.RIGHT:
                target_x = (target_x + 1) if target_x < (_WIDTH - 1) else (_WIDTH - 1)
            target_state = target_y * _WIDTH + target_x

            wind_x, wind_y = target_x, target_y
            if wind_y < (_HEIGHT - 1):
                wind_y += 1
            wind_state = wind_y * _WIDTH + wind_x

            probs[target_state] += (1 - _WIND_PROB)
            probs[wind_state] += _WIND_PROB

            if self._is_end(target_x, target_y):
                rewards[target_state] = 1
            elif self._is_dead(target_x, target_y):
                rewards[target_state] = -100
            else:
                rewards[target_state] = -1

            if self._is_end(wind_x, wind_y):
                rewards[wind_state] = 1
            elif self._is_dead(wind_x, wind_y):
                rewards[wind_state] = -100
            else:
                rewards[wind_state] = -1
            
        #print(probs)
        #print(rewards)

        return probs[next_state] if rewards[next_state] == reward else 0

def plot_world():
    plt.figure(figsize=(_WIDTH,_HEIGHT))
    plt.ylim((_HEIGHT - 0.5, -0.5))
    plt.xlim((-0.5, _WIDTH - 0.5))
    E = 2
    S = 0
    T = 8
    G = 1
    d = [[T,T,T,T,T,T,T],[T,G,G,G,G,G,T],[T,G,G,G,G,G,T],[T,G,G,G,G,G,T],[T,S,T,T,T,E,T],[T,T,T,T,T,T,T]]
    plt.imshow(d, cmap='Set1')

    for x in range(_WIDTH+1):
        plt.vlines(x - 0.5, -0.5, _HEIGHT - 0.5, linewidth=1)

    for y in range(_HEIGHT+1):
        plt.hlines(y - 0.5, -0.5, _WIDTH - 0.5, linewidth=1)    
    
def _plot_grid(ax):
    ax.set_ylim(_HEIGHT - 0.5, -0.5)
    ax.set_xlim(-0.5, _WIDTH - 0.5)
    for x in range(_WIDTH+1):
        ax.vlines(x - 0.5, -0.5, _HEIGHT + 0.5, linewidth=1, alpha=0.3)
    for y in range(_HEIGHT+1):
        ax.hlines(y - 0.5, -0.5, _WIDTH + 0.5, linewidth=1, alpha=0.3)
    
def _plot_values(ax, v):
    ax.set_title('state values')
    _plot_grid(ax)

    for row in range(_HEIGHT):
        for column in range(_WIDTH):
            state = row * _WIDTH + column
            ax.text(column, row, f'{v[state]:5.1f}', ha='center', va='center', size='x-large')

def _plot_policy(ax, env, pi):
    ax.set_title('policy')
    _plot_grid(ax)

    for state in env.STATES:
        for action in env.ACTIONS:
            if pi[state, action] > 0:
                x, y = state % _WIDTH, state // _WIDTH
                if action == env.UP:
                    dx, dy = 0, -0.4
                elif action == env.DOWN:
                    dx, dy = 0, 0.4
                elif action == env.RIGHT:
                    dx, dy = 0.4, 0
                elif action == env.LEFT:
                    dx, dy = -0.4, 0
                ax.arrow(x, y, dx, dy, length_includes_head=True, head_width=0.1, head_length=0.1)
            
def plot_values(v):
    fig, (ax1) = plt.subplots(1, 1, figsize=(_WIDTH,_HEIGHT))
    
    _plot_values(ax1, v)

def plot_values_and_policy(v, env, pi):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(_WIDTH*2,_HEIGHT))
    
    _plot_values(ax1, v)
    _plot_policy(ax2, env, pi)

def argmax(v):
    if len(v) < 1: return []
    max_v = v[0]
    max_i = [0]
    for i in range(1, len(v)):
        if abs(v[i] - max_v) < 1e-7:
            max_i.append(i)
        elif v[i] > max_v:
            max_v = v[i]
            max_i = [i] 
    return max_i