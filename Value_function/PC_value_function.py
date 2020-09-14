###
# Group Members
# Clarise Poobalan : 383321
# Nicolaas Cawood: 2376182
# Shikash Algu: 2373769
# Byron Gomes: 0709942R
###


#######################################################################
# Following are some utilities for tile coding from Rich.
# To make each file self-contained, I copied them from
# http://incompleteideas.net/tiles/tiles3.py-remove
# with some naming convention changes
#
# Tile coding starts
from math import floor, exp
import numpy as np
from gym.spaces import Discrete
import gym
from gym import wrappers
from collections import defaultdict
import matplotlib.pyplot as plt


class IHT:
    "Structure to handle collisions"

    def __init__(self, size_val):
        self.size = size_val
        self.overfull_count = 0
        self.dictionary = {}

    def count(self):
        return len(self.dictionary)

    def full(self):
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfull_count == 0: print('IHT full, starting to allow collisions')
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count


def hash_coords(coordinates, m, read_only=False):
    if isinstance(m, IHT): return m.get_index(tuple(coordinates), read_only)
    if isinstance(m, int): return hash(tuple(coordinates)) % m
    if m is None: return coordinates


def tiles(iht_or_size, num_tilings, floats, ints=None, read_only=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    if ints is None:
        ints = []
    qfloats = [floor(f * num_tilings) for f in floats]
    tiles = []
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tilingX2
        coords.extend(ints)
        tiles.append(hash_coords(coords, iht_or_size, read_only))
    return tiles


# Tile coding ends
#######################################################################

# bound for position and velocity
POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07


# wrapper class for state action value function
class ValueFunction:
    # In this example I use the tiling software instead of implementing standard tiling by myself
    # One important thing is that tiling is only a map from (state, action) to a series of indices
    # It doesn't matter whether the indices have meaning, only if this map satisfy some property
    # View the following webpage for more information
    # http://incompleteideas.net/sutton/tiles/tiles3.html
    # @max_size: the maximum # of indices
    def __init__(self, alpha, n_actions, num_of_tilings=8, max_size=2048):
        self.action_space = Discrete(n_actions)
        self.max_size = max_size
        self.num_of_tilings = num_of_tilings

        # divide step size equally to each tiling
        self.step_size = alpha / num_of_tilings

        self.hash_table = IHT(max_size)

        # weight for each tile
        self.weights = np.zeros(max_size)

        # position and velocity needs scaling to satisfy the tile software
        self.position_scale = self.num_of_tilings / (POSITION_MAX - POSITION_MIN)
        self.velocity_scale = self.num_of_tilings / (VELOCITY_MAX - VELOCITY_MIN)

    # get indices of active tiles for given state and action
    def _get_active_tiles(self, position, velocity, action):
        # I think positionScale * (position - position_min) would be a good normalization.
        # However positionScale * position_min is a constant, so it's ok to ignore it.
        active_tiles = tiles(self.hash_table, self.num_of_tilings,
                             [self.position_scale * position, self.velocity_scale * velocity],
                             [action])
        return active_tiles

    # estimate the value of given state and action
    def __call__(self, state, action):
        position, velocity = tuple(state)
        if position == POSITION_MAX:
            return 0.0
        active_tiles = self._get_active_tiles(position, velocity, action)
        return np.sum(self.weights[active_tiles])

    # learn with given state, action and target
    # Target = R + gamma*target
    def update(self, target, state, action):
        active_tiles = self._get_active_tiles(state[0], state[1], action)
        estimation = np.sum(self.weights[active_tiles])
        delta = self.step_size * (target - estimation)
        for active_tile in active_tiles:
            self.weights[active_tile] += delta

    # Epsilon greedy action selection
    def act(self, state, epsilon=0):
        if np.random.random() < epsilon:
            return self.action_space.sample()
        return np.argmax([self(state, action) for action in range(self.action_space.n)])


def semi_gradient_sarsa(n_episodes, n_runs, epsilon, alpha, discount_factor, print_=True):

    def _record_episode(episode):
        if episode == n_episodes - 1:
            return True
        return False

    number_of_steps = defaultdict(float)

    for run_number in range(n_runs):
        env = gym.make('MountainCar-v0')
        # Only video first run
        if run_number == 0:
            env = wrappers.Monitor(env, './output',
                                   video_callable=_record_episode,
                                   force=True)
        # Make tiling value function
        q_hat = ValueFunction(alpha=alpha,
                              n_actions=env.action_space.n,
                              num_of_tilings=8)

        for episode_number in range(n_episodes):
            # Initial state & action
            current_state = env.reset()
            action = q_hat.act(current_state, epsilon)
            # Loop for each step of episode
            step_count = 0
            while True:
                step_count += 1
                next_state, reward, done, info = env.step(action)
                # Only render final go
                if episode_number > (n_episodes - 1):
                    env.render()
                # Terminal
                if done:
                    q_hat.update(reward, current_state, action)
                    number_of_steps[episode_number] += step_count
                    break
                # Not yet terminal
                next_action = q_hat.act(next_state, epsilon)
                target = reward + discount_factor * q_hat.__call__(next_state, next_action)
                q_hat.update(target, current_state, action)
                # Update
                current_state = next_state
                action = next_action
        env.close()
        if print_:
            print('Runs complete: {:d}'.format(run_number + 1))
    average_steps = [value/n_runs for key, value in number_of_steps.items()]
    return average_steps


if __name__ == "__main__":
    average_steps = semi_gradient_sarsa(n_episodes=500,
                                        n_runs=100,
                                        epsilon=0.1,
                                        alpha=0.1,
                                        discount_factor=0.99)
    plt.title('Average steps per episode')
    plt.semilogy(range(len(average_steps)), average_steps)
    plt.xlabel('Episode number')
    plt.ylim([100,300])
    plt.savefig('./output/plot.png')
    plt.show()