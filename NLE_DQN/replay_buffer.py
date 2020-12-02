"""
Modified the DQN method from https://github.com/raillab/dqn

"""

import numpy as np

class ReplayBuffer:
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self, size):
        """
        Initialise a buffer of a given size for storing transitions
        :param size: the maximum number of transitions that can be stored
        """
        self._storage = []
        self._batch_size = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, glyph_state, stat_state, action, reward, glyph_state_, stat_state_, done):
        """
        Add a transition to the buffer. Old transitions will be overwritten if the buffer is full.
        :param state: the agent's initial state
        :param action: the action taken by the agent
        :param reward: the reward the agent received
        :param next_state: the subsequent state
        :param done: whether the episode terminated
        """
        data = (glyph_state, stat_state, action, reward, glyph_state_, stat_state_, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._batch_size

    def _encode_sample(self, indices):
        glyph_states, stat_states, actions, rewards, glyph_states_, stat_states_, dones = [], [], [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            glyph_state, stat_state, action, reward, glyph_state_, stat_state_, done = data
            glyph_states.append(np.array(glyph_state, copy=False))
            stat_states.append(np.array(stat_state, copy=False))
            actions.append(action)
            rewards.append(reward)
            glyph_states_.append(np.array(glyph_state_, copy=False))
            stat_states_.append(np.array(stat_state_, copy=False))
            dones.append(done)
        return (
            np.array(glyph_states),
            np.array(stat_states),
            np.array(actions),
            np.array(rewards),
            np.array(glyph_states_),
            np.array(stat_states_),
            np.array(dones),
        )

    def sample(self):
        """
        Randomly sample a batch of transitions from the buffer.
        :param batch_size: the number of transitions to sample
        :return: a mini-batch of sampled transitions
        """
        indices = np.random.randint(0, len(self._storage) - 1, size=self._batch_size)
        return self._encode_sample(indices)