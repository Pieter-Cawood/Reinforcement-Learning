from gym import spaces
import numpy as np
import torch

from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer

device = "cuda"

class DQNAgent:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma,
    ):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """

        self.replay_buffer = replay_buffer
        self.use_double_dqn = use_double_dqn
        self.lr = lr
        self.batch_size = batch_size
        self.discount_factor = gamma
        self.target_network_1 = DQN(observation_space, action_space).to(device) 
        self.target_network_2 = None
        
        if self.use_double_dqn:
            self.target_network_2 = DQN(observation_space, action_space).to(device)
            

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        # Get mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # To torch
        # Normalize data
        states = states / 255
        next_state = next_states / 255
        # Create tensors
        states = torch.from_numpy(states).to(device)
        actions = torch.from_numpy(actions).to(device)
        rewards = torch.from_numpy(rewards).to(device)
        next_states = torch.from_numpy(next_states).to(device)
        dones = torch.from_numpy(dones).to(device)

        Q = self.



        stop = True

        raise NotImplementedError

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        # TODO update target_network parameters with policy_network parameters
        raise NotImplementedError

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        # TODO Select action greedily from the Q-network given the state
        raise NotImplementedError
