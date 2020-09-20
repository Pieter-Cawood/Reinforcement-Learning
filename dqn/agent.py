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
        optimizer_momentum,
        batch_size,
        discount_factor,
    ):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param discount_factor: the discount factor
        """

        self.replay_buffer = replay_buffer
        self.use_double_dqn = use_double_dqn
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.policy_network = DQN(observation_space, action_space).to(device)
        # Target network used to estimate next best Q-vals
        self.target_network = DQN(observation_space, action_space).to(device)
        # Optimizer used in double trouble dqn paper
        self.optimizer = torch.optim.RMSprop(self.policy_network.parameters(), lr= lr, momentum=optimizer_momentum)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        # Get mini-batch, dones as float (0 or 1)
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # To torch
        # Normalize data
        states = states / 255
        next_states = next_states / 255
        # Make tensors on GPU
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).int().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        Q_val_next = None
        Q_val_next_max = None
        Q_target = None

        # Forward pass to get q values
        # Standard DQN
        if not self.use_double_dqn:
            Q_vals_next = self.target_network(next_states)
            Q_val_next_max, action_next_max = Q_vals_next.max(1)
        # Double Trouble Deep QN
        else:
            Q_vals_next = self.policy_network(next_states)
            Q_val_next_max, action_next_max = Q_vals_next.max(1)

        # Set y_j for each mini-batch entry,
        # If terminal then Q = rewards only
        Q_target = rewards + (1 - dones) * self.discount_factor * Q_val_next_max

        # Perform a gradient descent step on (y_j - Q)^2 ( loss )
        # Before backward pass, use optimizer to zero all the gradients of the tensors
        # it has to update
        self.optimizer.zero_grad()
        # Compute the Hubert loss
        loss = self.loss_fn(Q_val_next_max, Q_target)
        # Backward pass: compute gradients
        loss.backward()
        # Update optimizer parameters
        self.optimizer.step()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())


    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        states = state / 255
        states = torch.from_numpy(states).float().unsqueeze(0).to(device)
        Q_vals = self.policy_network(states)
        Q_val_max, action_max = Q_vals.max(1)
        return action_max.item()
