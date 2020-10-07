from gym import spaces
import torch.nn as nn
import torch

class Flatten(nn.Module):
    """
    Flatten a multi dimensional output from the Conv2D to a single dimension
    """
    def forward(self, x):
        return x.view(-1, x.shape[0])


class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        assert (
            type(observation_space) == spaces.Box
        ), "observation_space must be of type Box"
        assert (
            len(observation_space.shape) == 3
        ), "observation space must have the form channels x width x height"
        assert (
            type(action_space) == spaces.Discrete
        ), "action_space must be of type Discrete"
        
        self.conv_nn = nn.Sequential(
            # The first hidden layer convolves 32 filters, 8x8 with stride 4
            nn.Conv2d(observation_space.shape[0], 32, 8, 4),
            nn.ReLU(),
            # The second hidden layer convolves 64 filters, 4x4 with stride 2
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            # The third layer convolves 64 filters, 3x3 with stride 1
            # Output dimensions: No padding, stride *
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            # Conv2D outputs 7x7x64, reshape to 3136 vector
            nn.Flatten(),
            # The final hidden layer is fully connected and consists of 512 rectifier units
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            # The output layer is fully-connected with single output for each action
            nn.Linear(512, action_space.n)
            )

        
    def forward(self, x: torch.Tensor):
        return self.conv_nn(x)
