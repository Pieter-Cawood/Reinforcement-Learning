from gym import spaces
import torch.nn as nn


class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete, 
                 filter_sizes=[32, 64, 64], kernel_sizes=[8, 4, 3], strides=[4, 2, 1],
                 last_layer_size = 512):
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
            nn.Conv2d(observation_space.shape[0],
                      filter_sizes[0],
                      kernel_sizes[0],
                      strides[0]),
            nn.ReLU(),
            # The second hidden layer convolves 64 filters, 4x4 with stride 2
            nn.Conv2d(filter_sizes[0],
                      filter_sizes[1],
                      kernel_sizes[1],
                      strides[1]),
            nn.ReLU(),
            # The third layer convolves 64 filters, 3x3 with stride 1
            nn.Conv2d(filter_sizes[1],
                      filter_sizes[2],
                      kernel_sizes[2],
                      strides[2]),
            nn.ReLU(),
            # The final hidden layer is fully connected and consists of 512 rectifier units
            nn.Linear(filter_sizes[2], 
                      last_layer_size),
            nn.ReLU(),
            # The output layer is fully-connected with single output for each action
            nn.Linear(last_layer_size, 
                      action_space.n)
            
            )
        
    def forward(self, x):
        return self.conv_nn(x)

        
