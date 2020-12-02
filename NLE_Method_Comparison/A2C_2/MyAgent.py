import torch.nn as nn
import pathlib
import argparse
import os
import numpy as np
import gym
from torch.distributions import Categorical
import torch.nn.functional as F
from nle import nethack
from a2c_model import ActorCritic
import torch
from torch.autograd import Variable

def parse_args():
    parser = argparse.ArgumentParser("A2C experiments for Atari games")
    # Environment
    parser.add_argument("--crop_dims", type=int, default=10, help="name of the game")
    parser.add_argument("--enable-lstm", type=bool, default=True, help="Enable recurrent cell.")
    parser.add_argument("--env", type=str, default="NetHackScore-v0", help="name of the game")
    parser.add_argument("--storage_path", type=str, default="/opt/project/",
                        help="System storage path")
    return parser.parse_args()

ACTIONS = [
    nethack.CompassCardinalDirection.N,
    nethack.CompassCardinalDirection.E,
    nethack.CompassCardinalDirection.S,
    nethack.CompassCardinalDirection.W
]
STATS_INDICES = {
    'x_coordinate': 0,
    'y_coordinate': 1,
    'score': 9,
    'health_points': 10,
    'health_points_max': 11,
    'hunger_level': 18,
}


class Crop():
    """Helper class for NetHackNet below."""

    def __init__(self, height, width, height_target, width_target):
        super(Crop, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        self.width_grid = self._step_to_range(2 / (self.width - 1), self.width_target)[
                     None, :
                     ].expand(self.height_target, -1)
        self.height_grid = self._step_to_range(2 / (self.height - 1), height_target)[
                      :, None
                      ].expand(-1, self.width_target)

        # "clone" necessary, https://github.com/pytorch/pytorch/issues/34880
      #  self.register_buffer("width_grid", width_grid.clone())
      #  self.register_buffer("height_grid", height_grid.clone())

    def _step_to_range(self, delta, num_steps):
        """Range of `num_steps` integers with distance `delta` centered around zero."""
        return delta * torch.arange(-num_steps // 2, num_steps // 2)

    def crop(self, inputs, coordinates):
        """Calculates centered crop around given x,y coordinates.
        Args:
           inputs [B x H x W]
           coordinates [B x 2] x,y coordinates
        Returns:
           [B x H' x W'] inputs cropped and centered around x,y coordinates.
        """
        assert inputs.shape[1] == self.height
        assert inputs.shape[2] == self.width

        inputs = inputs[:, None, :, :].float()

        x = float(coordinates[0])
        y = float(coordinates[1])

        x_shift = 2 / (self.width - 1) * (x - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y - self.height // 2)

        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift,
                self.height_grid[None, :, :] + y_shift,
            ],
            dim=3,
        )

        # TODO: only cast to int if original tensor was int
        return (
            torch.round(F.grid_sample(inputs, grid, align_corners=True))
                .squeeze(1)
                .long()
        )



def transform_observation(observation, crop_dims=10):
    """Process the state into the model input shape
    of ([glyphs, stats], )"""
    observed_glyphs = observation['chars']
    observed_glyphs[np.where((observed_glyphs != 45) & (observed_glyphs != 124) &
                             (observed_glyphs != 35) & (observed_glyphs != 43))] = 0.0
    observed_glyphs[np.where((observed_glyphs == 45) | (observed_glyphs == 43) |
                             (observed_glyphs == 124))] = 8.0  # Walls & Door
    observed_glyphs[np.where((observed_glyphs == 35))] = 16.0  # Corridor
    cropper = Crop(height=observed_glyphs.shape[0], width=observed_glyphs.shape[1], height_target=crop_dims, width_target=crop_dims)
    cropped_glyphs = cropper.crop(torch.from_numpy(observed_glyphs).unsqueeze(0), (crop_dims, crop_dims)) / 16.0
    stat_x_coord = observation['blstats'][STATS_INDICES['x_coordinate']]
    stat_y_coord = observation['blstats'][STATS_INDICES['y_coordinate']]
    stat_health = float(observation['blstats'][STATS_INDICES['health_points']]) / \
                  float(observation['blstats'][STATS_INDICES['health_points_max']])
    #stat_hunger = observation['blstats'][STATS_INDICES['hunger_level']]
    observed_stats = np.array([stat_x_coord, stat_y_coord, stat_health,]) #stat_hunger])

    return cropped_glyphs.float(), observed_stats.astype(np.float32)



class Flatten(nn.Module):
    """
    Flatten a multi dimensional output from the Conv2D to a single dimension
    """

    def forward(self, x):
        return x.view(x.shape[0], -1)

class ActorCritic(nn.Module):
    def __init__(self, glyph_shape, num_actions, enable_lstm, crop_dims=10):
        super(ActorCritic, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            nn.ReLU(),
          #  nn.Dropout(0.5),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.flatten = Flatten()
        if enable_lstm:
            self.lstm = nn.LSTMCell(256, 256)
        self.linear = nn.Linear(1152, 256)
        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)

        for m in self.features:
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.0)

        nn.init.orthogonal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

        nn.init.orthogonal_(self.critic.weight)
        nn.init.constant_(self.critic.bias, 0.0)

        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.constant_(self.actor.bias, 0.0)

    def forward(self, x_glyphs, hx, cx, enable_lstm):
        x_glyphs = x_glyphs.unsqueeze(0)
        x_glyphs = self.features(x_glyphs)
        x_glyphs = self.flatten(x_glyphs)
        x_glyphs = self.linear(x_glyphs)
        x_glyphs = F.relu(x_glyphs)
        if enable_lstm:
            hx, cx = self.lstm(x_glyphs, (hx, cx))
            x = hx
        else:
            x = x_glyphs
        return Categorical(logits=self.actor(x)), self.critic(x), hx, cx


class MyAgent:
    def __init__(self, observation_space, action_space, **kwargs):
        self.observation_space = observation_space
        self.action_space = action_space
        self.seeds = kwargs.get('seeds', None)
        self.cx = Variable(torch.zeros(1, 256))
        self.hx = Variable(torch.zeros(1, 256))
        path = str(pathlib.Path(__file__).parent.absolute()) + '/a2c_model.pt'
        print(path)
        self.net = ActorCritic(observation_space['chars'].shape, len(ACTIONS), enable_lstm=True)
        self.net.load_state_dict(torch.load(path))

    def act(self, observation):
        # Perform processing to observation
        observed_glyphs, observed_stats = transform_observation(observation)
        with torch.no_grad():
            actor, value, _, _ = self.net(observed_glyphs, self.hx, self.cx, enable_lstm=True)
        action = actor.sample()
        action = action + 1  # Not using action 0
        return action
