"""
Modified the A2C method from https://github.com/raillab/a2c

"""

import argparse
import os
import numpy as np
import gym
import torch.nn.functional as F
from nle import nethack
import time
from model import ActorCritic
import torch
import torch.optim as optim
from torch.autograd import Variable

def parse_args():
    parser = argparse.ArgumentParser("A2C experiments for Atari games")
    parser.add_argument("--seeds", type=list, default=[1,2,3,4,5], help="which seed to use")
    # Environment
    parser.add_argument("--enable-lstm", type=bool, default=True, help="Enable recurrent cell.")
    parser.add_argument("--crop-dims", type=int, default=10, help="name of the game")
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


def make_env(args, seed, actions):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    env = gym.make(args.env, actions=actions)
    env.seed(seed)
    return env

def transform_observation(observation, args):
    """Process the state into the model input shape
    of ([glyphs, stats], )"""
    observed_glyphs = observation['chars']
    observed_glyphs[np.where((observed_glyphs != 45) & (observed_glyphs != 124) &
                             (observed_glyphs != 35) & (observed_glyphs != 43) &
                             (observed_glyphs != 36))] = 0.0
    observed_glyphs[np.where((observed_glyphs == 45) | (observed_glyphs == 43) |
                             (observed_glyphs == 124))] = 8.0  # Walls & Door
    observed_glyphs[np.where((observed_glyphs == 35) | (observed_glyphs == 36))] = 16.0  # Corridor
   # observed_glyphs = observation['glyphs']
    cropper = Crop(height=observed_glyphs.shape[0], width=observed_glyphs.shape[1], height_target=10, width_target=10)
    cropped_glyphs = cropper.crop(torch.from_numpy(observed_glyphs).unsqueeze(0), (args.crop_dims, args.crop_dims)) / 16.0
    #observed_glyphs[np.where(observed_glyphs == 36)] = 32 # Gold
    stat_x_coord = observation['blstats'][STATS_INDICES['x_coordinate']]
    stat_y_coord = observation['blstats'][STATS_INDICES['y_coordinate']]
    stat_health = float(observation['blstats'][STATS_INDICES['health_points']]) / \
                  float(observation['blstats'][STATS_INDICES['health_points_max']])
    #stat_hunger = observation['blstats'][STATS_INDICES['hunger_level']]
    observed_stats = np.array([stat_x_coord, stat_y_coord, stat_health,]) #stat_hunger])

    return cropped_glyphs.float(), observed_stats.astype(np.float32)

if __name__ == '__main__':
    args = parse_args()
    env = make_env(args, 0, ACTIONS)
    actor_critic = ActorCritic(env.observation_space['chars'].shape, env.action_space.n, args.enable_lstm)
    actor_critic.load_state_dict(torch.load(args.storage_path + 'a2c_model.pt'))
    for seed in args.seeds:
        env = make_env(args, seed, ACTIONS)
        observation = env.reset()
        cx = Variable(torch.zeros(1, 256))
        hx = Variable(torch.zeros(1, 256))
        done = False
        rewards = []
        while not done:
            observed_glyphs, observed_stats = transform_observation(observation, args)
            with torch.no_grad():
                actor, value, hx, cx = actor_critic(observed_glyphs, hx, cx, args.enable_lstm)
            action = actor.sample()
            next_observation, reward, done, infos = env.step(action.unsqueeze(1))
            observation = next_observation
            rewards.append(reward)
        print("Seed {0}, Score : {1}".format(seed, np.sum(rewards)))


