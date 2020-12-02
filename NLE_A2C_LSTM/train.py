"""
Modified the A2C method from https://github.com/raillab/a2c

"""

import argparse
import numpy as np
import gym
import torch.nn.functional as F
from nle import nethack
import time
from model import ActorCritic
import torch
import torch.optim as optim
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from gym.envs import registration

registration.register(id="NetHackGoldRunner-v0", entry_point="nle_goldrunner:NetHackGoldRunner")

def parse_args():
    parser = argparse.ArgumentParser("A2C experiments for Atari games")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # Environment
    parser.add_argument("--env", type=str, default="NetHackGoldRunner-v0", help="name of the game")
    parser.add_argument("--storage-path", type=str, default="/opt/project/",
                        help="System storage path")
    parser.add_argument("--storage-freq", type=int, default=100,
                        help="Number of episodes to save weights.")
    parser.add_argument("--enable-lstm", type=bool, default=True,
                        help="Enable recurrent cell.")
    parser.add_argument("--load-pretrained", type=bool, default=True,
                        help="Continue training on stored weights")
    parser.add_argument("--crop_dims", type=int, default=10, help="name of the game")
    # Core A2C parameters
    parser.add_argument("--actor-loss-coefficient", type=float, default=1.0, help="actor loss coefficient")
    parser.add_argument("--critic-loss-coefficient", type=float, default=0.5, help="critic loss coefficient")
    parser.add_argument("--entropy-loss-coefficient", type=float, default=0.01, help="entropy loss coefficient")
    parser.add_argument("--lr", type=float, default=7e-4, help="learning rate for the RMSprop optimizer")
    parser.add_argument("--alpha", type=float, default=0.99, help="alpha term the RMSprop optimizer")
    parser.add_argument("--eps", type=float, default=0.1, help="eps term for the RMSprop optimizer")
    parser.add_argument("--max-grad-norm", type=float, default=40, help="maximum norm of gradients")
    parser.add_argument("--num_steps", type=int, default=500, help="maximum steps for episode")
    parser.add_argument("--num-episodes", type=int, default=50000, help="number of episodes to train the model")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--num-frames", type=int, default=int(10e6),
                        help="total number of steps to run the environment for")
    parser.add_argument("--log-dir", type=str, default="logs", help="where to save log files")
    parser.add_argument("--save-freq", type=int, default=0, help="updates between saving models (default 0 => no save)")
    # Reporting
    parser.add_argument("--print-freq", type=int, default=1000, help="evaluation frequency.")
    return parser.parse_args()

STATS_INDICES = {
    'x_coordinate': 0,
    'y_coordinate': 1,
    'score': 9,
    'health_points': 10,
    'health_points_max': 11,
    'hunger_level': 18,
}

ACTIONS = [
    nethack.CompassCardinalDirection.N,
    nethack.CompassCardinalDirection.E,
    nethack.CompassCardinalDirection.S,
    nethack.CompassCardinalDirection.W
]

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

def compute_returns(next_value, rewards, masks, gamma):
    r = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        r = rewards[step] + gamma * r * masks[step]
        returns.insert(0, r)
    return returns


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
                             (observed_glyphs != 35) & (observed_glyphs != 43))] = 0.0
    observed_glyphs[np.where((observed_glyphs == 45) | (observed_glyphs == 43) |
                             (observed_glyphs == 124))] = 8.0 # Walls & Door
    observed_glyphs[np.where((observed_glyphs == 35) | (observed_glyphs == 36))] = 16.0 # Corridor & Gold
    #observed_glyphs = observation['glyphs']
    cropper = Crop(height=observed_glyphs.shape[0], width=observed_glyphs.shape[1], height_target=10, width_target=10)
    cropped_glyphs = cropper.crop(torch.from_numpy(observed_glyphs).unsqueeze(0), (args.crop_dims, args.crop_dims)) /16.0
    stat_x_coord = observation['blstats'][STATS_INDICES['x_coordinate']]
    stat_y_coord = observation['blstats'][STATS_INDICES['y_coordinate']]
    stat_health = float(observation['blstats'][STATS_INDICES['health_points']])
    #stat_hunger = observation['blstats'][STATS_INDICES['hunger_level']]
    observed_stats = np.array([stat_x_coord, stat_y_coord, stat_health,]) #stat_hunger])

    return cropped_glyphs.float(), observed_stats.astype(np.float32)


if __name__ == '__main__':
    args = parse_args()

    env = make_env(args, 0, actions=ACTIONS)
    actor_critic = ActorCritic(env.observation_space['chars'].shape, env.action_space.n, args.enable_lstm).to(device)

    if args.load_pretrained:
        actor_critic.load_state_dict(torch.load(args.storage_path + 'a2c_model.pt'))

    optimizer = optim.RMSprop(actor_critic.parameters(), lr=args.lr, eps=args.eps, alpha=args.alpha)
    num_episodes = args.num_episodes

    for episode_n in range(num_episodes):

        log_probs = []
        values = []
        rewards = []
        actions = []
        masks = []
        entropies = []
        cx = Variable(torch.zeros(1, 256))
        hx = Variable(torch.zeros(1, 256))
        seed = episode_n % 1000   # Seeds up to 1000 (same as paper)
        env = make_env(args, seed, actions=ACTIONS)
        observation = env.reset()

        for step in range(args.num_steps):
            observed_glyphs, observed_stats = transform_observation(observation, args)
            actor, value, hx, cx = actor_critic(observed_glyphs, hx, cx, args.enable_lstm)

            action = actor.sample()
            next_observation, reward, done, infos = env.step(action.unsqueeze(1))

            log_prob = actor.log_prob(action)
            entropy = actor.entropy()

            mask = 1.0 - done

            entropies.append(actor.entropy())
            log_probs.append(log_prob)
            values.append(value.squeeze())
            rewards.append(reward)
            masks.append(mask)

            if done:
                break
            else:
                observation = next_observation

        with torch.no_grad():
            observed_glyphs, observed_stats = transform_observation(next_observation, args)
            _, next_values, _, _ = actor_critic(observed_glyphs, hx, cx, args.enable_lstm)
            returns = compute_returns(next_values.squeeze(), rewards, masks, args.gamma)
            returns = torch.FloatTensor(returns)

        log_probs = torch.cat(log_probs)
        values = torch.FloatTensor(values)
        entropies = torch.cat(entropies)

        advantages = returns - values

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = entropies.mean()

        loss = args.actor_loss_coefficient * actor_loss + \
               args.critic_loss_coefficient * critic_loss - \
               args.entropy_loss_coefficient * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), args.max_grad_norm)
        optimizer.step()

        if len(rewards) > 1 :
            end = time.time()
            total_num_steps = (episode_n + 1) * args.num_episodes * args.num_steps
            print("********************************************************")
            print("Episode: {0}, total steps: {1}".format(episode_n, total_num_steps))
            print("Episode rewards: {:.1f}".format(np.sum(rewards)))
            print("Actor loss: {:.5f}, Critic loss: {:.5f}, Entropy: {:.5f}".format(actor_loss.item(), critic_loss.item(), entropy_loss.item()))
            print("********************************************************")
        if episode_n % args.storage_freq == 0:
            torch.save(actor_critic.state_dict(),
                       args.storage_path + 'a2c_model.pt')