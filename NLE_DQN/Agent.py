"""
Modified the DQN method from https://github.com/raillab/dqn

"""

import gym
from gym.envs import registration
from nle import nethack
import numpy as np
import nle
import torch
import argparse
import random
import pickle
from torch.nn import functional as F
from torch import nn
from torch.autograd import Variable

from DQNModel import DQNModel
from replay_buffer import ReplayBuffer

registration.register(id="NetHackGoldRunner-v0", entry_point="nle_goldrunner:NetHackGoldRunner")

parser = argparse.ArgumentParser(description="DQN Agent")
parser.add_argument("--path", type=str, default="/opt/project/",
                    help="System storage path")
parser.add_argument("--test_seeds", type=list, default=[1,2,3,4,5], help="which seed to evaluate trained model.")
parser.add_argument("--load", type=bool, default=False,
                    help="Load weights from file")
parser.add_argument("--total_steps", default=3000000, type=int,
                    help="Total number of steps to train over")
parser.add_argument("--env", type=str, default="NetHackGoldRunner-v0",
                    help="Gym environment.")
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--learning_rate", default=0.02,
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--discount_factor", default=0.99,
                    type=float, help="The discount factor.")
parser.add_argument("--device", default="cpu",
                    type=str, help="Torch device")
parser.add_argument("--batch_size", default=32, type=int,
                    help="Number of transitions to optimize at the same time.")
parser.add_argument("--eps_start", default=1.0, type=float,
                    help="e-greedy start threshold.")
parser.add_argument("--eps_end", default=0.1, type=float,
                    help="e-greedy end threshold.")
parser.add_argument("--eps_frac", default=0.1, type=float,
                    help="e-greedy fraction of num-steps.")
parser.add_argument("--target_update_freq", default=1000, type=float,
                    help="number of iterations between every target network update.")
parser.add_argument("--print_freq", default=10, type=float,
                    help="number of episodes before print.")
parser.add_argument("--grad_norm_clipping", default=10.0, type=float,
                    help="Global gradient norm clip.")

STATS_DIM = 3  # Only including x coordinate, y coordinate, HP percentage and hunger level (REMOVED HUNGER)
               # x & y coordinates are only used to crop the window

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


def transform_observation(observation):
    """Process the state into the model input shape
    of ([glyphs, stats], )"""
    observed_glyphs = observation['chars']
    observed_glyphs[np.where((observed_glyphs != 45) & (observed_glyphs != 124) &
                             (observed_glyphs != 35) & (observed_glyphs != 43) &
                             (observed_glyphs != 36))] = 0.0
    observed_glyphs[np.where((observed_glyphs == 45) | (observed_glyphs == 43) |
                             (observed_glyphs == 124))] = 8.0 / 16.0# Walls & Door
    observed_glyphs[np.where((observed_glyphs == 35) | (observed_glyphs == 36))] = 16.0 / 16.0# Corridor

    stat_x_coord = observation['blstats'][STATS_INDICES['x_coordinate']]
    stat_y_coord = observation['blstats'][STATS_INDICES['y_coordinate']]
    stat_health = float(observation['blstats'][STATS_INDICES['health_points']])
    #stat_hunger = observation['blstats'][STATS_INDICES['hunger_level']]
    observed_stats = np.array([stat_x_coord, stat_y_coord, stat_health,]) #stat_hunger])

    return observed_glyphs, observed_stats.astype(np.float32)

def make_env(env, seed, actions):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    env = gym.make(env, actions=actions)
    env.seed(seed)
    return env

class Agent():
    def __init__(self, flags):
        self.flags = flags
        self.env = gym.make(flags.env, actions=ACTIONS)
        self.policy_network = DQNModel(self.env.observation_space["glyphs"].shape,
                                       STATS_DIM,
                                       self.env.action_space.n)

        self.target_network = DQNModel(self.env.observation_space["glyphs"].shape,
                                       STATS_DIM,
                                       self.env.action_space.n)

        self.buffer_memory = ReplayBuffer(self.flags.batch_size)

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.flags.learning_rate)
        self.start_step = 0

        # Load model weights from file system
        if self.flags.load or self.flags.mode != "train":
            self.policy_network.load_state_dict(torch.load(
                self.flags.path + 'policy.pt'))
            self.optimizer.load_state_dict(torch.load(
                self.flags.path + 'optimizer.pt'))
            self.update_target_network()
            try:
                with open(self.flags.path + 'last_step.pickle', 'rb') as pickle_file:
                    self.start_step = pickle.load(pickle_file)
                    if self.flags.mode == "train":
                        print('Continuing from step ', self.start_step)
            except:
                print('Error loading start step, starting from 0')
                self.start_step = 0

        # TODO Initialise your agent's models

        # for example, if your agent had a Pytorch model it must be load here
        # model.load_state_dict(torch.load( 'path_to_network_model_file', map_location=torch.device(device)))

    def optimize_td_loss(self):
        """
        Optimise the TD-error over a minibatch of transitions
        :return: the loss
        """
        glyphs_states, stats_states, actions, rewards, glyphs_states_, stats_states_, dones = self.buffer_memory.sample()
        glyphs_states = torch.from_numpy(glyphs_states).float().to(self.flags.device)
        stats_states = torch.from_numpy(stats_states).float().to(self.flags.device)
        glyphs_states_ = torch.from_numpy(glyphs_states_).float().to(self.flags.device)
        stats_states_ = torch.from_numpy(stats_states_).float().to(self.flags.device)
        actions = torch.from_numpy(actions).long().to(self.flags.device)
        rewards = torch.from_numpy(rewards).float().to(self.flags.device)
        dones = torch.from_numpy(dones).float().to(self.flags.device)

        with torch.no_grad():
            Q_vals_next = self.policy_network(glyphs_states_, stats_states_)
            _, action_next_max = Q_vals_next.max(1)
            Q_val_next_max = self.target_network(glyphs_states_, stats_states_)
            Q_val_next_max = Q_val_next_max.gather(1, action_next_max.unsqueeze(1)).squeeze()

        # Set y_j for each mini-batch entry,
        # If terminal then Q = rewards only
        Q_target = rewards + (1 - dones) * self.flags.discount_factor * Q_val_next_max
        # Recompute gradients and get values
        Q_current = self.policy_network(glyphs_states, stats_states)
        Q_current = Q_current.gather(1, actions.unsqueeze(1)).squeeze()


        # Perform a gradient descent step on (y_j - Q)^2 ( loss )
        # Before backward pass, use optimizer to zero all the gradients of the tensors
        # it has to update
        # Compute the Hubert loss
        loss = F.smooth_l1_loss(Q_current, Q_target)
        # Backward pass: compute gradients
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.flags.grad_norm_clipping)
        # Update optimizer parameters
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def _act(self, observed_glyphs, observed_stats):
        observed_glyphs = torch.from_numpy(observed_glyphs).float().unsqueeze(0).to(self.flags.device)
        observed_stats = torch.from_numpy(observed_stats).float().unsqueeze(0).to(self.flags.device)
        # Don't compute gradients, just get values
        with torch.no_grad():
            Q_vals = self.policy_network(observed_glyphs, observed_stats)
        Q_val_max, action_max = Q_vals.max(1)
        return action_max.item()

    def train(self):
        episode_rewards = [0.0]
        average_rewards = []
        high_score = -float('inf')
        losses = []
        current_state = self.env.reset()
        for time_step in range(self.start_step, self.flags.total_steps):
            observed_glyphs, observed_stats = transform_observation(current_state)
            # Annealing e-greedy
            eps_timesteps = self.flags.eps_frac * self.flags.total_steps
            fraction = min(1.0, float(time_step) / eps_timesteps)
            eps_threshold = self.flags.eps_start + fraction * (self.flags.eps_end - self.flags.eps_start)
            if random.random() <= eps_threshold:
                action = self.env.action_space.sample()
            else:
                action = self._act(observed_glyphs, observed_stats)

            # Take a leap of faith in the environment, store transition
            # info for mini-batch SGD
            state_, reward, done, info = self.env.step(action)

            # Add step reward
            episode_rewards[-1] += reward

            high_score = max(current_state['blstats'][STATS_INDICES['score']], high_score)

            # Game-over
            if done:
                current_state = self.env.reset()
                episode_rewards.append(-1.0)
            else:
                glyphs_states_, stats_states_ = transform_observation(state_)
                self.buffer_memory.add(observed_glyphs,
                                       observed_stats,
                                       action,
                                       reward,
                                       glyphs_states_,
                                       stats_states_,
                                       float(done))
                current_state = state_
                high_score = max(current_state['blstats'][STATS_INDICES['score']], high_score)
            # Optimise TD loss
            if time_step > self.flags.batch_size + self.start_step:
                #if len(self.buffer_memory) >= self.flags.batch_size:
                loss = self.optimize_td_loss()
                losses.append(loss)

            # Target network every specified run
            if time_step > self.flags.batch_size and \
                time_step % self.flags.target_update_freq == 0:
                self.update_target_network()
                torch.save(self.policy_network.state_dict(),
                            self.flags.path + 'policy.pt')
                torch.save(self.optimizer.state_dict(),
                            self.flags.path + 'optimizer.pt')
                with open(self.flags.path + 'last_step.pickle', 'wb') as f:
                    pickle.dump(time_step, f)
            if done:
                num_episodes = len(episode_rewards)
                mean_100ep_reward = round(np.mean(episode_rewards[-5:-1]), 1)
                average_rewards.append(round(np.mean(episode_rewards[-self.flags.print_freq - 1:-1]), 1))
                print("********************************************************")
                print("Average loss: {}".format(np.array(losses).mean()))
                print("steps: {}".format(time_step))
                print("episodes: {}".format(num_episodes))
                print("mean 5 episode reward: {}".format(mean_100ep_reward))
                print("current high score: {}".format(high_score))
                print("% time spent exploring: {}".format(int(100 * eps_threshold)))
                print("********************************************************")
                losses = []

    def test(self):
        for seed in self.flags.test_seeds:
            episode_rewards = [0.0]
            env = make_env('NetHackScore-v0', seed, ACTIONS)
            current_state = env.reset()
            done = False
            steps = 0
            while not done and steps < 5000:
                observed_glyphs, observed_stats = transform_observation(current_state)
                # Sample action from dqn
                action = self._act(observed_glyphs, observed_stats)
                # Take a leap of faith in the environment
                state_, reward, done, info = env.step(action)
                current_state = state_
                # Add step reward
                episode_rewards[-1] += reward
                # Game-over
                steps += 1
                if done:
                    current_state = env.reset()
                if self.flags.mode == "test_render":
                    env.render()
            print("Episode rewards {}".format(np.sum(episode_rewards)))


def main(flags):
    agent = Agent(flags)
    if flags.mode == "train":
        agent.train()
    else:
        agent.test()


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
