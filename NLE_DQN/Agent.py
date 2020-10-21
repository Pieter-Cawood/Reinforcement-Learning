import gym
import numpy as np
import nle
import torch
import argparse
import random
from torch.nn import functional as F
from torch import nn
from nle import nethack

from DQNModel import DQNModel
from replay_buffer import ReplayBuffer


parser = argparse.ArgumentParser(description="PyTorch Agent")
parser.add_argument("--env", type=str, default="NetHackScore-v0",
                    help="Gym environment.")
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--learning_rate", default=0.00048,
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--discount_factor", default=0.99,
                    type=float, help="The discount factor.")
parser.add_argument("--device", default="cpu",
                    type=str, help="Torch device")
parser.add_argument("--total_steps", default=10000000, type=int,
                    help="Total number of steps to train over")
parser.add_argument("--batch_size", default=32, type=int,
                    help="Number of transitions to optimize at the same time.")
parser.add_argument("--eps_start", default=1.0, type=float,
                    help="e-greedy start threshold.")
parser.add_argument("--eps_end", default=0.1, type=float,
                    help="e-greedy end threshold.")
parser.add_argument("--eps_frac", default=0.1, type=float,
                    help="e-greedy fraction of num-steps.")
parser.add_argument("--target_update_freq", default=500, type=float,
                    help="number of iterations between every target network update.")
parser.add_argument("--print_freq", default=10, type=float,
                    help="number of episodes before print.")

STATS_DIM = 4  # Only including x coordinate, y coordinate, HP percentage and hunger level

STATS_INDICES = {
                    'x_coordinate' : 0,
                    'y_coordinate' : 1,
                    'score' : 9,
                    'health_points' : 10,
                    'health_points_max' : 11,
                    'hunger_level' : 18,
                }

ACTIONS = [
            nethack.CompassCardinalDirection.N,
            nethack.CompassCardinalDirection.E,
            nethack.CompassCardinalDirection.S,
            nethack.CompassCardinalDirection.W,
            nethack.CompassIntercardinalDirection.NE,
            nethack.CompassIntercardinalDirection.SE,
            nethack.CompassIntercardinalDirection.SW,
            nethack.CompassIntercardinalDirection.NW,
            nethack.MiscDirection.UP,
            nethack.MiscDirection.DOWN,
            nethack.MiscDirection.WAIT,
            nethack.Command.KICK,
            nethack.Command.EAT,
            nethack.Command.SEARCH
          ]


def transform_observation(observation):
    """Process the state into the model input shape
    of ([glyphs, stats], )"""
    observed_glyphs = observation['glyphs']

    stat_x_coord = observation['blstats'][STATS_INDICES['x_coordinate']]
    stat_y_coord = observation['blstats'][STATS_INDICES['y_coordinate']]
    stat_health = float(observation['blstats'][STATS_INDICES['health_points']]) /\
                  float(observation['blstats'][STATS_INDICES['health_points_max']])
    stat_hunger = observation['blstats'][STATS_INDICES['hunger_level']]
    observed_stats = np.array([stat_x_coord, stat_y_coord, stat_health, stat_hunger])
    return observed_glyphs.astype(np.float32), observed_stats.astype(np.float32)

class Agent():
    def __init__(self, flags):
        self.flags = flags
        self.env = gym.make(flags.env, actions=ACTIONS)
        self.policy_network  = DQNModel(self.env.observation_space["glyphs"].shape,
                                        STATS_DIM,
                                        self.env.action_space.n)
        self.target_network  = DQNModel(self.env.observation_space["glyphs"].shape,
                                        STATS_DIM,
                                        self.env.action_space.n)

        self.buffer_memory = ReplayBuffer(self.flags.batch_size)

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr= self.flags.learning_rate)



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
            Q_val_next_max = self.target_network(glyphs_states_, stats_states_).\
                gather(1, action_next_max.unsqueeze(1)).squeeze()

        # Set y_j for each mini-batch entry,
        # If terminal then Q = rewards only
        Q_target = rewards + (1 - dones) * self.flags.discount_factor * Q_val_next_max
        # Recompute gradients and get values
        Q_current = self.policy_network(glyphs_states, stats_states).\
            gather(1, actions.unsqueeze(1)).squeeze()

        # Perform a gradient descent step on (y_j - Q)^2 ( loss )
        # Before backward pass, use optimizer to zero all the gradients of the tensors
        # it has to update
        # Compute the Hubert loss
        loss = F.smooth_l1_loss(Q_current, Q_target)
        # Backward pass: compute gradients
        self.optimizer.zero_grad()
        loss.backward()
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
        episode_scores = [0.0]
        average_rewards_epochs = []
        losses = []
        current_state = self.env.reset()
        for time_step in range(flags.total_steps):
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

            # Sum step reward for episode's total
            episode_rewards[-1] += reward

            # Game-over
            if done:
                episode_scores[-1] = current_state['blstats'][STATS_INDICES['score']]
                current_state = self.env.reset()
                episode_rewards.append(0.0)
                episode_scores.append(0.0)
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

            # Optimise TD loss
            if time_step > self.flags.batch_size:
                loss = self.optimize_td_loss()
                losses.append(loss)

            # Target network every specified run
            if time_step > self.flags.batch_size and\
               time_step % self.flags.target_update_freq == 0:
                self.update_target_network()
            if done:
                num_episodes = len(episode_rewards)
                mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
                average_rewards.append(round(np.mean(episode_rewards[-self.flags.print_freq - 1:-1]), 1))
                average_rewards_epochs.append(num_episodes)
                print("********************************************************")
                print("steps: {}".format(time_step))
                print("episodes: {}".format(num_episodes))
                print("mean 100 episode reward: {}".format(mean_100ep_reward))
                print("% time spent exploring: {}".format(int(100 * eps_threshold)))
                print("********************************************************")


def main(flags):
    agent = Agent(flags)

    agent.train()

    # Training


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
