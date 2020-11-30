import numpy as np
import argparse
import torch
import gym
from copy import deepcopy
#from nle_goldrunner import *
from option_critic import OptionCriticFeatures, OptionCriticConv
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn
from nle import nethack

from experience_replay import ReplayBuffer
from logger import Logger

import time

#registration.register(id="NetHackGoldRunner-v0", entry_point="nle_goldrunner:NetHackGoldRunner")

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
parser.add_argument('--env', default='NetHackScore-v0', help='ROM to run')
parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
parser.add_argument('--frame-skip', default=4, type=int, help='Every how many frames to process')
parser.add_argument('--learning-rate', type=float, default=.0005, help='Learning rate')
parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
parser.add_argument('--epsilon-start', type=float, default=1.0, help=('Starting value for epsilon.'))
parser.add_argument('--epsilon-min', type=float, default=.1, help='Minimum epsilon.')
parser.add_argument('--epsilon-decay', type=float, default=20000, help=('Number of steps to minimum epsilon.'))
parser.add_argument('--max-history', type=int, default=10000, help=('Maximum number of steps stored in replay'))
parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
parser.add_argument('--freeze-interval', type=int, default=200, help=('Interval between target freezes.'))
parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update.'))
parser.add_argument('--termination-reg', type=float, default=0.01,
                    help=('Regularization to decrease termination prob.'))
parser.add_argument('--entropy-reg', type=float, default=0.01, help=('Regularization to increase policy entropy.'))
parser.add_argument('--num-options', type=int, default=3, help=('Number of options to create.'))
parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param.')

parser.add_argument('--max_steps_ep', type=int, default=18000, help='number of maximum steps per episode.')
parser.add_argument('--max_steps_total', type=int, default=int(4e6),
                    help='number of maximum steps to take.')  # bout 4 million
parser.add_argument('--cuda', type=bool, default=False, help='Enable CUDA training (recommended if possible).')
parser.add_argument('--seed', type=int, default=0, help='Random seed for numpy, torch, random.')
parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')
parser.add_argument('--exp', type=str, default=None, help='optional experiment name')
parser.add_argument('--switch-goal', type=bool, default=False, help='switch goal after 2k eps')

ACTIONS = [
    nethack.CompassCardinalDirection.N,
    nethack.CompassCardinalDirection.E,
    nethack.CompassCardinalDirection.S,
    nethack.CompassCardinalDirection.W,
    # nethack.CompassIntercardinalDirection.NE,
    # nethack.CompassIntercardinalDirection.SE,
    #  nethack.CompassIntercardinalDirection.SW,
    #nethack.CompassIntercardinalDirection.NW,
    #nethack.MiscDirection.UP,
    #nethack.MiscDirection.DOWN,
    #nethack.MiscDirection.WAIT,
    #nethack.CompassIntercardinalDirection.NW # Yes answer
    nethack.Command.KICK,
    #nethack.Command.EAT,
    nethack.Command.SEARCH
]

STATS_INDICES = {
    'x_coordinate': 0,
    'y_coordinate': 1,
    'score': 9,
    'health_points': 10,
    'health_points_max': 11,
    'hunger_level': 18,
}

def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs

def crop_glyphs(glyphs, x, y, size=10):
    x_max = 79
    y_max = 21
    x_diff = max(size - (x_max - x) + 1, 0)
    y_diff = max(size - (y_max - y) + 1, 0)
    x_start = max(x - size - x_diff, 0)
    y_start = max(y - size - y_diff, 0)
    x_diff_s = max(size - x, 0)
    y_diff_s = max(size - y, 0)
    x_end = min(x + size - x_diff + x_diff_s, x_max) + 1
    y_end = min(y + size - y_diff + y_diff_s, y_max) + 1
    crop = glyphs[y_start:y_end, x_start:x_end]
    return crop


def run(args):
    env = gym.make(args.env, actions=ACTIONS)
    option_critic = OptionCriticFeatures
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    option_critic = option_critic(
        in_features= 1, # 1 channel env.observation_space.shape[0],
        num_actions=env.action_space.n,
        num_options=args.num_options,
        temperature=args.temp,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        eps_test=args.optimal_eps,
        device=device
    )
    # Create a prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)

    optim = torch.optim.RMSprop(option_critic.parameters(), lr=args.learning_rate)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
    logger = Logger(logdir=args.logdir,
                   run_name=f"{OptionCriticFeatures.__name__}-{args.env}-{args.exp}-{time.ctime()}")

    steps = 0;
    if args.switch_goal: print(f"Current goal {env.goal}")
    while steps < args.max_steps_total:

        rewards = 0;
        option_lengths = {opt: [] for opt in range(args.num_options)}

        obs = env.reset()
        stat_x_coord = obs['blstats'][STATS_INDICES['x_coordinate']]
        stat_y_coord = obs['blstats'][STATS_INDICES['y_coordinate']]
        state = option_critic.get_state(to_tensor(crop_glyphs(obs['chars'],stat_x_coord,stat_y_coord)))
        greedy_option = option_critic.greedy_option(state)
        current_option = 0

        # Goal switching experiment: run for 1k episodes in fourrooms, switch goals and run for another
        # 2k episodes. In option-critic, if the options have some meaning, only the policy-over-options
        # should be finedtuned (this is what we would hope).
        if args.switch_goal and logger.n_eps == 1000:
            torch.save({'model_params': option_critic.state_dict(),
                        'goal_state': env.goal},
                       'models/option_critic_{args.seed}_1k')
            env.switch_goal()
            print(f"New goal {env.goal}")

        if args.switch_goal and logger.n_eps > 2000:
            torch.save({'model_params': option_critic.state_dict(),
                        'goal_state': env.goal},
                       'models/option_critic_{args.seed}_2k')
            break

        done = False;
        ep_steps = 0;
        option_termination = True;
        curr_op_len = 0
        while not done and ep_steps < args.max_steps_ep:
            epsilon = option_critic.epsilon

            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option
                curr_op_len = 0

            action, logp, entropy = option_critic.get_action(state, current_option)

            next_obs, reward, done, _ = env.step(action)
            buffer.push(obs['chars'], current_option, reward, next_obs['chars'], done)

            old_state = state
            state = option_critic.get_state(to_tensor(next_obs['chars']))

            option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)
            rewards += reward

            actor_loss, critic_loss = None, None
            if len(buffer) > args.batch_size:
                actor_loss = actor_loss_fn(obs, current_option, logp, entropy, \
                                           reward, done, next_obs, option_critic, option_critic_prime, args)
                loss = actor_loss

                if steps % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, args)
                    loss += critic_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                if steps % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            # update global steps etc
            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs

            logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)

        logger.log_episode(steps, rewards, option_lengths, ep_steps, epsilon)


if __name__ == "__main__":
    args = parser.parse_args()
    run(args)
