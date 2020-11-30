import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", 'matplotlib'])

import numpy as np
import gym
import nle
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import matplotlib.pyplot as plt
import random
import pickle
from collections import deque

device = torch.device("cpu")


STATS_INDICES = {
    'x_coordinate': 0,
    'y_coordinate': 1,
    'score': 9,
    'health_points': 10,
    'health_points_max': 11,
    'hunger_level': 18,
}

def crop_glyphs(glyphs, x, y, size=5):
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
    return crop.flatten()


def transform_observation(observation):
    """Process the state into the model input shape
    of ([glyphs, stats], )"""
    observed_glyphs = observation['glyphs']

    stat_x_coord = observation['blstats'][STATS_INDICES['x_coordinate']]
    stat_y_coord = observation['blstats'][STATS_INDICES['y_coordinate']]
    stat_health = float(observation['blstats'][STATS_INDICES['health_points']]) - float(observation['blstats'][STATS_INDICES['health_points_max']]/2)
    stat_hunger = observation['blstats'][STATS_INDICES['hunger_level']]
    observed_stats = np.array([stat_x_coord, stat_y_coord, stat_health, stat_hunger])

    flat_glyphs = crop_glyphs(observed_glyphs, stat_x_coord, stat_y_coord)

    final_obs = np.concatenate((observed_stats, flat_glyphs)).astype(np.float32)
    return final_obs


class SimplePolicy(nn.Module):
    def __init__(self, s_size=4, h1_size=16, h2_size=16, a_size=2):
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(s_size, h1_size),
                                    nn.ReLU(),
                                    nn.Linear(h1_size, h2_size),
                                    nn.ReLU(),
                                    nn.Linear(h2_size, a_size)).to(device)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return tnf.softmax(self.policy(x), dim=0)


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def compute_returns(rewards, gamma):
    returns = sum([(gamma**i) * reward for i, reward in enumerate(rewards)])
    return returns


def reinforce(env, policy_model, seed, learning_rate,
              number_episodes,
              max_episode_length,
              gamma, verbose=True):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    env.seed(seed)
    
    scores = []
    losses = []
    for episode in range(number_episodes):
        
        rewards = []
        log_probs = []
        state = transform_observation(env.reset())
        done = False
        # for step in range(max_episode_length):
        while not done:
            action_probs = policy_model(torch.from_numpy(state).float().to(device))
            action_sampler = torch.distributions.Categorical(action_probs)
            action = action_sampler.sample()
            log_probs.append(action_sampler.log_prob(action))
            new_state, reward, done, info = env.step(action.item())
            rewards.append(reward)
            state = transform_observation(new_state)
            # if done:
            #     break
            
        scores.append(sum(rewards))
        returns = compute_returns(rewards, gamma)
        loss = -1 * (returns * np.sum(np.array(log_probs)))
        losses.append(loss)
        policy_model.optimizer.zero_grad()
        loss.backward()
        policy_model.optimizer.step()
        
        window = 1
        if verbose and episode % window == 0 and episode != 0:
            print("Episode " + str(episode) + "/" + str(number_episodes) +
                  " Score: " + str(np.mean(scores[episode - window: episode])))
                                   
    policy = policy_model.parameters()
                                   
    return policy_model, scores, losses


# def compute_returns_naive_baseline(rewards, gamma):
#     r = 0
#     returns = []
#     for step in reversed(range(len(rewards))):
#         r = rewards[step] + gamma * r
#         returns.insert(0, r)
#     returns = np.array(returns)
#     returns = returns - returns.mean()
#     returns = returns / returns.std()
#     returns = torch.from_numpy(returns).float().to(device)
#     return returns
#
#
# def reinforce_naive_baseline(env, policy_model, seed, learning_rate,
#                              number_episodes,
#                              max_episode_length,
#                              gamma, verbose=True):
#     # set random seeds (for reproducibility)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     env.seed(seed)
#
#     scores = []
#     for episode in range(number_episodes):
#
#         rewards = []
#         log_probs = []
#         state = env.reset()['glyphs'][:,0]
#         for step in range(max_episode_length):
#
#             action_probs = policy_model(torch.from_numpy(state).float().to(device))
#             action_sampler = torch.distributions.Categorical(action_probs)
#             action = action_sampler.sample()
#             log_probs.append(action_sampler.log_prob(action).unsqueeze(0))
#             new_state, reward, done, info = env.step(action.item())
#             env.render()
#             rewards.append(reward)
#             state = new_state['glyphs'][:,0]
#             if done:
#                 break
#
#         scores.append(sum(rewards))
#         returns = compute_returns_naive_baseline(rewards, gamma)
#         log_probs = torch.cat(log_probs)
#         loss = -1 * torch.sum(returns * log_probs)
#         policy_model.optimizer.zero_grad()
#         loss.backward()
#         policy_model.optimizer.step()
#
#         window = 50
#         if verbose and episode % window == 0 and episode != 0:
#             print("Episode " + str(episode) + "/" + str(number_episodes) +
#                   " Score: " + str(np.mean(scores[episode - window: episode])))
#
#     policy = policy_model.parameters()
#     return policy, scores

def run_reinforce():
    env = gym.make("NetHackScore-v0")
    print("Run REINFORCE.")
    obs_space = transform_observation(env.reset())
    obs_space_size = obs_space.shape[0]
    policy_model = SimplePolicy(s_size=obs_space_size,
                                h1_size=80,
                                h2_size=40,
                                a_size=env.action_space.n)
    policy, scores, losses = reinforce(env=env,
                                       policy_model=policy_model,
                                       seed=42,
                                       learning_rate=1e-2,
                                       number_episodes=10,
                                       max_episode_length=5000,
                                       gamma=1.0,
                                       verbose=True)
    with open('/opt/project/objs.pkl', 'wb') as f:
        pickle.dump([policy_model, scores, losses], f)

    moving_avg = moving_average(scores, 50)
    
    # Plot learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(scores)
    plt.plot(moving_avg, '--')
    plt.legend(['Score', 'Moving Average (w=50)'])
    plt.title("REINFORCE learning curve")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig('/opt/project/Question 1.4.png')

# def investigate_variance_in_reinforce():
#     env = gym.make("NetHackScore-v0")
#     seeds = np.random.randint(1000, size=5)
#
#     print("Average REINFORCE over 5 runs.")
#     run_scores = []
#     for i, seed in enumerate(seeds):
#
#         policy_model = SimplePolicy(s_size=env.observation_space['glyphs'].shape[0],
#                                     h_size=50,
#                                     a_size=env.action_space.n)
#         print("Running REINFORCE for run", i+1, "of 5.")
#         policy, scores = reinforce(env=env,
#                                policy_model=policy_model,
#                                seed=42,
#                                learning_rate=1e-2,
#                                number_episodes=1000,
#                                max_episode_length=1000,
#                                gamma=1.0,
#                                verbose=True)
#         run_scores.append(moving_average(scores, 50))
#
#     mean = np.mean(run_scores, axis=0)
#     std = np.std(run_scores, axis=0)
#
#     # Plot learning curve with one standard deviation bounds
#     x_fill = np.arange(len(mean))
#     y_upper = mean + std
#     y_lower = mean - std
#     # plt.figure(figsize=(12, 6))
#     # plt.plot(mean)
#     # plt.fill_between(x_fill, y_upper, y_lower, alpha=0.2)
#     # plt.title("REINFORCE averaged over 5 seeds")
#     # plt.xlabel("Episode")
#     # plt.ylabel("Score")
#     # plt.savefig('Question 1.5.png')
#     return mean, std, x_fill, y_upper, y_lower


# def run_reinforce_with_naive_baseline(mean, std, x_fill, y_upper, y_lower):
#     env = gym.make("NetHackScore-v0")
#
#     print("Run REINFORCE with naive baseline.")
#     policy_model = SimplePolicy(s_size=env.observation_space['glyphs'].shape[0],
#                                 h_size=50,
#                                 a_size=env.action_space.n)
#     policy, scores = reinforce_naive_baseline(env=env,
#                                               policy_model=policy_model,
#                                               seed=42,
#                                               learning_rate=1e-2,
#                                               number_episodes=1500,
#                                               max_episode_length=1000,
#                                               gamma=1.0,
#                                               verbose=True)
#     moving_avg = moving_average(scores, 50)
#
#     # Plot learning curve
#     # plt.figure(figsize=(12, 6))
#     # plt.plot(scores)
#     # plt.plot(moving_avg, '--')
#     # plt.legend(['Score', 'Moving Average (w=50)'])
#     # plt.title("REINFORCE with naive baseline")
#     # plt.xlabel("Episode")
#     # plt.ylabel("Score")
#     # plt.savefig('Question 2.3.1.png')
#
#     np.random.seed(53)
#     seeds = np.random.randint(1000, size=5)
#
#     print("Average REINFORCE over 5 runs.")
#     run_scores = []
#     for i, seed in enumerate(seeds):
#
#         policy_model = SimplePolicy(s_size=env.observation_space['glyphs'].shape[0],
#                                     h_size=50,
#                                     a_size=env.action_space.n)
#         print("Running REINFORCE with naive baseline for run", i+1, "of 5.")
#         policy, scores = reinforce_naive_baseline(env=env,
#                                                   policy_model=policy_model,
#                                                   seed=42,
#                                                   learning_rate=1e-2,
#                                                   number_episodes=1500,
#                                                   max_episode_length=1000,
#                                                   gamma=1.0,
#                                                   verbose=True)
#         run_scores.append(moving_average(scores, 50))
#
#     mean_b = np.mean(run_scores, axis=0)
#     std_b = np.std(run_scores, axis=0)
#
#     # Plot learning curve with one standard deviation bounds
#     x_fill_b = np.arange(len(mean_b))
#     y_upper_b = mean_b + std_b
#     y_lower_b = mean_b - std_b
#     # plt.figure(figsize=(12, 6))
#     # plt.plot(mean)
#     # plt.fill_between(x_fill, y_upper, y_lower, alpha=0.2)
#     # plt.plot(mean_b)
#     # plt.fill_between(x_fill_b, y_upper_b, y_lower_b, alpha=0.2)
#     # plt.title("REINFORCE and REINFORCE with naive baseline averaged over 5 seeds")
#     # plt.xlabel("Episode")
#     # plt.ylabel("Score")
#     # plt.legend(['REINFORCE', 'REINFORCE with naive baseline'])
#     # plt.savefig('Question 2.3.2.png')


if __name__ == '__main__':
    run_reinforce()
    # mean, std, x_fill, y_upper, y_lower = investigate_variance_in_reinforce()
    # run_reinforce_with_naive_baseline(mean, std, x_fill, y_upper, y_lower)
