import numpy as np

import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimplePolicy(nn.Module):
    """Simple policy network"""

    def __init__(self, learning_rate=1e-2, s_size=4, h_size=16, a_size=2):
        super(SimplePolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(s_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, a_size)
        ) \
            .to(device)

        # Initialise linear layer weights
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)

    def forward(self, x: torch.Tensor):
        return F.softmax(self.model(x), dim=0)


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def compute_returns(rewards, gamma):
    """Compute's the return over all of the rewards"""
    returns = 0
    for k in range(len(rewards)):
        returns += gamma ** rewards[k]
    return returns


def compute_returns_naive_baseline(rewards, gamma):
    """Compute's the return over all of the rewards.
    Uses a Naive approach: Average of rewards as the constant baseline.
    """
    r = 0
    returns = []
    for step in reversed(range(len(rewards))):
        r = rewards[step] + gamma * r
        returns.insert(0, r)
    returns = np.array(returns)
    # Subtract mean as baseline
    returns = returns - returns.mean()
    # Normalize returns
    returns = returns / returns.std()
    return returns


def reinforce(env, policy_model, seed,
              number_episodes,
              max_episode_length,
              gamma,
              verbose=True,
              verbose_window_size=50
              ):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    scores = []
    for episode in range(number_episodes):
        rewards = []
        log_probs = []
        state = env.reset()

        """
        Sample the episode's trajectories
        """
        for step in range(max_episode_length):
            _state = torch.from_numpy(state).float().to(device)
            action_probs = policy_model(_state)
            # https://pytorch.org/docs/stable/distributions.html
            m = torch.distributions.Categorical(action_probs)
            # Choose action from prob dist
            action = m.sample()
            log_probs.append(m.log_prob(action))
            next_state, reward, done, info = env.step(action.item())
            rewards.append(reward)
            state = next_state
            # Terminal
            if done:
                break

        # Store to history
        scores.append(sum(rewards))

        """
        Get returns for full trajectory run, and update the parameters
        """
        returns = compute_returns(rewards, gamma)
        # Gradient ascent - negative of the loss function to maximize returns
        loss = -1 * (returns * np.sum(np.array(log_probs)))
        # Backward pass: compute gradients
        policy_model.optimizer.zero_grad()
        loss.backward()
        # Update optimizer parameters
        policy_model.optimizer.step()

        if verbose and episode % verbose_window_size == 0:
            print("Episode " + str(episode + 1) + "/" + str(number_episodes) +
                  " Score: " + str(sum(scores[episode - verbose_window_size: episode]) / verbose_window_size))

    return policy_model, scores


def reinforce_naive_baseline(env, policy_model, seed,
                             number_episodes,
                             max_episode_length,
                             gamma,
                             verbose=True,
                             verbose_window_size=50
                             ):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    scores = []
    for episode in range(number_episodes):
        rewards = []
        log_probs = []
        state = env.reset()

        """
        Sample the episode's trajectories
        """
        for step in range(max_episode_length):
            _state = torch.from_numpy(state).float().to(device)
            action_probs = policy_model(_state)
            # https://pytorch.org/docs/stable/distributions.html
            m = torch.distributions.Categorical(action_probs)
            # Choose action from prob dist
            action = m.sample()
            log_probs.append(m.log_prob(action).unsqueeze(0))
            next_state, reward, done, info = env.step(action.item())
            rewards.append(reward)
            state = next_state
            # Terminal
            if done:
                break

        # Store to history
        scores.append(sum(rewards))

        """
        Get returns for full trajectory run, and update the parameters
        """
        returns = compute_returns_naive_baseline(rewards, gamma)
        returns = torch.from_numpy(returns).float().to(device)
        log_probs = torch.cat(log_probs)
        # Gradient ascent - negative of the loss function to maximize returns
        loss = -1 * torch.sum(log_probs * returns)
        # Backward pass: compute gradients
        policy_model.optimizer.zero_grad()
        loss.backward()
        # Update optimizer parameters
        policy_model.optimizer.step()

        if verbose and episode % verbose_window_size == 0:
            print("Episode " + str(episode + 1) + "/" + str(number_episodes) +
                  " Score: " + str(sum(scores[episode - verbose_window_size: episode]) / verbose_window_size))

    return policy_model, scores


def run_reinforce():
    env = gym.make('CartPole-v1')

    policy_model = SimplePolicy()

    policy, scores = reinforce(env=env,
                               policy_model=policy_model,
                               seed=42,
                               number_episodes=1500,
                               max_episode_length=1000,
                               gamma=1.0,
                               verbose=True)

    moving_avg = moving_average(scores, 50)
    # Plot learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(scores)
    plt.plot(moving_avg, '--')
    plt.title("REINFORCE learning curve - CartPole-v1")
    plt.xlabel("Episode #")
    plt.ylabel("Score")
    plt.savefig('REINFORCE_LEARNING')


def investigate_variance_in_reinforce(verbose=True, n_runs=5):
    """ Run REINFORCE to determine the variance in results over multiple
     runs."""
    env = gym.make('CartPole-v1')
    policy_model = SimplePolicy()
    seeds = np.random.randint(1000, size=n_runs)
    all_scores = []
    for index, seed in enumerate(seeds):
        print("Running seed #" + str(index + 1) + "/" + str(n_runs) + " (" + str(int(seed)) + ")")
        policy, scores = reinforce(env=env,
                                   policy_model=policy_model,
                                   seed=int(seed),
                                   number_episodes=1500,
                                   max_episode_length=1000,
                                   gamma=1.0,
                                   verbose=verbose)
        all_scores.append(scores)
    mean = np.mean(all_scores, axis=0)
    std = np.std(all_scores, axis=0)
    return mean, std


def run_reinforce_with_naive_baseline(verbose=True, n_runs=5):
    """ Run REINFORCE to determine the variance in results over multiple
     runs."""
    env = gym.make('CartPole-v1')
    policy_model = SimplePolicy()
    seeds = np.random.randint(1000, size=n_runs)
    all_scores = []
    for index, seed in enumerate(seeds):
        print("Running seed #" + str(index + 1) + "/" + str(n_runs) + " (" + str(int(seed)) + ")")
        policy, scores = reinforce_naive_baseline(env=env,
                                                  policy_model=policy_model,
                                                  seed=int(seed),
                                                  number_episodes=1500,
                                                  max_episode_length=1000,
                                                  gamma=1.0,
                                                  verbose=verbose)
        all_scores.append(scores)
    mean = np.mean(all_scores, axis=0)
    std = np.std(all_scores, axis=0)
    return mean, std


if __name__ == '__main__':

    # REINFORCE
    print("Running REINFORCE")
    run_reinforce()

    # REINFORCE over 5 runs
    print("Running REINFORCE over 5 seeds")
    mean, std = investigate_variance_in_reinforce()
    x_fill = np.arange(len(mean))
    y_upper = mean + std
    y_lower = mean - std
    plt.figure(figsize=(12, 6))
    plt.plot(mean)
    plt.fill_between(x_fill, y_upper, y_lower, alpha=0.2)
    plt.title("REINFORCE averaged over 5 seeds")
    plt.xlabel("Episode #")
    plt.ylabel("Score")
    plt.savefig('REINFORCE_AVERAGED_LEARNING')

    # REINFORCE with naive baseline over 5 runs
    print("Running REINFORCE with baseline over 5 seeds")
    mean_baseline, std_baseline = run_reinforce_with_naive_baseline()
    x_fill_baseline = np.arange(len(mean_baseline))
    y_upper_baseline = mean_baseline + std_baseline
    y_lower_baseline = mean_baseline - std_baseline
    plt.figure(figsize=(12, 6))
    plt.plot(mean_baseline)
    plt.fill_between(x_fill_baseline, y_upper_baseline, y_lower_baseline, alpha=0.2)
    plt.title("REINFORCE with baseline averaged over 5 seeds")
    plt.xlabel("Episode #")
    plt.ylabel("Score")
    plt.savefig('BASELINE_REINFORCE_AVERAGED_LEARNING')

    # Combined plot
    plt.figure(figsize=(12, 6))
    plt.plot(mean, color='red', label= 'REINFORCE mean')
    plt.plot(mean_baseline, color='blue', label= 'Baseline mean')
    plt.fill_between(x_fill, y_upper, y_lower,
                     color='red', label='REINFORCE bounds', alpha=0.2)
    plt.fill_between(x_fill_baseline, y_upper_baseline, y_lower_baseline,
                     label='Baseline bounds', color='blue', alpha=0.2)
    plt.title("REINFORCE vs REINFORCE with baseline averaged over 5 seeds")
    plt.xlabel("Episode #")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig('REINFORCE_VS_AVERAGED_LEARNING')


