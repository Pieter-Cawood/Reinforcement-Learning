import numpy as np
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn
import torch.nn.functional as tnf

import random
from collections import deque

device = torch.device("cpu")


class SimplePolicy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(s_size, h_size),
                                    nn.ReLU(),
                                    nn.Linear(h_size, a_size)).to(device)
        
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
    for episode in range(number_episodes):
        
        rewards = []
        log_probs = []
        state = env.reset()
        for step in range(max_episode_length):
            
            action_probs = policy_model(torch.from_numpy(state).float().to(device))
            action_sampler = torch.distributions.Categorical(action_probs)
            action = action_sampler.sample()
            log_probs.append(action_sampler.log_prob(action))
            new_state, reward, done, info = env.step(action.item())
            rewards.append(reward)
            state = new_state
            if done:
                break
            
        scores.append(sum(rewards))
        returns = compute_returns(rewards, gamma)
        loss = -1 * (returns * np.sum(np.array(log_probs)))
        policy_model.optimizer.zero_grad()
        loss.backward()
        policy_model.optimizer.step()
        
        window = 250
        if verbose and episode % window == 0 and episode != 0:
            print("Episode " + str(episode) + "/" + str(number_episodes) +
                  " Score: " + str(np.mean(scores[episode - window: episode])))
                                   
    policy = policy_model.parameters()
                                   
    return policy, scores


def compute_returns_naive_baseline(rewards, gamma):
    returns = sum([(gamma**i) * reward for i, reward in enumerate(rewards)])
    r = 0
    returns = []
    for step in reversed(range(len(rewards))):
        r = rewards[step] + gamma * r
        returns.insert(0, r)
    returns = np.array(returns)
    returns = returns - returns.mean()
    returns = returns / returns.std()
    returns = torch.from_numpy(returns).float().to(device)
    return returns


def reinforce_naive_baseline(env, policy_model, seed, learning_rate,
                             number_episodes,
                             max_episode_length,
                             gamma, verbose=True):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.seed(seed)

    scores = []
    for episode in range(number_episodes):
        
        rewards = []
        log_probs = []
        state = env.reset()
        for step in range(max_episode_length):
            
            action_probs = policy_model(torch.from_numpy(state).float().to(device))
            action_sampler = torch.distributions.Categorical(action_probs)
            action = action_sampler.sample()
            log_probs.append(action_sampler.log_prob(action).unsqueeze(0))
            new_state, reward, done, info = env.step(action.item())
            rewards.append(reward)
            state = new_state
            if done:
                break
            
        scores.append(sum(rewards))
        returns = compute_returns_naive_baseline(rewards, gamma)
        log_probs = torch.cat(log_probs)
        loss = -1 * torch.sum(returns * log_probs)
        policy_model.optimizer.zero_grad()
        loss.backward()
        policy_model.optimizer.step()
        
        window = 250
        if verbose and episode % window == 0 and episode != 0:
            print("Episode " + str(episode) + "/" + str(number_episodes) +
                  " Score: " + str(np.mean(scores[episode - window: episode])))
                                   
    policy = policy_model.parameters()
    return policy, scores


def run_reinforce():
    env = gym.make('CartPole-v1')
    
    print("Run REINFORCE.")
    policy_model = SimplePolicy(s_size=env.observation_space.shape[0],
                                h_size=50,
                                a_size=env.action_space.n)
    policy, scores = reinforce(env=env,
                               policy_model=policy_model,
                               seed=42,
                               learning_rate=1e-2,
                               number_episodes=1500,
                               max_episode_length=1000,
                               gamma=1.0,
                               verbose=True)
    moving_avg = moving_average(scores, 50)
    
    # Plot learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(scores)
    plt.plot(moving_avg, '--')
    plt.title("REINFORCE learning curve")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig('Question 1.4.png')

def investigate_variance_in_reinforce():
    env = gym.make('CartPole-v1')
    seeds = np.random.randint(1000, size=5)
    
    print("Average REINFORCE over 5 runs.")
    run_scores = []
    for i, seed in enumerate(seeds):

        policy_model = SimplePolicy(s_size=env.observation_space.shape[0],
                                    h_size=50,
                                    a_size=env.action_space.n)
        print("Running REINFORCE for run", i+1, "of 5.")
        policy, scores = reinforce(env=env,
                               policy_model=policy_model,
                               seed=42,
                               learning_rate=1e-2,
                               number_episodes=1500,
                               max_episode_length=1000,
                               gamma=1.0,
                               verbose=True)
        run_scores.append(moving_average(scores, 50))
    
    mean = np.mean(run_scores, axis=0)
    std = np.std(run_scores, axis=0)
    
    # Plot learning curve with one standard deviation bounds
    x_fill = np.arange(len(mean))
    y_upper = mean + std
    y_lower = mean - std
    plt.figure(figsize=(12, 6))
    plt.plot(mean)
    plt.fill_between(x_fill, y_upper, y_lower, alpha=0.2)
    plt.title("REINFORCE averaged over 5 seeds")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig('Question 1.5.png')
    return mean, std


def run_reinforce_with_naive_baseline(mean, std):
    env = gym.make('CartPole-v1')
    
    print("Run REINFORCE with naive baseline.")
    policy_model = SimplePolicy(s_size=env.observation_space.shape[0],
                                h_size=50,
                                a_size=env.action_space.n)
    policy, scores = reinforce_naive_baseline(env=env,
                                              policy_model=policy_model,
                                              seed=42,
                                              learning_rate=1e-2,
                                              number_episodes=1500,
                                              max_episode_length=1000,
                                              gamma=1.0,
                                              verbose=True)
    moving_avg = moving_average(scores, 50)
    
    # Plot learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(scores)
    plt.plot(moving_avg, '--')
    plt.title("REINFORCE with naive baseline")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig('Question 2.3.1.png')

    np.random.seed(53)
    seeds = np.random.randint(1000, size=5)
    
    print("Average REINFORCE over 5 runs.")
    run_scores = []
    for i, seed in enumerate(seeds):

        policy_model = SimplePolicy(s_size=env.observation_space.shape[0],
                                    h_size=50,
                                    a_size=env.action_space.n)
        print("Running REINFORCE with naive baseline for run", i+1, "of 5.")
        policy, scores = reinforce_naive_baseline(env=env,
                                                  policy_model=policy_model,
                                                  seed=42,
                                                  learning_rate=1e-2,
                                                  number_episodes=1500,
                                                  max_episode_length=1000,
                                                  gamma=1.0,
                                                  verbose=True)
        run_scores.append(moving_average(scores, 50))
    
    mean = np.mean(run_scores, axis=0)
    std = np.std(run_scores, axis=0)
    
    # Plot learning curve with one standard deviation bounds
    x_fill = np.arange(len(mean))
    y_upper = mean + std
    y_lower = mean - std
    plt.figure(figsize=(12, 6))
    plt.plot(mean)
    plt.fill_between(x_fill, y_upper, y_lower, alpha=0.2)
    plt.title("REINFORCE with naive baseline averaged over 5 seeds")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig('Question 2.3.2.png')


if __name__ == '__main__':
    run_reinforce()
    mean, std = investigate_variance_in_reinforce()
    run_reinforce_with_naive_baseline(mean, std)
