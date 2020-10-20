import numpy as np

import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCriticNetwork(nn.Module):
    def __init__(self, learning_rate, input_dims, hidden_dims, a_size):
        super(ActorCriticNetwork, self).__init__()
        # Actor network -> Estimates the policy function (thetha params)
        self.actor_network = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, a_size),
            nn.ReLU(),
        ) \
            .to(device)

        # Critic network -> Estimates the value function (w params)
        self.critic_network = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 1),
            nn.ReLU(),
        ) \
            .to(device)

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate)

    def forward(self, state: np.ndarray):
        state_ = torch.from_numpy(state).float().to(device)
        # Actor, policy function estimate
        policy = self.actor_network(state_)
        # Critic, value function estimate
        value = self.critic_network(state_)
        return F.softmax(policy, dim=0), value

class Agent():
    def __init__(self, learning_rate, input_dims, discount_factor, hidden_dims, a_size):
        self.actor_critic = ActorCriticNetwork(learning_rate,
                                               input_dims,
                                               hidden_dims,
                                               a_size)

        self.discount_factor = discount_factor

    def new_state(self, state):
        actor_probs, critic_value = self.actor_critic.forward(state)
        actor_dist = torch.distributions.Categorical(actor_probs)
        return actor_dist, critic_value


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    env = gym.wrappers.Monitor(env, './output/', video_callable=lambda episode_id: episode_id % 5 == 0, force=True)
    discount_factor = 1.00
    agent = Agent(learning_rate=1e-2,
                  input_dims=env.observation_space.shape[0],
                  discount_factor=discount_factor,
                  hidden_dims=512,
                  a_size=env.action_space.n
                  )

    score_history = []
    n_episodes = 2000
    n_steps = 300

    for episode in range(n_episodes):
        state = env.reset()
        score = 0
        score_history = []
        log_probs = []

        for step in range(n_steps):

            actor_dist, critic_value = agent.new_state(state)
            # Choose action from prob dist
            action = actor_dist.sample()

            log_probs = actor_dist.log_prob(action)

            next_state, reward, done, info = env.step(action.item())

            with torch.no_grad():
                _, critic_value_ = agent.new_state(next_state)

            score += reward

            reward = torch.Tensor([reward]).to(device)

            delta = reward + discount_factor * critic_value_ * (1 - int(done)) - critic_value

            actor_loss = -log_probs * delta
            critic_loss = delta.pow(2).mean()

            agent.actor_critic.optimizer.zero_grad()
            (actor_loss + critic_loss).backward()
            agent.actor_critic.optimizer.step()

            state = next_state
            # While not terminal
            if done:
                break

        print("Episode #" + str(episode) + "Episode score: " + str(score))
