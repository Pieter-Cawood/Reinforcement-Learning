# https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/

PLOT = False

if PLOT:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", 'matplotlib'])
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

import torch
import torch.nn as nn
import torch.nn.functional as fun
# import copy
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from collections import OrderedDict
import gym
import nle
import pickle
from abc import ABC
import pathlib

STATS_INDICES = {
    'x_coordinate': 0,
    'y_coordinate': 1,
    'score': 9,
    'health_points': 10,
    'health_points_max': 11,
    'hunger_level': 18,
}


def crop_glyphs(glyphs, x, y, size=7):
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


def crop_state(observation):
    """Process the state into the model input shape
    of ([glyphs, stats], )"""

    stat_x_coord = observation['blstats'][STATS_INDICES['x_coordinate']]
    stat_y_coord = observation['blstats'][STATS_INDICES['y_coordinate']]
    stat_health = float(observation['blstats'][STATS_INDICES['health_points']]) - float(
        observation['blstats'][STATS_INDICES['health_points_max']] / 2)
    stat_hunger = observation['blstats'][STATS_INDICES['hunger_level']]
    observed_stats = np.array([stat_x_coord, stat_y_coord, stat_health, stat_hunger])
    # observed_stats = observation['blstats'][:].flatten()

    # observed_glyphs = observation['glyphs'][:,:].flatten()
    observed_glyphs = crop_glyphs(observation['glyphs'], stat_x_coord, stat_y_coord)

    final_obs = np.concatenate((observed_stats, observed_glyphs)).astype(np.float32)
    norm_obs = (final_obs - final_obs.mean()) / final_obs.std()
    return norm_obs


class ActorCriticNet(nn.Module, ABC):
    def __init__(self, in_shape, out_shape, hidden_layers=None, learning_rate=0.01, bias=False, device='cpu'):

        super(ActorCriticNet, self).__init__()

        if hidden_layers is None:
            hidden_layers = [32, 32, 32, 32]
        self.device = device
        self.n_inputs = in_shape
        self.n_outputs = out_shape
        self.hidden_layers = hidden_layers
        layer_list = [in_shape]
        layer_list.extend(hidden_layers)
        self.layer_list = layer_list
        self.n_hidden_layers = len(self.hidden_layers)
        self.n_hidden_nodes = self.layer_list[-1]
        self.learning_rate = learning_rate
        self.bias = bias
        self.action_space = np.arange(self.n_outputs)

        # Generate network according to hidden layer settings
        self.layers = OrderedDict()
        self.n_layers = 2 * self.n_hidden_layers
        j = 1
        for i in range(self.n_layers - 1):
            # Define single linear layer
            if self.n_hidden_layers == 0:
                self.layers[str(i)] = nn.Linear(
                    self.n_inputs,
                    self.n_outputs,
                    bias=self.bias)
            # Define layers
            elif i % 2 == 0:
                self.layers[str(i)] = nn.Linear(
                    self.layer_list[j - 1],
                    self.layer_list[j],
                    bias=self.bias)
                j += 1
            else:
                self.layers[str(i)] = nn.ReLU()

        self.body = nn.Sequential(self.layers)

        # Define policy head
        self.policy = nn.Sequential(
            nn.Linear(self.n_hidden_nodes, self.n_hidden_nodes, bias=self.bias),
            nn.ReLU(),
            nn.Linear(self.n_hidden_nodes, self.n_outputs, bias=self.bias))
        # Define value head
        self.value = nn.Sequential(
            nn.Linear(self.n_hidden_nodes, self.n_hidden_nodes, bias=self.bias),
            nn.ReLU(),
            nn.Linear(self.n_hidden_nodes, 1, bias=self.bias))

        if self.device == 'cuda':
            self.net.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.learning_rate)

    def predict(self, state):
        body_output = self.get_body_output(state)
        probs = fun.softmax(self.policy(body_output), dim=-1)
        return probs, self.value(body_output)

    def get_body_output(self, state):
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.body(state_t)

    def get_action(self, state):
        probs = self.predict(state)[0].detach().numpy()
        action = np.random.choice(self.action_space, p=probs)
        return action

    def act(self, state):
        probs = self.predict(state)[0].detach().numpy()
        ind = np.argmax(probs)
        action = self.action_space[ind]
        return action

    def get_log_probs(self, state):
        body_output = self.get_body_output(state)
        logprobs = fun.log_softmax(self.policy(body_output), dim=-1)
        return logprobs


class A2C:
    def __init__(self, envi, network):

        self.env = envi
        self.network = network
        self.action_space = np.arange(envi.action_space.n)
        self.s_0 = 0
        self.reward = 0
        self.n_steps = 0
        self.gamma = 0
        self.num_episodes = 0
        self.beta = 0
        self.zeta = 0
        self.batch_size = 0
        self.ep_counter = 0

        # Set up lists to log data
        self.ep_rewards = []
        self.kl_div = []
        self.policy_loss = []
        self.value_loss = []
        self.entropy_loss = []
        self.total_policy_loss = []
        self.total_loss = []

    def generate_episode(self):
        states, actions, rewards, dones, next_states = [], [], [], [], []
        counter = 0
        total_count = self.batch_size * self.n_steps
        while counter < total_count:
            done = False
            while not done:
                action = self.network.get_action(self.s_0)
                s_1, r, done, _ = self.env.step(action)
                s_1 = crop_state(s_1)
                self.reward += r
                states.append(self.s_0)
                next_states.append(s_1)
                actions.append(action)
                rewards.append(r)
                dones.append(done)
                self.s_0 = s_1

                if done:
                    self.ep_rewards.append(self.reward)
                    self.s_0 = crop_state(self.env.reset())
                    self.reward = 0
                    self.ep_counter += 1
                    if self.ep_counter >= self.num_episodes:
                        counter = total_count
                        break

                counter += 1
                if counter >= total_count:
                    break
        return states, actions, rewards, dones, next_states

    def calc_rewards(self, batch):
        states, actions, rewards, dones, next_states = batch
        rewards = np.array(rewards)
        total_steps = len(rewards)

        state_values = self.network.predict(states)[1]
        next_state_values = self.network.predict(next_states)[1]
        done_mask = torch.ByteTensor(dones).to(self.network.device)
        next_state_values[done_mask] = 0.0
        state_values = state_values.detach().numpy().flatten()
        next_state_values = next_state_values.detach().numpy().flatten()

        g = np.zeros_like(rewards, dtype=np.float32)
        # td_delta = np.zeros_like(rewards, dtype=np.float32)
        dones = np.array(dones)

        for t in range(total_steps):
            last_step = min(self.n_steps, total_steps - t)

            # Look for end of episode
            check_episode_completion = dones[t:t + last_step]
            if check_episode_completion.size > 0:
                if True in check_episode_completion:
                    next_ep_completion = np.where(check_episode_completion == True)[0][0]
                    last_step = next_ep_completion

            # Sum and discount rewards
            g[t] = sum([rewards[t + n:t + n + 1] * self.gamma ** n for
                        n in range(last_step)])

        if total_steps > self.n_steps:
            g[:total_steps - self.n_steps] += next_state_values[self.n_steps:] \
                                              * self.gamma ** self.n_steps
        td_delta = g - state_values
        return g, td_delta

    def train(self, n_steps=5, batch_size=10, num_episodes=2000, gamma=1, beta=1e-3, zeta=0.5):
        self.n_steps = n_steps
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.beta = beta
        self.zeta = zeta

        # Set up lists to log data
        self.ep_rewards = []
        self.kl_div = []
        self.policy_loss = []
        self.value_loss = []
        self.entropy_loss = []
        self.total_policy_loss = []
        self.total_loss = []

        self.s_0 = crop_state(self.env.reset())
        self.reward = 0
        self.ep_counter = 0
        while self.ep_counter < num_episodes:
            batch = self.generate_episode()
            g, td_delta = self.calc_rewards(batch)
            states = batch[0]
            actions = batch[1]
            current_probs = self.network.predict(states)[0].detach().numpy()

            self.update(states, actions, g, td_delta)

            new_probs = self.network.predict(states)[0].detach().numpy()
            kl = -np.sum(current_probs * np.log(new_probs / current_probs))
            self.kl_div.append(kl)

            print("\rMean Rewards: {:.2f} Episode: {:d}    ".format(
                np.mean(self.ep_rewards), self.ep_counter), end="")

    def plot_results(self):
        avg_rewards = [np.mean(self.ep_rewards[i:i + self.batch_size]) if i > self.batch_size
                       else np.mean(self.ep_rewards[:i + 1]) for i in range(len(self.ep_rewards))]

        plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 2)
        ax0 = plt.subplot(gs[0, :])
        ax0.plot(self.ep_rewards)
        ax0.plot(avg_rewards)
        ax0.set_xlabel('Episode')
        plt.title('Rewards')
        plt.savefig('/opt/project/A-C_Rewards.png')

        ax1 = plt.subplot(gs[1, 0])
        ax1.plot(self.policy_loss)
        plt.title('Policy Loss')
        plt.xlabel('Update Number')
        plt.savefig('/opt/project/A-C_Policy_Loss.png')

        ax2 = plt.subplot(gs[1, 1])
        ax2.plot(self.entropy_loss)
        plt.title('Entropy Loss')
        plt.xlabel('Update Number')
        plt.savefig('/opt/project/A-C_Entropy_Loss.png')

        ax3 = plt.subplot(gs[2, 0])
        ax3.plot(self.value_loss)
        plt.title('Value Loss')
        plt.xlabel('Update Number')
        plt.savefig('/opt/project/A-C_Value_Loss.png')

        ax4 = plt.subplot(gs[2, 1])
        ax4.plot(self.kl_div)
        plt.title('KL Divergence')
        plt.xlabel('Update Number')
        plt.savefig('/opt/project/A-C_KL_Divergence.png')

        plt.tight_layout()

    def calc_loss(self, states, actions, rewards, advantages):
        # actions_t = torch.LongTensor(actions).to(self.network.device)
        rewards_t = torch.FloatTensor(rewards).to(self.network.device)
        advantages_t = torch.FloatTensor(advantages).to(self.network.device)

        log_probs = self.network.get_log_probs(states)
        log_prob_actions = advantages_t * log_probs[range(len(actions)), actions]
        policy_loss = -log_prob_actions.mean()

        action_probs, values = self.network.predict(states)
        entropy_loss = -self.beta * (action_probs * log_probs).sum(dim=1).mean()

        value_loss = self.zeta * nn.MSELoss()(values.squeeze(-1), rewards_t)

        # Append values
        self.policy_loss.append(policy_loss)
        self.value_loss.append(value_loss)
        self.entropy_loss.append(entropy_loss)

        return policy_loss, entropy_loss, value_loss

    def update(self, states, actions, rewards, advantages):
        self.network.optimizer.zero_grad()
        policy_loss, entropy_loss, value_loss = self.calc_loss(states, actions, rewards, advantages)

        total_policy_loss = policy_loss - entropy_loss
        self.total_policy_loss.append(total_policy_loss)
        total_policy_loss.backward(retain_graph=True)

        value_loss.backward()

        total_loss = policy_loss + value_loss + entropy_loss
        self.total_loss.append(total_loss)
        self.network.optimizer.step()

    def play(self):
        state = crop_state(self.env.reset())
        self.env.render()
        rewards = []
        done = False
        while not done:
            action = self.network.act(state)
            new_state, r, done, _ = self.env.step(action)
            self.env.render()
            rewards.append(r)
            state = crop_state(new_state)
        print("Total rewards:", sum(rewards))


def load_agent():
    path = str(pathlib.Path(__file__).parent.absolute())
    param_path = path + '/a2c_net.pkl'
    state_path = path + '/a2c_state.pkl'
    with open(param_path, 'rb') as f:
        in_s, out_s, hidden, lr, b = pickle.load(f)
    neural_net = ActorCriticNet(in_shape=in_s, out_shape=out_s, hidden_layers=hidden, learning_rate=lr, bias=b)
    neural_net.load_state_dict(torch.load(state_path))
    return neural_net


if __name__ == '__main__':
    env = gym.make("NetHackScore-v0")
    net = load_agent()
    a2c = A2C(env, net)
    for j in range(9):
        print("Training session:", j, end='')
        a2c.train(n_steps=1000000, num_episodes=100)
        if PLOT:
            a2c.plot_results()
        file_name = '/opt/project/a2c_state' + str(j+1) + '.pkl'
        torch.save(net.state_dict(), file_name)
