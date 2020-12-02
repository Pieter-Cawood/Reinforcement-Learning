import pickle
from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fun
from collections import OrderedDict
import pathlib

np.random.seed(42)

STATS_INDICES = {
    'x_coordinate': 0,
    'y_coordinate': 1,
    'score': 9,
    'health_points': 10,
    'health_points_max': 11,
    'hunger_level': 18
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

    def get_greedy_action(self, state):
        probs = self.predict(state)[0].detach().numpy()
        ind = np.argmax(probs)
        action = self.action_space[ind]
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


def load_agent():
    path = str(pathlib.Path(__file__).parent.absolute())
    param_path = path + '/a2c_net.pkl'
    state_path = path + '/a2c_state.pkl'
    with open(param_path, 'rb') as f:
        in_s, out_s, hidden, lr, b = pickle.load(f)
    net = ActorCriticNet(in_shape=in_s, out_shape=out_s, hidden_layers=hidden, learning_rate=lr, bias=b)
    net.load_state_dict(torch.load(state_path))
    return net


class MyAgent:
    def __init__(self, observation_space, action_space, **kwargs):
        self.observation_space = observation_space
        self.action_space = action_space
        self.seeds = kwargs.get('seeds', None)
        self.net = load_agent()

    def act(self, observation):
        # Perform processing to observation
        state = crop_state(observation)
        action = self.net.get_action(state)
        return action
