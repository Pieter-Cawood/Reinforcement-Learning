import pickle
import torch
import numpy as np
import gym
import nle
import torch.nn as nn
import torch.nn.functional as tnf

device = torch.device('cpu')

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
    stat_health = float(observation['blstats'][STATS_INDICES['health_points']]) / float(observation['blstats'][STATS_INDICES['health_points_max']])
    stat_hunger = observation['blstats'][STATS_INDICES['hunger_level']]
    observed_stats = np.array([stat_x_coord, stat_y_coord, stat_health, stat_hunger])

    flat_glyphs = crop_glyphs(observed_glyphs, stat_x_coord, stat_y_coord)

    final_obs = np.concatenate((observed_stats, flat_glyphs)).astype(np.float32)
    return final_obs


class SimplePolicy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(s_size, h_size),
                                    nn.ReLU(),
                                    nn.Linear(h_size, a_size)).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return tnf.softmax(self.policy(x), dim=0)

class MyAgent:
    def __init__(self, observation_space, action_space):
        with open('/opt/project/objs.pkl', 'rb') as f:
            policy_model, scores, losses = pickle.load(f)
        self.model = policy_model


    def act(self, state):
        #does more stuff
        obs = transform_observation(state)
        action_probs = self.model(torch.from_numpy(obs).float().to(device)).detach().numpy()
        action = np.argmax(action_probs)
        return action

    def play(self, env):

        state = env.reset()
        done = False
        rewards = []

        while not done:
            action = self.act(state)
            state, reward, done, info = env.step(action)
            rewards.append(reward)

        return rewards


if __name__ == '__main__':
    env = gym.make("NetHackScore-v0")
    #env = gym.wrappers.Monitor(env, './reinforce-vid/sample_vid',video_callable=lambda episode_id: True,force=True)
    agent = MyAgent(0,0)
    rewards = agent.play(env)
    print('rewards: ', sum(rewards))