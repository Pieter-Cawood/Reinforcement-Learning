#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from nle import nethack
from torch.distributions import Categorical

import pathlib

device = torch.device('cpu')

STATS_INDICES = {
    'x_coordinate': 0,
    'y_coordinate': 1,
    'score': 9,
    'health_points': 10,
    'health_points_max': 11,
    'hunger_level': 18,
}

#Function for redeucing the observation space
def transform_observation(observation): 
    """Process the state into the model input shape
    of ([glyphs, stats], )"""

    msg = observation['message']
    msg_norm = msg/256
    return msg_norm

#Policy Class
class Policy(nn.Module): 
    def __init__(self, obs_size, act_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(obs_size, 512)
#         self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(512, 128)
#         self.dropout = nn.Dropout(p=0.5)
        self.affine3 = nn.Linear(128, 64)
#         self.dropout = nn.Dropout(p=0.4)
        self.affine4 = nn.Linear(64, act_size)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
#         x = self.dropout(x)
        x = F.relu(x)
        x = self.affine2(x)
#         x = self.dropout(x)
        x = F.relu(x)
        x = self.affine3(x)
#         x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine4(x)
        return F.softmax(action_scores, dim=1)
        
#Agent Class
class MyAgent:
    def __init__(self, observation_space, action_space, seeds):
        path = str(pathlib.Path(__file__).parent.absolute())
        state_path = path + '/data.pkl'
        #policy_model = torch.load('/home/clarise/Desktop/COMS7053A - RL/mod_msg4.pt')
        policy_model = Policy(256,23)
        policy_model.load_state_dict(torch.load(state_path))
        self.model = policy_model.eval()
        self.obs_space = observation_space
        self.act_space = action_space
        self.seeds = seeds


    def act(self, state):
        obs = transform_observation(state)
        state_obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)
        action_probs = self.model(state_obs) #policy_model(state_obs) #self.model(state_obs) #[0].detach().numpy() #torch.from_numpy(obs).float().to(device)).detach().numpy()
        m = Categorical(action_probs)
        action = m.sample()
        return action.item()


# In[ ]:




