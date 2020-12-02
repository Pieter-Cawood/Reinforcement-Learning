"""
Modified the A2C method from https://github.com/raillab/a2c

"""

import torch.nn as nn
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

class Flatten(nn.Module):
    """
    Flatten a multi dimensional output from the Conv2D to a single dimension
    """

    def forward(self, x):
        return x.view(x.shape[0], -1)

class ActorCritic(nn.Module):
    def __init__(self, glyph_shape, num_actions, enable_lstm, crop_dims=10):
        super(ActorCritic, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.flatten = Flatten()
        if enable_lstm:
            self.lstm = nn.LSTMCell(256, 256)
        self.linear = nn.Linear(1152, 256)
        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)

        for m in self.features:
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.0)

        nn.init.orthogonal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

        nn.init.orthogonal_(self.critic.weight)
        nn.init.constant_(self.critic.bias, 0.0)

        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.constant_(self.actor.bias, 0.0)

    def forward(self, x_glyphs, hx, cx, enable_lstm):
        x_glyphs = x_glyphs.unsqueeze(0)
        x_glyphs = self.features(x_glyphs)
        x_glyphs = self.flatten(x_glyphs)
        x_glyphs = self.linear(x_glyphs)
        x_glyphs = F.relu(x_glyphs)
        if enable_lstm:
            hx, cx = self.lstm(x_glyphs, (hx, cx))
            x = hx
        else:
            x = x_glyphs
        return Categorical(logits=self.actor(x)), self.critic(x), hx, cx