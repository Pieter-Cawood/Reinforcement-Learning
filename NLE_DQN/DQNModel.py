import torch
from torch.nn import functional as F
from torch import nn
from nle import nethack

class Crop(nn.Module):
    """Helper class to crop observations around an agent's position.
    Borrowed from
    https://github.com/facebookresearch/nle/blob/master/nle/agent/agent.py

    """

    def __init__(self, height, width, height_target, width_target):
        super(Crop, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        width_grid = self._step_to_range(2 / (self.width - 1), self.width_target)[
                     None, :
                     ].expand(self.height_target, -1)
        height_grid = self._step_to_range(2 / (self.height - 1), height_target)[
                      :, None
                      ].expand(-1, self.width_target)

        # "clone" necessary, https://github.com/pytorch/pytorch/issues/34880
        self.register_buffer("width_grid", width_grid.clone())
        self.register_buffer("height_grid", height_grid.clone())

    def _step_to_range(self, delta, num_steps):
        """Range of `num_steps` integers with distance `delta` centered around zero."""
        return delta * torch.arange(-num_steps // 2, num_steps // 2)

    def forward(self, inputs, coordinates):
        """Calculates centered crop around given x,y coordinates.
        Args:
           inputs [B x H x W]
           coordinates [B x 2] x,y coordinates
        Returns:
           [B x H' x W'] inputs cropped and centered around x,y coordinates.
        """
        assert inputs.shape[1] == self.height
        assert inputs.shape[2] == self.width

        inputs = inputs[:, None, :, :].float()

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        # TODO: only cast to int if original tensor was int
        return (
            torch.round(F.grid_sample(inputs, grid, align_corners=True))
                .squeeze(1)
                .long()
        )


class Flatten(nn.Module):
    """
    Flatten a multi dimensional output from the Conv2D to a single dimension
    """

    def forward(self, x):
        return x.view(-1, x.shape[0])


class DQNModel(nn.Module):
    def __init__(self, glyph_shape, stats_dim, num_actions, embedding_dim=32, crop_dim=15, final_layer_dims=512):
        super(DQNModel, self).__init__()

        self.glyph_shape = glyph_shape  # Should be  (21, 79)
        self.num_actions = num_actions
        self.h = self.glyph_shape[0]
        self.w = self.glyph_shape[1]
        self.k_dim = embedding_dim
        self.glyph_crop = Crop(self.h, self.w, crop_dim, crop_dim)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.glyph_flatten = nn.Flatten()
        self.final_linear_1 = nn.Linear(in_features=3872,
                                        out_features=final_layer_dims)
        self.output_linear = nn.Linear(in_features=final_layer_dims,
                                       out_features=self.num_actions)

    def forward(self, observed_glyphs, observed_stats):
        coordinates = observed_stats[:, :2]
        x_glyphs = self.glyph_crop(observed_glyphs, coordinates).unsqueeze(1).float()
        x_glyphs = self.features(x_glyphs)
        x_glyphs = self.glyph_flatten(x_glyphs)
        x = self.final_linear_1(x_glyphs)
        x = F.relu(x)
        x = self.output_linear(x)
        return x
