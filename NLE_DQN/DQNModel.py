import torch
from torch.nn import functional as F
from torch import nn
from nle import nethack


class Crop(nn.Module):
    """Helper class for NetHackNet below."""

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

    def __init__(self, glyph_shape, stats_dim, num_actions, embedding_dim=32, crop_dim=9, final_layer_dims=512):
        super(DQNModel, self).__init__()

        self.glyph_shape = glyph_shape  # Should be  (21, 79)
        self.stats_dim = stats_dim - 2 # Exclude 2 coordinate features, they are used to crop to origin only
        self.num_actions = num_actions
        self.h = self.glyph_shape[0]
        self.w = self.glyph_shape[1]
        self.k_dim = embedding_dim

        # Crop the glyphs, use embedding to reduce sparsity
        self.glyph_crop = Crop(self.h, self.w, crop_dim, crop_dim)
        self.glyph_embedding = nn.Embedding(nethack.MAX_GLYPH, self.k_dim)
        # Convolutional NN for the cropped glyphs
        self.glyph_conv_1 = nn.Conv2d(in_channels=crop_dim,#self.h,
                                      out_channels=self.k_dim // 2,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        self.glyph_conv_2 = nn.Conv2d(in_channels=self.k_dim // 2,
                                      out_channels=self.k_dim,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        self.glyph_conv_3 = nn.Conv2d(in_channels=self.k_dim,
                                      out_channels=self.k_dim,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)

        self.glyph_flatten = nn.Flatten()

        # MLP network for the character statistics
        self.stats_linear_1 = nn.Linear(in_features=self.stats_dim,
                                        out_features=self.k_dim)
        self.stats_linear_2 = nn.Linear(in_features=self.k_dim,
                                        out_features=self.k_dim)

        # Final MLP network for merging glyph network and stats network
        self.final_linear_1 = nn.Linear(in_features=9248,
                                        out_features=final_layer_dims)
        self.final_linear_2 = nn.Linear(in_features=final_layer_dims,
                                        out_features=final_layer_dims)
        # The output layer is fully-connected with single output for each action
        self.output_linear = nn.Linear(in_features=final_layer_dims,
                                       out_features=self.num_actions)

    def forward(self, observed_glyphs, observed_stats):
        # Crop glyphs to a fixed 9x9 dimension around agent coordinates
        # of each mini-batch
        coordinates = observed_stats[:, :2]
        x_glyphs = self.glyph_crop(observed_glyphs, coordinates)
        # Embeddings to reduce sparsity
        x_glyphs = self.glyph_embedding(x_glyphs)
        # Apply convolution
        x_glyphs = self.glyph_conv_1(x_glyphs)
        x_glyphs = F.relu(x_glyphs)
        x_glyphs = self.glyph_conv_2(x_glyphs)
        x_glyphs = F.relu(x_glyphs)
        x_glyphs = self.glyph_conv_3(x_glyphs)
        x_glyphs = F.relu(x_glyphs)
        x_glyphs = self.glyph_flatten(x_glyphs)

        # Learn from statistic features, excluding coordinates
        x_stats = observed_stats[:, 2:]
        x_stats = self.stats_linear_1(x_stats)
        x_stats = F.relu(x_stats)
        x_stats = self.stats_linear_2(x_stats)
        x_stats = F.relu(x_stats)

        # Merger MLP
        x = torch.cat([x_glyphs, x_stats], dim=1)
        x = self.final_linear_1(x)
        x = F.relu(x)
        x = self.final_linear_2(x)
        x = F.relu(x)
        x = self.output_linear(x)

        return x
