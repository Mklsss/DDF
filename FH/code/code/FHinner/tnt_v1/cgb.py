import einops
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class Layer_norm_process(nn.Module):  # n, h, w, c
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.zeros(c), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.ones(c), requires_grad=True)
        self.eps = eps

    def forward(self, feature):
        var_mean = torch.var_mean(feature, dim=-1, unbiased=False)
        mean = var_mean[1]
        var = var_mean[0]
        # layer norm process
        feature = (feature - mean[..., None]) / torch.sqrt(var[..., None] + self.eps)
        gamma = self.gamma.expand_as(feature)
        beta = self.beta.expand_as(feature)
        feature = feature * gamma + beta
        return feature


def block_images_einops(x, patch_size):  # n, h, w, c
    """Image to patches."""
    batch, height, width, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    x = einops.rearrange(
        x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x


def unblock_images_einops(x, grid_size, patch_size):
    """patches to images."""
    x = einops.rearrange(
        x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
        gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x


class GetSpatialGatingWeights(nn.Module):  # n, h, w, c
    """Get gating weights for cross-gating MLP block."""

    def __init__(self, num_channels, grid_size, block_size, input_proj_factor=2, use_bias=True, dropout_rate=0):
        super().__init__()
        self.num_channels = num_channels
        self.grid_size = grid_size
        self.block_size = block_size
        self.gh = self.grid_size[0]
        self.gw = self.grid_size[1]
        self.fh = self.block_size[0]
        self.fw = self.block_size[1]
        self.input_proj_factor = input_proj_factor
        self.use_bias = use_bias
        self.drop = dropout_rate
        self.LayerNorm_in = Layer_norm_process(self.num_channels)
        self.in_project = nn.Linear(self.num_channels, self.num_channels * self.input_proj_factor, bias=self.use_bias)
        self.gelu = nn.GELU()
        self.Dense_0 = nn.Linear(self.gh * self.gw, self.gh * self.gw, bias=self.use_bias)
        self.Dense_1 = nn.Linear(self.fh * self.fw, self.fh * self.fw, bias=self.use_bias)
        self.out_project = nn.Linear(self.num_channels * self.input_proj_factor, self.num_channels, bias=self.use_bias)
        self.dropout = nn.Dropout(self.drop)

    def forward(self, x):
        _, h, w, _ = x.shape
        # input projection
        x = self.LayerNorm_in(x)
        x = self.in_project(x)  # channel projection
        x = self.gelu(x)
        c = x.size(-1) // 2
        u, v = torch.split(x, c, dim=-1)
        # get grid MLP weights
        fh, fw = h // self.gh, w // self.gw
        u = block_images_einops(u, patch_size=(fh, fw))  # n, (gh gw) (fh fw) c
        u = u.permute(0, 3, 2, 1)  # n, c, (fh fw) (gh gw)
        u = self.Dense_0(u)
        u = u.permute(0, 3, 2, 1)  # n, (gh gw) (fh fw) c
        u = unblock_images_einops(u, grid_size=(self.gh, self.gw), patch_size=(fh, fw))
        # get block MLP weights
        gh, gw = h // self.fh, w // self.fw
        v = block_images_einops(v, patch_size=(self.fh, self.fw))  # n, (gh gw) (fh fw) c
        v = v.permute(0, 1, 3, 2)  # n (gh gw) c (fh fw)
        v = self.Dense_1(v)
        v = v.permute(0, 1, 3, 2)  # n, (gh gw) (fh fw) c
        v = unblock_images_einops(v, grid_size=(gh, gw), patch_size=(self.fh, self.fw))

        x = torch.cat([u, v], dim=-1)
        x = self.out_project(x)
        x = self.dropout(x)
        return x


class CrossGatingBlock(nn.Module):  # input shape: n, c, h, w
    """Cross-gating MLP block."""

    def __init__(self, x_features=1, num_channels=1, block_size=(2, 2), grid_size=(2, 2), cin_y=0, upsample_y=True,
                 use_bias=True,
                 use_global_mlp=True, dropout_rate=0):
        super().__init__()
        self.cin_y = cin_y
        self.x_features = x_features
        self.num_channels = num_channels
        self.block_size = block_size
        self.grid_size = grid_size
        self.upsample_y = upsample_y
        self.use_bias = use_bias
        self.use_global_mlp = use_global_mlp
        self.drop = dropout_rate
        self.ConvTranspose_0 = nn.ConvTranspose2d(self.cin_y, self.num_channels, kernel_size=(2, 2), stride=2,
                                                  bias=self.use_bias)
        self.Conv_0 = nn.Conv2d(self.x_features, self.num_channels, kernel_size=(1, 1), stride=1, bias=self.use_bias)
        self.Conv_1 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=(1, 1), stride=1, bias=self.use_bias)
        self.LayerNorm_x = Layer_norm_process(self.num_channels)
        self.in_project_x = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.gelu1 = nn.GELU()
        self.SplitHeadMultiAxisGating_x = GetSpatialGatingWeights(num_channels=self.num_channels,
                                                                  block_size=self.block_size, grid_size=self.grid_size,
                                                                  dropout_rate=self.drop, use_bias=self.use_bias)
        self.LayerNorm_y = Layer_norm_process(self.num_channels)
        self.in_project_y = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.gelu2 = nn.GELU()
        self.SplitHeadMultiAxisGating_y = GetSpatialGatingWeights(num_channels=self.num_channels,
                                                                  block_size=self.block_size, grid_size=self.grid_size,
                                                                  dropout_rate=self.drop, use_bias=self.use_bias)
        self.out_project_y = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.dropout1 = nn.Dropout(self.drop)
        self.out_project_x = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.dropout2 = nn.Dropout(self.drop)

    def forward(self, x, y):
        # Upscale Y signal, y is the gating signal.
        x = self.Conv_0(x)
        y = self.Conv_1(y)
        assert y.shape == x.shape
        x = x.permute(0, 2, 3, 1)  # n,h,w,c
        y = y.permute(0, 2, 3, 1)  # n,h,w,c
        shortcut_x = x
        shortcut_y = y
        # Get gating weights from X
        x = self.LayerNorm_x(x)
        x = self.in_project_x(x)
        x = self.gelu1(x)
        gx = self.SplitHeadMultiAxisGating_x(x)
        # Get gating weights from Y
        y = self.LayerNorm_y(y)
        y = self.in_project_y(y)
        y = self.gelu2(y)
        gy = self.SplitHeadMultiAxisGating_y(y)
        # Apply cross gating
        y = y * gx  ## gating y using x
        y = self.out_project_y(y)
        y = self.dropout1(y)
        y = y + shortcut_y
        x = x * gy  # gating x using y
        x = self.out_project_x(x)
        x = self.dropout2(x)
        x = x + y + shortcut_x  # get all aggregated signals
        return x.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2)  # n,c,h,w


# m = CrossGatingBlock()
# x = torch.rand(3, 1, 256, 256)
# y = torch.rand(3, 1, 256, 256)

# a, b = m(x, y)
# print(1)
