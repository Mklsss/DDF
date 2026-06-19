"""Projection-domain residual U-Net used only by the universal-backbone study.

The default DDF projection interface is ``(N, 360, 357) -> (N, 360, 357)``.
This module deliberately preserves that interface and does not contain any CT,
FBP, FP, feedback, or fusion logic.
"""

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=True)
        )
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=True)
        self.block = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        return self.block(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = ResidualBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class ResUNetSino(nn.Module):
    """P-CNN: residual U-Net for a 360-by-357 sinogram.

    The residual output makes the network a repair model: it predicts a
    correction to the interpolated sparse-view sinogram supplied by DDF's
    unchanged data pipeline.
    """

    def __init__(self, base_channels=32):
        super().__init__()
        c = int(base_channels)
        if c < 1:
            raise ValueError("base_channels must be positive")
        self.encoder0 = ResidualBlock(1, c)
        self.encoder1 = DownBlock(c, 2 * c)
        self.encoder2 = DownBlock(2 * c, 4 * c)
        self.bottleneck = DownBlock(4 * c, 8 * c)
        self.decoder2 = UpBlock(8 * c, 4 * c, 4 * c)
        self.decoder1 = UpBlock(4 * c, 2 * c, 2 * c)
        self.decoder0 = UpBlock(2 * c, c, c)
        self.out = nn.Conv2d(c, 1, 3, padding=1, bias=True)

    def forward(self, sinogram):
        if sinogram.ndim != 3 or sinogram.shape[1:] != (360, 357):
            raise ValueError(
                "ResUNetSino expects default DDF sinograms with shape "
                f"(N, 360, 357), received {tuple(sinogram.shape)}"
            )
        residual = sinogram.unsqueeze(1)
        x0 = self.encoder0(residual)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        xb = self.bottleneck(x2)
        x = self.decoder2(xb, x2)
        x = self.decoder1(x, x1)
        x = self.decoder0(x, x0)
        return (residual + self.out(x)).squeeze(1)
