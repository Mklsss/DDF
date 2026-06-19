"""SwinIR projection-domain backbone for the universal DDF ablation.

The DDF projection interface is fixed to ``(N, 360, 357) -> (N, 360, 357)``.
SwinIR internally pads the non-window-aligned detector width and crops its
output back to this shape, so no FBP, FP, feedback, or fusion code changes.
"""

import sys
from pathlib import Path

import torch
from torch import nn


DEFAULT_ROOT = Path("/autodl-fs/data/FH/code")
if str(DEFAULT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_ROOT))

from FHinner.tnt_v1.swinir import SwinIR


class SwinIRSino(nn.Module):
    """P-Swin: residual SwinIR restoration for a 360-by-357 sinogram."""

    def __init__(self, config):
        super().__init__()
        self.model = SwinIR(
            img_size=config["img_size"],
            patch_size=1,
            in_chans=1,
            embed_dim=config["embed_dim"],
            depths=config["depths"],
            num_heads=config["num_heads"],
            window_size=config["window_size"],
            mlp_ratio=config["mlp_ratio"],
            drop_path_rate=config["drop_path_rate"],
            use_checkpoint=config["use_checkpoint"],
            upscale=1,
            img_range=1.0,
            upsampler="",
            resi_connection=config["resi_connection"],
        )

    def forward(self, sinogram):
        if sinogram.ndim != 3 or sinogram.shape[1:] != (360, 357):
            raise ValueError(
                "SwinIRSino expects default DDF sinograms with shape "
                f"(N, 360, 357), received {tuple(sinogram.shape)}"
            )
        return self.model(sinogram.unsqueeze(1)).squeeze(1)
