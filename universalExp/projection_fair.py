"""Fair original-DDF protocol for controlled backbone replacements.

P-CNN/P-Swin replace ``sin``; I-CNN/I-Restor replace ``ct``; Both-CNN/Mixed
replace both. Every untouched component is instantiated from the original
training script and loaded from the same original checkpoint.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn


THIS_DIR = Path(__file__).resolve().parent
LEGACY_ROOT = Path("/autodl-fs/data/FH/code")
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))


@contextmanager
def _legacy_cwd():
    """The original DDF script loads matrices through cwd-relative paths."""
    previous = Path.cwd()
    os.chdir(LEGACY_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


@lru_cache(maxsize=None)
def legacy_namespace(sparse_factor: int):
    """Load definitions from the exact original S-specific DDF script only."""
    script = LEGACY_ROOT / f"DDP_run_c{int(sparse_factor)}.py"
    source = script.read_text(encoding="utf-8")
    source = source[:source.index("model = mymodel().to(device)")]
    namespace = {"__name__": "__fair_projection_ddf__"}
    with _legacy_cwd():
        exec(compile(source, str(script), "exec"), namespace)
    return namespace


class OriginalDDFWithReplacement(nn.Module):
    """Original DDF with optional projection and image replacements."""

    def __init__(
        self,
        sparse_factor: int,
        projection: nn.Module | None = None,
        image: nn.Module | None = None,
    ):
        super().__init__()
        namespace = legacy_namespace(int(sparse_factor))
        with _legacy_cwd():
            self.sin = projection or namespace["sin_angle"](
                num_sensor=357, angle=360 // int(sparse_factor), num_heads=1
            )
            # Keep these classes and their dimensions exactly as in DDP_run_c12.py.
            self.fbp = namespace["FbpLayer"]()
            self.fp = namespace["fp"]()
            self.gmlp = namespace["gmlp"]()
            self.ct = image or namespace["NAFNet"](
                img_channel=1, width=32, middle_blk_num=1,
                enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1],
            )
            self.fus_ct1 = namespace["CrossGatingBlock"]()

    def forward(self, sparse_sinogram):
        sin1 = self.sin(sparse_sinogram)
        ct1 = self.ct(self.fbp(sin1).permute(0, 3, 1, 2))
        feedback = self.fp(ct1).unsqueeze(1)

        # The legacy implementation uses squeeze(), which corrupts a final
        # single-item batch.  squeeze(1) is identical for normal batches while
        # preserving batch dimension, so all methods share one valid evaluator.
        g = self.gmlp.con1(feedback.squeeze(1))
        g = self.gmlp.act(g)
        restored = self.gmlp.con2(sin1)
        fused = self.gmlp.con3(restored * g)
        output, _ = self.fus_ct1(ct1, self.fbp(fused).permute(0, 3, 1, 2))
        return output, {"sinogram": sin1, "cascade": ct1}


def load_original_weights(model: nn.Module, checkpoint: str | Path, *, replaced_prefixes=()):
    """Load all original weights except deliberately replaced modules."""
    if isinstance(replaced_prefixes, str):
        replaced_prefixes = (replaced_prefixes,)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {
        key: value for key, value in state.items()
        if not any(key.startswith(prefix) for prefix in replaced_prefixes)
    }
    incompat = model.load_state_dict(state, strict=False)
    unexpected = list(incompat.unexpected_keys)
    missing = list(incompat.missing_keys)
    if unexpected:
        raise RuntimeError(f"unexpected original-checkpoint keys: {unexpected[:5]}")
    allowed_missing = set()
    for prefix in replaced_prefixes:
        module_name = prefix.removesuffix(".")
        allowed_missing.update(f"{prefix}{key}" for key in getattr(model, module_name).state_dict())
    if set(missing) != allowed_missing:
        raise RuntimeError(f"checkpoint did not load every shared DDF weight; missing={missing[:5]}")


def freeze_shared_ddf(model: OriginalDDFWithReplacement, train_prefixes):
    """Train only deliberately substituted backbone(s) in the controlled stage."""
    if isinstance(train_prefixes, str):
        train_prefixes = (train_prefixes,)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(tuple(train_prefixes))
