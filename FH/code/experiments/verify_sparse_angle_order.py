"""Verify that experiment loaders reproduce the legacy DDF angular order."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import ddf_experiment_lib as standard
from experiments.newablation import ddf_experiment_lib as ablation


def legacy_reference(sparse_channels):
    batch_size, channels, angle_count, sensor_count = sparse_channels.shape
    joined = sparse_channels[:, 0]
    for channel_index in range(1, channels):
        joined = torch.cat((joined, sparse_channels[:, channel_index]), dim=2)
    return joined.reshape(batch_size, angle_count * channels, sensor_count)


def verify_library(library, factor):
    full = torch.arange(360, dtype=torch.float32).reshape(1, 360, 1)
    channels = library.interpolate_sparse_views(full, factor)
    actual = library.reshape_sparse_channels(channels)
    expected = legacy_reference(channels)
    if not torch.equal(actual, expected):
        raise AssertionError(f"{library.__name__} failed legacy ordering for S={factor}")
    if not torch.equal(actual[:, 0::factor], full[:, 0::factor]):
        raise AssertionError(f"{library.__name__} misplaced measured views for S={factor}")


def main():
    for factor in (2, 4, 8, 12):
        verify_library(standard, factor)
        verify_library(ablation, factor)
    print("[preflight] sparse angular order matches legacy DDF for S=2,4,8,12")


if __name__ == "__main__":
    main()
