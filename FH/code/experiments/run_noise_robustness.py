#!/usr/bin/env python3
"""Evaluate pretrained sparse-view CT models under projection-domain noise.

Noise is applied to the full-view line-integral sinogram before the original
uniform sparse-view sampling and linear interpolation.  No model is retrained.
"""

import argparse
import csv
import gc
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from ddf_experiment_lib import load_config, reshape_interpolated_sinogram
import run_quantitative_metrics as quantitative_metrics
from run_quantitative_metrics import (
    DEFAULT_TEST_DATA,
    evaluate_cascade,
    evaluate_cascade_original,
    evaluate_ddf,
    evaluate_sparse_fbp,
    finalize,
    find_checkpoint,
)


PROGRESS_DESCRIPTION = "evaluation"
PLAIN_MAKE_BATCHES = quantitative_metrics.make_batches


def tqdm_make_batches(dataset, batch_size, max_samples):
    total_samples = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    total_batches = (total_samples + batch_size - 1) // batch_size
    return tqdm(
        PLAIN_MAKE_BATCHES(dataset, batch_size, max_samples),
        total=total_batches,
        desc=PROGRESS_DESCRIPTION,
        unit="batch",
        dynamic_ncols=True,
    )


# The reused evaluators look up make_batches in their defining module. Replacing
# that iterator only adds display; it does not alter samples or metric values.
quantitative_metrics.make_batches = tqdm_make_batches


class ProjectionNoiseDataset(torch.utils.data.Dataset):
    """Fixed noisy sinograms shared by every method in one test condition."""

    def __init__(
        self,
        npz_path,
        sparse_factor,
        photon_count=None,
        electronic_sigma=0.0,
        seed=2026,
    ):
        with np.load(npz_path) as data:
            full_sinogram = torch.tensor(data["sin357"], dtype=torch.float32)
            self.ct_label = torch.tensor(data["ct_label"], dtype=torch.float32).permute(0, 3, 1, 2)

        if photon_count is not None:
            full_sinogram = add_measurement_noise(
                full_sinogram,
                photon_count=float(photon_count),
                electronic_sigma=float(electronic_sigma),
                seed=int(seed),
            )
        elif electronic_sigma != 0:
            raise ValueError("electronic noise is defined in photon counts and requires photon_count")

        self.sinogram_input = reshape_interpolated_sinogram(full_sinogram, int(sparse_factor))

    def __len__(self):
        return len(self.sinogram_input)

    def __getitem__(self, index):
        return self.sinogram_input[index], self.ct_label[index]


def add_measurement_noise(line_integrals, photon_count, electronic_sigma, seed):
    """Apply Poisson quantum noise and optional additive electronic noise.

    For a noiseless line integral p, detected counts follow
    y ~ Poisson(I0 * exp(-p)).  Optional N(0, sigma_e^2) electronic noise is
    added in the photon-count domain, followed by the log transform
    p_noisy = -log(max(y, 1) / I0).
    """
    if photon_count <= 0:
        raise ValueError("photon_count must be positive")
    if electronic_sigma < 0:
        raise ValueError("electronic_sigma must be non-negative")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    attenuation = line_integrals.clamp_min(0.0)
    expected_counts = photon_count * torch.exp(-attenuation)
    measured_counts = torch.poisson(expected_counts, generator=generator)
    if electronic_sigma > 0:
        measured_counts.add_(
            torch.randn(measured_counts.shape, generator=generator) * electronic_sigma
        )
    measured_counts.clamp_min_(1.0)
    return -torch.log(measured_counts / photon_count)


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def condition_name(photon_count, electronic_sigma):
    if photon_count is None:
        return "clean"
    photon_label = f"{photon_count:.0e}".replace("+", "")
    if electronic_sigma == 0:
        return f"poisson_I0_{photon_label}"
    return f"poisson_I0_{photon_label}_gaussian_sigma_{electronic_sigma:g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_data", default=str(DEFAULT_TEST_DATA))
    parser.add_argument("--sparse_factor", type=int, default=12)
    parser.add_argument("--photon_counts", default="1e5,1e4")
    parser.add_argument(
        "--electronic_sigmas",
        default="0",
        help="Photon-count standard deviations. Use 0,5 to add optional Gaussian conditions.",
    )
    parser.add_argument("--include_clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--methods", default="sparse_fbp,cascade,ddf")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_csv", default="results/noise_robustness_S12.csv")
    args = parser.parse_args()

    test_path = Path(args.test_data)
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]

    conditions = []
    if args.include_clean:
        conditions.append((None, 0.0))
    for photon_count in parse_float_list(args.photon_counts):
        for electronic_sigma in parse_float_list(args.electronic_sigmas):
            conditions.append((photon_count, electronic_sigma))

    rows = []
    for condition_index, (photon_count, electronic_sigma) in enumerate(conditions):
        name = condition_name(photon_count, electronic_sigma)
        print(f"[condition] {name}")
        dataset = ProjectionNoiseDataset(
            test_path,
            sparse_factor=args.sparse_factor,
            photon_count=photon_count,
            electronic_sigma=electronic_sigma,
            seed=args.seed + condition_index,
        )

        for method in methods:
            global PROGRESS_DESCRIPTION
            PROGRESS_DESCRIPTION = f"{name} | {method}"
            if method == "sparse_fbp":
                checkpoint = ""
                acc = evaluate_sparse_fbp(
                    dataset, config, args.batch_size, args.max_samples, device
                )
            elif method == "cascade":
                checkpoint = find_checkpoint("cascade", args.sparse_factor)
                if checkpoint is None:
                    raise FileNotFoundError(
                        f"Cascade checkpoint not found for S={args.sparse_factor}"
                    )
                if "cascade_original" in str(checkpoint):
                    acc = evaluate_cascade_original(
                        dataset,
                        args.sparse_factor,
                        checkpoint,
                        args.batch_size,
                        args.max_samples,
                        device,
                    )
                else:
                    acc = evaluate_cascade(
                        dataset,
                        args.sparse_factor,
                        checkpoint,
                        config,
                        args.batch_size,
                        args.max_samples,
                        device,
                    )
            elif method == "ddf":
                checkpoint = find_checkpoint("ddf", args.sparse_factor)
                if checkpoint is None:
                    raise FileNotFoundError(
                        f"DDF checkpoint not found for S={args.sparse_factor}"
                    )
                acc = evaluate_ddf(
                    dataset,
                    args.sparse_factor,
                    checkpoint,
                    args.batch_size,
                    args.max_samples,
                    device,
                )
            else:
                raise ValueError(f"unsupported method: {method}")

            row = finalize(method, args.sparse_factor, acc, checkpoint)
            row.update(
                {
                    "condition": name,
                    "photon_count": "" if photon_count is None else photon_count,
                    "electronic_sigma": electronic_sigma,
                    "seed": args.seed + condition_index,
                }
            )
            rows.append(row)
            print(
                f"[result] {method}: PSNR={row['psnr']:.6f}, SSIM={row['ssim']:.6f}, "
                f"RMSE={row['rmse']:.6f}, MAE={row['mae']:.6f}"
            )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del dataset

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "photon_count",
        "electronic_sigma",
        "seed",
        "method",
        "sparse_factor",
        "psnr",
        "ssim",
        "rmse",
        "mae",
        "num_samples",
        "checkpoint",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved CSV: {output_path}")


if __name__ == "__main__":
    main()
