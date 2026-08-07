"""Merge native DuDoTrans checkpoint evaluations with the unified DDF results.

The script does not train or alter a checkpoint.  It imports the per-slice
metrics written by ``dudotrans_autodl/test_dudotrans.py``, pairs them with the
same ordered test slices in the unified evaluation, and reports two-sided
Wilcoxon signed-rank tests with Holm correction and paired bootstrap CIs.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon


PAPER_METHODS = ("sparse_fbp", "redcnn", "dudotrans", "cascade", "ddf")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_dudo_spec(spec):
    factor, views, csv_path, checkpoint = spec.split(":", 3)
    return int(factor), int(views), Path(csv_path), Path(checkpoint)


def holm_adjust(p_values):
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running_max = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[original_index]))
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def rank_biserial(differences):
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def bootstrap_ci(differences, repeats, seed):
    rng = np.random.default_rng(seed)
    sample_count = differences.size
    means = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 1000):
        size = min(1000, repeats - start)
        indices = rng.integers(0, sample_count, size=(size, sample_count))
        means[start : start + size] = differences[indices].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def analyse(rows, bootstrap_repeats, seed):
    lookup = {
        (int(row["sparse_factor"]), row["method"], int(row["sample_index"])): row
        for row in rows
    }
    output = []
    for factor in sorted({int(row["sparse_factor"]) for row in rows}):
        ddf_indices = sorted(
            int(row["sample_index"])
            for row in rows
            if int(row["sparse_factor"]) == factor and row["method"] == "ddf"
        )
        for metric in ("psnr", "ssim"):
            family = []
            for comparator in PAPER_METHODS:
                if comparator == "ddf":
                    continue
                comparator_indices = sorted(
                    int(row["sample_index"])
                    for row in rows
                    if int(row["sparse_factor"]) == factor and row["method"] == comparator
                )
                if not ddf_indices or comparator_indices != ddf_indices:
                    continue
                ddf_values = np.asarray(
                    [float(lookup[(factor, "ddf", index)][metric]) for index in ddf_indices]
                )
                comparator_values = np.asarray(
                    [float(lookup[(factor, comparator, index)][metric]) for index in ddf_indices]
                )
                differences = ddf_values - comparator_values
                result = wilcoxon(
                    differences,
                    zero_method="wilcox",
                    correction=False,
                    alternative="two-sided",
                    method="auto",
                )
                ci_low, ci_high = bootstrap_ci(
                    differences,
                    repeats=bootstrap_repeats,
                    seed=seed + factor * 100 + len(family),
                )
                family.append(
                    {
                        "sparse_factor": factor,
                        "metric": metric,
                        "comparison": f"DDF vs {comparator}",
                        "n_pairs": differences.size,
                        "ddf_mean": float(ddf_values.mean()),
                        "comparator_mean": float(comparator_values.mean()),
                        "mean_paired_difference": float(differences.mean()),
                        "ci95_low": float(ci_low),
                        "ci95_high": float(ci_high),
                        "wilcoxon_statistic": float(result.statistic),
                        "p_raw": float(result.pvalue),
                        "rank_biserial": rank_biserial(differences),
                        "ddf_better_count": int(np.count_nonzero(differences > 0)),
                        "tie_count": int(np.count_nonzero(differences == 0)),
                        "ddf_worse_count": int(np.count_nonzero(differences < 0)),
                    }
                )
            if family:
                adjusted = holm_adjust([row["p_raw"] for row in family])
                for row, p_adjusted in zip(family, adjusted):
                    row["p_holm"] = float(p_adjusted)
                    row["significant_0_05"] = bool(p_adjusted < 0.05)
                output.extend(family)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_metrics", required=True)
    parser.add_argument(
        "--dudotrans",
        action="append",
        default=[],
        metavar="FACTOR:VIEWS:CSV:CHECKPOINT",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    base_path = Path(args.base_metrics).resolve()
    rows = []
    for row in read_csv(base_path):
        if row["method"] in PAPER_METHODS and row["method"] != "dudotrans":
            rows.append(row)

    dudo_metadata = {}
    for spec in args.dudotrans:
        factor, views, csv_path, checkpoint = parse_dudo_spec(spec)
        csv_path = csv_path.resolve()
        checkpoint = checkpoint.resolve()
        imported = read_csv(csv_path)
        if len(imported) != 200:
            raise ValueError(f"Expected 200 DuDoTrans rows for S={factor}, got {len(imported)}")
        for row in imported:
            rows.append(
                {
                    "sample_index": int(row["index"]),
                    "sparse_factor": factor,
                    "method": "dudotrans",
                    "psnr": float(row["pred_psnr"]),
                    "ssim": float(row["pred_ssim"]),
                    "checkpoint": str(checkpoint),
                }
            )
        dudo_metadata[str(factor)] = {
            "views": views,
            "metrics_csv": str(csv_path),
            "metrics_csv_sha256": sha256(csv_path),
            "checkpoint": str(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256(checkpoint),
            "native_image_size": 512,
        }

    rows.sort(key=lambda row: (int(row["sparse_factor"]), PAPER_METHODS.index(row["method"]), int(row["sample_index"])))
    paired = analyse(rows, args.bootstrap_repeats, args.seed)

    means = []
    for factor in sorted({int(row["sparse_factor"]) for row in rows}):
        for method in PAPER_METHODS:
            subset = [row for row in rows if int(row["sparse_factor"]) == factor and row["method"] == method]
            if subset:
                means.append(
                    {
                        "sparse_factor": factor,
                        "method": method,
                        "n": len(subset),
                        "psnr_mean": float(np.mean([float(row["psnr"]) for row in subset])),
                        "ssim_mean": float(np.mean([float(row["ssim"]) for row in subset])),
                    }
                )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "per_slice_metrics.csv", rows)
    write_csv(output_dir / "method_means.csv", means)
    write_csv(output_dir / "paired_significance.csv", paired)
    metadata = {
        "analysis_scope": "paired image-slice-level comparisons on one held-out patient",
        "sample_count": 200,
        "base_metrics": str(base_path),
        "base_metrics_sha256": sha256(base_path),
        "dudotrans_protocol": "original checkpoint, native view count, native 512x512 preprocessing",
        "dudotrans": dudo_metadata,
        "unavailable_original_checkpoints": {
            "2": "the local 120-view checkpoint is truncated and cannot be loaded",
            "8": "the local 60-view checkpoint is truncated and cannot be loaded",
        },
        "wilcoxon": "two-sided paired Wilcoxon signed-rank test",
        "multiplicity": "Holm correction within each sparse-factor/metric family",
        "confidence_interval": f"paired percentile bootstrap of mean difference, {args.bootstrap_repeats} resamples",
        "seed": args.seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
