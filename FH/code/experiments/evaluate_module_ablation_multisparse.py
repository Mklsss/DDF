"""Evaluate completed multi-sparsity module ablations on the held-out test set."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.newablation.ddf_experiment_lib import (
    SinogramCTDataset,
    build_model,
    load_config,
    psnr_batch,
    set_seed,
    ssim_batch,
)


METHODS = {
    "no_sine": "ddf_no_sine_fusion",
    "no_ct": "ddf_ct_conv_fusion",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--variants", default="no_sine,no_ct")
    parser.add_argument("--sparse_factors", default="2,4,8")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def parse_csv_values(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_per_slice_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=("task", "sparse_factor", "slice_index", "psnr", "ssim"),
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def evaluate_task(task, method, factor, checkpoint, test_npz, config, device, batch_size):
    dataset = SinogramCTDataset(test_npz, factor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    model = build_model(method, factor, config).to(device)
    payload = torch.load(checkpoint, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    model.eval()

    per_slice = []
    sample_index = 0
    with torch.no_grad():
        for sparse_sinogram, target_ct in tqdm(loader, desc=f"test {task}", ncols=110):
            prediction, _ = model(sparse_sinogram.to(device=device, dtype=torch.float32))
            prediction = prediction.clamp(0, 1)
            target_ct = target_ct.to(device=device, dtype=torch.float32)
            for batch_index in range(prediction.shape[0]):
                pred_item = prediction[batch_index : batch_index + 1]
                target_item = target_ct[batch_index : batch_index + 1]
                per_slice.append(
                    {
                        "task": task,
                        "sparse_factor": factor,
                        "slice_index": sample_index,
                        "psnr": psnr_batch(pred_item, target_item),
                        "ssim": ssim_batch(pred_item, target_item),
                    }
                )
                sample_index += 1

    result = {
        "task": task,
        "sparse_factor": factor,
        "best_epoch": int(payload.get("epoch", -1)) if isinstance(payload, dict) else -1,
        "best_val_psnr": float(payload.get("best_val_psnr", float("nan")))
        if isinstance(payload, dict)
        else float("nan"),
        "test_psnr": float(np.mean([row["psnr"] for row in per_slice])),
        "test_ssim": float(np.mean([row["ssim"] for row in per_slice])),
        "test_slices": len(per_slice),
        "checkpoint": str(checkpoint),
    }
    del model, loader, dataset, payload, state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, per_slice


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    config = load_config(args.config)
    config["nafnet"] = dict(config["nafnet"])
    config["nafnet"]["width"] = 32

    results = []
    per_slice_rows = []
    for variant in parse_csv_values(args.variants):
        if variant not in METHODS:
            raise ValueError(f"unknown variant: {variant}")
        for factor in parse_csv_values(args.sparse_factors, int):
            task = f"module_{variant}_S{factor}"
            checkpoint = checkpoint_dir / f"{task}.pth.tar"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            result, rows = evaluate_task(
                task,
                METHODS[variant],
                factor,
                checkpoint,
                args.test_npz,
                config,
                device,
                args.batch_size,
            )
            results.append(result)
            per_slice_rows.extend(rows)
            task_dir = output_dir / "tasks"
            atomic_write_text(task_dir / f"{task}.json", json.dumps(result, indent=2))
            write_per_slice_csv(task_dir / f"{task}_per_slice.csv", rows)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    atomic_write_text(output_dir / "module_ablation_multisparse.json", json.dumps(results, indent=2))

    summary_lines = [
        "task,sparse_factor,best_epoch,best_val_psnr,test_psnr,test_ssim,test_slices,checkpoint"
    ]
    for row in results:
        summary_lines.append(
            f"{row['task']},{row['sparse_factor']},{row['best_epoch']},"
            f"{row['best_val_psnr']:.8f},{row['test_psnr']:.8f},{row['test_ssim']:.8f},"
            f"{row['test_slices']},{row['checkpoint']}"
        )
    atomic_write_text(output_dir / "module_ablation_multisparse.csv", "\n".join(summary_lines) + "\n")

    write_per_slice_csv(output_dir / "module_ablation_multisparse_per_slice.csv", per_slice_rows)


if __name__ == "__main__":
    main()
