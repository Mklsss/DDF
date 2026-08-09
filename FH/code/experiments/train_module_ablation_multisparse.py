"""Train one reviewer-requested module ablation with a fixed 1600/200 split."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.newablation.ddf_experiment_lib import (
    SinogramCTDataset,
    build_model,
    evaluate_model,
    freeze_projection_layers,
    load_config,
    set_seed,
)


METHODS = {
    "no_sine": "ddf_no_sine_fusion",
    "no_ct": "ddf_ct_conv_fusion",
}


def split_loaders(npz_path, sparse_factor, train_count, val_count, batch_size):
    dataset = SinogramCTDataset(npz_path, sparse_factor)
    required = train_count + val_count
    if len(dataset) < required:
        raise ValueError(f"dataset has {len(dataset)} samples, but {required} are required")
    train_set = Subset(dataset, range(train_count))
    val_set = Subset(dataset, range(train_count, required))
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader


def test_loader(npz_path, sparse_factor, batch_size):
    return DataLoader(
        SinogramCTDataset(npz_path, sparse_factor),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )


def checkpoint_model_state(payload):
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def atomic_torch_save(payload, checkpoint):
    """Write a complete checkpoint before replacing the previous best file."""
    checkpoint = Path(checkpoint)
    temporary = checkpoint.with_name(f".{checkpoint.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=sorted(METHODS))
    parser.add_argument("--sparse_factor", required=True, type=int, choices=(2, 4, 8))
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--result_json", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.33)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--train_count", type=int, default=1600)
    parser.add_argument("--val_count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab_project", default="DDF-reviewer-ablation-2026")
    parser.add_argument("--swanlab_run_name", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    task_name = f"module_{args.variant}_S{args.sparse_factor}"
    checkpoint = Path(args.checkpoint)
    result_json = Path(args.result_json)
    for required in (Path(args.train_npz), Path(args.test_npz)):
        if not required.is_file():
            raise FileNotFoundError(required)
    if checkpoint.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(f"checkpoint already exists: {checkpoint}; pass --resume to continue")
    if args.dry_run:
        print(json.dumps({
            "task": task_name,
            "method": METHODS[args.variant],
            "sparse_factor": args.sparse_factor,
            "train_count": args.train_count,
            "val_count": args.val_count,
            "epochs": args.epochs,
            "checkpoint": str(checkpoint),
            "swanlab_run": args.swanlab_run_name or task_name,
        }, indent=2))
        return

    set_seed(args.seed)
    config = load_config(args.config)
    config["nafnet"] = dict(config["nafnet"])
    config["nafnet"]["width"] = 32
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = split_loaders(
        args.train_npz, args.sparse_factor, args.train_count, args.val_count, args.batch_size
    )
    model = build_model(METHODS[args.variant], args.sparse_factor, config).to(device)
    freeze_projection_layers(model)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    start_epoch = 0
    best_psnr = -float("inf")
    if args.resume and checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint_model_state(payload), strict=True)
        if isinstance(payload, dict):
            if "optimizer" in payload:
                optimizer.load_state_dict(payload["optimizer"])
            if "scheduler" in payload:
                scheduler.load_state_dict(payload["scheduler"])
            start_epoch = int(payload.get("epoch", 0))
            best_psnr = float(payload.get("best_val_psnr", -float("inf")))
        print(f"resumed {checkpoint} at epoch {start_epoch}, best_val_psnr={best_psnr:.6f}")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    result_json.parent.mkdir(parents=True, exist_ok=True)
    run = None
    if args.swanlab:
        import swanlab
        run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_run_name or task_name,
            config={
                "family": "module",
                "variant": args.variant,
                "sparse_factor": args.sparse_factor,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "train_count": args.train_count,
                "validation_count": args.val_count,
                "seed": args.seed,
            },
        )

    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            progress = tqdm(
                train_loader,
                desc=f"{task_name} epoch {epoch}/{args.epochs}",
                ncols=120,
            )
            for sparse_sinogram, target_ct in progress:
                optimizer.zero_grad(set_to_none=True)
                prediction, _ = model(sparse_sinogram.to(device, dtype=torch.float32))
                loss = torch.nn.functional.mse_loss(
                    prediction, target_ct.to(device, dtype=torch.float32)
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at epoch {epoch}: {loss.item()}")
                loss.backward()
                optimizer.step()
                loss_sum += loss.detach().item()
                progress.set_postfix(loss=f"{loss.item():.6f}")
            scheduler.step()
            val_psnr, val_ssim = evaluate_model(model, val_loader, device)
            mean_loss = loss_sum / len(train_loader)
            metrics = {
                "epoch": epoch,
                "train/loss": mean_loss,
                "train/lr": optimizer.param_groups[0]["lr"],
                "val/psnr": val_psnr,
                "val/ssim": val_ssim,
            }
            print(
                f"epoch={epoch} {task_name} train_loss={mean_loss:.8f} "
                f"val_psnr={val_psnr:.6f} val_ssim={val_ssim:.6f}",
                flush=True,
            )
            if run is not None:
                run.log(metrics, step=epoch)
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                atomic_torch_save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_psnr": best_psnr,
                    "task": task_name,
                }, checkpoint)
                print(f"updated best checkpoint: {checkpoint}", flush=True)

        payload = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint_model_state(payload), strict=True)
        test_psnr, test_ssim = evaluate_model(
            model, test_loader(args.test_npz, args.sparse_factor, args.batch_size), device
        )
        result = {
            "task": task_name,
            "sparse_factor": args.sparse_factor,
            "best_epoch": int(payload["epoch"]),
            "best_val_psnr": float(payload["best_val_psnr"]),
            "test_psnr": test_psnr,
            "test_ssim": test_ssim,
            "checkpoint": str(checkpoint),
        }
        result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        if run is not None:
            run.log({"test/psnr": test_psnr, "test/ssim": test_ssim})
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
