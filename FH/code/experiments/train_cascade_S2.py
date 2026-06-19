import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from ddf_experiment_lib import (
    ROOT,
    build_model,
    check_required_files,
    evaluate_model,
    freeze_projection_layers,
    get_loader,
    load_config,
    resolve_path,
    set_seed,
)


SPARSE_FACTOR = 2


def maybe_init_swanlab(args):
    if not args.swanlab:
        return None
    import swanlab

    swanlab.init(
        project=args.project,
        experiment_name=f"Cascade_S{SPARSE_FACTOR}_{args.epochs}ep",
        config={
            "method": "Cascade",
            "sparse_factor": SPARSE_FACTOR,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "val_batch_size": args.val_batch_size,
            "lr": args.lr,
            "checkpoint": args.output,
        },
    )
    return swanlab


def main():
    parser = argparse.ArgumentParser(description=f"Train Cascade baseline for S={SPARSE_FACTOR}.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.33)
    parser.add_argument("--output", default=f"weights/cascade_S{SPARSE_FACTOR}.pth")
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--project", default="DDF-reproduction")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = torch.device(args.device)

    missing_files = check_required_files(config, "cascade")
    if missing_files:
        for missing_file in missing_files:
            print(f"required file not found: {missing_file}")
        return

    train_loader = get_loader(resolve_path(config["train_data"]), SPARSE_FACTOR, args.batch_size, shuffle=False)
    val_loader = get_loader(resolve_path(config["test_data"]), SPARSE_FACTOR, args.val_batch_size, shuffle=False)

    model = build_model("cascade", SPARSE_FACTOR, config).to(device)
    freeze_projection_layers(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    swanlab = maybe_init_swanlab(args)
    best_psnr = -float("inf")

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f"Cascade S={SPARSE_FACTOR} epoch {epoch}/{args.epochs - 1}", ncols=120)

        for step, data in enumerate(pbar):
            sin_in, label = data
            sin_in_device = sin_in.to(device=device, dtype=torch.float32)
            label_device = label.to(device=device, dtype=torch.float32)

            pred, _ = model(sin_in_device)
            loss = nn.MSELoss()(pred, label_device)

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch}, step={step}: {loss.item()}")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_sum += float(loss.detach().item())
            avg_loss = loss_sum / (step + 1)
            pbar.set_postfix(loss=f"{avg_loss:.6f}")

        scheduler.step()

        val_psnr, val_ssim = evaluate_model(model, val_loader, device)
        avg_loss = loss_sum / max(len(train_loader), 1)
        print(
            f"Cascade, S={SPARSE_FACTOR}, epoch={epoch}, "
            f"train_loss={avg_loss:.6f}, val_psnr={val_psnr:.6f}, val_ssim={val_ssim:.6f}"
        )

        if swanlab is not None:
            swanlab.log(
                {
                    "train_loss": avg_loss,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                },
                step=epoch,
            )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), output_path)
            print(f"saved best checkpoint: {output_path}")

    if swanlab is not None:
        swanlab.finish()


if __name__ == "__main__":
    main()
