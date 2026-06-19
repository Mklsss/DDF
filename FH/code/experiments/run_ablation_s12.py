import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    import swanlab
except ImportError:
    swanlab = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ssim
from ddf_experiment_lib import (
    build_model,
    get_loader,
    load_config,
    resolve_path,
    set_seed,
)


SPARSE_FACTOR = 12


def load_original_ddf_namespace():
    text = Path("DDP_run_c12.py").read_text(encoding="utf-8")
    stop = text.index("model = mymodel().to(device)")
    ns = {"__name__": "__ablation_s12__"}
    exec(text[:stop], ns)
    return ns


def unwrap_state_dict(obj):
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def psnr_batch(prediction, target):
    prediction = prediction.detach().cpu()
    target = target.detach().cpu()
    scores = []
    for i in range(prediction.shape[0]):
        img1 = prediction[i]
        img2 = target[i]
        data_range = torch.max(img2) - torch.min(img2)
        mse = torch.mean((img1 - img2) ** 2)
        if mse.item() <= 0:
            scores.append(float("inf"))
        else:
            scores.append((10.0 * torch.log10((data_range ** 2) / mse)).item())
    return float(np.mean(scores))


def evaluate_model(model, loader, device):
    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    num_samples = 0
    with torch.no_grad():
        for sin_in, label in loader:
            original_n = sin_in.shape[0]
            if original_n == 1:
                sin_in = torch.cat([sin_in, sin_in], dim=0)
                label = torch.cat([label, label], dim=0)

            sin_in = sin_in.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            out = model(sin_in)
            pred = out[0] if isinstance(out, tuple) else out
            pred = pred[:original_n].clamp(0, 1).detach().cpu()
            label = label[:original_n].detach().cpu()

            psnr_sum += psnr_batch(pred, label) * original_n
            ssim_sum += float(ssim.ssim(pred, label).item()) * original_n
            num_samples += original_n
    return psnr_sum / num_samples, ssim_sum / num_samples, num_samples


def train_variant(method_key, checkpoint, args, device, config):
    run = None
    if args.swanlab:
        if swanlab is None:
            print("swanlab is not installed; continue without SwanLab logging.")
        else:
            run = swanlab.init(
                project=args.project,
                experiment_name=f"{method_key}_S12_{args.epochs}ep",
                config={
                    "method_key": method_key,
                    "sparse_factor": SPARSE_FACTOR,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "val_batch_size": args.val_batch_size,
                    "lr": args.lr,
                    "step_size": args.step_size,
                    "gamma": args.gamma,
                    "checkpoint": str(checkpoint),
                },
            )
    train_loader = get_loader(resolve_path(config["train_data"]), SPARSE_FACTOR, args.batch_size, shuffle=False)
    val_loader = get_loader(resolve_path(config["test_data"]), SPARSE_FACTOR, args.val_batch_size, shuffle=False)

    model = build_model(method_key, SPARSE_FACTOR, config).to(device)
    if args.resume and Path(checkpoint).exists():
        state = unwrap_state_dict(torch.load(checkpoint, map_location=device))
        print(model.load_state_dict(state, strict=False))
        print(f"resumed model weights from: {checkpoint}")
    for name in ("fbp", "fp"):
        module = getattr(model, name, None)
        if module is not None:
            for p in module.parameters():
                p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_psnr = -1.0
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f"{method_key} S=12 epoch {epoch}/{args.epochs - 1}", ncols=120)
        for sin_in, label in pbar:
            sin_in = sin_in.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            out = model(sin_in)
            pred = out[0] if isinstance(out, tuple) else out
            loss = torch.nn.MSELoss()(pred, label)

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss: {loss}")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss_sum += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        scheduler.step()
        val_psnr, val_ssim, num_samples = evaluate_model(model, val_loader, device)
        avg_train_loss = train_loss_sum / max(len(train_loader), 1)
        print(f"{method_key}, S=12, epoch={epoch}, train_loss={avg_train_loss:.6f}, val_psnr={val_psnr:.6f}, val_ssim={val_ssim:.6f}")
        if run is not None:
            swanlab.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "best_psnr": max(best_psnr, val_psnr),
            })

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_state = {k: v for k, v in model.state_dict().items() if not (k.startswith('fbp.') or k.startswith('fp.'))}
            torch.save(save_state, checkpoint)
            print(f"saved best checkpoint: {checkpoint}")

    if run is not None:
        swanlab.finish()
    return checkpoint


def evaluate_original_proposed(checkpoint, args, device):
    ns = load_original_ddf_namespace()
    test_path = args.test_data if args.test_data is not None else "./data/test_meiaonew.npz"
    dataset = ns["load_data"](test_path)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    model = ns["mymodel"]().to(device)
    state = unwrap_state_dict(torch.load(checkpoint, map_location=device))
    model.load_state_dict(state, strict=False)
    return evaluate_model(model, loader, device)


def evaluate_lib_variant(method_key, checkpoint, args, device, config):
    loader = get_loader(resolve_path(config["test_data"]), SPARSE_FACTOR, args.val_batch_size, shuffle=False)
    model = build_model(method_key, SPARSE_FACTOR, config).to(device)
    state = unwrap_state_dict(torch.load(checkpoint, map_location=device))
    model.load_state_dict(state, strict=False)
    return evaluate_model(model, loader, device)


def write_csv(rows, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["module", "variant", "sparse_factor", "psnr", "ssim", "num_samples", "checkpoint"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved CSV: {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "eval", "train_eval"), default="eval")
    parser.add_argument("--target", choices=("sine", "ct", "all"), default="all")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.33)
    parser.add_argument("--config", default=None)
    parser.add_argument("--test_data", default=None)
    parser.add_argument("--output_csv", default="results/ablation_s12.csv")
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--project", default="DDF-reproduction")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--proposed_checkpoint", default="weights/DDF_c12_best.pth")
    parser.add_argument("--sine_checkpoint", default="weights/DDF_c12_sine_ablation_best.pth")
    parser.add_argument("--ct_checkpoint", default="weights/DDF_c12_ct_ablation_best.pth")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.test_data is not None:
        config["test_data"] = args.test_data
    config["nafnet"] = dict(config["nafnet"])
    config["nafnet"]["width"] = 32
    set_seed(int(config["seed"]))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.mode in ("train", "train_eval"):
        if args.target in ("sine", "all"):
            train_variant("ddf_no_sine_fusion", args.sine_checkpoint, args, device, config)
        if args.target in ("ct", "all"):
            train_variant("ddf_ct_conv_fusion", args.ct_checkpoint, args, device, config)

    if args.mode in ("eval", "train_eval"):
        rows = []

        if Path(args.proposed_checkpoint).exists():
            psnr, ssim_value, n = evaluate_original_proposed(args.proposed_checkpoint, args, device)
            rows.append({
                "module": "sine_fusion",
                "variant": "proposed",
                "sparse_factor": 12,
                "psnr": f"{psnr:.6f}",
                "ssim": f"{ssim_value:.6f}",
                "num_samples": n,
                "checkpoint": args.proposed_checkpoint,
            })
            rows.append({
                "module": "ct_fusion",
                "variant": "proposed",
                "sparse_factor": 12,
                "psnr": f"{psnr:.6f}",
                "ssim": f"{ssim_value:.6f}",
                "num_samples": n,
                "checkpoint": args.proposed_checkpoint,
            })
        else:
            print("checkpoint for S=12 not found, skip.")

        if Path(args.sine_checkpoint).exists():
            psnr, ssim_value, n = evaluate_lib_variant("ddf_no_sine_fusion", args.sine_checkpoint, args, device, config)
            rows.append({
                "module": "sine_fusion",
                "variant": "replacement",
                "sparse_factor": 12,
                "psnr": f"{psnr:.6f}",
                "ssim": f"{ssim_value:.6f}",
                "num_samples": n,
                "checkpoint": args.sine_checkpoint,
            })
        else:
            print("checkpoint for S=12 not found, skip.")

        if Path(args.ct_checkpoint).exists():
            psnr, ssim_value, n = evaluate_lib_variant("ddf_ct_conv_fusion", args.ct_checkpoint, args, device, config)
            rows.append({
                "module": "ct_fusion",
                "variant": "replacement",
                "sparse_factor": 12,
                "psnr": f"{psnr:.6f}",
                "ssim": f"{ssim_value:.6f}",
                "num_samples": n,
                "checkpoint": args.ct_checkpoint,
            })
        else:
            print("checkpoint for S=12 not found, skip.")

        write_csv(rows, args.output_csv)


if __name__ == "__main__":
    main()
