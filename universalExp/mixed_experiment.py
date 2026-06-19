"""Mixed universal DDF experiment: SwinIR-sino projection + RED-CNN image."""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = Path("/autodl-fs/data/FH/code")
if str(DEFAULT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_ROOT))

from cgb import CrossGatingBlock
from experiments.ddf_experiment_lib import (
    FbpLayer,
    ForwardProjectionLayer,
    GMLPSineFusion,
    SinogramCTDataset,
    evaluate_model,
    freeze_projection_layers,
    resolve_path,
    set_seed,
)
from image_backbones import RedCNN
from pswin import SwinIRSino


def load_config(path=None):
    config_path = Path(path) if path else THIS_DIR / "configs" / "mixed_default.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle), config_path.resolve()


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


class DDFMixed(nn.Module):
    """Complete DDF with SwinIR-sino and RED-CNN replacing its two backbones."""

    def __init__(self, config):
        super().__init__()
        self.sin = SwinIRSino(config["pswin"])
        self.fbp = FbpLayer(resolve_path(config["fbp_matrix"]))
        self.fp = ForwardProjectionLayer(
            resolve_path(config["fp_index"]), resolve_path(config["fp_data"])
        )
        self.ct = RedCNN(config["redcnn"]["channels"])
        self.sine_fusion = GMLPSineFusion()
        self.ct_fusion = CrossGatingBlock()

    def forward(self, sparse_sinogram):
        restored_sinogram = self.sin(sparse_sinogram)
        sparse_fbp_nhwc = self.fbp(sparse_sinogram)
        fbp1 = self.fbp(restored_sinogram).permute(0, 3, 1, 2)
        ct_nd = self.ct(fbp1)
        feedback_sinogram = self.fp(ct_nd).unsqueeze(1)
        fused_sinogram = self.sine_fusion(
            feedback_sinogram, restored_sinogram.unsqueeze(1)
        )
        fbp2 = self.fbp(fused_sinogram.squeeze(1)).permute(0, 3, 1, 2)
        ct_pre, _ = self.ct_fusion(ct_nd, fbp2)
        return ct_pre, {
            "sparse_fbp": sparse_fbp_nhwc.permute(0, 3, 1, 2),
            "cascade": ct_nd,
            "fused_sinogram": fused_sinogram,
        }


def make_loader(data_path, sparse_factor, batch_size, shuffle=False):
    dataset = SinogramCTDataset(resolve_path(data_path), sparse_factor)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def checkpoint_path(sparse_factor):
    return THIS_DIR / "checkpoints" / "ddf" / f"Mixed_P-Swin_REDCNN_S{sparse_factor}.pth"


def append_summary(row):
    path = THIS_DIR / "results" / "summary.csv"
    fields = [
        "architecture", "projection_backbone", "image_backbone", "sparse_factor",
        "psnr", "ssim", "checkpoint", "config",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate(sparse_factor, config, checkpoint, batch_size, device):
    model = DDFMixed(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    loader = make_loader(config["test_data"], sparse_factor, batch_size)
    return evaluate_model(model, loader, device)


def train(sparse_factor, config, batch_size, device, checkpoint, swanlab_run=None):
    print(f"[setup] building ddf Mixed model on {device}", flush=True)
    model = DDFMixed(config).to(device)
    freeze_projection_layers(model)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["train"]["learning_rate"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["train"]["step_size"],
        gamma=config["train"]["gamma"],
    )
    print("[setup] loading train dataset", flush=True)
    train_loader = make_loader(config["train_data"], sparse_factor, batch_size, shuffle=True)
    print("[setup] loading test dataset", flush=True)
    test_loader = make_loader(config["test_data"], sparse_factor, batch_size)
    print(
        f"[setup] ready: {len(train_loader)} train batches, {len(test_loader)} test batches",
        flush=True,
    )
    best_psnr = -float("inf")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(int(config["train"]["epochs"])):
        model.train()
        running_loss = 0.0
        progress = tqdm(
            train_loader,
            desc=f"ddf Mixed S={sparse_factor} epoch {epoch + 1}/{config['train']['epochs']}",
            unit="batch",
            dynamic_ncols=True,
        )
        for step, (sparse_sinogram, target_ct) in enumerate(progress, start=1):
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(sparse_sinogram.to(device=device, dtype=torch.float32))
            loss = nn.functional.mse_loss(
                prediction, target_ct.to(device=device, dtype=torch.float32)
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1}: {loss.item()}")
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().item()
            progress.set_postfix(
                loss=f"{running_loss / step:.6f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        scheduler.step()
        psnr, ssim = evaluate_model(model, test_loader, device)
        metrics = {
            "epoch": epoch + 1,
            "train/loss": running_loss / len(train_loader),
            "eval/psnr": psnr,
            "eval/ssim": ssim,
            "train/lr": optimizer.param_groups[0]["lr"],
        }
        print(
            f"epoch={epoch + 1} architecture=ddf-Mixed S={sparse_factor} "
            f"psnr={psnr:.6f} ssim={ssim:.6f}",
            flush=True,
        )
        if swanlab_run is not None:
            swanlab_run.log(metrics)
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(model.state_dict(), checkpoint)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Mixed universal DDF experiment")
    parser.add_argument("--mode", required=True, choices=("train", "test"))
    parser.add_argument("--sparse_factor", type=int, default=12)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab_project", default="universalExp")
    parser.add_argument("--swanlab_mode", choices=("cloud", "local", "offline"), default="cloud")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    set_seed(int(config["seed"]))
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else checkpoint_path(args.sparse_factor)
    if not checkpoint.is_absolute():
        checkpoint = THIS_DIR / checkpoint
    swanlab_run = None
    if args.swanlab:
        import swanlab
        swanlab_run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=f"ddf-Mixed-P-Swin-REDCNN-S{args.sparse_factor}",
            config={"architecture": "ddf-Mixed", "sparse_factor": args.sparse_factor, **config},
            mode=args.swanlab_mode,
        )
    if args.mode == "train":
        print(f"P-Swin parameters: {parameter_count(SwinIRSino(config['pswin'])):,}")
        print(f"RED-CNN parameters: {parameter_count(RedCNN(config['redcnn']['channels'])):,}")
        train(args.sparse_factor, config, args.batch_size, device, checkpoint, swanlab_run)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    psnr, ssim = evaluate(args.sparse_factor, config, checkpoint, args.batch_size, device)
    print(f"ddf Mixed S={args.sparse_factor}: PSNR={psnr:.6f} SSIM={ssim:.6f}")
    append_summary({
        "architecture": "ddf-Mixed",
        "projection_backbone": "SwinIR-sino/P-Swin",
        "image_backbone": "RED-CNN",
        "sparse_factor": args.sparse_factor,
        "psnr": f"{psnr:.6f}",
        "ssim": f"{ssim:.6f}",
        "checkpoint": str(checkpoint.resolve()),
        "config": str(config_path),
    })


if __name__ == "__main__":
    main()
