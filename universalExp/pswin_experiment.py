"""P-Swin experiment: replace only DDF's projection backbone with SwinIR."""

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
    DEFAULT_FACTORS, FbpLayer, ForwardProjectionLayer, GMLPSineFusion,
    SinogramCTDataset, build_nafnet, evaluate_model, freeze_projection_layers,
    resolve_path, set_seed,
)
from pswin import SwinIRSino


def load_config(path=None):
    config_path = Path(path) if path else THIS_DIR / "configs" / "pswin_default.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle), config_path.resolve()


def factor_list(value):
    return list(DEFAULT_FACTORS) if value is None else [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


class DDFPSwin(nn.Module):
    """Original DDF path with only ``sin_angle`` replaced by ``SwinIRSino``."""

    def __init__(self, config):
        super().__init__()
        self.sin = SwinIRSino(config["pswin"])
        self.fbp = FbpLayer(resolve_path(config["fbp_matrix"]))
        self.fp = ForwardProjectionLayer(resolve_path(config["fp_index"]), resolve_path(config["fp_data"]))
        self.ct = build_nafnet(config)
        self.sine_fusion = GMLPSineFusion()
        self.ct_fusion = CrossGatingBlock()

    def forward(self, sparse_sinogram):
        swinir_sinogram = self.sin(sparse_sinogram)
        sparse_fbp_nhwc = self.fbp(sparse_sinogram)
        fbp1 = self.fbp(swinir_sinogram).permute(0, 3, 1, 2)
        ct_nd = self.ct(fbp1)
        feedback_sinogram = self.fp(ct_nd).unsqueeze(1)
        fused_sinogram = self.sine_fusion(feedback_sinogram, swinir_sinogram.unsqueeze(1))
        fbp2 = self.fbp(fused_sinogram.squeeze(1)).permute(0, 3, 1, 2)
        ct_pre, _ = self.ct_fusion(ct_nd, fbp2)
        return ct_pre, {"sparse_fbp": sparse_fbp_nhwc.permute(0, 3, 1, 2), "cascade": ct_nd, "fused_sinogram": fused_sinogram}


def make_loader(data_path, sparse_factor, batch_size, shuffle=False):
    dataset = SinogramCTDataset(resolve_path(data_path), sparse_factor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


def checkpoint_path(sparse_factor):
    return THIS_DIR / "checkpoints" / "ddf" / f"P-Swin_NAFNet_S{sparse_factor}.pth"


def append_summary(row):
    path = THIS_DIR / "results" / "summary.csv"
    fields = ["architecture", "projection_backbone", "image_backbone", "sparse_factor", "psnr", "ssim", "checkpoint", "config"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def evaluate(sparse_factor, config, checkpoint, batch_size, device):
    model = DDFPSwin(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    return evaluate_model(model, make_loader(config["test_data"], sparse_factor, batch_size), device)


def train(sparse_factor, config, batch_size, device, checkpoint, swanlab_run=None):
    print(f"[setup] building ddf P-Swin model on {device}", flush=True)
    model = DDFPSwin(config).to(device)
    freeze_projection_layers(model)
    optimizer = torch.optim.Adam([item for item in model.parameters() if item.requires_grad], lr=config["train"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["train"]["step_size"], gamma=config["train"]["gamma"])
    print("[setup] loading train dataset", flush=True)
    train_loader = make_loader(config["train_data"], sparse_factor, batch_size, shuffle=True)
    print("[setup] loading test dataset", flush=True)
    test_loader = make_loader(config["test_data"], sparse_factor, batch_size, shuffle=False)
    print(f"[setup] ready: {len(train_loader)} train batches, {len(test_loader)} test batches", flush=True)
    best_psnr = -float("inf")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(int(config["train"]["epochs"])):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"ddf P-Swin S={sparse_factor} epoch {epoch + 1}/{config['train']['epochs']}", unit="batch", dynamic_ncols=True)
        for step, (sparse_sinogram, target_ct) in enumerate(progress, start=1):
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(sparse_sinogram.to(device=device, dtype=torch.float32))
            loss = nn.functional.mse_loss(prediction, target_ct.to(device=device, dtype=torch.float32))
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1}: {loss.item()}")
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().item()
            progress.set_postfix(loss=f"{running_loss / step:.6f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step()
        psnr, ssim = evaluate_model(model, test_loader, device)
        metrics = {"epoch": epoch + 1, "train/loss": running_loss / len(train_loader), "eval/psnr": psnr, "eval/ssim": ssim, "train/lr": optimizer.param_groups[0]["lr"]}
        print(f"epoch={epoch + 1} architecture=ddf-P-Swin S={sparse_factor} psnr={psnr:.6f} ssim={ssim:.6f}", flush=True)
        if swanlab_run is not None:
            swanlab_run.log(metrics)
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(model.state_dict(), checkpoint)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="P-Swin universal DDF experiment")
    parser.add_argument("--mode", required=True, choices=("train", "test"))
    parser.add_argument("--sparse_factor", default="12")
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
    swanlab_run = None
    if args.swanlab:
        import swanlab
        swanlab_run = swanlab.init(project=args.swanlab_project, experiment_name=f"ddf-P-Swin-S{args.sparse_factor}", config={"architecture": "ddf-P-Swin", "sparse_factor": args.sparse_factor, **config}, mode=args.swanlab_mode)
    for sparse_factor in factor_list(args.sparse_factor):
        checkpoint = Path(args.checkpoint.format(S=sparse_factor)) if args.checkpoint else checkpoint_path(sparse_factor)
        checkpoint = checkpoint if checkpoint.is_absolute() else THIS_DIR / checkpoint
        if args.mode == "train":
            print(f"P-Swin parameters: {parameter_count(SwinIRSino(config['pswin'])):,}")
            train(sparse_factor, config, args.batch_size, device, checkpoint, swanlab_run)
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        psnr, ssim = evaluate(sparse_factor, config, checkpoint, args.batch_size, device)
        print(f"ddf P-Swin S={sparse_factor}: PSNR={psnr:.6f} SSIM={ssim:.6f}")
        append_summary({"architecture": "ddf-P-Swin", "projection_backbone": "SwinIR-sino/P-Swin", "image_backbone": "NAFNet", "sparse_factor": sparse_factor, "psnr": f"{psnr:.6f}", "ssim": f"{ssim:.6f}", "checkpoint": str(checkpoint.resolve()), "config": str(config_path)})


if __name__ == "__main__":
    main()
