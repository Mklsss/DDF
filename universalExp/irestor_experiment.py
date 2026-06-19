"""I-Restor universal-backbone experiment: original DDF plus Restormer image backbone.

Only DDF's image-domain ``ct`` backbone changes.  The original UTR/UTR+Angle
sinogram network, FBP, forward projection, GMLP sinogram fusion, and CGB image
fusion remain in the path exactly as in the default DDF implementation.
"""

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
from model.sin_angle import sin_angle
from FHinner.tnt_v1.restormor import Restormer


def load_config(path=None):
    config_path = Path(path) if path else THIS_DIR / "configs" / "irestor_default.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle), config_path.resolve()


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def build_restormer(config):
    args = config["restormer"]
    return Restormer(
        inp_channels=args["inp_channels"],
        out_channels=args["out_channels"],
        dim=args["dim"],
        num_blocks=args["num_blocks"],
        num_refinement_blocks=args["num_refinement_blocks"],
        heads=args["heads"],
        ffn_expansion_factor=args["ffn_expansion_factor"],
        bias=args["bias"],
        LayerNorm_type=args["layer_norm_type"],
    )


class DDFIRestor(nn.Module):
    """Full DDF with only NAFNet replaced by Restormer."""

    def __init__(self, sparse_factor, config):
        super().__init__()
        angle = 360 // int(sparse_factor)
        self.sin = sin_angle(num_sensor=357, angle=angle, num_heads=1)
        self.fbp = FbpLayer(resolve_path(config["fbp_matrix"]))
        self.fp = ForwardProjectionLayer(
            resolve_path(config["fp_index"]), resolve_path(config["fp_data"])
        )
        self.ct = build_restormer(config)
        self.sine_fusion = GMLPSineFusion()
        self.ct_fusion = CrossGatingBlock()

    def forward(self, sparse_sinogram):
        utr_sinogram = self.sin(sparse_sinogram)
        sparse_fbp_nhwc = self.fbp(sparse_sinogram)
        fbp1 = self.fbp(utr_sinogram).permute(0, 3, 1, 2)
        ct_nd = self.ct(fbp1)
        feedback_sinogram = self.fp(ct_nd).unsqueeze(1)
        fused_sinogram = self.sine_fusion(feedback_sinogram, utr_sinogram.unsqueeze(1))
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
        dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available()
    )


def checkpoint_path(sparse_factor):
    return THIS_DIR / "checkpoints" / "ddf" / f"I-Restor_Restormer_S{sparse_factor}.pth"


def append_summary(row):
    path = THIS_DIR / "results" / "summary.csv"
    fields = [
        "architecture", "projection_backbone", "image_backbone", "sparse_factor",
        "psnr", "ssim", "checkpoint", "config",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def evaluate(sparse_factor, config, checkpoint, batch_size, device):
    model = DDFIRestor(sparse_factor, config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    loader = make_loader(config["test_data"], sparse_factor, batch_size)
    return evaluate_model(model, loader, device)


def train(sparse_factor, config, batch_size, device, checkpoint, swanlab_run=None):
    print(f"[setup] building DDF I-Restor model on {device}", flush=True)
    model = DDFIRestor(sparse_factor, config).to(device)
    freeze_projection_layers(model)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["train"]["learning_rate"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["train"]["step_size"], gamma=config["train"]["gamma"]
    )
    print("[setup] loading train dataset", flush=True)
    train_loader = make_loader(config["train_data"], sparse_factor, batch_size, shuffle=True)
    print("[setup] loading test dataset", flush=True)
    test_loader = make_loader(config["test_data"], sparse_factor, batch_size)
    print(f"[setup] ready: {len(train_loader)} train batches, {len(test_loader)} test batches", flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_psnr = -float("inf")
    for epoch in range(int(config["train"]["epochs"])):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"ddf I-Restor S={sparse_factor} epoch {epoch + 1}/{config['train']['epochs']}", unit="batch", dynamic_ncols=True)
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
        metrics = {
            "epoch": epoch + 1,
            "train/loss": running_loss / len(train_loader),
            "train/lr": optimizer.param_groups[0]["lr"],
            "eval/psnr": psnr,
            "eval/ssim": ssim,
        }
        print(f"epoch={epoch + 1} architecture=ddf I-Restor S={sparse_factor} psnr={psnr:.6f} ssim={ssim:.6f}", flush=True)
        if swanlab_run is not None:
            swanlab_run.log(metrics)
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(model.state_dict(), checkpoint)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="I-Restor (Restormer) universal DDF experiment")
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
    checkpoint = Path(args.checkpoint) if args.checkpoint else checkpoint_path(args.sparse_factor)
    if not checkpoint.is_absolute():
        checkpoint = THIS_DIR / checkpoint
    swanlab_run = None
    if args.swanlab:
        import swanlab
        swanlab_run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=f"ddf-I-Restor-Restormer-S{args.sparse_factor}",
            config={"architecture": "ddf-I-Restor", "sparse_factor": args.sparse_factor, **config},
            mode=args.swanlab_mode,
        )
    if args.mode == "train":
        print(f"Restormer parameters: {parameter_count(build_restormer(config)):,}")
        train(args.sparse_factor, config, args.batch_size, torch.device(args.device), checkpoint, swanlab_run)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    psnr, ssim = evaluate(args.sparse_factor, config, checkpoint, args.batch_size, torch.device(args.device))
    print(f"ddf I-Restor (Restormer) S={args.sparse_factor}: PSNR={psnr:.6f} SSIM={ssim:.6f}")
    append_summary({
        "architecture": "ddf-I-Restor", "projection_backbone": "UTR / UTR+Angle",
        "image_backbone": "Restormer", "sparse_factor": args.sparse_factor,
        "psnr": f"{psnr:.6f}", "ssim": f"{ssim:.6f}",
        "checkpoint": str(checkpoint.resolve()), "config": str(config_path),
    })


if __name__ == "__main__":
    main()
