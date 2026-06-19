"""Both-CNN ablation: ResUNet-sino and RED-CNN in the full DDF path.

The FBP/FP operators and both DDF fusion modules are imported unchanged from
the original implementation.  ``sin`` and ``ct`` are the only replaced
backbones; ``ct`` is deliberately shared by DDF's first reconstruction and
its final reconstruction, matching the original DDF parameter-sharing rule.
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
    DEFAULT_FACTORS,
    FbpLayer,
    ForwardProjectionLayer,
    GMLPSineFusion,
    SinogramCTDataset,
    evaluate_model,
    freeze_projection_layers,
    resolve_path,
    set_seed,
)
from icnn import REDCNN
from pcnn import ResUNetSino


def load_config(path=None):
    config_path = Path(path) if path else THIS_DIR / "configs" / "bothcnn_default.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle), config_path.resolve()


def factor_list(value):
    if value is None:
        return list(DEFAULT_FACTORS)
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


class DDFBothCNN(nn.Module):
    """Full DDF with ResUNet backbones in both projection and image domains."""

    def __init__(self, config):
        super().__init__()
        self.sin = ResUNetSino(config["pcnn"]["base_channels"])
        self.fbp = FbpLayer(resolve_path(config["fbp_matrix"]))
        self.fp = ForwardProjectionLayer(
            resolve_path(config["fp_index"]), resolve_path(config["fp_data"])
        )
        self.ct = REDCNN(config["redcnn"]["channels"])
        self.sine_fusion = GMLPSineFusion()
        self.ct_fusion = CrossGatingBlock()

    def forward(self, sparse_sinogram):
        repaired_sinogram = self.sin(sparse_sinogram)
        sparse_fbp = self.fbp(sparse_sinogram).permute(0, 3, 1, 2)
        fbp1 = self.fbp(repaired_sinogram).permute(0, 3, 1, 2)
        ct_nd = self.ct(fbp1)
        feedback_sinogram = self.fp(ct_nd).unsqueeze(1)
        fused_sinogram = self.sine_fusion(feedback_sinogram, repaired_sinogram.unsqueeze(1))
        fbp2 = self.fbp(fused_sinogram.squeeze(1)).permute(0, 3, 1, 2)
        ct_pre, _ = self.ct_fusion(ct_nd, fbp2)
        return ct_pre, {
            "sparse_fbp": sparse_fbp,
            "cascade": ct_nd,
            "fused_sinogram": fused_sinogram,
        }


def build_model(config):
    return DDFBothCNN(config)


def make_loader(data_path, sparse_factor, batch_size, shuffle=False):
    dataset = SinogramCTDataset(resolve_path(data_path), sparse_factor)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available()
    )


def checkpoint_path(sparse_factor):
    return THIS_DIR / "checkpoints" / "ddf" / f"Both-CNN_REDCNN_S{sparse_factor}.pth"


def append_summary(row):
    path = THIS_DIR / "results" / "summary.csv"
    fields = [
        "architecture", "projection_backbone", "image_backbone", "sparse_factor",
        "psnr", "ssim", "checkpoint", "config",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate(sparse_factor, config, checkpoint, batch_size, device):
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    loader = make_loader(config["test_data"], sparse_factor, batch_size, shuffle=False)
    return evaluate_model(model, loader, device)


def train(sparse_factor, config, batch_size, device, checkpoint, swanlab_run=None):
    print(f"[setup] building DDF Both-CNN model on {device}", flush=True)
    model = build_model(config).to(device)
    freeze_projection_layers(model)
    optimizer = torch.optim.Adam(
        [item for item in model.parameters() if item.requires_grad],
        lr=config["train"]["learning_rate"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["train"]["step_size"], gamma=config["train"]["gamma"]
    )
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
        progress = tqdm(
            train_loader,
            desc=f"ddf Both-CNN S={sparse_factor} epoch {epoch + 1}/{config['train']['epochs']}",
            unit="batch",
            dynamic_ncols=True,
        )
        for step, (sparse_sinogram, target_ct) in enumerate(progress, start=1):
            prediction, _ = model(sparse_sinogram.to(device=device, dtype=torch.float32))
            loss = nn.functional.mse_loss(prediction, target_ct.to(device=device, dtype=torch.float32))
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1}: {loss.item()}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.detach().item()
            progress.set_postfix(
                loss=f"{running_loss / step:.6f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}"
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
        print(f"epoch={epoch + 1} architecture=ddf-Both-CNN S={sparse_factor} psnr={psnr:.6f} ssim={ssim:.6f}", flush=True)
        if swanlab_run is not None:
            swanlab_run.log(metrics)
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(model.state_dict(), checkpoint)
    return checkpoint


def parser():
    result = argparse.ArgumentParser(description="Both-CNN universal full-DDF experiment")
    result.add_argument("--mode", required=True, choices=("train", "test"))
    result.add_argument("--sparse_factor", default="12")
    result.add_argument("--config", default=None)
    result.add_argument("--checkpoint", default=None)
    result.add_argument("--batch_size", type=int, default=3)
    result.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    result.add_argument("--swanlab", action="store_true", help="log epoch metrics to SwanLab")
    result.add_argument("--swanlab_project", default="universalExp")
    result.add_argument("--swanlab_mode", choices=("cloud", "local", "offline"), default="cloud")
    return result


def main():
    args = parser().parse_args()
    config, config_path = load_config(args.config)
    set_seed(int(config["seed"]))
    device = torch.device(args.device)
    swanlab_run = None
    if args.swanlab:
        import swanlab
        swanlab_run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=f"ddf-Both-CNN-S{args.sparse_factor or 'all'}",
            config={"architecture": "ddf-Both-CNN", "sparse_factor": args.sparse_factor, **config},
            mode=args.swanlab_mode,
        )
    for sparse_factor in factor_list(args.sparse_factor):
        checkpoint = Path(args.checkpoint.format(S=sparse_factor)) if args.checkpoint else checkpoint_path(sparse_factor)
        if not checkpoint.is_absolute():
            checkpoint = THIS_DIR / checkpoint
        if args.mode == "train":
            print(f"P-CNN parameters: {parameter_count(ResUNetSino(config['pcnn']['base_channels'])):,}")
            print(f"I-CNN parameters: {parameter_count(REDCNN(config['redcnn']['channels'])):,}")
            train(sparse_factor, config, args.batch_size, device, checkpoint, swanlab_run)
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        psnr, ssim = evaluate(sparse_factor, config, checkpoint, args.batch_size, device)
        print(f"ddf Both-CNN S={sparse_factor}: PSNR={psnr:.6f} SSIM={ssim:.6f}")
        append_summary({
            "architecture": "ddf-Both-CNN",
            "projection_backbone": "ResUNet-sino/P-CNN",
            "image_backbone": "RED-CNN/I-CNN",
            "sparse_factor": sparse_factor,
            "psnr": f"{psnr:.6f}", "ssim": f"{ssim:.6f}",
            "checkpoint": str(checkpoint.resolve()), "config": str(config_path),
        })


if __name__ == "__main__":
    main()
