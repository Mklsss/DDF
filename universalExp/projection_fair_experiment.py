"""Train/evaluate a fair P-CNN, P-Swin, or I-CNN single-domain replacement."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from projection_fair import OriginalDDFWithReplacement, freeze_shared_ddf, load_original_weights

THIS_DIR = Path(__file__).resolve().parent
LEGACY_ROOT = Path("/autodl-fs/data/FH/code")
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

from experiments.ddf_experiment_lib import SinogramCTDataset, evaluate_model, set_seed
from pcnn import ResUNetSino
from pswin import SwinIRSino
from icnn import REDCNN


def build_replacement(name, config):
    if name == "pcnn":
        return ResUNetSino(config["pcnn"]["base_channels"])
    if name == "pswin":
        return SwinIRSino(config["pswin"])
    if name == "icnn":
        return REDCNN(config["redcnn"]["channels"])
    raise ValueError(name)


def loader(path, sparse_factor, batch_size, shuffle=False):
    dataset = SinogramCTDataset(path, sparse_factor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=("original", "pcnn", "pswin", "icnn"))
    parser.add_argument("--mode", required=True, choices=("train", "test"))
    parser.add_argument("--config", required=True, help="pcnn_default.json, pswin_default.json, or icnn_default.json")
    parser.add_argument("--sparse_factor", type=int, default=12)
    parser.add_argument("--original_checkpoint", default="/autodl-fs/data/FH/code/weights/DDF_c12_best.pth")
    parser.add_argument("--checkpoint", default=None, help="Replacement checkpoint; defaults under checkpoints/fair_single_domain")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--swanlab", action="store_true", help="log epoch metrics to SwanLab")
    parser.add_argument("--swanlab_project", default="universalExp")
    parser.add_argument("--swanlab_mode", choices=("cloud", "local", "offline"), default="cloud")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    device = torch.device(args.device)
    replacement = None if args.backbone == "original" else build_replacement(args.backbone, config)
    is_image_replacement = args.backbone == "icnn"
    model = OriginalDDFWithReplacement(
        args.sparse_factor,
        image=replacement if is_image_replacement else None,
        projection=None if is_image_replacement else replacement,
    ).to(device)
    replaced_prefix = None if args.backbone == "original" else ("ct." if is_image_replacement else "sin.")
    load_original_weights(model, args.original_checkpoint, replaced_prefix=replaced_prefix)

    test_loader = loader(config["test_data"], args.sparse_factor, args.batch_size)
    if args.backbone == "original":
        psnr, ssim = evaluate_model(model, test_loader, device)
        print(f"original DDF S={args.sparse_factor}: PSNR={psnr:.6f} SSIM={ssim:.6f}")
        return

    checkpoint = Path(args.checkpoint) if args.checkpoint else THIS_DIR / "checkpoints" / "fair_single_domain" / f"{args.backbone}_S{args.sparse_factor}.pth"
    if args.mode == "test":
        model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
        psnr, ssim = evaluate_model(model, test_loader, device)
        print(f"fair {args.backbone} S={args.sparse_factor}: PSNR={psnr:.6f} SSIM={ssim:.6f}")
        return

    freeze_shared_ddf(model, replaced_prefix)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    train_loader = loader(config["train_data"], args.sparse_factor, args.batch_size, shuffle=True)
    best = -float("inf")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    run = None
    if args.swanlab:
        import swanlab
        run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=f"fair-{args.backbone}-S{args.sparse_factor}",
            config={"backbone": args.backbone, "sparse_factor": args.sparse_factor,
                    "protocol": "original-ddf-single-replacement-frozen-shared", **config},
            mode=args.swanlab_mode,
        )
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            for sino, target in tqdm(train_loader, desc=f"fair {args.backbone} {epoch}/{args.epochs}"):
                optimizer.zero_grad(set_to_none=True)
                prediction, _ = model(sino.to(device, dtype=torch.float32))
                loss = F.mse_loss(prediction, target.to(device, dtype=torch.float32))
                loss.backward()
                optimizer.step()
                loss_sum += loss.detach().item()
            psnr, ssim = evaluate_model(model, test_loader, device)
            metrics = {
                "epoch": epoch,
                "train/loss": loss_sum / len(train_loader),
                "eval/psnr": psnr,
                "eval/ssim": ssim,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            print(f"epoch={epoch} fair-{args.backbone} PSNR={psnr:.6f} SSIM={ssim:.6f}", flush=True)
            if run is not None:
                run.log(metrics)
            if psnr > best:
                best = psnr
                torch.save(model.state_dict(), checkpoint)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
