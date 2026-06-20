"""Train/evaluate fair DDF replacements, including RED-CNN warm-starting."""

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
from FHinner.tnt_v1.restormor import Restormer


def build_restormer(config):
    args = config["restormer"]
    return Restormer(
        inp_channels=args["inp_channels"], out_channels=args["out_channels"], dim=args["dim"],
        num_blocks=args["num_blocks"], num_refinement_blocks=args["num_refinement_blocks"],
        heads=args["heads"], ffn_expansion_factor=args["ffn_expansion_factor"],
        bias=args["bias"], LayerNorm_type=args["layer_norm_type"],
    )


def build_replacements(name, config):
    if name == "pcnn":
        return ResUNetSino(config["pcnn"]["base_channels"]), None
    if name == "pswin":
        return SwinIRSino(config["pswin"]), None
    if name == "icnn":
        return None, REDCNN(config["redcnn"]["channels"])
    if name == "irestor":
        return None, build_restormer(config)
    if name == "bothcnn":
        return ResUNetSino(config["pcnn"]["base_channels"]), REDCNN(config["redcnn"]["channels"])
    if name == "mixed":
        return SwinIRSino(config["pswin"]), REDCNN(config["redcnn"]["channels"])
    raise ValueError(name)


def loader(path, sparse_factor, batch_size, shuffle=False):
    dataset = SinogramCTDataset(path, sparse_factor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


def warmstart_path(backbone, sparse_factor):
    """Standalone RED-CNN distillation checkpoint used before fair tuning."""
    return THIS_DIR / "checkpoints" / "warmstart" / f"{backbone}_REDCNN_S{sparse_factor}.pth"


def supports_redcnn_warmstart(backbone):
    return backbone in {"icnn", "bothcnn", "mixed"}


def load_redcnn_warmstart(image, path, device):
    """Load a RED-CNN-only distillation checkpoint with strict compatibility checks."""
    payload = torch.load(path, map_location=device)
    state = payload["redcnn"] if isinstance(payload, dict) and "redcnn" in payload else payload
    image.load_state_dict(state, strict=True)


def train_redcnn_warmstart(student, teacher, train_loader, *, device, epochs, learning_rate, checkpoint):
    """Distil original DDF's exact FBP-to-NAFNet mapping into RED-CNN.

    Crucially, inputs come from the original restored sinogram, then its FBP.
    The student therefore learns the distribution that reaches DDF's original
    NAFNet, not an unrelated raw sparse-FBP reconstruction.
    """
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        student.train()
        loss_sum = 0.0
        for sino, _ in tqdm(train_loader, desc=f"REDCNN warm-start {epoch}/{epochs}"):
            sino = sino.to(device, dtype=torch.float32)
            with torch.no_grad():
                teacher_fbp = teacher.fbp(teacher.sin(sino)).permute(0, 3, 1, 2)
                teacher_ct = teacher.ct(teacher_fbp)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(student(teacher_fbp), teacher_ct)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite RED-CNN warm-start loss at epoch {epoch}: {loss.item()}")
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().item()
        mean_loss = loss_sum / len(train_loader)
        print(f"epoch={epoch} REDCNN-warmstart distill_mse={mean_loss:.8f}", flush=True)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                {"redcnn": student.state_dict(), "distill_mse": best_loss, "teacher": "original-ddf-NAFNet"},
                checkpoint,
            )
    return best_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=("original", "pcnn", "pswin", "icnn", "irestor", "bothcnn", "mixed"))
    parser.add_argument("--mode", required=True, choices=("warmstart", "train", "test"))
    parser.add_argument("--config", required=True, help="a matching config from configs/")
    parser.add_argument("--sparse_factor", type=int, default=12)
    parser.add_argument("--original_checkpoint", default="/autodl-fs/data/FH/code/weights/DDF_c12_best.pth")
    parser.add_argument("--checkpoint", default=None, help="Replacement checkpoint; defaults under checkpoints/fair_protocol")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmstart_checkpoint", default=None, help="REDCNN-only distillation checkpoint")
    parser.add_argument("--warmstart_epochs", type=int, default=None)
    parser.add_argument("--warmstart_learning_rate", type=float, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--swanlab", action="store_true", help="log epoch metrics to SwanLab")
    parser.add_argument("--swanlab_project", default="universalExp")
    parser.add_argument("--swanlab_mode", choices=("cloud", "local", "offline"), default="cloud")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    device = torch.device(args.device)
    projection, image = (None, None) if args.backbone == "original" else build_replacements(args.backbone, config)
    model = OriginalDDFWithReplacement(
        args.sparse_factor,
        image=image, projection=projection,
    ).to(device)
    replaced_prefixes = tuple(prefix for prefix, module in (("sin.", projection), ("ct.", image)) if module is not None)
    load_original_weights(model, args.original_checkpoint, replaced_prefixes=replaced_prefixes)

    default_warmstart = warmstart_path(args.backbone, args.sparse_factor)
    warmstart_checkpoint = Path(args.warmstart_checkpoint) if args.warmstart_checkpoint else default_warmstart
    if not warmstart_checkpoint.is_absolute():
        warmstart_checkpoint = THIS_DIR / warmstart_checkpoint

    test_loader = loader(config["test_data"], args.sparse_factor, args.batch_size)
    if args.backbone == "original":
        psnr, ssim = evaluate_model(model, test_loader, device)
        print(f"original DDF S={args.sparse_factor}: PSNR={psnr:.6f} SSIM={ssim:.6f}")
        return

    if args.mode == "warmstart":
        if not supports_redcnn_warmstart(args.backbone):
            raise ValueError(f"{args.backbone} has no RED-CNN image branch to warm-start")
        warmstart_config = config.get("warmstart", {})
        epochs = args.warmstart_epochs or int(warmstart_config.get("epochs", 30))
        learning_rate = args.warmstart_learning_rate or float(warmstart_config.get("learning_rate", 1e-4))
        teacher = OriginalDDFWithReplacement(args.sparse_factor).to(device)
        load_original_weights(teacher, args.original_checkpoint)
        train_loader = loader(config["train_data"], args.sparse_factor, args.batch_size, shuffle=True)
        best_loss = train_redcnn_warmstart(
            model.ct, teacher, train_loader, device=device, epochs=epochs,
            learning_rate=learning_rate, checkpoint=warmstart_checkpoint,
        )
        print(f"saved RED-CNN warm-start to {warmstart_checkpoint} (best distill MSE={best_loss:.8f})")
        return

    checkpoint = Path(args.checkpoint) if args.checkpoint else THIS_DIR / "checkpoints" / "fair_protocol" / f"{args.backbone}_S{args.sparse_factor}.pth"
    if args.mode == "test":
        model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
        psnr, ssim = evaluate_model(model, test_loader, device)
        print(f"fair {args.backbone} S={args.sparse_factor}: PSNR={psnr:.6f} SSIM={ssim:.6f}")
        return

    if supports_redcnn_warmstart(args.backbone):
        if not warmstart_checkpoint.exists():
            raise FileNotFoundError(
                "I-CNN fair training requires a RED-CNN warm-start because a random image "
                "backbone corrupts the frozen original DDF feedback path. Run: "
                f"python {Path(__file__).name} --backbone {args.backbone} --mode warmstart "
                f"--config {args.config} --sparse_factor {args.sparse_factor}"
            )
        load_redcnn_warmstart(model.ct, warmstart_checkpoint, device)
        print(f"[setup] loaded RED-CNN warm-start: {warmstart_checkpoint}", flush=True)

    freeze_shared_ddf(model, replaced_prefixes)
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
