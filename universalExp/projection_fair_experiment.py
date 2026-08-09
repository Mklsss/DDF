"""Train/evaluate fair DDF replacements, including RED-CNN warm-starting."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from projection_fair import OriginalDDFWithReplacement, freeze_shared_ddf, load_original_weights

THIS_DIR = Path(__file__).resolve().parent
LEGACY_ROOT = Path("/root/autodl-fs/PAPER2/FH/code")
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

from experiments.ddf_experiment_lib import SinogramCTDataset, evaluate_model, set_seed
from pcnn import ResUNetSino
from pswin import SwinIRSino
from icnn import REDCNN
from FHinner.tnt_v1.restormor import Restormer


def atomic_torch_save(payload, checkpoint):
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(f".{checkpoint.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def split_train_validation_loaders(path, sparse_factor, batch_size, train_count, val_count):
    dataset = SinogramCTDataset(path, sparse_factor)
    required = train_count + val_count
    if len(dataset) < required:
        raise ValueError(f"dataset has {len(dataset)} samples, but {required} are required")
    train_set = torch.utils.data.Subset(dataset, range(train_count))
    val_set = torch.utils.data.Subset(dataset, range(train_count, required))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        torch.utils.data.DataLoader(train_set, **kwargs),
        torch.utils.data.DataLoader(val_set, **kwargs),
    )


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


def checkpoint_state(payload):
    """Accept both legacy raw state dicts and resumable fair-training checkpoints."""
    if not isinstance(payload, dict):
        return payload
    for key in ("model", "model_state", "state_dict"):
        if key in payload:
            return payload[key]
    if any(key in payload for key in ("optimizer", "scheduler", "epoch", "best_psnr")):
        raise KeyError(f"checkpoint contains training metadata but no model weights; keys={list(payload)}")
    return payload


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

    # RED-CNN retains an FBP skip connection while the original NAFNet teacher
    # has a substantially different signed output mean.  Aligning only the
    # final bias gives the student a useful first prediction without encoding
    # any target image into its weights.  It also makes an incorrect output
    # activation obvious immediately rather than silently plateauing.
    with torch.no_grad():
        calibration_sino, _ = next(iter(train_loader))
        calibration_sino = calibration_sino.to(device, dtype=torch.float32)
        calibration_fbp = teacher.fbp(teacher.sin(calibration_sino)).permute(0, 3, 1, 2)
        calibration_target = teacher.ct(calibration_fbp)
        student.tconv5.bias.add_((calibration_target - student(calibration_fbp)).mean())
        calibration_loss = F.mse_loss(student(calibration_fbp), calibration_target).item()
    print(f"[setup] REDCNN output-bias calibrated; initial distill_mse={calibration_loss:.8f}", flush=True)
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
            atomic_torch_save(
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
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Accumulate this many micro-batches before each optimizer step.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--resume_checkpoint", default=None, help="Resume model/optimizer/scheduler state from this checkpoint")
    parser.add_argument("--resume_epoch", type=int, default=0, help="Epoch represented by a legacy raw-state checkpoint")
    parser.add_argument("--lr_patience", type=int, default=15, help="Epochs without PSNR improvement before reducing LR")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="Multiplicative LR reduction after a plateau")
    parser.add_argument("--min_learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_threshold", type=float, default=0.01, help="Minimum absolute PSNR gain treated as an improvement")
    parser.add_argument("--warmstart_checkpoint", default=None, help="REDCNN-only distillation checkpoint")
    parser.add_argument("--warmstart_epochs", type=int, default=None)
    parser.add_argument("--warmstart_learning_rate", type=float, default=None)
    parser.add_argument("--auto_warmstart", action="store_true", help="Create a missing RED-CNN warm-start before fair training")
    parser.add_argument("--train_data", default=None, help="Override config train_data")
    parser.add_argument("--test_data", default=None, help="Override config test_data")
    parser.add_argument("--train_count", type=int, default=1600)
    parser.add_argument("--val_count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--result_json", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--swanlab", action="store_true", help="log epoch metrics to SwanLab")
    parser.add_argument("--swanlab_project", default="universalExp")
    parser.add_argument("--swanlab_run_name", default=None)
    parser.add_argument("--swanlab_mode", choices=("cloud", "local", "offline"), default="cloud")
    args = parser.parse_args()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be at least 1")

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.train_data:
        config["train_data"] = args.train_data
    if args.test_data:
        config["test_data"] = args.test_data
    checkpoint = Path(args.checkpoint) if args.checkpoint else THIS_DIR / "checkpoints" / "fair_protocol" / f"{args.backbone}_S{args.sparse_factor}.pth"
    if not checkpoint.is_absolute():
        checkpoint = THIS_DIR / checkpoint
    if args.dry_run:
        required = [Path(config["train_data"]), Path(config["test_data"]), Path(args.original_checkpoint)]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing required files: {missing}")
        print(json.dumps({
            "task": f"backbone_{args.backbone}_S{args.sparse_factor}",
            "backbone": args.backbone,
            "sparse_factor": args.sparse_factor,
            "epochs": args.epochs,
            "micro_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "train_count": args.train_count,
            "validation_count": args.val_count,
            "checkpoint": str(checkpoint),
            "auto_warmstart": bool(args.auto_warmstart),
            "swanlab_project": args.swanlab_project,
        }, indent=2))
        return
    if args.mode == "train" and checkpoint.exists() and not args.resume_checkpoint:
        raise FileExistsError(
            f"checkpoint already exists: {checkpoint}; pass --resume_checkpoint to continue"
        )
    set_seed(args.seed)
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
        train_loader, _ = split_train_validation_loaders(
            config["train_data"], args.sparse_factor, args.batch_size, args.train_count, args.val_count
        )
        best_loss = train_redcnn_warmstart(
            model.ct, teacher, train_loader, device=device, epochs=epochs,
            learning_rate=learning_rate, checkpoint=warmstart_checkpoint,
        )
        print(f"saved RED-CNN warm-start to {warmstart_checkpoint} (best distill MSE={best_loss:.8f})")
        return

    if args.mode == "test":
        model.load_state_dict(checkpoint_state(torch.load(checkpoint, map_location=device)), strict=True)
        psnr, ssim = evaluate_model(model, test_loader, device)
        print(f"fair {args.backbone} S={args.sparse_factor}: PSNR={psnr:.6f} SSIM={ssim:.6f}")
        return

    if supports_redcnn_warmstart(args.backbone):
        if not warmstart_checkpoint.exists():
            if not args.auto_warmstart:
                raise FileNotFoundError(
                    "I-CNN fair training requires a RED-CNN warm-start. "
                    "Pass --auto_warmstart or run warmstart mode first."
                )
            warmstart_config = config.get("warmstart", {})
            warmstart_epochs = args.warmstart_epochs or int(warmstart_config.get("epochs", 30))
            warmstart_lr = args.warmstart_learning_rate or float(warmstart_config.get("learning_rate", 1e-4))
            teacher = OriginalDDFWithReplacement(args.sparse_factor).to(device)
            load_original_weights(teacher, args.original_checkpoint)
            warmstart_train_loader, _ = split_train_validation_loaders(
                config["train_data"], args.sparse_factor, args.batch_size, args.train_count, args.val_count
            )
            train_redcnn_warmstart(
                model.ct, teacher, warmstart_train_loader, device=device,
                epochs=warmstart_epochs, learning_rate=warmstart_lr,
                checkpoint=warmstart_checkpoint,
            )
        load_redcnn_warmstart(model.ct, warmstart_checkpoint, device)
        print(f"[setup] loaded RED-CNN warm-start: {warmstart_checkpoint}", flush=True)

    freeze_shared_ddf(model, replaced_prefixes)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience,
        min_lr=args.min_learning_rate, threshold=args.lr_threshold, threshold_mode="abs",
    )
    train_loader, val_loader = split_train_validation_loaders(
        config["train_data"], args.sparse_factor, args.batch_size, args.train_count, args.val_count
    )
    best = -float("inf")
    start_epoch = 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if args.resume_checkpoint:
        resume_checkpoint = Path(args.resume_checkpoint)
        if not resume_checkpoint.is_absolute():
            resume_checkpoint = THIS_DIR / resume_checkpoint
        payload = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint_state(payload), strict=True)
        if isinstance(payload, dict) and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if isinstance(payload, dict) and "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        # CLI policy intentionally overrides threshold saved by older runs.
        scheduler.threshold = args.lr_threshold
        scheduler.threshold_mode = "abs"
        start_epoch = int(payload.get("epoch", args.resume_epoch)) if isinstance(payload, dict) else args.resume_epoch
        best = float(payload.get("best_psnr", -float("inf"))) if isinstance(payload, dict) else -float("inf")
        if best == -float("inf"):
            best, _ = evaluate_model(model, val_loader, device)
        print(
            f"[setup] resumed {resume_checkpoint} at epoch={start_epoch}; "
            f"best_psnr={best:.6f}; lr={optimizer.param_groups[0]['lr']:.2e}", flush=True,
        )
    if start_epoch >= args.epochs:
        raise ValueError(f"resume epoch ({start_epoch}) must be smaller than --epochs ({args.epochs})")
    run = None
    if args.swanlab:
        import swanlab
        run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_run_name or f"fair-{args.backbone}-S{args.sparse_factor}",
            config={"backbone": args.backbone, "sparse_factor": args.sparse_factor,
                    "protocol": "original-ddf-single-replacement-frozen-shared",
                    "train_count": args.train_count, "validation_count": args.val_count,
                    "micro_batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
                    "seed": args.seed, **config},
            mode=args.swanlab_mode,
        )
    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            optimizer.zero_grad(set_to_none=True)
            accumulation_steps = args.gradient_accumulation_steps
            loader_length = len(train_loader)
            for batch_index, (sino, target) in enumerate(
                tqdm(train_loader, desc=f"fair {args.backbone} {epoch}/{args.epochs}")
            ):
                prediction, _ = model(sino.to(device, dtype=torch.float32))
                loss = F.mse_loss(prediction, target.to(device, dtype=torch.float32))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at epoch {epoch}: {loss.item()}")
                group_start = (batch_index // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, loader_length - group_start)
                (loss / group_size).backward()
                if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == loader_length:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                loss_sum += loss.detach().item()
            psnr, ssim = evaluate_model(model, val_loader, device)
            metrics = {
                "epoch": epoch,
                "train/loss": loss_sum / len(train_loader),
                "val/psnr": psnr,
                "val/ssim": ssim,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            print(f"epoch={epoch} fair-{args.backbone} PSNR={psnr:.6f} SSIM={ssim:.6f}", flush=True)
            scheduler.step(psnr)
            if run is not None:
                run.log(metrics, step=epoch)
            if psnr > best:
                best = psnr
                atomic_torch_save(
                    {
                        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch, "best_psnr": best,
                    },
                    checkpoint,
                )
        best_payload = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint_state(best_payload), strict=True)
        test_psnr, test_ssim = evaluate_model(model, test_loader, device)
        result = {
            "task": f"backbone_{args.backbone}_S{args.sparse_factor}",
            "sparse_factor": args.sparse_factor,
            "best_epoch": int(best_payload["epoch"]),
            "best_val_psnr": float(best_payload["best_psnr"]),
            "test_psnr": test_psnr,
            "test_ssim": test_ssim,
            "checkpoint": str(checkpoint),
        }
        print(json.dumps(result, indent=2), flush=True)
        if args.result_json:
            result_path = Path(args.result_json)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if run is not None:
            run.log({"test/psnr": test_psnr, "test/ssim": test_ssim})
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
