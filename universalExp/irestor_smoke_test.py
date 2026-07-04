"""One-batch forward/backward admission test for DDF I-Restor."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from irestor_experiment import (
    THIS_DIR, DDFIRestor, build_restormer, load_config, make_loader, parameter_count, set_seed,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse_factor", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--micro_batch_size", type=int, default=None, help="split the batch for gradient accumulation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="use CUDA automatic mixed precision")
    parser.add_argument("--output", default=str(THIS_DIR / "results" / "smoke_ddf_irestor_S12_B3.json"))
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    set_seed(int(config["seed"]))
    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    sinogram, target = next(iter(make_loader(config["train_data"], args.sparse_factor, args.batch_size)))
    model = DDFIRestor(args.sparse_factor, config).to(device).train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    micro_batch_size = args.micro_batch_size or args.batch_size
    if micro_batch_size <= 0:
        raise ValueError("--micro_batch_size must be positive")
    predictions = []
    auxiliary = None
    total_loss = 0.0
    for start in range(0, sinogram.shape[0], micro_batch_size):
        end = min(start + micro_batch_size, sinogram.shape[0])
        weight = (end - start) / sinogram.shape[0]
        with torch.cuda.amp.autocast(enabled=use_amp):
            prediction, auxiliary = model(sinogram[start:end].to(device=device, dtype=torch.float32))
            loss = nn.functional.mse_loss(prediction, target[start:end].to(device=device, dtype=torch.float32))
        scaler.scale(loss * weight).backward()
        total_loss += loss.detach().item() * weight
        predictions.append(prediction.detach().cpu())
    prediction = torch.cat(predictions, dim=0)
    loss = torch.tensor(total_loss, device=device)
    restormer_has_grad = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.ct.parameters()
    )
    report = {
        "config": str(config_path), "architecture": "ddf-I-Restor", "image_backbone": "Restormer",
        "sparse_factor": args.sparse_factor, "batch_size": args.batch_size,
        "micro_batch_size": micro_batch_size, "amp": use_amp,
        "input_shape": list(sinogram.shape), "target_shape": list(target.shape),
        "prediction_shape": list(prediction.shape), "loss": float(loss.detach().cpu()),
        "finite_loss": bool(torch.isfinite(loss).item()),
        "restormer_has_finite_gradient": restormer_has_grad,
        "restormer_parameters": parameter_count(build_restormer(config)),
        "total_parameters": parameter_count(model),
        "auxiliary_shapes": {key: list(value.shape) for key, value in auxiliary.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
