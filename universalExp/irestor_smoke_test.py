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
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=str(THIS_DIR / "results" / "smoke_ddf_irestor_S12_B3.json"))
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    set_seed(int(config["seed"]))
    device = torch.device(args.device)
    sinogram, target = next(iter(make_loader(config["train_data"], args.sparse_factor, args.batch_size)))
    model = DDFIRestor(args.sparse_factor, config).to(device).train()
    prediction, auxiliary = model(sinogram.to(device=device, dtype=torch.float32))
    loss = nn.functional.mse_loss(prediction, target.to(device=device, dtype=torch.float32))
    loss.backward()
    restormer_has_grad = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.ct.parameters()
    )
    report = {
        "config": str(config_path), "architecture": "ddf-I-Restor", "image_backbone": "Restormer",
        "sparse_factor": args.sparse_factor, "batch_size": args.batch_size,
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
