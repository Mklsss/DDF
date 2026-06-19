"""One-batch forward/backward validation for the full DDF Both-CNN ablation."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from bothcnn_experiment import THIS_DIR, build_model, load_config, make_loader, parameter_count, set_seed
from icnn import REDCNN
from pcnn import ResUNetSino


def has_finite_gradient(module):
    return any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def main():
    parser = argparse.ArgumentParser(description="Smoke test for full-DDF Both-CNN")
    parser.add_argument("--sparse_factor", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=str(THIS_DIR / "results" / "smoke_bothcnn_ddf_S12_B3.json"))
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    set_seed(int(config["seed"]))
    device = torch.device(args.device)
    loader = make_loader(config["train_data"], args.sparse_factor, args.batch_size, shuffle=False)
    sinogram, target = next(iter(loader))
    sinogram, target = sinogram.to(device), target.to(device)
    model = build_model(config).to(device)
    model.train()
    prediction, auxiliary = model(sinogram)
    loss = nn.functional.mse_loss(prediction, target)
    loss.backward()
    report = {
        "config": str(config_path),
        "architecture": "ddf-Both-CNN",
        "sparse_factor": args.sparse_factor,
        "batch_size": args.batch_size,
        "input_shape": list(sinogram.shape),
        "target_shape": list(target.shape),
        "prediction_shape": list(prediction.shape),
        "loss": float(loss.detach().cpu()),
        "finite_loss": bool(torch.isfinite(loss).item()),
        "pcnn_parameters": parameter_count(ResUNetSino(config["pcnn"]["base_channels"])),
        "icnn_parameters": parameter_count(REDCNN(config["redcnn"]["channels"])),
        "pcnn_has_finite_gradient": has_finite_gradient(model.sin),
        "icnn_has_finite_gradient": has_finite_gradient(model.ct),
        "auxiliary_shapes": {key: list(value.shape) for key, value in auxiliary.items()},
        "total_parameters": parameter_count(model),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
