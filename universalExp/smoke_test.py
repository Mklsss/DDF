"""One-batch forward/backward validation for Cascade-P-CNN and DDF-P-CNN."""

import argparse
import json

import torch
from torch import nn

from pcnn_experiment import (
    THIS_DIR, build_model, load_config, make_loader, parameter_count, set_seed,
)
from pcnn import ResUNetSino


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse_factor", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--architecture",
        choices=("cascade", "ddf", "all"),
        default="all",
        help="Model to validate; 'all' validates both architectures.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=str(THIS_DIR / "results" / "smoke_test.json"))
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    set_seed(int(config["seed"]))
    device = torch.device(args.device)
    loader = make_loader(config["train_data"], args.sparse_factor, batch_size=args.batch_size, shuffle=False)
    sinogram, target = next(iter(loader))
    sinogram, target = sinogram.to(device), target.to(device)
    report = {
        "config": str(config_path), "sparse_factor": args.sparse_factor,
        "batch_size": args.batch_size, "architecture": args.architecture,
        "input_shape": list(sinogram.shape), "target_shape": list(target.shape),
        "pcnn_parameters": parameter_count(ResUNetSino(config["pcnn"]["base_channels"])),
        "models": {},
    }
    architectures = ("cascade", "ddf") if args.architecture == "all" else (args.architecture,)
    for architecture in architectures:
        model = build_model(architecture, config).to(device)
        model.train()
        prediction, auxiliary = model(sinogram)
        loss = nn.MSELoss()(prediction, target)
        loss.backward()
        pcnn_has_grad = any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.sin.parameters()
        )
        report["models"][architecture] = {
            "prediction_shape": list(prediction.shape),
            "loss": float(loss.detach().cpu()),
            "finite_loss": bool(torch.isfinite(loss).item()),
            "pcnn_has_finite_gradient": pcnn_has_grad,
            "auxiliary_shapes": {key: list(value.shape) for key, value in auxiliary.items()},
            "total_parameters": parameter_count(model),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output = __import__("pathlib").Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
