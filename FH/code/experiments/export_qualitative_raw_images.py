import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ddf_experiment_lib import (
    SinogramCTDataset,
    build_model,
    load_config,
    parse_factors,
    resolve_path,
)


def unwrap_state_dict(obj):
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def infer_width(state_dict, default_width):
    key = "ct.intro.weight"
    if key in state_dict:
        return int(state_dict[key].shape[0])
    return int(default_width)


def save_gray_png(path, tensor):
    arr = tensor.detach().cpu().float().squeeze().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, arr, cmap="gray", vmin=0.0, vmax=1.0)


def load_prediction(method, sparse_factor, base_config, checkpoint_path, sample_sinogram, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"missing checkpoint: {checkpoint_path}")
        return None, None

    state_dict = unwrap_state_dict(torch.load(checkpoint_path, map_location=device))
    config = dict(base_config)
    config["nafnet"] = dict(base_config["nafnet"])
    config["nafnet"]["width"] = infer_width(state_dict, base_config["nafnet"]["width"])

    model = build_model(method, sparse_factor, config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    with torch.no_grad():
        pred, extras = model(sample_sinogram.to(device=device, dtype=torch.float32))
    return pred.clamp(0, 1), extras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--sparse_factors", default="2,4,8,12")
    parser.add_argument("--ddf_checkpoint", default="weights/DDF_c{S}_best.pth")
    parser.add_argument("--cascade_checkpoint", default="weights/cascade_S{S}.pth")
    parser.add_argument("--output_dir", default="fig/qualitative_raw")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    root_out = Path(args.output_dir) / f"sample_{args.sample_index:03d}"

    for sparse_factor in parse_factors(args.sparse_factors):
        dataset = SinogramCTDataset(resolve_path(config["test_data"]), sparse_factor)
        sample_sinogram, gt_ct = dataset[args.sample_index]
        sample_sinogram = sample_sinogram.unsqueeze(0)
        gt_ct = gt_ct.unsqueeze(0)

        s_out = root_out / f"S{sparse_factor}"
        save_gray_png(s_out / "ground_truth.png", gt_ct)

        cascade_ckpt = args.cascade_checkpoint.format(S=sparse_factor, sparse_factor=sparse_factor)
        cascade_pred, cascade_extras = load_prediction("cascade", sparse_factor, config, cascade_ckpt, sample_sinogram, device)
        if cascade_pred is not None:
            save_gray_png(s_out / "cascade.png", cascade_pred)
            save_gray_png(s_out / "sparse_fbp.png", cascade_extras["sparse_fbp"])

        ddf_ckpt = args.ddf_checkpoint.format(S=sparse_factor, sparse_factor=sparse_factor)
        ddf_pred, ddf_extras = load_prediction("ddf", sparse_factor, config, ddf_ckpt, sample_sinogram, device)
        if ddf_pred is not None:
            save_gray_png(s_out / "ddf.png", ddf_pred)
            if not (s_out / "sparse_fbp.png").exists():
                save_gray_png(s_out / "sparse_fbp.png", ddf_extras["sparse_fbp"])

        print(f"saved raw images for S={sparse_factor}: {s_out}")


if __name__ == "__main__":
    main()
