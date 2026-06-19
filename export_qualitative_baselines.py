#!/usr/bin/env python
# coding: utf-8
"""
Export qualitative reconstruction images for external baselines.

This script reuses the existing `test_redcnn.py` / `test_dudotrans.py`
code structure and saves one PNG per sparse factor for a chosen sample index,
so the paper-style figure script can stitch them together later.

Example:
    python export_qualitative_baselines.py ^
      --sample_index 33 ^
      --sparse_factors 2,4,8,12 ^
      --redcnn_checkpoint ./weights_redcnn/RED_CNN_c{S}_best.pth ^
      --dudotrans_checkpoint ./weights_dudotrans/DuDoTrans_c{S}_best.pth ^
      --output_root ../FH/code/fig/qualitative_raw
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_factors(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def tensor_to_image(tensor):
    image = tensor.detach().cpu().float().squeeze().numpy()
    return np.clip(image, 0.0, 1.0)


def ensure_parent(path_obj):
    path_obj.parent.mkdir(parents=True, exist_ok=True)


def save_png(image, path_obj):
    ensure_parent(path_obj)
    plt.imsave(path_obj, image, cmap="gray", vmin=0.0, vmax=1.0)
    print(f"saved {path_obj}")


def resolve_template(template, sparse_factor):
    path_obj = Path(template.format(S=sparse_factor, sparse_factor=sparse_factor))
    if not path_obj.is_absolute():
        path_obj = PROJECT_ROOT / path_obj
    return path_obj


def export_redcnn_sample(redcnn_module, checkpoint_path, sparse_factor, sample_index, device):
    dataset = redcnn_module.load_data("./data/test_meiaonew.npz", sparse_factor)
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index={sample_index} outside dataset length {len(dataset)} for RED-CNN")

    sin_in, _ = dataset[sample_index]
    sin_in = sin_in.unsqueeze(0).to(device).to(torch.float32)

    model = redcnn_module.RED_CNN(out_ch=96).to(device)
    fbp_operator = redcnn_module.FbpLayer(at_npz="./model/A_new.npz", device=device).to(device)
    for parameter in fbp_operator.parameters():
        parameter.requires_grad = False
    fbp_operator.eval()

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        fbp_u = fbp_operator(sin_in).permute(0, 3, 1, 2)
        recon = model(fbp_u).clamp(0, 1)
    return tensor_to_image(recon)


def export_dudotrans_sample(dudotrans_module, checkpoint_path, sparse_factor, sample_index, device):
    dataset = dudotrans_module.load_data("./data/test_meiaonew.npz", sparse_factor)
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index={sample_index} outside dataset length {len(dataset)} for DuDoTrans")

    sin_in, label = dataset[sample_index]
    sin_in = sin_in.unsqueeze(0).to(device).to(torch.float32)
    label = label.unsqueeze(0).to(device).to(torch.float32)

    model = dudotrans_module.DuDoTrans(num_view=sparse_factor, img_size=256).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        fbp_u = model.fbp(sin_in)
        _, _, _, recon = model(fbp_u, label, sin_in)
        recon = recon.clamp(0, 1)
    return tensor_to_image(recon)


def main():
    parser = argparse.ArgumentParser(description="Export qualitative baseline images for RED-CNN and DuDoTrans.")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--sparse_factors", default="2,4,8,12")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_root", default="../FH/code/fig/qualitative_raw")
    parser.add_argument("--redcnn_script", default="test_redcnn.py")
    parser.add_argument("--dudotrans_script", default="test_dudotrans.py")
    parser.add_argument("--redcnn_checkpoint", default="./weights_redcnn/RED_CNN_c{S}_best.pth")
    parser.add_argument("--dudotrans_checkpoint", default="./weights_dudotrans/DuDoTrans_c{S}_best.pth")
    parser.add_argument("--skip_redcnn", action="store_true")
    parser.add_argument("--skip_dudotrans", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sparse_factors = parse_factors(args.sparse_factors)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (SCRIPT_DIR / output_root).resolve()

    redcnn_module = None
    dudotrans_module = None

    if not args.skip_redcnn:
        redcnn_script = Path(args.redcnn_script)
        if not redcnn_script.is_absolute():
            redcnn_script = (SCRIPT_DIR / redcnn_script).resolve()
        redcnn_module = load_module("redcnn_test_module", redcnn_script)

    if not args.skip_dudotrans:
        dudotrans_script = Path(args.dudotrans_script)
        if not dudotrans_script.is_absolute():
            dudotrans_script = (SCRIPT_DIR / dudotrans_script).resolve()
        dudotrans_module = load_module("dudotrans_test_module", dudotrans_script)

    original_cwd = Path.cwd()
    try:
        os_cwd = PROJECT_ROOT
        import os
        os.chdir(os_cwd)

        for sparse_factor in sparse_factors:
            sample_dir = output_root / f"sample_{args.sample_index:03d}" / f"S{sparse_factor}"

            if redcnn_module is not None:
                redcnn_ckpt = resolve_template(args.redcnn_checkpoint, sparse_factor)
                redcnn_image = export_redcnn_sample(
                    redcnn_module=redcnn_module,
                    checkpoint_path=redcnn_ckpt,
                    sparse_factor=sparse_factor,
                    sample_index=args.sample_index,
                    device=device,
                )
                save_png(redcnn_image, sample_dir / "redcnn.png")

            if dudotrans_module is not None:
                dudotrans_ckpt = resolve_template(args.dudotrans_checkpoint, sparse_factor)
                dudotrans_image = export_dudotrans_sample(
                    dudotrans_module=dudotrans_module,
                    checkpoint_path=dudotrans_ckpt,
                    sparse_factor=sparse_factor,
                    sample_index=args.sample_index,
                    device=device,
                )
                save_png(dudotrans_image, sample_dir / "dudotrans.png")
    finally:
        import os
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
