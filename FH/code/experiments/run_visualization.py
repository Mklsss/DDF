import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ddf_experiment_lib import (
    ROOT,
    build_model,
    check_required_files,
    load_config,
    parse_factors,
    resolve_checkpoint,
    resolve_path,
    set_seed,
    SinogramCTDataset,
)


def parse_roi(roi_text):
    values = [int(value.strip()) for value in roi_text.split(",") if value.strip()]
    if len(values) != 4:
        raise ValueError("--roi must contain four integers: x1,y1,x2,y2")
    return values


def tensor_to_image(tensor):
    array = tensor.detach().cpu().float().squeeze().numpy()
    return np.clip(array, 0.0, 1.0)


def load_prediction(method, method_key, sparse_factor, config, checkpoint_arg, sample_sinogram, device):
    checkpoint_path = resolve_checkpoint(checkpoint_arg, config, method_key, sparse_factor)
    if checkpoint_path is None:
        print(f"checkpoint for S={sparse_factor} not found, please train first.")
        return None, None, None
    model = build_model(method, sparse_factor, config).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    with torch.no_grad():
        prediction_ct, extras = model(sample_sinogram.to(device=device, dtype=torch.float32))
    return prediction_ct.clamp(0, 1), extras, checkpoint_path


def add_roi_box(axis, roi):
    x1, y1, x2, y2 = roi
    axis.add_patch(
        plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor="white", facecolor="none", linewidth=0.8)
    )


def save_comparison_figure(sparse_factor, gt_img, sparse_fbp_img, cascade_img, ddf_img, roi, output_dir):
    error_cascade = np.abs(cascade_img - gt_img)
    error_ddf = np.abs(ddf_img - gt_img)
    x1, y1, x2, y2 = roi
    roi_images = [
        gt_img[y1:y2, x1:x2],
        sparse_fbp_img[y1:y2, x1:x2],
        cascade_img[y1:y2, x1:x2],
        ddf_img[y1:y2, x1:x2],
    ]

    fig, axes = plt.subplots(3, 4, figsize=(9.2, 6.8), constrained_layout=True)
    titles = ["GT", "Sparse FBP", "Cascade", "DDF"]
    images = [gt_img, sparse_fbp_img, cascade_img, ddf_img]
    for col, (title, image) in enumerate(zip(titles, images)):
        axes[0, col].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(title, fontsize=9)
        axes[0, col].axis("off")
        add_roi_box(axes[0, col], roi)

    error_titles = ["", "abs(Sparse FBP-GT)", "abs(Cascade-GT)", "abs(DDF-GT)"]
    error_images = [np.zeros_like(gt_img), np.abs(sparse_fbp_img - gt_img), error_cascade, error_ddf]
    error_max = max(float(np.max(error_images[1])), float(np.max(error_images[2])), float(np.max(error_images[3])), 1e-6)
    for col, (title, image) in enumerate(zip(error_titles, error_images)):
        axes[1, col].imshow(image, cmap="magma", vmin=0, vmax=error_max)
        axes[1, col].set_title(title, fontsize=9)
        axes[1, col].axis("off")

    for col, image in enumerate(roi_images):
        axes[2, col].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[2, col].set_title("Zoomed ROI" if col == 0 else "", fontsize=9)
        axes[2, col].axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"reconstruction_comparison_S{sparse_factor}.pdf"
    png_path = output_dir / f"reconstruction_comparison_S{sparse_factor}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {pdf_path}")
    print(f"saved {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative DDF/Cascade comparison figures.")
    parser.add_argument("--sparse_factor", default=None, help="Sparse factor, e.g. 2 or 2,4,8,12. Defaults to all.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None, help="DDF checkpoint path or template with {S}.")
    parser.add_argument("--cascade_checkpoint", default=None, help="Cascade checkpoint path or template with {S}.")
    parser.add_argument("--output_dir", default="fig")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test_sample_index", type=int, default=0)
    parser.add_argument("--roi", default="96,96,160,160")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = torch.device(args.device)
    roi = parse_roi(args.roi)
    output_dir = Path(args.output_dir)

    missing_files = check_required_files(config, "ddf")
    if missing_files:
        for missing_file in missing_files:
            print(f"required file not found: {missing_file}")
        print("TODO: update experiments/config_default.json or pass --config with the correct paths.")
        return

    for sparse_factor in parse_factors(args.sparse_factor):
        dataset = SinogramCTDataset(resolve_path(config["test_data"]), sparse_factor)
        sample_index = int(args.test_sample_index)
        if sample_index < 0 or sample_index >= len(dataset):
            raise IndexError(f"--test_sample_index={sample_index} is outside test dataset length {len(dataset)}")
        sample_sinogram, gt_ct = dataset[sample_index]
        sample_sinogram = sample_sinogram.unsqueeze(0)
        gt_ct = gt_ct.unsqueeze(0)

        cascade_pred, cascade_extras, cascade_checkpoint = load_prediction(
            "cascade", "cascade", sparse_factor, config, args.cascade_checkpoint, sample_sinogram, device
        )
        ddf_pred, ddf_extras, ddf_checkpoint = load_prediction(
            "ddf", "ddf", sparse_factor, config, args.checkpoint, sample_sinogram, device
        )
        if cascade_pred is None or ddf_pred is None:
            continue
        sparse_fbp = cascade_extras["sparse_fbp"] if cascade_extras and "sparse_fbp" in cascade_extras else ddf_extras["sparse_fbp"]

        print(f"S={sparse_factor}, Cascade checkpoint={cascade_checkpoint}")
        print(f"S={sparse_factor}, DDF checkpoint={ddf_checkpoint}")
        save_comparison_figure(
            sparse_factor,
            tensor_to_image(gt_ct),
            tensor_to_image(sparse_fbp),
            tensor_to_image(cascade_pred),
            tensor_to_image(ddf_pred),
            roi,
            output_dir,
        )


if __name__ == "__main__":
    main()
