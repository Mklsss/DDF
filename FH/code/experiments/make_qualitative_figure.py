from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


# =========================
# Configurable settings
# =========================

input_root = Path("fig/qualitative_raw_no_overlap/sample_000")
output_pdf = Path("fig/qualitative_results.pdf")
output_png = Path("fig/qualitative_results.png")

use_cascade_placeholder = True

method_files = {
    "Ground Truth": "ground_truth.png",
    "Sparse FBP": "sparse_fbp.png",
    "Cascade": "cascade.png",
    "DDF": "ddf.png",
}

sparse_factors = [2, 4, 8, 12]

roi_by_factor = {
    2: (96, 96, 160, 160),
    4: (96, 96, 160, 160),
    8: (96, 96, 160, 160),
    12: (96, 96, 160, 160),
}

placeholder_gray = 0.55
zoom_location = [0.58, 0.04, 0.38, 0.38]  # x, y, w, h in panel coordinates


# =========================
# Helper functions
# =========================

def load_image(path):
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype(np.float32)
    return arr


def normalize_for_display(arr):
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        return np.clip(arr, 0, 1)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def draw_placeholder(ax):
    ax.imshow(np.full((256, 256), placeholder_gray), cmap="gray", vmin=0, vmax=1)
    ax.text(
        0.5,
        0.5,
        "Cascade\nPlaceholder",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=11,
        color="white",
        fontweight="bold",
    )


def draw_panel(ax, arr, roi):
    show = normalize_for_display(arr)
    ax.imshow(show, cmap="gray", vmin=0, vmax=1)

    x1, y1, x2, y2 = roi
    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1.2,
            edgecolor="lime",
            facecolor="none",
        )
    )

    patch = show[y1:y2, x1:x2]
    inset = ax.inset_axes(zoom_location)
    inset.imshow(patch, cmap="gray", vmin=0, vmax=1)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("lime")
        spine.set_linewidth(1.0)


def main():
    methods = list(method_files.keys())
    fig, axes = plt.subplots(
        len(sparse_factors),
        len(methods),
        figsize=(10.2, 10.8),
        constrained_layout=False,
    )

    for col, method in enumerate(methods):
        axes[0, col].set_title(method, fontsize=13, pad=10)

    for row, sparse_factor in enumerate(sparse_factors):
        roi = roi_by_factor[sparse_factor]
        axes[row, 0].set_ylabel(
            f"S={sparse_factor}",
            fontsize=13,
            rotation=0,
            labelpad=34,
            va="center",
        )

        for col, method in enumerate(methods):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])

            image_path = input_root / f"S{sparse_factor}" / method_files[method]

            if method == "Cascade" and use_cascade_placeholder:
                draw_placeholder(ax)
                continue

            if not image_path.exists():
                print(f"WARNING: missing image, use placeholder: {image_path}")
                draw_placeholder(ax)
                continue

            arr = load_image(image_path)
            draw_panel(ax, arr, roi)

    plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.03, wspace=0.02, hspace=0.08)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {output_pdf}")
    print(f"saved {output_png}")


if __name__ == "__main__":
    main()
