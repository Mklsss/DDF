from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


# =========================
# Configurable settings
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
FH_ROOT = CODE_ROOT.parent
PROJECT_ROOT = FH_ROOT.parent

# Put the sample index you want to plot here.
sample_name = "sample_000"

# One figure will be generated for each sparse factor into this directory.
output_dir = CODE_ROOT / "fig/qualitative_two_row"

# Each method points to its own root. The script will read:
#   method_root / sample_name / S{factor} / filename
# This avoids "套娃" and also works when each baseline is stored separately.
method_specs = [
    {
        "title": "Ground Truth",
        "root": CODE_ROOT / "fig/qualitative_raw_no_overlap_cascade_original",
        "filename": "ground_truth.png",
    },
    {
        "title": "Sparse FBP",
        "root": CODE_ROOT / "fig/qualitative_raw_no_overlap_cascade_original",
        "filename": "sparse_fbp.png",
    },
    {
        "title": "RED-CNN",
        "root": CODE_ROOT / "fig/qualitative_result/redcnn",
        "filename": "redcnn.png",
    },
    {
        "title": "DuDoTrans",
        "root": CODE_ROOT / "fig/qualitative_result/dudotrans",
        "filename": "dudotrans.png",
    },
    {
        "title": "Cascade",
        "root": CODE_ROOT / "fig/qualitative_raw_no_overlap_cascade_original",
        "filename": "cascade.png",
    },
    {
        "title": "DDF",
        "root": CODE_ROOT / "fig/qualitative_raw_no_overlap_cascade_original",
        "filename": "ddf.png",
    },
]

sparse_factors = [2, 4, 8, 12]

roi_by_factor = {
    2: (96, 96, 160, 160),
    4: (96, 96, 160, 160),
    8: (96, 96, 160, 160),
    12: (96, 96, 160, 160),
}

placeholder_gray = 0.55
zoom_location = [0.03, 0.60, 0.35, 0.35]  # left-top inset


# =========================
# Helpers
# =========================

def resolve_path(path_value):
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return (CODE_ROOT / path_obj).resolve()


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


def draw_placeholder(ax, text):
    ax.imshow(np.full((256, 256), placeholder_gray), cmap="gray", vmin=0, vmax=1)
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=11,
        color="white",
        fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def add_roi_box(ax, roi, color="lime"):
    x1, y1, x2, y2 = roi
    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1.2,
            edgecolor=color,
            facecolor="none",
        )
    )


def add_zoom_inset(ax, image, roi):
    x1, y1, x2, y2 = roi
    patch = image[y1:y2, x1:x2]
    inset = ax.inset_axes(zoom_location)
    inset.imshow(patch, cmap="gray", vmin=0, vmax=1)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("lime")
        spine.set_linewidth(1.0)


def compute_error_map(pred, gt):
    return np.abs(pred - gt)


def resolve_image_path(method_spec, sparse_factor):
    root = resolve_path(method_spec["root"])
    return root / sample_name / f"S{sparse_factor}" / method_spec["filename"]


def load_method_images(sparse_factor):
    gt_spec = method_specs[0]
    gt_path = resolve_image_path(gt_spec, sparse_factor)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth image not found: {gt_path}")

    gt = normalize_for_display(load_image(gt_path))
    rows = []

    for method_spec in method_specs:
        image_path = resolve_image_path(method_spec, sparse_factor)
        if image_path.exists():
            image = normalize_for_display(load_image(image_path))
            missing = False
        else:
            image = np.full_like(gt, placeholder_gray, dtype=np.float32)
            missing = True
            print(f"WARNING: missing image -> {image_path}")

        if method_spec["title"] == "Ground Truth":
            error = np.zeros_like(gt, dtype=np.float32)
        else:
            error = compute_error_map(image, gt)

        rows.append(
            {
                "title": method_spec["title"],
                "image": image,
                "error": error,
                "missing": missing,
            }
        )

    return gt, rows


def save_figure_for_factor(sparse_factor):
    roi = roi_by_factor[sparse_factor]
    _, entries = load_method_images(sparse_factor)

    method_count = len(entries)
    fig, axes = plt.subplots(
        2,
        method_count,
        figsize=(2.6 * method_count, 5.2),
        constrained_layout=False,
    )

    if method_count == 1:
        axes = np.array(axes).reshape(2, 1)

    error_max = max(float(np.max(item["error"])) for item in entries)
    error_max = max(error_max, 1e-6)

    for col, item in enumerate(entries):
        img_ax = axes[0, col]
        err_ax = axes[1, col]

        if item["missing"]:
            draw_placeholder(img_ax, f"{item['title']}\nMissing")
        else:
            img_ax.imshow(item["image"], cmap="gray", vmin=0, vmax=1)
            add_roi_box(img_ax, roi)
            add_zoom_inset(img_ax, item["image"], roi)
            img_ax.set_xticks([])
            img_ax.set_yticks([])

        img_ax.set_title(item["title"], fontsize=12, pad=10)

        err_ax.imshow(item["error"], cmap="magma", vmin=0, vmax=error_max)
        err_ax.set_xticks([])
        err_ax.set_yticks([])

    axes[0, 0].set_ylabel(f"S={sparse_factor}\nImage", fontsize=12, rotation=0, labelpad=32, va="center")
    axes[1, 0].set_ylabel("Error", fontsize=12, rotation=0, labelpad=32, va="center")

    plt.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.05, wspace=0.03, hspace=0.06)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"qualitative_two_row_S{sparse_factor}.pdf"
    png_path = output_dir / f"qualitative_two_row_S{sparse_factor}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {pdf_path}")
    print(f"saved {png_path}")


def main():
    for sparse_factor in sparse_factors:
        save_figure_for_factor(sparse_factor)


if __name__ == "__main__":
    main()
