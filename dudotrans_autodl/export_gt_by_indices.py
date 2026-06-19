import argparse
import hashlib
import os

import numpy as np
from PIL import Image


def md5_of_array(arr: np.ndarray) -> str:
    m = hashlib.md5()
    m.update(arr.tobytes())
    return m.hexdigest()


def to_uint8_img(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx > mn:
        x = (x - mn) / (mx - mn)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def load_ct_label(npz_path: str) -> np.ndarray:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    d = np.load(npz_path)
    if "ct_label" not in d.files:
        raise KeyError(f"'ct_label' not found. keys={d.files}")

    ct = d["ct_label"]
    if ct.ndim == 4:
        if ct.shape[1] == 1:
            ct = ct[:, 0, :, :]
        elif ct.shape[-1] == 1:
            ct = ct[:, :, :, 0]
        else:
            raise ValueError(f"Unsupported ct_label shape: {ct.shape}")
    elif ct.ndim != 3:
        raise ValueError(f"Unsupported ct_label shape: {ct.shape}")

    return ct.astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Export selected ground-truth slices from an NPZ file.")
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, default="./export_check_gt")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ct_all = load_ct_label(args.npz)
    n = ct_all.shape[0]
    print(f"[OK] Loaded ct_label from: {args.npz}")
    print(f"ct_label shape: {ct_all.shape} (N={n})")
    print(f"Export indices: {args.indices}")

    for idx in args.indices:
        if idx < 0 or idx >= n:
            print(f"[SKIP] idx={idx} out of range 0..{n - 1}")
            continue

        img = ct_all[idx]
        h, w = img.shape
        md5 = md5_of_array(img)
        save_path = os.path.join(args.out_dir, f"gt_idx{idx}.png")
        Image.fromarray(to_uint8_img(img)).save(save_path)

        print(f"[SAVE] idx={idx} shape=({h},{w}) min={img.min():.6f} max={img.max():.6f} md5={md5}")
        print(f"       -> {save_path}")


if __name__ == "__main__":
    main()
