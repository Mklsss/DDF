import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from networks import RED_CNN
from npz_loader import NPZRedCNNDataset


def save_png(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.clip(img, 0.0, 1.0)
    Image.fromarray((img * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Export RED-CNN NPZ predictions.")
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--views", type=int, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./exports/redcnn")
    parser.add_argument("--input_key", type=str, default="")
    parser.add_argument("--no_generate_fbp", action="store_true")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    dataset = NPZRedCNNDataset(
        npz_path=args.npz,
        views=args.views,
        img_size=args.img_size,
        input_key=args.input_key,
        generate_fbp=(not args.no_generate_fbp),
        patch_n=0,
        patch_size=0,
    )

    model = RED_CNN().to(device)
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    export_root = os.path.join(args.out_dir, f"view_{args.views:03d}")
    input_dir = os.path.join(export_root, "input")
    gt_dir = os.path.join(export_root, "gt")
    pred_dir = os.path.join(export_root, "pred")

    with torch.no_grad():
        for idx in args.indices:
            x, y = dataset[idx]
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            padded = F.pad(x_tensor, (10, 10, 10, 10), mode="reflect")
            pred = model(padded)[:, :, 10:-10, 10:-10]

            save_png(x, os.path.join(input_dir, f"idx_{idx:04d}.png"))
            save_png(y, os.path.join(gt_dir, f"idx_{idx:04d}.png"))
            save_png(pred.detach().cpu().numpy()[0, 0], os.path.join(pred_dir, f"idx_{idx:04d}.png"))
            print(f"[OK] idx={idx} -> {export_root}")


if __name__ == "__main__":
    main()
