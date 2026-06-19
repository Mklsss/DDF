import argparse
import csv
import json
import os
import random
import time
from glob import glob

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from loaders.load_dataset import NPZ_CTSlice_Provider
from modules.reconstructor import reconstructor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DuDoTrans checkpoints on an NPZ test set.")
    parser.add_argument("--test_npz", type=str, default="/root/autodl-fs/dataset/test_meiaonew.npz")
    parser.add_argument("--views", type=int, nargs="+", default=[30, 60, 90, 120])
    parser.add_argument("--ckpt", type=str, default=None, help="Single checkpoint path. Only valid with one --views value.")
    parser.add_argument("--ckpt_root", type=str, default="./results/models")
    parser.add_argument("--out_dir", type=str, default="./results/test")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--vis_indices", type=int, nargs="*", default=[0, 66, 133, 199])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--poission_level", type=float, default=1e5)
    parser.add_argument("--gaussian_level", type=float, default=0.05)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_latest_ckpt(ckpt_root, views):
    view_dir = os.path.join(ckpt_root, f"view_{views:03d}")
    candidates = sorted(glob(os.path.join(view_dir, "*.pth.tar")), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under: {view_dir}")
    return candidates[-1]


def load_model(dataset, ckpt_path, device):
    model = reconstructor(dataset).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if "reconstructor_state" not in state:
        raise KeyError(f"Missing 'reconstructor_state' in checkpoint. Keys: {list(state.keys())}")
    model.load_state_dict(state["reconstructor_state"], strict=True)
    model.eval()
    return model


def tensor_to_numpy(x):
    x = torch.nan_to_num(x.detach(), nan=0.0, posinf=1.0, neginf=0.0)
    x = torch.clamp(x, 0.0, 1.0)
    return x.cpu().numpy()


def calc_metrics(pred, gt):
    pred_np = tensor_to_numpy(pred)
    gt_np = tensor_to_numpy(gt)

    rows = []
    for b in range(pred_np.shape[0]):
        for c in range(pred_np.shape[1]):
            x = np.nan_to_num(pred_np[b, c], nan=0.0, posinf=1.0, neginf=0.0)
            y = np.nan_to_num(gt_np[b, c], nan=0.0, posinf=1.0, neginf=0.0)
            rmse = float(np.sqrt(np.mean((x - y) ** 2)))
            rows.append(
                {
                    "psnr": float(peak_signal_noise_ratio(y, x, data_range=1.0)),
                    "ssim": float(
                        structural_similarity(
                            y,
                            x,
                            gaussian_weights=True,
                            win_size=11,
                            data_range=1.0,
                            sigma=1.5,
                        )
                    ),
                    "rmse": rmse,
                }
            )
    return rows


def average_metrics(rows):
    if not rows:
        return {"psnr": 0.0, "ssim": 0.0, "rmse": 0.0}
    return {
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "rmse": float(np.mean([r["rmse"] for r in rows])),
    }


def to_uint8(img):
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def load_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def make_panel(gt, fbp, pred, out_path, title):
    gt_img = gt[0, 0]
    fbp_img = fbp[0, 0]
    pred_img = pred[0, 0]
    err_img = np.abs(pred_img - gt_img)
    if err_img.max() > 0:
        err_img = err_img / err_img.max()

    tiles = [
        ("GT", gt_img),
        ("FBP", fbp_img),
        ("DuDoTrans", pred_img),
        ("Abs Error", err_img),
    ]

    cell = 256
    header_h = 62
    title_h = 44
    pad = 10
    width = len(tiles) * cell + (len(tiles) + 1) * pad
    height = title_h + header_h + cell + 2 * pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font_title = load_font(24)
    font_head = load_font(22)

    draw.text((pad, 10), title, fill=(0, 0, 0), font=font_title)
    for i, (name, img) in enumerate(tiles):
        x = pad + i * (cell + pad)
        bbox = draw.textbbox((0, 0), name, font=font_head)
        tw = bbox[2] - bbox[0]
        draw.text((x + (cell - tw) // 2, title_h + 18), name, fill=(0, 0, 0), font=font_head)
        tile = Image.fromarray(to_uint8(img), mode="L").resize((cell, cell), Image.BICUBIC).convert("RGB")
        canvas.paste(tile, (x, title_h + header_h))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def evaluate_view(args, views, ckpt_path, device):
    set_seed(args.seed)
    dataset = NPZ_CTSlice_Provider(
        npz_path=args.test_npz,
        poission_level=args.poission_level,
        gaussian_level=args.gaussian_level,
        num_view=views,
        img_size=512,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model = load_model(dataset, ckpt_path, device)

    all_pred_rows = []
    all_fbp_rows = []
    per_slice_rows = []
    vis_set = set(args.vis_indices or [])
    view_tag = f"view_{views:03d}"
    vis_dir = os.path.join(args.out_dir, view_tag, "vis")

    total = len(dataset)
    if args.max_samples is not None:
        total = min(total, args.max_samples)

    infer_time = 0.0
    seen = 0
    with torch.no_grad():
        pbar = tqdm(loader, total=(total + args.batch_size - 1) // args.batch_size, desc=f"test {view_tag}", ncols=120)
        for batch_idx, (gt, fbp_u, projs_noisy) in enumerate(pbar):
            start_idx = batch_idx * args.batch_size
            if start_idx >= total:
                break
            keep = min(gt.shape[0], total - start_idx)
            gt = gt[:keep].to(device, non_blocking=True)
            fbp_u = fbp_u[:keep].to(device, non_blocking=True)
            projs_noisy = projs_noisy[:keep].float().to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            tic = time.time()
            _, _, _, pred = model(fbp_u, gt, projs_noisy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_time += time.time() - tic

            pred_rows = calc_metrics(pred, gt)
            fbp_rows = calc_metrics(fbp_u, gt)
            all_pred_rows.extend(pred_rows)
            all_fbp_rows.extend(fbp_rows)

            pred_np = tensor_to_numpy(pred)
            gt_np = tensor_to_numpy(gt)
            fbp_np = tensor_to_numpy(fbp_u)
            for local_i in range(keep):
                sample_idx = start_idx + local_i
                row = {
                    "view": views,
                    "index": sample_idx,
                    "pred_psnr": pred_rows[local_i]["psnr"],
                    "pred_ssim": pred_rows[local_i]["ssim"],
                    "pred_rmse": pred_rows[local_i]["rmse"],
                    "fbp_psnr": fbp_rows[local_i]["psnr"],
                    "fbp_ssim": fbp_rows[local_i]["ssim"],
                    "fbp_rmse": fbp_rows[local_i]["rmse"],
                }
                per_slice_rows.append(row)
                if sample_idx in vis_set:
                    out_path = os.path.join(vis_dir, f"idx_{sample_idx:04d}.png")
                    title = (
                        f"{view_tag} idx_{sample_idx:04d} | "
                        f"PSNR {row['pred_psnr']:.2f} SSIM {row['pred_ssim']:.4f}"
                    )
                    make_panel(
                        gt_np[local_i : local_i + 1],
                        fbp_np[local_i : local_i + 1],
                        pred_np[local_i : local_i + 1],
                        out_path,
                        title,
                    )

            seen += keep
            avg = average_metrics(all_pred_rows)
            pbar.set_postfix({"PSNR": f"{avg['psnr']:.2f}", "SSIM": f"{avg['ssim']:.4f}"})

    pred_avg = average_metrics(all_pred_rows)
    fbp_avg = average_metrics(all_fbp_rows)
    return {
        "view": views,
        "ckpt": ckpt_path,
        "num_samples": seen,
        "pred_psnr": pred_avg["psnr"],
        "pred_ssim": pred_avg["ssim"],
        "pred_rmse": pred_avg["rmse"],
        "fbp_psnr": fbp_avg["psnr"],
        "fbp_ssim": fbp_avg["ssim"],
        "fbp_rmse": fbp_avg["rmse"],
        "avg_infer_time_sec": infer_time / max(1, seen),
        "per_slice": per_slice_rows,
    }


def write_outputs(out_dir, summaries):
    os.makedirs(out_dir, exist_ok=True)
    summary_csv = os.path.join(out_dir, "summary.csv")
    detail_csv = os.path.join(out_dir, "per_slice_metrics.csv")
    summary_json = os.path.join(out_dir, "summary.json")

    summary_fields = [
        "view",
        "num_samples",
        "pred_psnr",
        "pred_ssim",
        "pred_rmse",
        "fbp_psnr",
        "fbp_ssim",
        "fbp_rmse",
        "avg_infer_time_sec",
        "ckpt",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for item in summaries:
            writer.writerow({k: item[k] for k in summary_fields})

    detail_fields = [
        "view",
        "index",
        "pred_psnr",
        "pred_ssim",
        "pred_rmse",
        "fbp_psnr",
        "fbp_ssim",
        "fbp_rmse",
    ]
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        for item in summaries:
            writer.writerows(item["per_slice"])

    compact = [{k: v for k, v in item.items() if k != "per_slice"} for item in summaries]
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)

    return summary_csv, detail_csv, summary_json


def main():
    args = parse_args()
    if args.ckpt and len(args.views) != 1:
        raise ValueError("--ckpt can only be used when exactly one view is specified.")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    summaries = []
    for views in args.views:
        ckpt_path = args.ckpt or find_latest_ckpt(args.ckpt_root, views)
        print(f"[INFO] views={views} ckpt={ckpt_path}")
        summaries.append(evaluate_view(args, views, ckpt_path, device))

    summary_csv, detail_csv, summary_json = write_outputs(args.out_dir, summaries)
    print(f"[OK] summary_csv: {summary_csv}")
    print(f"[OK] detail_csv: {detail_csv}")
    print(f"[OK] summary_json: {summary_json}")
    for item in summaries:
        print(
            "[RESULT] view={view:3d} n={num_samples} "
            "DuDoTrans PSNR={pred_psnr:.2f} SSIM={pred_ssim:.4f} RMSE={pred_rmse:.4f} | "
            "FBP PSNR={fbp_psnr:.2f} SSIM={fbp_ssim:.4f} RMSE={fbp_rmse:.4f}".format(**item)
        )


if __name__ == "__main__":
    main()
