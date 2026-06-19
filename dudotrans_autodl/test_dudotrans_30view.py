import argparse
import os

import torch

from test_dudotrans import evaluate_view, find_latest_ckpt, write_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate only the 30-view DuDoTrans checkpoint.")
    parser.add_argument("--test_npz", type=str, default="/root/autodl-fs/dataset/test_meiaonew.npz")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional 30-view checkpoint path.")
    parser.add_argument("--ckpt_root", type=str, default="./results/models")
    parser.add_argument("--out_dir", type=str, default="./results/test_30view")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--vis_indices", type=int, nargs="*", default=[0, 66, 133, 199])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--poission_level", type=float, default=1e5)
    parser.add_argument("--gaussian_level", type=float, default=0.05)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    views = 30
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ckpt_path = args.ckpt or find_latest_ckpt(args.ckpt_root, views)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"30-view checkpoint not found: {ckpt_path}")

    print(f"[INFO] testing only 30 views")
    print(f"[INFO] ckpt={ckpt_path}")
    summary = evaluate_view(args, views, ckpt_path, device)
    summary_csv, detail_csv, summary_json = write_outputs(args.out_dir, [summary])

    print(f"[OK] summary_csv: {summary_csv}")
    print(f"[OK] detail_csv: {detail_csv}")
    print(f"[OK] summary_json: {summary_json}")
    print(
        "[RESULT] 30 views | n={num_samples} | "
        "DuDoTrans PSNR={pred_psnr:.2f} SSIM={pred_ssim:.4f} RMSE={pred_rmse:.4f} | "
        "FBP PSNR={fbp_psnr:.2f} SSIM={fbp_ssim:.4f} RMSE={fbp_rmse:.4f}".format(**summary)
    )


if __name__ == "__main__":
    main()
