import argparse

from ablation_metric_utils import (
    cleanup,
    eval_lib_variant,
    eval_original_ddf,
    finalize,
    find_checkpoint,
    get_device,
    load_config,
    write_rows,
)

DEFAULT_DDF_PATTERNS = [
    "weights/DDF_c{S}_best.pth",
    "weights/DDF_c{S}_ckpt.pth",
]

CONV_FUSION_PATTERNS = [
    "weights/ddf_ct_conv_fusion_S{S}.pth",
    "weights/ct_conv_fusion_S{S}.pth",
    "weights/conv_fusion_S{S}.pth",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse_factors", default="2,4,8,12")
    parser.add_argument("--methods", default="CONV Fusion,CGB Fusion")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_csv", default="results/ablation_ct_fusion.csv")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    config = load_config(args.config)
    factors = [int(x.strip()) for x in args.sparse_factors.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    rows = []

    for s in factors:
        for method in methods:
            if method == "CGB Fusion":
                ckpt = find_checkpoint(DEFAULT_DDF_PATTERNS, s)
                if ckpt is None:
                    print(f"checkpoint for S={s} not found, skip.")
                    continue
                acc = eval_original_ddf(s, ckpt, args.batch_size, args.max_samples, device)
            elif method == "CONV Fusion":
                ckpt = find_checkpoint(CONV_FUSION_PATTERNS, s)
                if ckpt is None:
                    print(f"checkpoint for S={s} not found, skip.")
                    continue
                acc = eval_lib_variant("ddf_ct_conv_fusion", s, ckpt, args.batch_size, args.max_samples, device, config)
            else:
                print(f"unknown method: {method}, skip.")
                continue

            row = finalize(method, s, acc, ckpt)
            rows.append(row)
            print(f"{method}, S={s}, PSNR={row['psnr']:.6f}, SSIM={row['ssim']:.6f}, num_samples={row['num_samples']}, checkpoint={row['checkpoint']}")
            cleanup()

    write_rows(rows, args.output_csv)


if __name__ == "__main__":
    main()
