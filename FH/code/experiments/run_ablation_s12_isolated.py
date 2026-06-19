import argparse
import csv
import subprocess
import sys
from pathlib import Path


def run_one(name, args, extra):
    tmp_csv = Path(f"results/.ablation_s12_{name}.csv")
    cmd = [
        sys.executable,
        "experiments/run_ablation_s12.py",
        "--mode", "eval",
        "--device", args.device,
        "--test_data", args.test_data,
        "--output_csv", str(tmp_csv),
        *extra,
    ]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return tmp_csv


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--test_data", default="data/test_meiaonew_no_overlap.npz")
    parser.add_argument("--output_csv", default="results/ablation_s12_no_overlap.csv")
    parser.add_argument("--proposed_checkpoint", default="weights/DDF_c12_best.pth")
    parser.add_argument("--sine_checkpoint", default="weights/DDF_c12_sine_ablation_best.pth")
    parser.add_argument("--ct_checkpoint", default="weights/DDF_c12_ct_ablation_best.pth")
    args = parser.parse_args()

    missing = "weights/__missing_checkpoint__.pth"

    proposed_csv = run_one("proposed", args, [
        "--proposed_checkpoint", args.proposed_checkpoint,
        "--sine_checkpoint", missing,
        "--ct_checkpoint", missing,
    ])
    sine_csv = run_one("sine", args, [
        "--proposed_checkpoint", missing,
        "--sine_checkpoint", args.sine_checkpoint,
        "--ct_checkpoint", missing,
    ])
    ct_csv = run_one("ct", args, [
        "--proposed_checkpoint", missing,
        "--sine_checkpoint", missing,
        "--ct_checkpoint", args.ct_checkpoint,
    ])

    rows = []
    for p in [proposed_csv, sine_csv, ct_csv]:
        rows.extend(read_rows(p))

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["module", "variant", "sparse_factor", "psnr", "ssim", "num_samples", "checkpoint"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved CSV: {out}")
    with out.open() as f:
        print(f.read())


if __name__ == "__main__":
    main()
