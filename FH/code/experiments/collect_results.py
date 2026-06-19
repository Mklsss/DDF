import argparse
import csv
from pathlib import Path


DEFAULT_FILES = [
    "ddf_fixed_factors.csv",
    "cascade_fixed_factors.csv",
    "ablation_sine_fusion.csv",
    "ablation_ct_fusion.csv",
]


def read_rows(csv_path):
    if not csv_path.exists():
        print(f"missing result file: {csv_path}")
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def main():
    parser = argparse.ArgumentParser(description="Collect experiment csv files into one summary csv.")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output", default="results/summary.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for file_name in DEFAULT_FILES:
        rows.extend(read_rows(results_dir / file_name))
    if not rows:
        print("no rows collected")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "sparse_factor", "psnr", "ssim", "checkpoint", "config"]
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
