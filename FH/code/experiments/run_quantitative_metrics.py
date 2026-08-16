import argparse
import csv
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ssim
from ddf_experiment_lib import FbpLayer, build_model, load_config, resolve_path


DEFAULT_REDCNN_ROOT = ROOT.parents[1] / "redcnn_autodl"
DEFAULT_TEST_DATA = ROOT.parents[1] / "dataset" / "test_meiaonew.npz"


def unwrap_state_dict(obj):
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def find_checkpoint(method, sparse_factor):
    patterns = {
        "ddf": [f"weights/DDF_c{sparse_factor}_best.pth", f"weights/DDF_c{sparse_factor}_ckpt.pth"],
        "cascade": [
            f"weights/cascade_original_S{sparse_factor}.pth",
            f"weights/cascade_S{sparse_factor}.pth",
            f"weights/Cascade_{sparse_factor}.pth",
        ],
    }
    for item in patterns.get(method, []):
        path = Path(item)
        if path.exists():
            return path
    return None


def load_original_ddf_namespace(sparse_factor):
    text = Path(f"DDP_run_c{sparse_factor}.py").read_text(encoding="utf-8")
    stop = text.index("model = mymodel().to(device)")
    ns = {"__name__": "__quantitative_metrics__"}
    exec(text[:stop], ns)
    return ns


def infer_width(state_dict, default_width):
    if "ct.intro.weight" in state_dict:
        return int(state_dict["ct.intro.weight"].shape[0])
    return int(default_width)


def psnr_batch(prediction, target):
    scores = []
    for i in range(prediction.shape[0]):
        data_range = torch.max(target[i]) - torch.min(target[i])
        mse = torch.mean((prediction[i] - target[i]) ** 2)
        scores.append(float(10.0 * torch.log10((data_range ** 2) / mse)))
    return scores


def make_batches(dataset, batch_size, max_samples):
    total = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        sin_list, label_list = [], []
        for idx in range(start, end):
            sin_in, label = dataset[idx]
            sin_list.append(sin_in)
            label_list.append(label)
        yield torch.stack(sin_list, dim=0), torch.stack(label_list, dim=0)


def empty_metrics():
    return {
        "psnr_sum": 0.0,
        "ssim_sum": 0.0,
        "rmse_sum": 0.0,
        "mae_sum": 0.0,
        "num_samples": 0,
    }


def update_metrics(pred, label, acc):
    pred = pred.clamp(0, 1)
    label = label.to(pred.device)
    acc["psnr_sum"] += float(np.sum(psnr_batch(pred, label)))
    acc["ssim_sum"] += float(ssim.ssim(pred, label).item()) * pred.shape[0]
    error = pred - label
    per_slice_mse = torch.mean(error.square(), dim=(1, 2, 3))
    per_slice_mae = torch.mean(error.abs(), dim=(1, 2, 3))
    acc["rmse_sum"] += float(torch.sqrt(per_slice_mse).sum().item())
    acc["mae_sum"] += float(per_slice_mae.sum().item())
    acc["num_samples"] += int(pred.shape[0])


def evaluate_sparse_fbp(dataset, config, batch_size, max_samples, device):
    fbp = FbpLayer(resolve_path(config["fbp_matrix"])).to(device)
    fbp.eval()
    acc = empty_metrics()
    with torch.no_grad():
        for sin_in, label in make_batches(dataset, batch_size, max_samples):
            sin_in = sin_in.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = fbp(sin_in).permute(0, 3, 1, 2)
            update_metrics(pred, label, acc)
    return acc


def evaluate_cascade_original(dataset, sparse_factor, checkpoint, batch_size, max_samples, device):
    ns = load_original_ddf_namespace(sparse_factor)

    class CascadeOriginal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sin = ns["sin_angle"](num_sensor=357, angle=int(360 / sparse_factor), num_heads=1)
            self.fbp = ns["FbpLayer"]()
            self.ct = ns["NAFNet"](
                img_channel=1,
                width=32,
                middle_blk_num=1,
                enc_blk_nums=[1, 1, 1, 28],
                dec_blk_nums=[1, 1, 1, 1],
            )

        def forward(self, x):
            sin1 = self.sin(x)
            fbp1 = self.fbp(sin1).permute(0, 3, 1, 2)
            return self.ct(fbp1)

    model = CascadeOriginal().to(device)
    state = unwrap_state_dict(torch.load(checkpoint, map_location=device))
    print(model.load_state_dict(state, strict=False))
    model.eval()

    acc = empty_metrics()
    with torch.no_grad():
        for sin_in, label in make_batches(dataset, batch_size, max_samples):
            sin_in = sin_in.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = model(sin_in)
            update_metrics(pred, label, acc)
    return acc


def evaluate_cascade(dataset, sparse_factor, checkpoint, config, batch_size, max_samples, device):
    state = unwrap_state_dict(torch.load(checkpoint, map_location=device))
    model_config = dict(config)
    model_config["nafnet"] = dict(config["nafnet"])
    model_config["nafnet"]["width"] = infer_width(state, config["nafnet"]["width"])
    model = build_model("cascade", sparse_factor, model_config).to(device)
    print(model.load_state_dict(state, strict=False))
    model.eval()

    acc = empty_metrics()
    with torch.no_grad():
        for sin_in, label in make_batches(dataset, batch_size, max_samples):
            sin_in = sin_in.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred, _ = model(sin_in)
            update_metrics(pred, label, acc)
    return acc


def evaluate_redcnn(dataset, sparse_factor, checkpoint, batch_size, max_samples, device):
    redcnn_root = checkpoint.parent.parent
    if str(redcnn_root) not in sys.path:
        sys.path.insert(0, str(redcnn_root))
    from networks import RED_CNN

    ns = load_original_ddf_namespace(sparse_factor)

    class REDCNNPipeline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fbp = ns["FbpLayer"]()
            self.net = RED_CNN()

        def forward(self, sinogram):
            fbp = self.fbp(sinogram).permute(0, 3, 1, 2)
            padded = F.pad(fbp, (10, 10, 10, 10), mode="reflect")
            return self.net(padded)[:, :, 10:-10, 10:-10]

    model = REDCNNPipeline().to(device)
    state = unwrap_state_dict(torch.load(checkpoint, map_location=device))
    print(model.net.load_state_dict(state, strict=True))
    model.eval()

    acc = empty_metrics()
    with torch.no_grad():
        for sin_in, label in make_batches(dataset, batch_size, max_samples):
            sin_in = sin_in.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            update_metrics(model(sin_in), label, acc)
    return acc


def evaluate_ddf(dataset, sparse_factor, checkpoint, batch_size, max_samples, device):
    ns = load_original_ddf_namespace(sparse_factor)
    model = ns["mymodel"]().to(device)
    state = unwrap_state_dict(torch.load(checkpoint, map_location=device))
    print(model.load_state_dict(state, strict=False))
    model.eval()

    acc = empty_metrics()
    with torch.no_grad():
        for sin_in, label in make_batches(dataset, batch_size, max_samples):
            n = sin_in.shape[0]
            if n == 1:
                sin_in = torch.cat([sin_in, sin_in], dim=0)
                label = torch.cat([label, label], dim=0)
            sin_in = sin_in.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = model(sin_in)[:n]
            update_metrics(pred, label[:n], acc)
    return acc


def finalize(method, sparse_factor, acc, checkpoint):
    n = max(acc["num_samples"], 1)
    return {
        "method": method,
        "sparse_factor": sparse_factor,
        "psnr": acc["psnr_sum"] / n,
        "ssim": acc["ssim_sum"] / n,
        "rmse": acc["rmse_sum"] / n,
        "mae": acc["mae_sum"] / n,
        "num_samples": acc["num_samples"],
        "checkpoint": str(checkpoint) if checkpoint else "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse_factors", default="2,4,8,12")
    parser.add_argument("--methods", default="sparse_fbp,redcnn,cascade,ddf")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_csv", default="results/quantitative_results.csv")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--test_data", default=str(DEFAULT_TEST_DATA))
    parser.add_argument(
        "--redcnn_checkpoint",
        default=str(DEFAULT_REDCNN_ROOT / "save" / "REDCNN_100000iter.ckpt"),
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    factors = [int(x.strip()) for x in args.sparse_factors.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    rows = []

    for s in factors:
        ns = load_original_ddf_namespace(s)
        lib_dataset = ns["load_data"](args.test_data)

        for method in methods:
            if method == "sparse_fbp":
                acc = evaluate_sparse_fbp(lib_dataset, config, args.batch_size, args.max_samples, device)
                ckpt = ""
            elif method == "redcnn":
                ckpt = Path(args.redcnn_checkpoint)
                if not ckpt.exists():
                    raise FileNotFoundError(ckpt)
                acc = evaluate_redcnn(
                    lib_dataset, s, ckpt, args.batch_size, args.max_samples, device
                )
            elif method == "cascade":
                ckpt = find_checkpoint("cascade", s)
                if ckpt is None:
                    print(f"checkpoint for S={s} not found, skip.")
                    continue
                if "cascade_original" in str(ckpt):
                    acc = evaluate_cascade_original(
                        lib_dataset, s, ckpt, args.batch_size, args.max_samples, device
                    )
                else:
                    acc = evaluate_cascade(lib_dataset, s, ckpt, config, args.batch_size, args.max_samples, device)
            elif method == "ddf":
                ckpt = find_checkpoint("ddf", s)
                if ckpt is None:
                    print(f"checkpoint for S={s} not found, skip.")
                    continue
                acc = evaluate_ddf(
                    lib_dataset, s, ckpt, args.batch_size, args.max_samples, device
                )
            else:
                print(f"unknown method: {method}, skip.")
                continue

            row = finalize(method, s, acc, ckpt)
            rows.append(row)
            print(
                f"{method}, S={s}, PSNR={row['psnr']:.6f}, "
                f"SSIM={row['ssim']:.6f}, RMSE={row['rmse']:.6f}, "
                f"MAE={row['mae']:.6f}, num_samples={row['num_samples']}, "
                f"checkpoint={row['checkpoint']}"
            )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "sparse_factor",
                "psnr",
                "ssim",
                "rmse",
                "mae",
                "num_samples",
                "checkpoint",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved CSV: {output_csv}")


if __name__ == "__main__":
    main()
