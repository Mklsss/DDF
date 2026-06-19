import csv
import gc
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ssim
from ddf_experiment_lib import build_model, load_config


def unwrap_state_dict(obj):
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def infer_width(state_dict, default_width):
    if "ct.intro.weight" in state_dict:
        return int(state_dict["ct.intro.weight"].shape[0])
    return int(default_width)


def load_original_ddf_namespace(sparse_factor):
    script = Path(f"DDP_run_c{sparse_factor}.py")
    if not script.exists():
        script = Path("DDP_run.py.py")
    text = script.read_text(encoding="utf-8")
    stop = text.index("model = mymodel().to(device)")
    ns = {"__name__": "__ablation_metrics__"}
    exec(text[:stop], ns)
    return ns


def find_checkpoint(patterns, sparse_factor):
    for pattern in patterns:
        path = Path(pattern.format(S=sparse_factor, sparse_factor=sparse_factor))
        if path.exists():
            return path
    return None


def psnr_batch(prediction, target):
    scores = []
    for i in range(prediction.shape[0]):
        img1 = prediction[i].detach()
        img2 = target[i].detach()
        data_range = torch.max(img2) - torch.min(img2)
        mse = torch.mean((img1 - img2) ** 2)
        if mse.item() <= 0:
            scores.append(float("inf"))
        else:
            scores.append((10.0 * torch.log10((data_range ** 2) / mse)).item())
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


def update_metrics(pred, label, acc):
    pred = pred.clamp(0, 1)
    label = label.to(pred.device)
    acc["psnr_sum"] += float(np.sum(psnr_batch(pred, label)))
    acc["ssim_sum"] += float(ssim.ssim(pred, label).item()) * pred.shape[0]
    acc["num_samples"] += int(pred.shape[0])


def eval_original_ddf(sparse_factor, checkpoint, batch_size, max_samples, device):
    ns = load_original_ddf_namespace(sparse_factor)
    dataset = ns["load_data"]("./data/test_meiaonew.npz")
    model = ns["mymodel"]().to(device)
    state = unwrap_state_dict(torch.load(checkpoint, map_location=device))
    model.load_state_dict(state, strict=False)
    model.eval()

    acc = {"psnr_sum": 0.0, "ssim_sum": 0.0, "num_samples": 0}
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


def eval_lib_variant(method_key, sparse_factor, checkpoint, batch_size, max_samples, device, config):
    ns = load_original_ddf_namespace(sparse_factor)
    dataset = ns["load_data"]("./data/test_meiaonew.npz")
    state = unwrap_state_dict(torch.load(checkpoint, map_location=device))

    variant_config = dict(config)
    variant_config["nafnet"] = dict(config["nafnet"])
    variant_config["nafnet"]["width"] = infer_width(state, config["nafnet"]["width"])

    model = build_model(method_key, sparse_factor, variant_config).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    acc = {"psnr_sum": 0.0, "ssim_sum": 0.0, "num_samples": 0}
    with torch.no_grad():
        for sin_in, label in make_batches(dataset, batch_size, max_samples):
            n = sin_in.shape[0]
            if n == 1:
                sin_in = torch.cat([sin_in, sin_in], dim=0)
                label = torch.cat([label, label], dim=0)
            sin_in = sin_in.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            out = model(sin_in)
            pred = out[0] if isinstance(out, tuple) else out
            update_metrics(pred[:n], label[:n], acc)
    return acc


def finalize(method, sparse_factor, acc, checkpoint):
    n = max(acc["num_samples"], 1)
    return {
        "method": method,
        "sparse_factor": sparse_factor,
        "psnr": acc["psnr_sum"] / n,
        "ssim": acc["ssim_sum"] / n,
        "num_samples": acc["num_samples"],
        "checkpoint": str(checkpoint),
    }


def write_rows(rows, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "sparse_factor", "psnr", "ssim", "num_samples", "checkpoint"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved CSV: {output_csv}")


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_device(device_text):
    return torch.device(device_text if torch.cuda.is_available() else "cpu")
