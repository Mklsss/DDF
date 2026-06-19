import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ssim
from ddf_experiment_lib import build_model, load_config, resolve_path, SinogramCTDataset


def save_img(path, x, cmap="gray", vmin=0, vmax=1):
    arr = x.detach().cpu().float().squeeze().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, arr, cmap=cmap, vmin=vmin, vmax=vmax)


def stats(name, x):
    x = x.detach().cpu().float()
    return {
        "name": name,
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
    }


def psnr_one(pred, gt):
    data_range = torch.max(gt) - torch.min(gt)
    mse = torch.mean((pred - gt) ** 2)
    return float(10.0 * torch.log10((data_range ** 2) / mse))


def unwrap_state(obj):
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def infer_width(state, default_width):
    if "ct.intro.weight" in state:
        return int(state["ct.intro.weight"].shape[0])
    return int(default_width)


def load_original_ddf_namespace(s):
    text = Path(f"DDP_run_c{s}.py").read_text(encoding="utf-8")
    stop = text.index("model = mymodel().to(device)")
    ns = {"__name__": "__diagnose__"}
    exec(text[:stop], ns)
    return ns


def main():
    config = load_config(None)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    samples = [0, 1, 2]
    factors = [2, 4, 8, 12]

    for s in factors:
        ckpt = Path(f"weights/cascade_S{s}.pth")
        print(f"\n=== S={s} ===")
        print(f"cascade checkpoint: {ckpt}, exists={ckpt.exists()}")

        state = unwrap_state(torch.load(ckpt, map_location=device))
        print(f"checkpoint keys: {len(state)}")
        print(f"checkpoint ct.intro.weight shape: {tuple(state['ct.intro.weight'].shape) if 'ct.intro.weight' in state else 'missing'}")

        cascade_config = dict(config)
        cascade_config["nafnet"] = dict(config["nafnet"])
        cascade_config["nafnet"]["width"] = infer_width(state, config["nafnet"]["width"])
        print(f"cascade model width used: {cascade_config['nafnet']['width']}")

        model = build_model("cascade", s, cascade_config).to(device)
        load_ret = model.load_state_dict(state, strict=False)
        print(f"missing keys: {len(load_ret.missing_keys)}")
        print(f"unexpected keys: {len(load_ret.unexpected_keys)}")
        if load_ret.missing_keys[:5]:
            print(f"first missing keys: {load_ret.missing_keys[:5]}")
        if load_ret.unexpected_keys[:5]:
            print(f"first unexpected keys: {load_ret.unexpected_keys[:5]}")
        model.eval()

        ns = load_original_ddf_namespace(s)
        ddf = ns["mymodel"]().to(device)
        ddf_state = unwrap_state(torch.load(f"weights/DDF_c{s}_best.pth", map_location=device))
        ddf.load_state_dict(ddf_state, strict=False)
        ddf.eval()

        dataset = SinogramCTDataset(resolve_path(config["test_data"]), s)

        for idx in samples:
            sin_in, gt = dataset[idx]
            sin_in = sin_in.unsqueeze(0).to(device=device, dtype=torch.float32)
            gt = gt.unsqueeze(0).to(device=device, dtype=torch.float32)

            with torch.no_grad():
                cascade, extra = model(sin_in)
                cascade = cascade.clamp(0, 1)
                sparse_fbp = extra["sparse_fbp"].clamp(0, 1)

                ddf_in, _ = ns["load_data"]("./data/test_meiaonew.npz")[idx]
                ddf_in = torch.stack([ddf_in, ddf_in], dim=0).to(device=device, dtype=torch.float32)
                ddf_pred = ddf(ddf_in)[:1].clamp(0, 1)

            cascade_plus_fbp = (cascade + sparse_fbp).clamp(0, 1)

            out_dir = Path(f"results/cascade_diagnostic/sample_{idx:03d}/S{s}")
            save_img(out_dir / "ground_truth.png", gt)
            save_img(out_dir / "sparse_fbp.png", sparse_fbp)
            save_img(out_dir / "cascade.png", cascade)
            save_img(out_dir / "cascade_plus_sparse_fbp.png", cascade_plus_fbp)
            save_img(out_dir / "ddf.png", ddf_pred)
            save_img(out_dir / "error_cascade.png", torch.abs(cascade - gt), cmap="magma", vmin=0, vmax=1)
            save_img(out_dir / "error_ddf.png", torch.abs(ddf_pred - gt), cmap="magma", vmin=0, vmax=1)

            for name, tensor in [
                ("ground_truth", gt),
                ("sparse_fbp", sparse_fbp),
                ("cascade", cascade),
                ("cascade_plus_sparse_fbp", cascade_plus_fbp),
                ("ddf", ddf_pred),
            ]:
                st = stats(name, tensor)
                row = {
                    "sample": idx,
                    "sparse_factor": s,
                    **st,
                    "psnr": psnr_one(tensor, gt),
                    "ssim": float(ssim.ssim(tensor, gt).item()),
                }
                rows.append(row)
                print(row)

    out_csv = Path("results/cascade_diagnostic/diagnostic_stats.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "sparse_factor", "name", "min", "max", "mean", "std", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved diagnostic csv: {out_csv}")
    print("saved diagnostic images: results/cascade_diagnostic/")


if __name__ == "__main__":
    main()
