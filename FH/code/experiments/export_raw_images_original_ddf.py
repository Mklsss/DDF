import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ddf_experiment_lib import build_model, load_config, resolve_path, SinogramCTDataset


def save_gray_png(path, tensor):
    arr = tensor.detach().cpu().float().squeeze().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, arr, cmap="gray", vmin=0.0, vmax=1.0)


def load_original_ddf_namespace(script_path):
    text = Path(script_path).read_text(encoding="utf-8")
    stop = text.index("model = mymodel().to(device)")
    ns = {"__name__": "__export_raw__"}
    exec(text[:stop], ns)
    return ns


def load_original_ddf_prediction(sparse_factor, checkpoint, sample_index, device):
    ns = load_original_ddf_namespace(f"DDP_run_c{sparse_factor}.py")
    model = ns["mymodel"]().to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    dataset = ns["load_data"]("./data/test_meiaonew.npz")
    sample_indices = [sample_index, min(sample_index + 1, len(dataset) - 1)]
    sin_list = []
    gt_list = []
    for idx in sample_indices:
        sin_i, gt_i = dataset[idx]
        sin_list.append(sin_i)
        gt_list.append(gt_i)
    sin_in = torch.stack(sin_list, dim=0).to(device=device, dtype=torch.float32)
    gt = torch.stack(gt_list, dim=0)

    with torch.no_grad():
        ddf = model(sin_in).clamp(0, 1)
        sparse_fbp = model.fbp(sin_in).permute(0, 3, 1, 2).clamp(0, 1)
    return gt[:1], sparse_fbp[:1], ddf[:1]


def load_cascade_original_prediction(sparse_factor, checkpoint, sample_index, device):
    ns = load_original_ddf_namespace(f"DDP_run_c{sparse_factor}.py")

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
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    dataset = ns["load_data"]("./data/test_meiaonew.npz")
    sin_in, _ = dataset[sample_index]
    sin_in = sin_in.unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        return model(sin_in).clamp(0, 1)


def load_cascade_prediction(sparse_factor, checkpoint, config, sample_index, device):
    dataset = SinogramCTDataset(resolve_path(config["test_data"]), sparse_factor)
    sin_in, _ = dataset[sample_index]
    sin_in = sin_in.unsqueeze(0).to(device=device, dtype=torch.float32)

    model = build_model("cascade", sparse_factor, config).to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        cascade, _ = model(sin_in)
    return cascade.clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--sparse_factors", default="2,4,8,12")
    parser.add_argument("--ddf_checkpoint", default="weights/DDF_c{S}_best.pth")
    parser.add_argument("--cascade_checkpoint", default="weights/cascade_S{S}.pth")
    parser.add_argument("--output_dir", default="fig/qualitative_raw_original")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    factors = [int(x.strip()) for x in args.sparse_factors.split(",") if x.strip()]
    root = Path(args.output_dir) / f"sample_{args.sample_index:03d}"

    for s in factors:
        ddf_ckpt = args.ddf_checkpoint.format(S=s, sparse_factor=s)
        cascade_ckpt = args.cascade_checkpoint.format(S=s, sparse_factor=s)
        gt, sparse_fbp, ddf = load_original_ddf_prediction(s, ddf_ckpt, args.sample_index, device)
        out = root / f"S{s}"
        save_gray_png(out / "ground_truth.png", gt)
        save_gray_png(out / "sparse_fbp.png", sparse_fbp)
        save_gray_png(out / "ddf.png", ddf)

        try:
            if "cascade_original" in cascade_ckpt:
                cascade = load_cascade_original_prediction(s, cascade_ckpt, args.sample_index, device)
            else:
                cascade = load_cascade_prediction(s, cascade_ckpt, config, args.sample_index, device)
            save_gray_png(out / "cascade.png", cascade)
        except Exception as exc:
            print(f"skip cascade for S={s}: {exc}")
        print(f"saved raw images for S={s}: {out}")


if __name__ == "__main__":
    main()
