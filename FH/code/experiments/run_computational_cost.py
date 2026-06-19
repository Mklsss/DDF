import argparse
import csv
import gc
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddf_experiment_lib import build_model, freeze_projection_layers, load_config


def load_original_ddf_namespace(sparse_factor):
    script = Path(f"DDP_run_c{sparse_factor}.py")
    if not script.exists():
        script = Path("DDP_run.py.py")
    text = script.read_text(encoding="utf-8")
    stop = text.index("model = mymodel().to(device)")
    ns = {"__name__": "__computational_cost__"}
    exec(text[:stop], ns)
    return ns


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_params(model):
    return sum(p.numel() for p in model.parameters())


def profile_flops(model, input_tensor):
    try:
        from thop import profile
    except Exception as exc:
        return None, f"thop import failed: {exc}"

    try:
        model.eval()
        with torch.no_grad():
            flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        return int(flops), "thop.profile"
    except Exception as exc:
        return None, f"thop profile failed: {exc}"


def build_ddf(sparse_factor, device):
    ns = load_original_ddf_namespace(sparse_factor)
    model = ns["mymodel"]().to(device)
    if hasattr(model, "fbp"):
        for p in model.fbp.parameters():
            p.requires_grad = False
    if hasattr(model, "fp"):
        for p in model.fp.parameters():
            p.requires_grad = False
    return model


def build_cascade(sparse_factor, config, device):
    model = build_model("cascade", sparse_factor, config).to(device)
    freeze_projection_layers(model)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse_factors", default="2,4,8,12")
    parser.add_argument("--methods", default="cascade,ddf")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_csv", default="results/computational_cost.csv")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    factors = [int(x.strip()) for x in args.sparse_factors.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    input_shape = (args.batch_size, 360, 357)
    rows = []

    for s in factors:
        for method in methods:
            if method == "cascade":
                model = build_cascade(s, config, device)
                definition = "SIN_ld -> projection-domain model -> FBP -> NafNet -> CT_nd"
                fixed_note = "FBP frozen: excluded from trainable params; THOP may not count sparse FBP matrix multiply FLOPs."
            elif method == "ddf":
                model = build_ddf(s, device)
                definition = "full DDF -> CT_pre"
                fixed_note = "FBP and FP frozen/non-trainable: excluded from trainable params; THOP may not count sparse FBP/FP matrix multiply FLOPs."
            else:
                print(f"unknown method: {method}, skip.")
                continue

            x = torch.randn(*input_shape, device=device)
            flops, tool_or_error = profile_flops(model, x)
            row = {
                "method": method,
                "sparse_factor": s,
                "trainable_params": trainable_params(model),
                "total_params": total_params(model),
                "flops": "" if flops is None else flops,
                "input_shape": str(input_shape),
                "flops_tool": tool_or_error,
                "definition": definition,
                "fixed_module_note": fixed_note,
            }
            rows.append(row)
            print(
                f"{method}, S={s}, trainable_params={row['trainable_params']}, "
                f"total_params={row['total_params']}, flops={row['flops']}, "
                f"input_shape={row['input_shape']}, tool={row['flops_tool']}"
            )

            del model, x
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "sparse_factor",
        "trainable_params",
        "total_params",
        "flops",
        "input_shape",
        "flops_tool",
        "definition",
        "fixed_module_note",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved CSV: {output_csv}")


if __name__ == "__main__":
    main()
