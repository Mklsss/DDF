import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch


def load_ns(sparse_factor):
    text = Path(f"DDP_run_c{sparse_factor}.py").read_text(encoding="utf-8")
    stop = text.index("model = mymodel().to(device)")
    ns = {"__name__": "__cost_original__"}
    exec(text[:stop], ns)
    return ns


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_params(model):
    return sum(p.numel() for p in model.parameters())


def profile_flops(model, x):
    from thop import profile
    model.eval()
    with torch.no_grad():
        flops, _ = profile(model, inputs=(x,), verbose=False)
    return int(flops)


def build_cascade_original(ns, sparse_factor, device):
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
    for p in model.fbp.parameters():
        p.requires_grad = False
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse_factors", default="2,4,8,12")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_csv", default="results/computational_cost_cascade_original.csv")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    factors = [int(x.strip()) for x in args.sparse_factors.split(",") if x.strip()]
    rows = []

    for s in factors:
        ns = load_ns(s)
        x = torch.randn(args.batch_size, 360, 357, device=device)

        cascade = build_cascade_original(ns, s, device)
        ddf = ns["mymodel"]().to(device)
        for name in ("fbp", "fp"):
            module = getattr(ddf, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = False

        for method, model, note in [
            ("cascade_original", cascade, "FBP frozen; sparse FBP matrix multiply may not be fully counted by THOP."),
            ("ddf", ddf, "FBP/FP frozen; sparse FBP/FP matrix multiply may not be fully counted by THOP."),
        ]:
            rows.append({
                "method": method,
                "sparse_factor": s,
                "trainable_params": trainable_params(model),
                "total_params": total_params(model),
                "flops": profile_flops(model, x),
                "input_shape": str(tuple(x.shape)),
                "flops_tool": "thop.profile",
                "fixed_module_note": note,
            })
            print(rows[-1])

        del cascade, ddf, x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "sparse_factor", "trainable_params", "total_params", "flops", "input_shape", "flops_tool", "fixed_module_note"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved CSV: {out}")


if __name__ == "__main__":
    main()
