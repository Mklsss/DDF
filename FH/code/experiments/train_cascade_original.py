import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import swanlab
except ImportError:
    swanlab = None


def load_ddf_namespace(sparse_factor):
    text = Path(f"DDP_run_c{sparse_factor}.py").read_text(encoding="utf-8")
    stop = text.index("model = mymodel().to(device)")
    ns = {"__name__": "__cascade_original__"}
    exec(text[:stop], ns)
    return ns


def psnr_batch(prediction, target):
    scores = []
    prediction = prediction.detach().cpu()
    target = target.detach().cpu()
    for i in range(prediction.shape[0]):
        img1 = prediction[i]
        img2 = target[i]
        data_range = torch.max(img2) - torch.min(img2)
        mse = torch.mean((img1 - img2) ** 2)
        scores.append(float(10.0 * torch.log10((data_range ** 2) / mse)))
    return float(np.mean(scores))


def evaluate(model, loader, device, ssim_module):
    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    num_samples = 0
    with torch.no_grad():
        for sin_in, label in loader:
            sin_in = sin_in.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = model(sin_in).clamp(0, 1).detach().cpu()
            label_cpu = label.detach().cpu()
            n = pred.shape[0]
            psnr_sum += psnr_batch(pred, label_cpu) * n
            ssim_sum += float(ssim_module.ssim(pred, label_cpu).item()) * n
            num_samples += n
    return psnr_sum / num_samples, ssim_sum / num_samples


def subset_dataset(dataset, indices):
    return torch.utils.data.Subset(dataset, indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse_factor", type=int, required=True, choices=[2, 4, 8, 12])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.33)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--project", default="DDF-reproduction")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ns = load_ddf_namespace(args.sparse_factor)
    ssim_module = __import__("ssim")

    full_train_data = ns["load_data"]("./data/train_meiaonew.npz")
    n_total = len(full_train_data)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_total))

    train_loader = torch.utils.data.DataLoader(
        subset_dataset(full_train_data, train_indices),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=full_train_data.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        subset_dataset(full_train_data, val_indices),
        batch_size=args.val_batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=full_train_data.collate_fn,
    )

    class CascadeOriginal(nn.Module):
        def __init__(self):
            super().__init__()
            self.sin = ns["sin_angle"](num_sensor=357, angle=int(360 / args.sparse_factor), num_heads=1)
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

    output = Path(args.output or f"weights/cascade_original_S{args.sparse_factor}.pth")
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and output.exists():
        state = torch.load(output, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        print(model.load_state_dict(state, strict=False))
        print(f"resumed model weights from: {output}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    run = None
    if args.swanlab:
        if swanlab is None:
            print("swanlab is not installed; continue without SwanLab logging.")
        else:
            run = swanlab.init(
                project=args.project,
                experiment_name=f"CascadeOriginal_S{args.sparse_factor}_{args.epochs}ep",
                config={**vars(args), "train_samples": n_train, "val_samples": n_val, "val_source": "last_10_percent_of_train"},
            )

    best_psnr = -float("inf")
    print(f"train_samples={n_train}, val_samples={n_val}, val_source=train_split")

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f"CascadeOriginal S={args.sparse_factor} epoch {epoch}/{args.epochs - 1}", ncols=120)
        for step, (sin_in, label) in enumerate(pbar):
            sin_in = sin_in.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = model(sin_in)
            loss = nn.MSELoss()(pred, label)

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss: {loss}")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_sum += float(loss.detach().item())
            pbar.set_postfix(loss=f"{loss_sum / (step + 1):.6f}")

        scheduler.step()
        val_psnr, val_ssim = evaluate(model, val_loader, device, ssim_module)
        train_loss = loss_sum / max(len(train_loader), 1)
        print(f"CascadeOriginal, S={args.sparse_factor}, epoch={epoch}, train_loss={train_loss:.6f}, val_psnr={val_psnr:.6f}, val_ssim={val_ssim:.6f}")

        if run is not None:
            swanlab.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "best_psnr": max(best_psnr, val_psnr),
                "lr": scheduler.get_last_lr()[0],
            })

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), output)
            print(f"saved best checkpoint: {output}")

    if run is not None:
        swanlab.finish()


if __name__ == "__main__":
    main()
