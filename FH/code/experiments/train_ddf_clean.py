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
    ns = {"__name__": "__train_ddf_clean__"}
    exec(text[:stop], ns)
    return ns


def psnr_batch(prediction, target):
    prediction = prediction.detach().cpu()
    target = target.detach().cpu()
    scores = []
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
            original_n = sin_in.shape[0]
            if original_n == 1:
                sin_in = torch.cat([sin_in, sin_in], dim=0)
                label = torch.cat([label, label], dim=0)

            sin_in = sin_in.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = model(sin_in)[:original_n].clamp(0, 1).detach().cpu()
            label_cpu = label[:original_n].detach().cpu()

            psnr_sum += psnr_batch(pred, label_cpu) * original_n
            ssim_sum += float(ssim_module.ssim(pred, label_cpu).item()) * original_n
            num_samples += original_n

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
    parser.add_argument("--output_best", default=None)
    parser.add_argument("--output_ckpt", default=None)
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

    model = ns["mymodel"]().to(device)
    for p in model.fbp.parameters():
        p.requires_grad = False

    output_best = Path(args.output_best or f"weights/DDF_c{args.sparse_factor}_clean_best.pth")
    output_ckpt = Path(args.output_ckpt or f"weights/DDF_c{args.sparse_factor}_clean_ckpt.pth")
    output_best.parent.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    start_epoch = 0
    best_psnr = -float("inf")

    if args.resume and output_ckpt.exists():
        ckpt = torch.load(output_ckpt, map_location=device)
        print(model.load_state_dict(ckpt["model"], strict=False))
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_psnr = float(ckpt.get("best_psnr", best_psnr))
        print(f"resumed from {output_ckpt}, start_epoch={start_epoch}, best_psnr={best_psnr:.6f}")
    elif args.resume and output_best.exists():
        state = torch.load(output_best, map_location=device)
        print(model.load_state_dict(state, strict=False))
        print(f"resumed model weights from {output_best}")

    run = None
    if args.swanlab:
        if swanlab is None:
            print("swanlab is not installed; continue without SwanLab logging.")
        else:
            run = swanlab.init(
                project=args.project,
                experiment_name=f"DDFClean_S{args.sparse_factor}_{args.epochs}ep",
                config={**vars(args), "train_samples": n_train, "val_samples": n_val, "val_source": "last_10_percent_of_train"},
            )

    print(f"train_samples={n_train}, val_samples={n_val}, val_source=train_split")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f"DDFClean S={args.sparse_factor} epoch {epoch}/{args.epochs - 1}", ncols=120)
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

        print(f"DDFClean, S={args.sparse_factor}, epoch={epoch}, train_loss={train_loss:.6f}, val_psnr={val_psnr:.6f}, val_ssim={val_ssim:.6f}")

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
            torch.save(model.state_dict(), output_best)
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_psnr": best_psnr,
            }, output_ckpt)
            print(f"saved best checkpoint: {output_best}")
            print(f"saved training checkpoint: {output_ckpt}")

    if run is not None:
        swanlab.finish()


if __name__ == "__main__":
    main()
