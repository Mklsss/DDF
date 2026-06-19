import argparse
import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from networks import RED_CNN
from npz_loader import get_npz_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Train original RED-CNN on NPZ data.")
    parser.add_argument("--train_npz", type=str, required=True)
    parser.add_argument("--views", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patch_n", type=int, default=10)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./save_npz")
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--input_key", type=str, default="")
    parser.add_argument("--no_generate_fbp", action="store_true")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--save_epochs", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def latest_checkpoint(model_dir):
    files = sorted(glob(os.path.join(model_dir, "REDCNN_epoch_*.ckpt")))
    return files[-1] if files else ""


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    view_tag = f"view_{args.views:03d}"
    model_dir = os.path.join(args.output_dir, view_tag)
    os.makedirs(model_dir, exist_ok=True)

    print(f"[train_redcnn_npz] views={args.views}")
    print(f"[train_redcnn_npz] train_npz={args.train_npz}")
    print(f"[train_redcnn_npz] output_dir={args.output_dir}")
    print(f"[train_redcnn_npz] device={device}")

    loader = get_npz_loader(
        npz_path=args.train_npz,
        views=args.views,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        input_key=args.input_key,
        generate_fbp=(not args.no_generate_fbp),
        patch_n=args.patch_n,
        patch_size=args.patch_size,
        shuffle=True,
    )

    model = RED_CNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if not args.restart:
        ckpt = args.resume_ckpt or latest_checkpoint(model_dir)
        if ckpt:
            print(f"[train_redcnn_npz] loading checkpoint: {ckpt}")
            state = torch.load(ckpt, map_location=device)
            if isinstance(state, dict) and "model" in state:
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                start_epoch = int(state.get("epoch", -1)) + 1
            else:
                model.load_state_dict(state)

    losses = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs - 1}", ncols=110)

        for x, y in pbar:
            x = x.float().to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)

            if args.patch_size > 0:
                x = x.view(-1, 1, args.patch_size, args.patch_size)
                y = y.view(-1, 1, args.patch_size, args.patch_size)
            else:
                x = x.unsqueeze(1)
                y = y.unsqueeze(1)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            losses.append(loss_value)
            epoch_loss += loss_value
            pbar.set_postfix({"loss": f"{loss_value:.6f}"})

        avg_loss = epoch_loss / max(1, len(loader))
        print(f"[train_redcnn_npz] epoch={epoch} avg_loss={avg_loss:.6f}")

        if epoch % args.save_epochs == 0 or epoch == args.epochs - 1:
            save_path = os.path.join(model_dir, f"REDCNN_epoch_{epoch:03d}.ckpt")
            torch.save(
                {
                    "epoch": epoch,
                    "views": args.views,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "avg_loss": avg_loss,
                },
                save_path,
            )
            np.save(os.path.join(model_dir, "loss.npy"), np.array(losses, dtype=np.float32))
            print(f"[train_redcnn_npz] saved: {save_path}")


if __name__ == "__main__":
    main()
