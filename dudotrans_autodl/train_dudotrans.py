import argparse
import random

import numpy as np
import torch

import main as dudotrans_main


PAPER_VIEWS = {2: 180, 4: 90, 8: 45, 12: 30}


def parse_args():
    parser = argparse.ArgumentParser(description="AutoDL launcher for DuDoTrans.")
    parser.add_argument("--train_npz", type=str, default="/root/autodl-fs/dataset/train_meiaonew.npz")
    parser.add_argument(
        "--sparse_factor",
        type=int,
        choices=sorted(PAPER_VIEWS),
        default=None,
        help="Paper sparse factor. S=2/4/8/12 maps to 180/90/45/30 views.",
    )
    parser.add_argument(
        "--views",
        type=int,
        default=None,
        help="Explicit view count. When --sparse_factor is set, this must match 360/S.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--train_count",
        type=int,
        default=1600,
        help="Use the first N slices for optimization; reserve the remainder for validation.",
    )
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--restart", action="store_true", help="Start from scratch.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--metric_interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry_run", action="store_true", help="Validate arguments without constructing the model.")
    parser.add_argument("--init_only", action="store_true", help="Construct data/model, then exit before training.")
    return parser.parse_args()


def resolve_views(sparse_factor, views):
    if sparse_factor is None:
        return 30 if views is None else views
    expected = PAPER_VIEWS[sparse_factor]
    if views is not None and views != expected:
        raise ValueError(
            f"S={sparse_factor} requires {expected} views according to the paper, "
            f"but --views={views} was supplied."
        )
    return expected


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    args.views = resolve_views(args.sparse_factor, args.views)
    set_seed(args.seed)
    print(f"[train_dudotrans] sparse_factor={args.sparse_factor}")
    print(f"[train_dudotrans] views={args.views}")
    print(f"[train_dudotrans] train_npz={args.train_npz}")
    print(f"[train_dudotrans] output_dir={args.output_dir}")
    print(f"[train_dudotrans] train_count={args.train_count}")
    print(f"[train_dudotrans] seed={args.seed}")
    if args.dry_run:
        print("[train_dudotrans] dry-run validation passed")
        return

    trainer = dudotrans_main.Trainer(
        learning_rate=args.lr,
        is_restart=args.restart,
        max_epoch=args.epochs,
        is_cuda=(not args.cpu),
        num_view=args.views,
        num_workers=args.num_workers,
        train_count=args.train_count,
        use_amp=args.amp,
        metric_interval=args.metric_interval,
        train_npz=args.train_npz,
        output_dir=args.output_dir,
        resume_ckpt=args.resume_ckpt,
        batch_size=args.batch_size,
    )
    if args.init_only:
        print("[train_dudotrans] initialization check passed")
        return
    trainer.train()
    print("[train_dudotrans] done")


if __name__ == "__main__":
    main()
