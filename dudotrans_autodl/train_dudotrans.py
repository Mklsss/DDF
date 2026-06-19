import argparse

import main as dudotrans_main


def parse_args():
    parser = argparse.ArgumentParser(description="AutoDL launcher for DuDoTrans.")
    parser.add_argument("--train_npz", type=str, default="/root/autodl-fs/dataset/train_meiaonew.npz")
    parser.add_argument("--views", type=int, default=30, help="Sparse view count, e.g. 30/60/90/120.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--restart", action="store_true", help="Start from scratch.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--metric_interval", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[train_dudotrans] views={args.views}")
    print(f"[train_dudotrans] train_npz={args.train_npz}")
    print(f"[train_dudotrans] output_dir={args.output_dir}")

    trainer = dudotrans_main.Trainer(
        learning_rate=args.lr,
        is_restart=args.restart,
        max_epoch=args.epochs,
        is_cuda=(not args.cpu),
        num_view=args.views,
        num_workers=args.num_workers,
        use_amp=args.amp,
        metric_interval=args.metric_interval,
        train_npz=args.train_npz,
        output_dir=args.output_dir,
        resume_ckpt=args.resume_ckpt,
        batch_size=args.batch_size,
    )
    trainer.train()
    print("[train_dudotrans] done")


if __name__ == "__main__":
    main()
