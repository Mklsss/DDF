#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?usage: bash run_reviewer1_scheduler_task.sh step|cosine|plateau}"
case "$TASK" in
  step|cosine|plateau) ;;
  *) echo "unknown scheduler: $TASK" >&2; exit 2 ;;
esac

ROOT="/root/autodl-fs/PAPER2"
CODE="$ROOT/FH/code"
OUT="$CODE/results/scheduler_r1_S12"
WEIGHTS="$CODE/weights/scheduler_r1_S12"
mkdir -p "$OUT" "$WEIGHTS"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true
export NO_PROXY='*'
export no_proxy='*'
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128'

cd "$CODE"
/root/miniconda3/bin/python -u experiments/train_ddf_clean.py \
  --sparse_factor 12 \
  --scheduler "$TASK" \
  --epochs 100 \
  --batch_size 3 \
  --val_batch_size 2 \
  --train_count 1600 \
  --val_count 200 \
  --lr 1e-4 \
  --step_size 10 \
  --gamma 0.33 \
  --cosine_t_max 100 \
  --min_lr 1e-8 \
  --plateau_factor 0.33 \
  --plateau_patience 5 \
  --plateau_threshold 0.01 \
  --seed 2026 \
  --initial_weights "$WEIGHTS/shared_initial_seed2026.pth" \
  --output_best "$WEIGHTS/${TASK}_best.pth" \
  --output_ckpt "$WEIGHTS/${TASK}_ckpt.pth" \
  --history_csv "$OUT/${TASK}_history.csv" \
  --metadata_json "$OUT/${TASK}_metadata.json" \
  --swanlab \
  --project DDF-reviewer1-scheduler
