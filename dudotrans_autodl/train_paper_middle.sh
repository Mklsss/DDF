#!/usr/bin/env bash
set -euo pipefail

# Paper mapping: S=2/4/8/12 -> 180/90/45/30 measured views.
# This launcher trains the two middle paper settings sequentially: S=4 and S=8.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TRAIN_NPZ="${TRAIN_NPZ:-/root/autodl-fs/PAPER2/dataset/train_meiaonew.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_paper_exact}"
LOG_DIR="${LOG_DIR:-./logs/paper_exact}"
EPOCHS="${EPOCHS:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TRAIN_COUNT="${TRAIN_COUNT:-1600}"
SEED="${SEED:-2026}"
FORCE="${FORCE:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$TRAIN_NPZ" ]]; then
  echo "Training data not found: $TRAIN_NPZ" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

"$PYTHON_BIN" -c "import torch, odl, SimpleITK, skimage, timm; print('dependency check passed')"

train_setting() {
  local sparse_factor="$1"
  local expected_views="$2"
  local target_dir="$OUTPUT_DIR/models/view_$(printf '%03d' "$expected_views")"
  local log_file="$LOG_DIR/train_S${sparse_factor}_view$(printf '%03d' "$expected_views").log"

  if compgen -G "$target_dir/*.pth.tar" > /dev/null && [[ "$FORCE" != "1" ]]; then
    echo "Refusing to overwrite existing checkpoints in $target_dir" >&2
    echo "Move them elsewhere or rerun with FORCE=1." >&2
    exit 1
  fi

  echo "Starting paper S=$sparse_factor (${expected_views} views)"
  "$PYTHON_BIN" -u train_dudotrans.py \
    --train_npz "$TRAIN_NPZ" \
    --sparse_factor "$sparse_factor" \
    --epochs "$EPOCHS" \
    --lr "$LEARNING_RATE" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --train_count "$TRAIN_COUNT" \
    --metric_interval 200 \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR" \
    --restart \
    2>&1 | tee "$log_file"
}

# Do not run these in parallel on the same GPU.
train_setting 4 90
train_setting 8 45

echo "Both paper-middle DuDoTrans trainings completed."
