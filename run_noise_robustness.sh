#!/usr/bin/env bash
set -euo pipefail

# Inference-only reviewer experiment: clean + Poisson I0={1e5,1e4}, S=12.
# The existing pretrained checkpoints are used; this script does not train.

ROOT="${PAPER2_ROOT:-/root/autodl-fs/PAPER2}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TEST_NPZ="${TEST_NPZ:-$ROOT/dataset/test_meiaonew.npz}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
PHOTON_COUNTS="${PHOTON_COUNTS:-1e5,1e4}"
ELECTRONIC_SIGMAS="${ELECTRONIC_SIGMAS:-0}"
OUTPUT_CSV="${OUTPUT_CSV:-$ROOT/FH/code/results/noise_robustness/noise_robustness_S12.csv}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$TEST_NPZ" ]]; then
  echo "Test dataset not found: $TEST_NPZ" >&2
  exit 1
fi

require_any() {
  local label="$1"
  shift
  local candidate
  for candidate in "$@"; do
    [[ -f "$candidate" ]] && return 0
  done
  echo "$label checkpoint not found. Checked: $*" >&2
  exit 1
}

require_any "DDF" \
  "$ROOT/FH/code/weights/DDF_c12_best.pth" \
  "$ROOT/FH/code/weights/DDF_c12_ckpt.pth"
require_any "Cascade" \
  "$ROOT/FH/code/weights/cascade_original_S12.pth" \
  "$ROOT/FH/code/weights/cascade_S12.pth" \
  "$ROOT/FH/code/weights/Cascade_12.pth"

args=(
  --test_data "$TEST_NPZ"
  --sparse_factor 12
  --photon_counts "$PHOTON_COUNTS"
  --electronic_sigmas "$ELECTRONIC_SIGMAS"
  --methods sparse_fbp,cascade,ddf
  --batch_size "$BATCH_SIZE"
  --device "$DEVICE"
  --seed 2026
  --output_csv "$OUTPUT_CSV"
)
if [[ -n "$MAX_SAMPLES" ]]; then
  args+=(--max_samples "$MAX_SAMPLES")
fi

cd "$ROOT/FH/code"
exec "$PYTHON_BIN" -u experiments/run_noise_robustness.py "${args[@]}"
