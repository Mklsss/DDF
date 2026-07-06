#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SPARSE_FACTOR="${SPARSE_FACTOR:-12}"
BATCH_SIZE="${BATCH_SIZE:-3}"
EPOCHS="${EPOCHS:-500}"
DEVICE="${DEVICE:-cuda:0}"
CONFIG="${CONFIG:-configs/mixed_default.json}"
ORIGINAL_CHECKPOINT="${ORIGINAL_CHECKPOINT:-/autodl-fs/data/FH/code/weights/DDF_c12_best.pth}"
CHECKPOINT="${CHECKPOINT:-checkpoints/fair_protocol/mixed_S${SPARSE_FACTOR}.pth}"
WARMSTART_CHECKPOINT="${WARMSTART_CHECKPOINT:-checkpoints/warmstart/mixed_REDCNN_S${SPARSE_FACTOR}.pth}"

SWANLAB_ARGS=()
if [[ "${SWANLAB:-0}" == "1" ]]; then
  SWANLAB_ARGS=(--swanlab --swanlab_project "${SWANLAB_PROJECT:-universalExp}" --swanlab_mode "${SWANLAB_MODE:-cloud}")
fi

echo "[mixed-fair] config=${CONFIG}"
echo "[mixed-fair] sparse_factor=${SPARSE_FACTOR} batch_size=${BATCH_SIZE} epochs=${EPOCHS} device=${DEVICE}"

if [[ ! -f "${WARMSTART_CHECKPOINT}" ]]; then
  echo "[mixed-fair] warm-start checkpoint not found; training RED-CNN warm-start first"
  python projection_fair_experiment.py \
    --backbone mixed \
    --mode warmstart \
    --config "${CONFIG}" \
    --sparse_factor "${SPARSE_FACTOR}" \
    --original_checkpoint "${ORIGINAL_CHECKPOINT}" \
    --warmstart_checkpoint "${WARMSTART_CHECKPOINT}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}"
else
  echo "[mixed-fair] found warm-start checkpoint: ${WARMSTART_CHECKPOINT}"
fi

START_EPOCH=0
if [[ -f "${CHECKPOINT}" ]]; then
  START_EPOCH="$(python - "${CHECKPOINT}" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu")
print(int(payload.get("epoch", 0)) if isinstance(payload, dict) else 0)
PY
)"
  echo "[mixed-fair] found existing checkpoint at epoch ${START_EPOCH}: ${CHECKPOINT}"
fi

if (( START_EPOCH < EPOCHS )); then
  RESUME_ARGS=()
  if [[ -f "${CHECKPOINT}" ]]; then
    RESUME_ARGS=(--resume_checkpoint "${CHECKPOINT}")
  fi

  python projection_fair_experiment.py \
    --backbone mixed \
    --mode train \
    --config "${CONFIG}" \
    --sparse_factor "${SPARSE_FACTOR}" \
    --original_checkpoint "${ORIGINAL_CHECKPOINT}" \
    --checkpoint "${CHECKPOINT}" \
    --warmstart_checkpoint "${WARMSTART_CHECKPOINT}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --device "${DEVICE}" \
    "${RESUME_ARGS[@]}" \
    "${SWANLAB_ARGS[@]}"
else
  echo "[mixed-fair] checkpoint already reached requested epochs; skipping training"
fi

python projection_fair_experiment.py \
  --backbone mixed \
  --mode test \
  --config "${CONFIG}" \
  --sparse_factor "${SPARSE_FACTOR}" \
  --original_checkpoint "${ORIGINAL_CHECKPOINT}" \
  --checkpoint "${CHECKPOINT}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}"
