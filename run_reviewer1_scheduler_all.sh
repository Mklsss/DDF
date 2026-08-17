#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-fs/PAPER2"
OUT="$ROOT/FH/code/results/scheduler_r1_S12"
mkdir -p "$OUT/logs"

for scheduler in step cosine plateau; do
  echo "[$(date --iso-8601=seconds)] starting $scheduler"
  CUDA_VISIBLE_DEVICES=0 bash "$ROOT/run_reviewer1_scheduler_task.sh" "$scheduler" \
    2>&1 | tee "$OUT/logs/${scheduler}.log"
  echo "[$(date --iso-8601=seconds)] completed $scheduler"
done

echo "[$(date --iso-8601=seconds)] all scheduler experiments completed"
