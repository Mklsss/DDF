#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-fs/PAPER2/FH/dudotrans_autodl

# Avoid stale AutoDL proxy variables breaking SwanLab login.
export NO_PROXY='*'
export no_proxy='*'
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

exec /root/miniconda3/bin/python -u train_dudotrans.py \
  --train_npz /root/autodl-fs/PAPER2/dataset/train_meiaonew.npz \
  --sparse_factor 2 \
  --views 180 \
  --epochs 200 \
  --lr 1e-4 \
  --batch_size 1 \
  --num_workers 0 \
  --train_count 1600 \
  --seed 2026 \
  --output_dir ./results_paper_S2 \
  --restart \
  --swanlab \
  --swanlab_project DDF-reproduction \
  --swanlab_experiment DuDoTrans_S2_180views_200ep \
  "$@"
