#!/usr/bin/env bash
set -euo pipefail

# Run exactly one of the 21 reviewer-requested ablation trainings in the foreground.
# Usage: bash run_reviewer_ablation_task.sh module_no_sine_S2

if [[ $# -ne 1 ]]; then
  echo "Usage: bash $0 TASK_ID" >&2
  exit 2
fi

TASK_ID="$1"
ROOT="${PAPER2_ROOT:-/root/autodl-fs/PAPER2}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TRAIN_NPZ="${TRAIN_NPZ:-$ROOT/dataset/train_meiaonew.npz}"
TEST_NPZ="${TEST_NPZ:-$ROOT/dataset/test_meiaonew.npz}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-3}"
TRAIN_COUNT="${TRAIN_COUNT:-1600}"
VAL_COUNT="${VAL_COUNT:-200}"
SEED="${SEED:-2026}"
RUN_TAG="${RUN_TAG:-reviewer21_corrected_v2}"
PROJECT="${SWANLAB_PROJECT:-DDF-reviewer-ablation-corrected-v2}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
CLEAR_NETWORK_PROXY="${CLEAR_NETWORK_PROXY:-1}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/FH/code:$ROOT/universalExp${PYTHONPATH:+:$PYTHONPATH}"

# AutoDL's acceleration helper can leave a dead localhost proxy behind after
# the proxy process stops. SwanLab uses Requests, which recognizes every
# environment variable whose name ends in _proxy, not only the common six.
# Bypass all such proxies by default so cloud logging connects directly.
if [[ "$CLEAR_NETWORK_PROXY" == "1" ]]; then
  while IFS= read -r env_name; do
    if [[ "${env_name,,}" == *_proxy ]]; then
      unset "$env_name"
    fi
  done < <(compgen -e)
  export NO_PROXY="*"
  export no_proxy="*"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$TRAIN_NPZ" || ! -f "$TEST_NPZ" ]]; then
  echo "Training or test NPZ is missing." >&2
  exit 1
fi

effective_proxies="$($PYTHON_BIN -c 'import requests; print(requests.utils.get_environ_proxies("https://api.swanlab.cn"))')"
echo "[setup] SwanLab effective proxies: $effective_proxies" >&2
"$PYTHON_BIN" "$ROOT/FH/code/experiments/verify_sparse_angle_order.py"

run_module() {
  local variant="$1"
  local factor="$2"
  local checkpoint="$ROOT/FH/code/weights/$RUN_TAG/$TASK_ID.pth.tar"
  local result="$ROOT/FH/code/results/$RUN_TAG/$TASK_ID.json"
  local extra=()
  [[ "$RESUME" == "1" ]] && extra+=(--resume)
  [[ "$DRY_RUN" == "1" ]] && extra+=(--dry_run)
  cd "$ROOT/FH/code"
  exec "$PYTHON_BIN" -u experiments/train_module_ablation_multisparse.py \
    --variant "$variant" \
    --sparse_factor "$factor" \
    --train_npz "$TRAIN_NPZ" \
    --test_npz "$TEST_NPZ" \
    --checkpoint "$checkpoint" \
    --result_json "$result" \
    --epochs "$EPOCHS" \
    --learning_rate 1e-4 \
    --step_size 10 \
    --gamma 0.33 \
    --batch_size "$BATCH_SIZE" \
    --train_count "$TRAIN_COUNT" \
    --val_count "$VAL_COUNT" \
    --seed "$SEED" \
    --swanlab \
    --swanlab_project "$PROJECT" \
    --swanlab_run_name "${TASK_ID}_corrected_v2" \
    "${extra[@]}"
}

run_backbone() {
  local backbone="$1"
  local factor="$2"
  local config="$ROOT/universalExp/configs/${backbone}_default.json"
  local original="$ROOT/FH/code/weights/DDF_c${factor}_best.pth"
  local checkpoint="$ROOT/universalExp/checkpoints/$RUN_TAG/$TASK_ID.pth.tar"
  local result="$ROOT/universalExp/results/$RUN_TAG/$TASK_ID.json"
  local learning_rate="1e-4"
  local extra=()
  [[ "$backbone" == "irestor" ]] && learning_rate="1e-5"
  if [[ "$backbone" == "icnn" || "$backbone" == "mixed" ]]; then
    extra+=(
      --auto_warmstart
      --warmstart_checkpoint "$ROOT/universalExp/checkpoints/$RUN_TAG/warmstart/${backbone}_S${factor}.pth.tar"
    )
  fi
  if [[ "$RESUME" == "1" ]]; then
    extra+=(--resume_checkpoint "$checkpoint")
  fi
  [[ "$DRY_RUN" == "1" ]] && extra+=(--dry_run)
  cd "$ROOT/universalExp"
  exec "$PYTHON_BIN" -u projection_fair_experiment.py \
    --backbone "$backbone" \
    --mode train \
    --config "$config" \
    --sparse_factor "$factor" \
    --original_checkpoint "$original" \
    --checkpoint "$checkpoint" \
    --result_json "$result" \
    --train_data "$TRAIN_NPZ" \
    --test_data "$TEST_NPZ" \
    --train_count "$TRAIN_COUNT" \
    --val_count "$VAL_COUNT" \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --learning_rate "$learning_rate" \
    --swanlab \
    --swanlab_project "$PROJECT" \
    --swanlab_run_name "${TASK_ID}_corrected_v2" \
    "${extra[@]}"
}

if [[ "$TASK_ID" =~ ^module_(no_sine|no_ct)_S(2|4|8)$ ]]; then
  run_module "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
elif [[ "$TASK_ID" =~ ^backbone_(pcnn|pswin|icnn|irestor|mixed)_S(2|4|8)$ ]]; then
  run_backbone "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
else
  echo "Unknown TASK_ID: $TASK_ID" >&2
  echo "Allowed module tasks: module_{no_sine,no_ct}_S{2,4,8}" >&2
  echo "Allowed backbone tasks: backbone_{pcnn,pswin,icnn,irestor,mixed}_S{2,4,8}" >&2
  exit 2
fi
