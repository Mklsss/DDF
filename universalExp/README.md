# Universal DDF backbone experiment

本目录独立于原始实现，当前已实现完整 DDF 中以 P-CNN 或 P-Swin 替换投影域
`sin_angle` backbone，以及以 I-CNN（RED-CNN）或 I-Restor（Restormer）替换图像域
NAFNet 的变体；也包括 Mixed（P-Swin + RED-CNN）双域替换。完整实验矩阵、固定协议、当前验证状态、正式命令和结果登记规则统一维护在
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。
## Fair DDF backbone comparison

`pcnn_experiment.py`, `pswin_experiment.py`, and `icnn_experiment.py` are earlier
small/reinitialized-DDF experiments and must not be compared to the published original DDF checkpoint.
experiments and must not be compared to the published original DDF checkpoint:
they use a different, much smaller image backbone.  Use the fair entry point
below for the projection-domain ablation.  It restores the original S12 DDF
(`width=32`, `[1,1,1,28]` image blocks) from one checkpoint and freezes every
shared component. P-CNN/P-Swin replace only `sin`; I-CNN/I-Restor replace only
`ct`; Both-CNN/Mixed replace both `sin` and `ct`.

```bash
cd /autodl-fs/data/universalExp

# Establish the exact original-DDF reference on the same test set/metric.
python projection_fair_experiment.py --backbone original --mode test \
  --config configs/pcnn_default.json --sparse_factor 12 --batch_size 2

# Controlled projection-only replacement runs.
python projection_fair_experiment.py --backbone pcnn --mode train \
  --config configs/pcnn_default.json --sparse_factor 12 --epochs 100
python projection_fair_experiment.py --backbone pswin --mode train \
  --config configs/pswin_default.json --sparse_factor 12 --epochs 100
python projection_fair_experiment.py --backbone icnn --mode train \
  --config configs/icnn_default.json --sparse_factor 12 --epochs 100
python projection_fair_experiment.py --backbone irestor --mode train \
  --config configs/irestor_default.json --sparse_factor 12 --epochs 100
python projection_fair_experiment.py --backbone bothcnn --mode train \
  --config configs/bothcnn_default.json --sparse_factor 12 --epochs 100
python projection_fair_experiment.py --backbone mixed --mode train \
  --config configs/mixed_default.json --sparse_factor 12 --epochs 100
```

The resulting checkpoints are saved in `checkpoints/fair_single_domain/`. Do not
mix their metrics with the legacy `checkpoints/ddf/` results.
