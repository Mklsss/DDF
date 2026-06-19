# Universal DDF 双域 Backbone 消融实验记录

## 1. 目标与固定协议

本实验研究投影域 `F_p` 与图像域 `F_i` 的 backbone 对完整 DDF 重建框架的影响。正式比较均使用完整 DDF：保留 FBP、FP 反馈、投影域融合和图像域融合；每一组只替换表中指定的 backbone。

固定协议如下：

- 稀疏系数：`S=12`，即从 360 个视角中每隔 12 个采样一次（30-view）；网络输入仍按原 DDF 流程插值为 `(N, 360, 357)`。
- 训练：`batch_size=3`、500 epochs、Adam、学习率与 scheduler 均采用 `configs/pcnn_default.json`。
- 评估：PSNR、SSIM；测试使用原 DDF 测试集与原指标实现。
- 追踪：SwanLab cloud；每轮记录训练 loss、学习率、PSNR、SSIM。
- 准入条件：每个组别必须先完成其正式设置下的 smoke test（前向、反向、有限 loss、有限 backbone 梯度），才能开始 500 epoch 训练。

> 范围说明：本文档只记录 S=12 的正式消融。任何其他稀疏系数或中断尝试均不属于结果表，也不可用于横向比较。

## 2. 完整实验矩阵

| 组别 | 投影域 `F_p` | 图像域 `F_i` | 目的 | 第一阶段 | 实现状态 | 正式结果 |
|---|---|---|---|---|---|---|
| Default | UTR / UTR+Angle | NAFNet | 原始 DDF 配置 | 是 | 待接入统一实验入口 | 待运行 |
| P-CNN | ResUNet-sino | NAFNet | 仅替换投影域为 CNN | 是 | 已实现：`DDFPCNN` | 待训练 |
| P-Swin | SwinIR-sino | NAFNet | 仅替换投影域为强 restoration backbone | 否 | 已实现，smoke 通过 | 待运行 |
| I-CNN | UTR / UTR+Angle | RED-CNN | 仅替换图像域为 CNN | 是 | 已实现，待 smoke | 待运行 |
| I-Restor | UTR / UTR+Angle | Restormer | 仅替换图像域为现代 restoration backbone | 否 | 已实现：`DDFIRestor` | 待运行 |
| Both-CNN | ResUNet-sino | RED-CNN | 两个域均替换为 CNN | 是 | 已实现，待 smoke | 待运行 |
| Mixed | SwinIR-sino | RED-CNN | 两个域均替换，比较跨类型组合 | 否 | 已实现，待 smoke | 待运行 |

第一阶段按以下顺序完成：**Default → P-CNN → I-CNN → Both-CNN**。该 2×2 CNN 设计能直接归因于投影域替换、图像域替换及两者共同替换的贡献。第二阶段再实现并比较 P-Swin、I-Restor 和 Mixed。

## 3. 当前实现与验证状态

当前已实现 P-CNN、P-Swin、I-CNN、I-Restor 与 Both-CNN 变体，且正式目标均为完整 DDF 路径：

- `DDFPCNN.sin`：`ResUNetSino`，以残差 U-Net 修复插值后的 sinogram，替换原 DDF 的投影域 backbone。
- `DDFPCNN.ct`：原 DDF 的 NAFNet，配置为 `width=16`、`middle_blk_num=1`、四级 encoder/decoder。
- 保留：`FbpLayer`、`ForwardProjectionLayer`、`GMLPSineFusion`、`CrossGatingBlock`。
- 已通过 smoke：DDF、S=12、batch size 3；输入 `(3, 360, 357)`，输出 `(3, 1, 256, 256)`，loss 与 P-CNN 梯度均为有限值。详见 `results/smoke_ddf_S12_B3.json`。
- 尚未完成任何正式 500 epoch 训练；smoke 中的 loss 仅用于正确性验证，不是模型性能结论。

I-Restor 保留原 DDF 的 `sin_angle`（UTR / UTR+Angle）、`FbpLayer`、`ForwardProjectionLayer`、`GMLPSineFusion` 和 `CrossGatingBlock`，仅将图像域 `ct` 从 NAFNet 替换为 Restormer。Restormer 的正式配置在 `configs/irestor_default.json`：单通道输入/输出、`dim=24`、blocks `[4,6,6,8]`。

P-Swin 已实现为独立入口，正式路径同样保持完整 DDF：

- `DDFPSwin.sin`：`SwinIRSino`，将 `(N, 360, 357)` 的插值 sinogram 视为单通道 restoration 图像；仅以 SwinIR 替换原投影域 `sin_angle`。
- SwinIR 配置：`embed_dim=60`、4 个 stage、每 stage 6 个 Swin block、6 heads、`window_size=8`；使用 activation checkpoint 以控制训练显存。
- 宽度 `357` 不是 8 的倍数，SwinIR 仅在其内部反射填充到窗口对齐尺寸并在输出裁剪回 `(N, 360, 357)`，因此 DDF 的 FBP/FP/融合接口不变。
- `FbpLayer`、`ForwardProjectionLayer`、`GMLPSineFusion`、`CrossGatingBlock` 和图像域 NAFNet 均未改动。
- 已通过 smoke：DDF、S=12、batch size 3；输入 `(3, 360, 357)`，输出 `(3, 1, 256, 256)`，loss `4.004617` 有限，P-Swin backbone 梯度有限。详见 `results/smoke_pswin_ddf_S12_B3.json`；该 loss 仅用于正确性验证，不构成性能结论。

Mixed 固定为可复现的跨类型组合：`DDFMixed.sin` 使用 `SwinIRSino`，`DDFMixed.ct`
使用原项目同构的 96-channel `RED-CNN`。完整 DDF 的 `FbpLayer`、
`ForwardProjectionLayer`、`GMLPSineFusion` 和 `CrossGatingBlock` 均保留不变。
正式配置为 `configs/mixed_default.json`，最佳 checkpoint 为
`checkpoints/ddf/Mixed_P-Swin_REDCNN_S12.pth`。

Both-CNN 以 `DDFBothCNN` 实现：`sin` 是修复 `(N, 360, 357)` sinogram 的 `ResUNetSino`，`ct` 是与 I-CNN 完全相同的 96-channel `REDCNN`，处理 `(N, 1, 256, 256)` 图像。两个图像域调用共享同一个 `ct` 模块及参数；`FbpLayer`、`ForwardProjectionLayer`、`GMLPSineFusion` 和 `CrossGatingBlock` 均未改变，仍使用 S=12、batch size 3、500 epochs 的固定协议。

## 4. P-CNN 正式运行

先运行（或复核）S=12、batch size 3 的 smoke：

```bash
cd /autodl-fs/data/universalExp

python -u smoke_test.py \
  --architecture ddf \
  --sparse_factor 12 \
  --batch_size 3 \
  --device cuda:0 \
  --output results/smoke_ddf_S12_B3.json
```

随后训练 DDF-P-CNN：

```bash
python -u pcnn_experiment.py \
  --architecture ddf \
  --mode train \
  --sparse_factor 12 \
  --batch_size 3 \
  --device cuda:0 \
  --swanlab \
  --swanlab_project universalExp \
  --swanlab_mode cloud
```

训练会把最佳权重写入 `checkpoints/ddf/P-CNN_NAFNet_S12.pth`，并在训练结束后评估；也可在已有 checkpoint 时单独测试：

```bash
python -u pcnn_experiment.py \
  --architecture ddf \
  --mode test \
  --sparse_factor 12 \
  --batch_size 3 \
  --device cuda:0 \
  --swanlab \
  --swanlab_project universalExp \
  --swanlab_mode cloud
```

## 5. P-Swin 正式运行

先在完整 DDF、S=12、batch size 3 的正式配置下运行 smoke（它会做一次前向、反向、有限 loss 和有限 P-Swin 梯度检查）：

```bash
cd /autodl-fs/data/universalExp

python -u smoke_pswin.py \
  --sparse_factor 12 \
  --batch_size 3 \
  --device cuda:0 \
  --output results/smoke_pswin_ddf_S12_B3.json
```

Smoke 成功后，再开始正式 500 epoch 训练：

```bash
python -u pswin_experiment.py \
  --mode train \
  --sparse_factor 12 \
  --batch_size 3 \
  --device cuda:0 \
  --swanlab \
  --swanlab_project universalExp \
  --swanlab_mode cloud
```

最佳权重写入 `checkpoints/ddf/P-Swin_NAFNet_S12.pth`。已有 checkpoint 时可单独测试：

```bash
python -u pswin_experiment.py \
  --mode test \
  --sparse_factor 12 \
  --batch_size 3 \
  --device cuda:0 \
  --swanlab \
  --swanlab_project universalExp \
  --swanlab_mode cloud
```

## 6. I-CNN 正式运行

I-CNN 固定使用原项目同构的 96-channel RED-CNN；保留原始 `sin_angle`
（UTR/UTR+Angle）、FBP、FP 反馈、`GMLPSineFusion` 和 `CrossGatingBlock`，仅将
图像域 `F_i` 从 NAFNet 替换为 RED-CNN。

先运行 S=12、batch size 3 的 smoke：

```bash
cd /autodl-fs/data/universalExp

python -u smoke_test_icnn.py \
  --sparse_factor 12 \
  --batch_size 3 \
  --device cuda:0 \
  --output results/smoke_icnn_ddf_S12_B3.json
```

Smoke 成功后，开始正式 500 epoch 训练（SwanLab 和 tqdm 会直接显示在当前终端）：

```bash
python -u icnn_experiment.py \
  --mode train \
  --sparse_factor 12 \
  --batch_size 3 \
  --device cuda:0 \
  --swanlab \
  --swanlab_project universalExp \
  --swanlab_mode cloud
```

最佳权重为 `checkpoints/ddf/I-CNN_REDCNN_S12.pth`；训练结束后自动评估并追加正式
结果。已有 checkpoint 时可单独测试：

```bash
python -u icnn_experiment.py --mode test --sparse_factor 12 --batch_size 3 --device cuda:0
```

## 7. I-Restor 正式运行

先以正式的 S=12、batch size 3 配置运行 smoke；它验证完整 DDF 前向/反向、有限 loss 和有限 Restormer 梯度：

```bash
cd /autodl-fs/data/universalExp

python -u irestor_smoke_test.py \
  --sparse_factor 12 \
  --batch_size 3 \
  --device cuda:0 \
  --output results/smoke_ddf_irestor_S12_B3.json
```

smoke 通过后开始 500 epoch 的正式 DDF I-Restor 训练：

```bash
python -u irestor_experiment.py \
  --mode train \
  --sparse_factor 12 \
  --batch_size 3 \
  --device cuda:0 \
  --swanlab \
  --swanlab_project universalExp \
  --swanlab_mode cloud
```

最佳权重为 `checkpoints/ddf/I-Restor_Restormer_S12.pth`。已有 checkpoint 时单独评估：

```bash
python -u irestor_experiment.py \
  --mode test \
  --sparse_factor 12 \
  --batch_size 3 \
  --device cuda:0 \
  --swanlab \
  --swanlab_project universalExp \
  --swanlab_mode cloud
```

## 8. Both-CNN 正式运行

先在完整 DDF、S=12、batch size 3 的正式配置下运行 smoke。它只执行一个 batch 的前向和反向，并检查 loss 以及两个 CNN backbone 的梯度是否有限：

```bash
cd /autodl-fs/data/universalExp

python -u smoke_bothcnn.py \
  --sparse_factor 12 \
  --batch_size 3 \
  --device cuda:0 \
  --output results/smoke_bothcnn_ddf_S12_B3.json
```

Smoke 成功后，开始正式 500 epoch 训练：

```bash
python -u bothcnn_experiment.py \
  --mode train \
  --sparse_factor 12 \
  --batch_size 3 \
  --device cuda:0 \
  --swanlab \
  --swanlab_project universalExp \
  --swanlab_mode cloud
```

最佳权重写入 `checkpoints/ddf/Both-CNN_REDCNN_S12.pth`。已有 checkpoint 时单独评估：

```bash
python -u bothcnn_experiment.py \
  --mode test \
  --sparse_factor 12 \
  --batch_size 3 \
  --device cuda:0 \
  --swanlab \
  --swanlab_project universalExp \
  --swanlab_mode cloud
```

## 9. Mixed 正式运行

Mixed 固定使用 P-Swin（SwinIR-sino）和 RED-CNN。先以正式的 S=12、batch size 3
配置运行 smoke；它会检查完整 DDF 的前向/反向、有限 loss，以及两个替换 backbone
的有限梯度：

```bash
cd /autodl-fs/data/universalExp

python -u smoke_mixed.py \
  --sparse_factor 12 \
  --batch_size 3 \
  --device cuda:0 \
  --output results/smoke_mixed_ddf_S12_B3.json
```

Smoke 成功后，开始正式 500 epoch 训练（SwanLab 和 tqdm 会直接显示在当前终端）：

```bash
python -u mixed_experiment.py \
  --mode train \
  --sparse_factor 12 \
  --batch_size 3 \
  --device cuda:0 \
  --swanlab \
  --swanlab_project universalExp \
  --swanlab_mode cloud
```

最佳权重为 `checkpoints/ddf/Mixed_P-Swin_REDCNN_S12.pth`。已有 checkpoint 时单独评估：

```bash
python -u mixed_experiment.py \
  --mode test \
  --sparse_factor 12 \
  --batch_size 3 \
  --device cuda:0 \
  --swanlab \
  --swanlab_project universalExp \
  --swanlab_mode cloud
```

## 10. 正式结果登记

仅在完整训练成功、最佳 checkpoint 可加载且测试完成后填写下表。每一行必须对应完整 DDF、S=12、batch size 3。

| 组别 | 实现版本 / checkpoint | 训练轮数 | PSNR | SSIM | SwanLab run | 状态 |
|---|---|---:|---:|---:|---|---|
| Default | — | — | — | — | — | 待实现 / 待运行 |
| P-CNN | `checkpoints/ddf/P-CNN_NAFNet_S12.pth` | — | — | — | — | smoke 通过，待训练 |
| P-Swin | `checkpoints/ddf/P-Swin_NAFNet_S12.pth` | — | — | — | — | smoke 通过，待训练 |
| I-CNN | `checkpoints/ddf/I-CNN_REDCNN_S12.pth` | — | — | — | — | 已实现，待 smoke / 训练 |
| I-Restor | `checkpoints/ddf/I-Restor_Restormer_S12.pth` | — | — | — | — | 已实现，待 smoke / 训练 |
| Both-CNN | `checkpoints/ddf/Both-CNN_REDCNN_S12.pth` | — | — | — | — | 已实现，待 smoke / 训练 |
| Mixed | `checkpoints/ddf/Mixed_P-Swin_REDCNN_S12.pth` | — | — | — | — | 已实现，待 smoke / 训练 |

结果文件 `results/summary.csv` 只能追加正式测试结果；训练中断、smoke、不同稀疏系数或不同 batch size 的数值不得填入上表。

## 11. GitHub 版本记录

本目录的代码、配置、实验计划和正式结果记录均通过仓库的 `origin` GitHub 远端保存。重要版本必须提交并推送，确保新窗口、新机器或其他协作者可从仓库恢复上下文。

- **实现里程碑**：新增或修改 backbone、数据流程、训练/评估逻辑、smoke 逻辑后，完成对应 smoke test，再提交并推送代码、配置、本文档和必要的 smoke 证据。
- **正式实验里程碑**：每组 500 epoch 训练完成并测试后，更新本文档的结果表、`results/summary.csv` 和 SwanLab run 链接，再提交并推送。
- **不进入 Git 的运行产物**：checkpoint、训练日志、SwanLab 本地运行文件和中断尝试产物。它们由本机或实验平台保存；Git 仅保存能定位和复现实验的配置、命令、指标与云端链接。
- **上下文恢复**：在新窗口或新环境中，先阅读 `universalExp/EXPERIMENT_PLAN.md`，再查看当前 Git 分支与最新提交；该文档是实验范围、状态和运行协议的唯一事实来源。
