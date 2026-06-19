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
| P-Swin | SwinIR-sino | NAFNet | 仅替换投影域为强 restoration backbone | 否 | 待实现 | 待运行 |
| I-CNN | UTR / UTR+Angle | RED-CNN / U-Net | 仅替换图像域为 CNN | 是 | 待实现 | 待运行 |
| I-Restor | UTR / UTR+Angle | Restormer / SwinIR | 仅替换图像域为现代 restoration backbone | 否 | 待实现 | 待运行 |
| Both-CNN | ResUNet-sino | RED-CNN / U-Net | 两个域均替换为 CNN | 是 | 待实现 | 待运行 |
| Mixed | SwinIR-sino | RED-CNN / U-Net 或 Restormer | 两个域均替换，比较跨类型组合 | 否 | 待实现 | 待运行 |

第一阶段按以下顺序完成：**Default → P-CNN → I-CNN → Both-CNN**。该 2×2 CNN 设计能直接归因于投影域替换、图像域替换及两者共同替换的贡献。第二阶段再实现并比较 P-Swin、I-Restor 和 Mixed。

## 3. 当前实现与验证状态

当前 `universalExp` 仅实现了 P-CNN 变体，且正式目标为完整 DDF 路径：

- `DDFPCNN.sin`：`ResUNetSino`，以残差 U-Net 修复插值后的 sinogram，替换原 DDF 的投影域 backbone。
- `DDFPCNN.ct`：原 DDF 的 NAFNet，配置为 `width=16`、`middle_blk_num=1`、四级 encoder/decoder。
- 保留：`FbpLayer`、`ForwardProjectionLayer`、`GMLPSineFusion`、`CrossGatingBlock`。
- 已通过 smoke：DDF、S=12、batch size 3；输入 `(3, 360, 357)`，输出 `(3, 1, 256, 256)`，loss 与 P-CNN 梯度均为有限值。详见 `results/smoke_ddf_S12_B3.json`。
- 尚未完成任何正式 500 epoch 训练；smoke 中的 loss 仅用于正确性验证，不是模型性能结论。

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

## 5. 正式结果登记

仅在完整训练成功、最佳 checkpoint 可加载且测试完成后填写下表。每一行必须对应完整 DDF、S=12、batch size 3。

| 组别 | 实现版本 / checkpoint | 训练轮数 | PSNR | SSIM | SwanLab run | 状态 |
|---|---|---:|---:|---:|---|---|
| Default | — | — | — | — | — | 待实现 / 待运行 |
| P-CNN | `checkpoints/ddf/P-CNN_NAFNet_S12.pth` | — | — | — | — | smoke 通过，待训练 |
| P-Swin | — | — | — | — | — | 待实现 |
| I-CNN | — | — | — | — | — | 待实现 |
| I-Restor | — | — | — | — | — | 待实现 |
| Both-CNN | — | — | — | — | — | 待实现 |
| Mixed | — | — | — | — | — | 待实现 |

结果文件 `results/summary.csv` 只能追加正式测试结果；训练中断、smoke、不同稀疏系数或不同 batch size 的数值不得填入上表。

## 6. GitHub 版本记录

本目录的代码、配置、实验计划和正式结果记录均通过仓库的 `origin` GitHub 远端保存。重要版本必须提交并推送，确保新窗口、新机器或其他协作者可从仓库恢复上下文。

- **实现里程碑**：新增或修改 backbone、数据流程、训练/评估逻辑、smoke 逻辑后，完成对应 smoke test，再提交并推送代码、配置、本文档和必要的 smoke 证据。
- **正式实验里程碑**：每组 500 epoch 训练完成并测试后，更新本文档的结果表、`results/summary.csv` 和 SwanLab run 链接，再提交并推送。
- **不进入 Git 的运行产物**：checkpoint、训练日志、SwanLab 本地运行文件和中断尝试产物。它们由本机或实验平台保存；Git 仅保存能定位和复现实验的配置、命令、指标与云端链接。
- **上下文恢复**：在新窗口或新环境中，先阅读 `universalExp/EXPERIMENT_PLAN.md`，再查看当前 Git 分支与最新提交；该文档是实验范围、状态和运行协议的唯一事实来源。
