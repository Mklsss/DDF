# DuDoTrans baseline

用于 sparse-view CT reconstruction 的 DuDoTrans baseline。已改成 AutoDL 可直接训练版本，不依赖外层 `config.py`。

## 数据

训练/测试数据为 `.npz`，必须包含：

```text
ct_label
```

支持形状：

```text
(N, H, W)
(N, H, W, 1)
(N, 1, H, W)
```

示例路径：

```text
/root/autodl-fs/dataset/train_meiaonew.npz
/root/autodl-fs/dataset/test_meiaonew.npz
```

## 安装

```bash
cd /root/autodl-fs/dudotrans_autodl
pip install -r requirements.txt

conda install -y --override-channels \
  -c astra-toolbox \
  -c nvidia \
  -c defaults \
  astra-toolbox
```

注意：使用 AutoDL 自带 PyTorch，不要用 `requirements.txt` 重装 torch。

## 训练

30 views：

```bash
python train_dudotrans.py \
  --train_npz /root/autodl-fs/dataset/train_meiaonew.npz \
  --views 30 \
  --epochs 100 \
  --restart \
  --output_dir ./results
```

其他视角只改：

```bash
--views 60
--views 90
--views 120
```

结果保存到：

```text
results/models/view_030/
results/models/view_060/
results/models/view_090/
results/models/view_120/
```

## 继续训练

不要加 `--restart`：

```bash
python train_dudotrans.py \
  --train_npz /root/autodl-fs/dataset/train_meiaonew.npz \
  --views 30 \
  --epochs 100 \
  --resume_ckpt ./results/models/view_030/epoch_019_iter_001799.pth.tar \
  --output_dir ./results
```

`--views` 要和 checkpoint 对应。

## 导出预测

```bash
python export_dudotrans_pred_by_indices.py \
  --npz /root/autodl-fs/dataset/test_meiaonew.npz \
  --views 30 \
  --indices 0 66 133 199 \
  --ckpt ./results/models/view_030/epoch_019_iter_001799.pth.tar \
  --out_dir ./exports/dudotrans
```

## 常见问题

如果缺 Python 包：

```bash
pip install -r requirements.txt
```

如果报 `astra_cuda` 找不到，重装 ASTRA：

```bash
conda install -y --override-channels \
  -c astra-toolbox \
  -c nvidia \
  -c defaults \
  astra-toolbox
```

如果报 `numpy.ndarray` 相关错误：

```bash
pip install --force-reinstall "numpy==1.26.4"
```
