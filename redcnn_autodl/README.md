# RED-CNN baseline

基于原始 `RED-CNN-master` 整理，已适配 AutoDL 和 `.npz` 数据训练。

## 数据

`.npz` 里必须有：

```text
ct_label
```

如果 `.npz` 里有现成 FBP 输入，用 `--input_key` 指定；没有就用 ODL/ASTRA 按 `--views` 生成 FBP。

## 安装

```bash
cd /root/autodl-fs/redcnn_autodl
pip install -r requirements.txt

conda install -y --override-channels \
  -c astra-toolbox \
  -c nvidia \
  -c defaults \
  astra-toolbox
```

使用 AutoDL 自带 PyTorch，不要重装 torch。

## 训练

```bash
python train_redcnn_npz.py \
  --train_npz /root/autodl-fs/dataset/train_meiaonew.npz \
  --views 30 \
  --epochs 100 \
  --restart \
  --output_dir ./save_npz
```

其他视角只改：

```bash
--views 60
--views 90
--views 120
```

## 继续训练

```bash
python train_redcnn_npz.py \
  --train_npz /root/autodl-fs/dataset/train_meiaonew.npz \
  --views 30 \
  --epochs 100 \
  --resume_ckpt ./save_npz/view_030/REDCNN_epoch_099.ckpt \
  --output_dir ./save_npz
```

## 导出预测

```bash
python export_redcnn_npz.py \
  --npz /root/autodl-fs/dataset/test_meiaonew.npz \
  --views 30 \
  --indices 0 66 133 199 \
  --ckpt ./save_npz/view_030/REDCNN_epoch_099.ckpt \
  --out_dir ./exports/redcnn
```

## 旧版原始入口

原项目文件 `main.py / solver.py / loader.py / prep.py` 保留。  
新数据集训练请用 `train_redcnn_npz.py`。
