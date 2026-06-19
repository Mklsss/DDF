# export_dudotrans_pred_by_indices.py
import os
import argparse
import numpy as np
import torch
from PIL import Image

# 关键：按你项目结构 import
from loaders.load_dataset import NPZ_CTSlice_Provider
from modules.reconstructor import reconstructor


def _ensure_dir(dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)


def _to_uint8_image(img_2d: np.ndarray) -> np.ndarray:
    # img_2d: [0,1] float
    img_clipped = np.clip(img_2d, 0.0, 1.0)
    img_uint8 = (img_clipped * 255.0 + 0.5).astype(np.uint8)
    return img_uint8


def _save_png(img_2d: np.ndarray, save_path: str) -> None:
    img_uint8 = _to_uint8_image(img_2d)
    im = Image.fromarray(img_uint8, mode="L")
    im.save(save_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True, help="test_meiaonew.npz 路径")
    parser.add_argument("--views", type=int, required=True, help="30/60/90/120")
    parser.add_argument("--indices", type=int, nargs="+", required=True, help="例如 0 66 133 199")
    parser.add_argument("--ckpt", type=str, required=True, help="epoch_xxx_iter_xxxxxx.pth.tar 路径")
    parser.add_argument("--out_dir", type=str, required=True, help="导出目录")
    parser.add_argument("--cpu", action="store_true", help="强制用 CPU（不推荐）")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")

    # 1) 构建数据集（这里的随机噪声会导致 fbp_u/sino 每次不完全一样）
    #    你现在的目的只是先“对齐 GT 是否同一张”，所以先导出 GT 和 pred 看是否对得上即可。
    dataset = NPZ_CTSlice_Provider(
        npz_path=args.npz,
        poission_level=1e5,
        gaussian_level=0.05,
        num_view=int(args.views),
        img_size=512
    )

    # 2) 构建模型并加载权重
    model = reconstructor(dataset).to(device)
    ckpt_obj = torch.load(args.ckpt, map_location=device)

    if not isinstance(ckpt_obj, dict):
        raise ValueError(f"checkpoint 不是 dict，实际类型: {type(ckpt_obj)}")

    if "reconstructor_state" not in ckpt_obj:
        raise KeyError(f"checkpoint keys: {list(ckpt_obj.keys())}，但没有 reconstructor_state")

    model.load_state_dict(ckpt_obj["reconstructor_state"], strict=True)
    model.eval()

    # 3) 导出
    export_root = os.path.join(args.out_dir, f"view_{int(args.views):03d}")
    out_gt_dir = os.path.join(export_root, "gt")
    out_pred_dir = os.path.join(export_root, "pred")
    _ensure_dir(out_gt_dir)
    _ensure_dir(out_pred_dir)

    with torch.no_grad():
        for idx in args.indices:
            gt_item, fbp_u_item, projs_noisy_item = dataset[idx]  # 直接按 index 取，保证一致

            # dataset 返回形状大概是：
            # gt_item: (1,H,W)   fbp_u_item: (1,H,W)   projs_noisy_item: (1,num_view,800)
            gt_tensor = gt_item.unsqueeze(0).to(device)            # (B=1,1,H,W)
            fbp_u_tensor = fbp_u_item.unsqueeze(0).to(device)      # (1,1,H,W)
            sino_tensor = projs_noisy_item.unsqueeze(0).to(device) # (1,1,num_view,800) or (1,num_view,800)
            sino_tensor = sino_tensor.float()

            # forward
            sinos_gt, sinos_enhanced, img_ril, reconstructed = model(fbp_u_tensor, gt_tensor, sino_tensor)

            # 保存 GT 与 pred（取 [0,0,:,:]）
            gt_np = gt_tensor.detach().cpu().numpy()[0, 0, :, :]
            pred_np = reconstructed.detach().cpu().numpy()[0, 0, :, :]

            gt_path = os.path.join(out_gt_dir, f"idx_{idx:04d}.png")
            pred_path = os.path.join(out_pred_dir, f"idx_{idx:04d}.png")

            _save_png(gt_np, gt_path)
            _save_png(pred_np, pred_path)

            print(f"[OK] views={args.views} idx={idx} -> {gt_path} , {pred_path}")


if __name__ == "__main__":
    main()
