import os

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader


def _as_nhw(arr, name):
    if arr.ndim == 4:
        if arr.shape[1] == 1:
            arr = arr[:, 0, :, :]
        elif arr.shape[-1] == 1:
            arr = arr[:, :, :, 0]
        else:
            raise ValueError(f"Unsupported {name} shape: {arr.shape}")
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported {name} shape: {arr.shape}")
    return arr.astype(np.float32)


def _resize(img, img_size):
    if img.shape == (img_size, img_size):
        return img.astype(np.float32)
    zoom_y = img_size / img.shape[0]
    zoom_x = img_size / img.shape[1]
    return ndimage.zoom(img, (zoom_y, zoom_x), order=1).astype(np.float32)


def _norm01(img):
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        return (img - mn) / (mx - mn)
    return np.zeros_like(img, dtype=np.float32)


def _find_input_key(npz_obj, views):
    candidates = [
        f"fbp_{views}",
        f"fbp_{views}views",
        f"fbp_{views:03d}",
        f"fbp_{views:03d}views",
        "fbp",
        "fbp_li",
        "input",
        "ldct",
    ]
    for key in candidates:
        if key in npz_obj.files:
            return key
    return None


class NPZRedCNNDataset(Dataset):
    def __init__(
        self,
        npz_path,
        views,
        img_size=512,
        target_key="ct_label",
        input_key="",
        generate_fbp=True,
        patch_n=10,
        patch_size=64,
    ):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ not found: {npz_path}")
        data = np.load(npz_path)
        if target_key not in data.files:
            raise KeyError(f"{target_key} not found in {npz_path}. keys={data.files}")

        self.gt = _as_nhw(data[target_key], target_key)
        self.views = int(views)
        self.img_size = int(img_size)
        self.patch_n = int(patch_n)
        self.patch_size = int(patch_size)

        resolved_input_key = input_key or _find_input_key(data, self.views)
        self.inputs = _as_nhw(data[resolved_input_key], resolved_input_key) if resolved_input_key else None
        self.fbp_ops = None

        if self.inputs is None:
            if not generate_fbp:
                raise ValueError("No FBP/input key found. Remove --no_generate_fbp or pass --input_key.")
            self.fbp_ops = self._build_fbp_ops()
            print("[NPZRedCNNDataset] No input key found; generating FBP with ODL/ASTRA.")
        else:
            print(f"[NPZRedCNNDataset] Using input key: {resolved_input_key}")

    def _build_fbp_ops(self):
        import odl
        from odl.contrib import torch as odl_torch

        xx = 200
        space = odl.uniform_discr(
            [-xx, -xx], [xx, xx], [self.img_size, self.img_size], dtype="float32"
        )
        angle_partition = odl.uniform_partition(0, 2 * np.pi, self.views)
        detector_partition = odl.uniform_partition(-480, 480, 800)
        geometry = odl.tomo.FanBeamGeometry(
            angle_partition, detector_partition, src_radius=600, det_radius=290
        )
        operator = odl.tomo.RayTransform(space, geometry, impl="astra_cuda")
        fbp = odl.tomo.fbp_op(operator, filter_type="Ram-Lak", frequency_scaling=0.9) * np.sqrt(2)
        return odl_torch.operator.OperatorModule(operator), odl_torch.operator.OperatorModule(fbp)

    def __len__(self):
        return self.gt.shape[0]

    def _full_pair(self, idx):
        target = _norm01(_resize(self.gt[idx], self.img_size))
        if self.inputs is not None:
            input_img = _norm01(_resize(self.inputs[idx], self.img_size))
        else:
            radon_op, fbp_op = self.fbp_ops
            with torch.no_grad():
                target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
                sino = radon_op(target_tensor)
                input_tensor = torch.clamp(fbp_op(sino), 0.0, 1.0)
                input_img = input_tensor.detach().cpu().numpy()[0]
        return input_img.astype(np.float32), target.astype(np.float32)

    def __getitem__(self, idx):
        input_img, target_img = self._full_pair(idx)
        if self.patch_size > 0:
            return self._patches(input_img, target_img)
        return input_img, target_img

    def _patches(self, input_img, target_img):
        patch_inputs = []
        patch_targets = []
        h, w = input_img.shape
        size = self.patch_size
        for _ in range(self.patch_n):
            top = np.random.randint(0, h - size)
            left = np.random.randint(0, w - size)
            patch_inputs.append(input_img[top:top + size, left:left + size])
            patch_targets.append(target_img[top:top + size, left:left + size])
        return np.array(patch_inputs, dtype=np.float32), np.array(patch_targets, dtype=np.float32)


def get_npz_loader(
    npz_path,
    views,
    batch_size,
    num_workers,
    img_size=512,
    input_key="",
    generate_fbp=True,
    patch_n=10,
    patch_size=64,
    shuffle=True,
):
    dataset = NPZRedCNNDataset(
        npz_path=npz_path,
        views=views,
        img_size=img_size,
        input_key=input_key,
        generate_fbp=generate_fbp,
        patch_n=patch_n,
        patch_size=patch_size,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
