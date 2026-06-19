import os
import odl
import torch
import pydicom
import random
import numpy as np
from PIL import Image
from glob import glob
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from odl.contrib import torch as odl_torch

class CTSlice_Provider(Dataset):
  def __init__(self, base_path, poission_level=5e6, gaussian_level=0.05, num_view=96):
    self.base_path=base_path
    self.slices_path=glob(os.path.join(self.base_path,'*/*.dcm'))
    self.radon_full, self.iradon_full, self.fbp_full, self.op_norm_full=self._radon_transform(num_view=360)
    self.radon_curr, self.iradon_curr, self.fbp_curr, self.op_norm_curr=self._radon_transform(num_view=num_view)
    self.poission_level=poission_level
    self.gaussian_level=gaussian_level
    self.num_view=num_view
    
  def _radon_transform(self, num_view=96, start_ang=0, end_ang=2*np.pi, num_detectors=512):
    # the function is used to generate fp, bp, fbp functions
    # the physical parameters is set as MetaInvNet and EPNet
    xx=200
    space=odl.uniform_discr([-xx, -xx], [xx, xx], [512,512], dtype='float32')
    angles=np.array(num_view).astype(int)
    angle_partition=odl.uniform_partition(start_ang, end_ang, angles)
    detector_partition=odl.uniform_partition(-480, 480, num_detectors)
    geometry=odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=600, det_radius=290)
    operator=odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

    op_norm=odl.operator.power_method_opnorm(operator)
    op_norm = torch.tensor(float(op_norm * 2 * np.pi), dtype=torch.double, device='cuda')

    op_layer=odl_torch.operator.OperatorModule(operator)
    op_layer_adjoint=odl_torch.operator.OperatorModule(operator.adjoint)
    fbp=odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9)*np.sqrt(2)
    op_layer_fbp=odl_torch.operator.OperatorModule(fbp)

    return op_layer, op_layer_adjoint, op_layer_fbp, op_norm

  def ril(self, num_view=96, start_ang=0, end_ang=2*np.pi, num_detectors=256):
  # def ril(self, num_view=96, start_ang=-5/12*np.pi, end_ang=5/12*np.pi, num_detectors=800):
    xx=200
    space=odl.uniform_discr([-xx, -xx], [xx, xx], [512,512], dtype='float32')
    angles=np.array(num_view).astype(int)
    angle_partition=odl.uniform_partition(start_ang, end_ang, angles)
    detector_partition=odl.uniform_partition(-480, 480, num_detectors)
    geometry=odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=600, det_radius=290)
    operator=odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

    op_norm=odl.operator.power_method_opnorm(operator)
    op_norm = torch.tensor(float(op_norm * 2 * np.pi), dtype=torch.double, device='cuda')

    op_layer=odl_torch.operator.OperatorModule(operator)
    op_layer_adjoint=odl_torch.operator.OperatorModule(operator.adjoint)
    fbp=odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9)*np.sqrt(2)
    op_layer_fbp=odl_torch.operator.OperatorModule(fbp)

    return op_layer, op_layer_adjoint, op_layer_fbp, op_norm

  def __getitem__(self, index):
    slice_path=self.slices_path[index]
    dcm=pydicom.read_file(slice_path)
    dcm.image=dcm.pixel_array*dcm.RescaleSlope+dcm.RescaleIntercept
    data_slice=dcm.image
    data_slice=np.array(data_slice).astype(float)
    data_slice=(data_slice-np.min(data_slice))/(np.max(data_slice)-np.min(data_slice))

    # the following code is used to generate projections with noise in the way of odl package
    phantom = torch.tensor(data_slice, dtype=torch.float32).unsqueeze(0)
    sino=self.radon_curr(phantom)

    # the following part code is used to randomly choose sinograms to satisfy the sparse-view requeirement

    # add poission noise
    intensityI0=self.poission_level
    scale_value = torch.tensor(float(intensityI0), dtype=torch.float32)
    normalized_sino=torch.exp(-sino/sino.max())
    th_data=np.random.poisson(scale_value*normalized_sino)
    sino_noisy = -torch.log(torch.tensor(th_data, dtype=torch.float32) / scale_value)
    sino_noisy = sino_noisy*sino.max()

    # add Gaussian noise
    noise_std=self.gaussian_level
    noise_std=np.array(noise_std).astype(np.float)
    nx,ny=np.array(self.num_view).astype(np.int),np.array(800).astype(np.int)
    noise = noise_std*np.random.randn(nx,ny)
    noise = torch.tensor(noise, dtype=torch.float32)
    sino_noisy = sino_noisy + noise

    fbp_u=self.fbp_curr(sino_noisy)
    phantom=phantom#.type(torch.DoubleTensor)
    fbp_u=fbp_u#.type(torch.DoubleTensor)
    sino_noisy=sino_noisy#.type(torch.DoubleTensor)

    return phantom, fbp_u, sino_noisy

  def __len__(self):
    return len(self.slices_path)


class NPZ_CTSlice_Provider(Dataset):
  """
  使用你的 npz 数据集（ct_label）来生成 DuDoTrans 所需的
  (gt, fbp_u, projs_noisy)，其它流程尽量保持和原版一致。
  """
  def __init__(self, npz_path, poission_level=5e6, gaussian_level=0.05,
               num_view=96, img_size=512):
    self.npz_path = npz_path
    if not os.path.exists(self.npz_path):
      raise FileNotFoundError(f"NPZ data file not found: {self.npz_path}")

    data = np.load(self.npz_path)
    if 'ct_label' not in data:
      raise KeyError(f"'ct_label' not found in {self.npz_path}")
    ct = data['ct_label']

    # 统一成 (N, H, W)
    if ct.ndim == 4:
      # 可能是 (N, H, W, 1) 或 (N, 1, H, W)
      if ct.shape[-1] == 1:
        ct = ct[..., 0]
      elif ct.shape[1] == 1:
        ct = ct[:, 0, ...]
      else:
        raise ValueError(f"Unsupported ct_label shape: {ct.shape}")
    elif ct.ndim != 3:
      raise ValueError(f"Unsupported ct_label shape: {ct.shape}")

    self.ct_images = ct.astype(np.float32)
    self.poission_level = poission_level
    self.gaussian_level = gaussian_level
    self.num_view = int(num_view)    # 稀疏视角数
    self.img_size = int(img_size)    # ODL 重建分辨率，保持 512 与原版一致

    # 构建 full-view 和 当前视角数的投影/FBP 算子
    self.radon_full, self.iradon_full, self.fbp_full, self.op_norm_full = \
        self._radon_transform(num_view=360)
    self.radon_curr, self.iradon_curr, self.fbp_curr, self.op_norm_curr = \
        self._radon_transform(num_view=self.num_view)

  def _radon_transform(self, num_view=96, start_ang=0, end_ang=2*np.pi, num_detectors=800):
    xx = 200
    # 注意这里的 [self.img_size, self.img_size]，默认为 512×512
    space = odl.uniform_discr([-xx, -xx], [xx, xx], [self.img_size, self.img_size], dtype='float32')
    angles = np.array(num_view).astype(int)
    angle_partition = odl.uniform_partition(start_ang, end_ang, angles)
    detector_partition = odl.uniform_partition(-480, 480, num_detectors)
    geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition,
                                        src_radius=600, det_radius=290)
    operator = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

    op_norm = odl.operator.power_method_opnorm(operator)
    op_norm = torch.tensor(float(op_norm * 2 * np.pi), dtype=torch.double, device='cuda')

    op_layer = odl_torch.operator.OperatorModule(operator)
    op_layer_adjoint = odl_torch.operator.OperatorModule(operator.adjoint)
    fbp = odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9) * np.sqrt(2)
    op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

    return op_layer, op_layer_adjoint, op_layer_fbp, op_norm

  def ril(self, num_view=None, start_ang=0, end_ang=2*np.pi, num_detectors=800):
    """
    给 reconstructor 用的 RIL/FBP 运算符。
    不传 num_view 时，默认用当前数据集的 self.num_view。
    """
    if num_view is None:
      num_view = self.num_view

    xx = 200
    space = odl.uniform_discr([-xx, -xx], [xx, xx], [self.img_size, self.img_size], dtype='float32')
    angles = np.array(num_view).astype(int)
    angle_partition = odl.uniform_partition(start_ang, end_ang, angles)
    detector_partition = odl.uniform_partition(-480, 480, num_detectors)
    geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition,
                                        src_radius=600, det_radius=290)
    operator = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

    op_norm = odl.operator.power_method_opnorm(operator)
    op_norm = torch.tensor(float(op_norm * 2 * np.pi), dtype=torch.double, device='cuda')

    op_layer = odl_torch.operator.OperatorModule(operator)
    op_layer_adjoint = odl_torch.operator.OperatorModule(operator.adjoint)
    fbp = odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9) * np.sqrt(2)
    op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

    return op_layer, op_layer_adjoint, op_layer_fbp, op_norm

  def __getitem__(self, index):
    # 取出你的 ct_label，对应的某一张切片
    data_slice = self.ct_images[index]   # (H, W)

    # 如有需要，把 256×256 resize 到 512×512（与 ODL 设置一致）
    if data_slice.shape[0] != self.img_size or data_slice.shape[1] != self.img_size:
      zoom_y = self.img_size / data_slice.shape[0]
      zoom_x = self.img_size / data_slice.shape[1]
      data_slice = ndimage.zoom(data_slice, (zoom_y, zoom_x), order=1)

    # 归一化到 [0, 1]，保持和原版 CTSlice_Provider 一致
    data_slice = data_slice.astype(np.float32)
    min_val, max_val = data_slice.min(), data_slice.max()
    if max_val > min_val:
      data_slice = (data_slice - min_val) / (max_val - min_val)
    else:
      data_slice = np.zeros_like(data_slice, dtype=np.float32)

    # 生成 phantom（ground truth 图像）
    phantom = torch.tensor(data_slice, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

    # 生成稀疏视角投影
    sino = self.radon_curr(phantom)   # (1, num_view, 800)

    # --- 加 Poisson 噪声 ---
    intensityI0 = self.poission_level
    scale_value = torch.tensor(float(intensityI0), dtype=torch.float32)
    normalized_sino = torch.exp(-sino / sino.max())
    th_data = np.random.poisson(scale_value.cpu().numpy() * normalized_sino.cpu().numpy())
    sino_noisy = -torch.log(torch.tensor(th_data, dtype=torch.float32) / scale_value)
    sino_noisy = sino_noisy * sino.max()

    # --- 加 Gaussian 噪声 ---
    noise_std = float(self.gaussian_level)
    nx, ny = int(self.num_view), int(800)
    noise = noise_std * np.random.randn(nx, ny)
    noise = torch.tensor(noise, dtype=torch.float32).float()
    sino_noisy = sino_noisy + noise   # broadcasting 到 (1, num_view, 800)

    # 用当前视角的 FBP 得到 fbp_u
    fbp_u = self.fbp_curr(sino_noisy)

    # DataLoader 会再加一个 batch 维度，所以这里保持 (1, H, W)、(1, H, W)、(1, num_view, 800)
    return phantom, fbp_u, sino_noisy

  def __len__(self):
    return self.ct_images.shape[0]

    
if __name__=='__main__':
  print('Reading CT slices Beginning')
  aapm_dataset=CTSlice_Provider('/home1/cewang/Data/NIH-AAPM-DuDo/train')
  aapm_dataloader=DataLoader(dataset=aapm_dataset, batch_size=2, shuffle=False)#True
  for index, (gt, fbpu, projs_noisy) in enumerate(aapm_dataloader):
    if index==1:
      img_save=sitk.GetImageFromArray(fbpu)
      print(gt.shape)
      print(fbpu.shape)
      print(projs_noisy.shape)
