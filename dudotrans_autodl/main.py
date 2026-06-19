import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import time
import torch
import shutil
from glob import glob
import numpy as np
import SimpleITK as sitk
from torch.utils.data import DataLoader
from modules.reconstructor import reconstructor, reconstructor_loss
from loaders.load_dataset import CTSlice_Provider, NPZ_CTSlice_Provider

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from sklearn import metrics
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


class Trainer:
    def __init__(self, 
                 learning_rate=1e-4,
                 is_restart=False,#############
                 max_epoch=20,
                 is_cuda=True,
                 num_view=30,
                 num_workers=0,
                 use_amp=False,
                 metric_interval=200,
                 train_npz='/root/autodl-fs/dataset/train_meiaonew.npz',
                 test_npz=None,
                 output_dir='./results',
                 resume_ckpt=None,
                 batch_size=1,
                 poission_level=1e5,
                 gaussian_level=0.05):
        
        # ---- 基本超参数 ----
        self.poission_level = poission_level
        self.gaussian_level = gaussian_level
        self.batch_size = batch_size
        self.lr = learning_rate
        self.is_cuda = is_cuda
        self.num_view = num_view
        self.is_restart = is_restart
        self.max_epoch = max_epoch
        self.num_workers = num_workers
        self.use_amp = use_amp and is_cuda  # 只在 CUDA 下启用 AMP
        self.metric_interval = metric_interval  # 每多少个 iter 计算一次 PSNR/SSIM
        self.train_npz = train_npz
        self.test_npz = test_npz
        self.output_dir = output_dir
        self.resume_ckpt = resume_ckpt

        if self.is_cuda:
            torch.backends.cudnn.benchmark = True  # 对固定输入大小加速

        # ---- Data Flow Pipeline ----
        print('Reading CT slices Beginning (NPZ version)')
        self.train_dataset = NPZ_CTSlice_Provider(
            npz_path=self.train_npz,
            poission_level=self.poission_level,
            gaussian_level=self.gaussian_level,
            num_view=self.num_view,  # 控制稀疏角度数
            img_size=512             # 与 reconstructor 的 img_h,img_w 保持一致
        )
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.is_cuda
        )

        # ---- 模型与损失 ----
        self.reconstructor_func = reconstructor(self.train_dataset)

        num_param = sum(p.numel() for p in self.reconstructor_func.parameters() if p.requires_grad)
        print('Number of parameters in model: {}'.format(num_param))

        self.reconstructor_loss = reconstructor_loss()
        self.ril_loss = reconstructor_loss()
        self.sinogram_loss = reconstructor_loss()

        if self.is_cuda:
            self.reconstructor_func = self.reconstructor_func.cuda()
            self.reconstructor_loss = self.reconstructor_loss.cuda()
            self.ril_loss = self.ril_loss.cuda()
            self.sinogram_loss = self.sinogram_loss.cuda()

        self.reconstructor_params = list(self.reconstructor_func.parameters())
        self.reconstructor_optimizer = torch.optim.Adam(self.reconstructor_params, self.lr)
        self.reconstructor_optimizer_20 = torch.optim.Adam(self.reconstructor_params, 0.1 * self.lr)

        # AMP 的梯度缩放器
        self.scaler = GradScaler(enabled=self.use_amp)

        # ---- 结果目录结构：按视角数划分子目录 ----
        models_root = os.path.join(self.output_dir, 'models')
        vis_root = os.path.join(self.output_dir, 'visualization')
        os.makedirs(models_root, exist_ok=True)
        os.makedirs(vis_root, exist_ok=True)

        self.view_tag = f"view_{self.num_view:03d}"
        self.model_dir = os.path.join(models_root, self.view_tag)
        self.vis_dir = os.path.join(vis_root, self.view_tag)

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        # ---- 训练状态初始化 / 恢复 ----
        if self.is_restart:
            self.epoch = 0
            self.global_iter = 0
            self.best_loss = np.inf
            print('Training process started')
        else:
            try:
                # 旧 checkpoint 依旧从旧目录读取（如果你有老的 ckpt）
                ckpt_path = './results/models/view_030/epoch_010_iter_001799.pth.tar'
                #ckpt_path = './results/models/view_030/epoch_000_iter_001799.pth.tar'
                ckpt_candidates = sorted(glob(os.path.join(self.model_dir, '*.pth.tar')))
                ckpt_path = self.resume_ckpt or (ckpt_candidates[-1] if ckpt_candidates else None)
                if ckpt_path is None:
                    raise FileNotFoundError(f'No checkpoint found in {self.model_dir}')
                print(f'Loading checkpoint: {ckpt_path}')
                state = torch.load(ckpt_path)
                self.epoch = state['epoch']
                self.reconstructor_func.load_state_dict(state['reconstructor_state'])
                self.reconstructor_optimizer.load_state_dict(state['reconstructor_optimizer'])
                print('Saved ckpt is loaded successfully')
            except:
                self.epoch = 0
                print('There is no saved ckpt file to load, the training process is restarted')

        print('Settings are finished')

    def train(self):
        for e in range(self.epoch, self.max_epoch):
            # ----- 1. 初始化本 epoch 的累加量 -----
            epoch_loss_recon = 0.0
            epoch_loss_ril = 0.0
            epoch_loss_sino = 0.0
            epoch_loss_total = 0.0
            num_batches = 0

            start_time = time.time()

            # 用 tqdm 包装 DataLoader，显示当前 epoch 的进度
            pbar = tqdm(self.train_loader,
                        desc=f"Epoch {e}/{self.max_epoch - 1}",
                        ncols=120)

            switch_epoch = self.max_epoch // 2  # 前半程大 lr，后半程 0.1×lr

            for num_iter, (gt, fbp_u, projs_noisy) in enumerate(pbar):
                # 搬数据到 GPU
                if self.is_cuda:
                    gt = gt.cuda(non_blocking=True)
                    fbp_u = fbp_u.cuda(non_blocking=True)
                    projs_noisy = projs_noisy.float().cuda(non_blocking=True)

                # ----- 前向 + 损失（支持 AMP） -----
                with autocast(enabled=self.use_amp):
                    sinos_gt, sinos_enhanced, img_ril, reconstructed_image = \
                        self.reconstructor_func(fbp_u, gt, projs_noisy)

                    loss_recon = self.reconstructor_loss(reconstructed_image, gt)
                    loss_ril = self.ril_loss(img_ril, gt)
                    loss_sino = self.sinogram_loss(sinos_enhanced, sinos_gt)
                    loss_reconstructor = loss_recon + loss_ril + loss_sino

                # ----- 优化器更新 -----
                if e < switch_epoch:
                    self.reconstructor_optimizer.zero_grad()
                    if self.use_amp:
                        self.scaler.scale(loss_reconstructor).backward()
                        self.scaler.step(self.reconstructor_optimizer)
                        self.scaler.update()
                    else:
                        loss_reconstructor.backward()
                        self.reconstructor_optimizer.step()
                else:
                    self.reconstructor_optimizer_20.zero_grad()
                    if self.use_amp:
                        self.scaler.scale(loss_reconstructor).backward()
                        self.scaler.step(self.reconstructor_optimizer_20)
                        self.scaler.update()
                    else:
                        loss_reconstructor.backward()
                        self.reconstructor_optimizer_20.step()

                # ----- 累加到 epoch 级 -----
                epoch_loss_recon += loss_recon.item()
                epoch_loss_ril += loss_ril.item()
                epoch_loss_sino += loss_sino.item()
                epoch_loss_total += loss_reconstructor.item()
                num_batches += 1

                # ----- 每若干 iter 计算一次指标，并更新 tqdm 显示 -----
                if num_iter % self.metric_interval == 0:
                    curr_psnr, curr_ssim, curr_rmse = self.calculate_metric(reconstructed_image, gt)
                    pbar.set_postfix({
                        'L_total': f"{loss_reconstructor.item():.4f}",
                        'PSNR': f"{curr_psnr:.2f}",
                        'SSIM': f"{curr_ssim:.4f}"
                    })

            # ----- 2. 一个 epoch 结束，打印平均 loss -----
            epoch_time = time.time() - start_time
            avg_recon = epoch_loss_recon / max(1, num_batches)
            avg_ril = epoch_loss_ril / max(1, num_batches)
            avg_sino = epoch_loss_sino / max(1, num_batches)
            avg_total = epoch_loss_total / max(1, num_batches)

            print(
                "Epoch {:3d}/{:3d} finished - Time: {:.2f}s | "
                "Loss_recon: {:.4f} | Loss_ril: {:.4f} | "
                "Loss_sino: {:.4f} | Loss_total: {:.4f}"
                .format(e, self.max_epoch - 1, epoch_time, avg_recon, avg_ril, avg_sino, avg_total)
            )

            # ----- 3. 定期保存 checkpoint（每 10 个 epoch / 最后一个 epoch） -----
            if e % 10 == 0 or e == self.max_epoch - 1:
                state = {
                    'epoch': e,
                    'reconstructor_state': self.reconstructor_func.state_dict(),
                    'reconstructor_optimizer': self.reconstructor_optimizer.state_dict(),
                }
                self.save_checkpoint(e, state, num_iter)

    def save_checkpoint(self, num_epoch, state, num_iter):
        save_path = os.path.join(
            self.model_dir,
            f"epoch_{num_epoch:03d}_iter_{num_iter:06d}.pth.tar"
        )
        torch.save(state, save_path)
        print(f"Save model after {num_epoch}-th epoch: {save_path}")

    # def calculate_metric(self, pred, gt):
    #     assert len(pred.shape) == 4 and pred.shape == gt.shape
    #     pred_np = torch.clamp(pred, 0.0, 1.0).cpu().data.numpy()
    #     gt_np = torch.clamp(gt, 0.0, 1.0).cpu().data.numpy()

    #     curr_psnr, curr_ssim = 0.0, 0.0
    #     for i in range(pred_np.shape[0]):
    #         for j in range(pred_np.shape[1]):
    #             curr_psnr = compare_psnr(pred_np[i, j, ...], gt_np[i, j, ...], data_range=1.0)
    #             curr_ssim = compare_ssim(
    #                 pred_np[i, j, ...],
    #                 gt_np[i, j, ...],
    #                 gaussian_weights=True,
    #                 win_size=11,
    #                 data_range=2.0,
    #                 sigma=1.5
    #             )
    #             curr_rmse = np.sqrt(metrics.mean_squared_error(pred_np[i, j, ...], gt_np[i, j, ...]))
    #     return curr_psnr, curr_ssim, curr_rmse
    def calculate_metric(self, pred, gt):
        """
        安全版指标计算：
        - 先用 torch.nan_to_num 把 pred / gt 里的 NaN, inf 处理掉
        - 再转成 numpy 计算 PSNR / SSIM / RMSE
        - 对 batch 和通道做平均，返回三个标量
        """
        assert len(pred.shape) == 4 and pred.shape == gt.shape

        # 1. 先在 torch 层面把 NaN / Inf 处理掉，避免传给 sklearn 出错
        pred_safe = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
        gt_safe   = torch.nan_to_num(gt,   nan=0.0, posinf=1.0, neginf=0.0)

        # 2. 限幅到 [0,1] 再转 numpy
        pred_np = torch.clamp(pred_safe, 0.0, 1.0).detach().cpu().numpy()
        gt_np   = torch.clamp(gt_safe,   0.0, 1.0).detach().cpu().numpy()

        # 3. 逐样本逐通道计算并做平均
        sum_psnr, sum_ssim, sum_rmse = 0.0, 0.0, 0.0
        count = 0

        for i in range(pred_np.shape[0]):      # batch 维
            for j in range(pred_np.shape[1]):  # 通道维
                x = pred_np[i, j, ...]
                y = gt_np[i, j, ...]

                # 再次保险：把 numpy 里的 NaN/Inf 也清一下
                x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
                y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)

                psnr = compare_psnr(x, y, data_range=1.0)
                ssim = compare_ssim(
                    x, y,
                    gaussian_weights=True,
                    win_size=11,
                    data_range=1.0,   # 你的图像已经限制在 [0,1]，这里用 1.0 更合理
                    sigma=1.5
                )

                mse = metrics.mean_squared_error(x, y)
                if np.isnan(mse) or np.isinf(mse):
                    mse = 0.0
                rmse = np.sqrt(mse)

                sum_psnr += psnr
                sum_ssim += ssim
                sum_rmse += rmse
                count += 1

        if count == 0:
            return 0.0, 0.0, 0.0

        avg_psnr = sum_psnr / count
        avg_ssim = sum_ssim / count
        avg_rmse = sum_rmse / count

        return avg_psnr, avg_ssim, avg_rmse


    def inference(self):
        print('Reading CT slices Beginning (NPZ version)')
        self.test_dataset = NPZ_CTSlice_Provider(
            npz_path=self.test_npz or self.train_npz,
            poission_level=self.poission_level,
            gaussian_level=self.gaussian_level,
            num_view=self.num_view,
            img_size=512
        )
        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.is_cuda
        )

        # 注意：如果你想测试新训练的模型，可以改这里的 ckpt 路径
        state = torch.load('./results/models/epoch_100iter1745.pth.tar')
        self.reconstructor_func.load_state_dict(state['reconstructor_state'])

        if self.is_cuda:
            self.reconstructor_func = self.reconstructor_func.cuda()

        aver_psnr, aver_ssim, aver_rmse = 0.0, 0.0, 0.0
        aver_time = 0.0
        time_start = time.time()

        with torch.no_grad():
            for num_iter, (gt, fbp_u, projs_noisy) in enumerate(self.test_loader):
                if self.is_cuda:
                    gt = gt.cuda(non_blocking=True)
                    fbp_u = fbp_u.cuda(non_blocking=True)
                    projs_noisy = projs_noisy.float().cuda(non_blocking=True)

                time_start_in = time.time()
                _, sino_enhanced, ___, reconstructed_image = self.reconstructor_func(fbp_u, gt, projs_noisy)
                time_end_in = time.time()

                curr_psnr, curr_ssim, curr_rmse = self.calculate_metric(reconstructed_image, gt)
                fbp_psnr, fbp_ssim, fbp_rmse = self.calculate_metric(fbp_u, gt)
                print('This is the {}-th infering slice, the psnr and ssim is {:.2f} {:.4f}'
                      .format(num_iter, curr_psnr, curr_ssim))

                aver_psnr += curr_psnr
                aver_ssim += curr_ssim
                aver_rmse += curr_rmse
                aver_time += (time_end_in - time_start_in)

            time_end = time.time()
            aver_psnr = aver_psnr / (num_iter + 1)
            aver_ssim = aver_ssim / (num_iter + 1)
            aver_rmse = aver_rmse / (num_iter + 1)
            aver_time = aver_time / (num_iter + 1)
            aver_time_out = (time_end - time_start) / (num_iter + 1)
            print('the average psnr and ssim rmse is {:.2f} {:.4f} {:.4f}'
                  .format(aver_psnr, aver_ssim, aver_rmse))
            print('the average time of inside and outside is {} and {}'
                  .format(aver_time, aver_time_out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DuDoTrans on AutoDL.')
    parser.add_argument('--train_npz', type=str, default='/root/autodl-fs/dataset/train_meiaonew.npz')
    parser.add_argument('--test_npz', type=str, default=None)
    parser.add_argument('--views', type=int, default=30, help='Sparse view count, e.g. 30/60/90/120.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--restart', action='store_true', help='Start from scratch instead of loading a checkpoint.')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--metric_interval', type=int, default=200)
    args = parser.parse_args()

    trainer = Trainer(
        learning_rate=args.lr,
        is_restart=args.restart,
        max_epoch=args.epochs,
        is_cuda=(not args.cpu),
        num_view=args.views,
        num_workers=args.num_workers,
        use_amp=args.amp,
        metric_interval=args.metric_interval,
        train_npz=args.train_npz,
        test_npz=args.test_npz,
        output_dir=args.output_dir,
        resume_ckpt=args.resume_ckpt,
        batch_size=args.batch_size,
    )
    trainer.train()
    # trainer.inference()
