import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cgb import CrossGatingBlock
from model.sin_angle import Attention, sin_angle
from nafnet.NAFNet_arch import NAFNet

DEFAULT_FACTORS = (2, 4, 8, 12)


def resolve_path(path_value):
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return ROOT / path_obj


def load_config(config_path=None):
    default_path = Path(__file__).with_name("config_default.json")
    with default_path.open("r", encoding="utf-8") as file_obj:
        config = json.load(file_obj)
    if config_path:
        with Path(config_path).open("r", encoding="utf-8") as file_obj:
            user_config = json.load(file_obj)
        config = merge_dict(config, user_config)
    return config


def merge_dict(base, update):
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SinogramCTDataset(Dataset):
    def __init__(self, npz_path, sparse_factor):
        self.sparse_factor = int(sparse_factor)
        data = np.load(npz_path)
        full_sinogram = torch.tensor(data["sin357"], dtype=torch.float32)
        ct_label = torch.tensor(data["ct_label"], dtype=torch.float32).permute(0, 3, 1, 2)
        self.sinogram_input = reshape_interpolated_sinogram(full_sinogram, self.sparse_factor)
        self.ct_label = ct_label

    def __len__(self):
        return len(self.sinogram_input)

    def __getitem__(self, index):
        return self.sinogram_input[index], self.ct_label[index]


def reshape_sparse_channels(sparse_channels):
    batch_size, channels, angle_count, sensor_count = sparse_channels.shape
    reshaped_sinogram = sparse_channels[:, 0, :, :]
    for channel_index in range(channels - 1):
        next_channel = sparse_channels[:, channel_index + 1, :, :]
        reshaped_sinogram = torch.cat((reshaped_sinogram, next_channel), dim=1)
    return torch.reshape(reshaped_sinogram, (batch_size, angle_count * channels, sensor_count))


def interpolate_sparse_views(full_sinogram, sparse_factor):
    sample_factor = int(sparse_factor)
    batch_size, _, sensor_count = full_sinogram.shape
    original_sparse = full_sinogram[:, 0::sample_factor, :]
    wrapped_sparse = torch.cat((original_sparse, original_sparse[:, 0, :].unsqueeze(1)), dim=1)
    interpolated_channels = torch.zeros(
        (batch_size, sample_factor, int(360 / sample_factor), sensor_count),
        dtype=full_sinogram.dtype,
    )
    for channel_index in range(sample_factor):
        if channel_index == 0:
            interpolated_channels[:, channel_index, :, :] = full_sinogram[:, channel_index::sample_factor, :]
        else:
            left_weight = sample_factor - channel_index
            right_weight = channel_index
            interpolated_channels[:, channel_index, :, :] = (
                left_weight * original_sparse + right_weight * wrapped_sparse[:, 1:, :]
            ) / sample_factor
    return interpolated_channels


def reshape_interpolated_sinogram(full_sinogram, sparse_factor):
    interpolated_channels = interpolate_sparse_views(full_sinogram, sparse_factor)
    return reshape_sparse_channels(interpolated_channels)


class FbpLayer(nn.Module):
    def __init__(self, matrix_path):
        super().__init__()
        raw_matrix = np.load(matrix_path)
        indice = raw_matrix["indice"].astype("int32")
        data = raw_matrix["data"].astype("float32").reshape(-1)
        self.register_buffer("cos", torch.tensor(raw_matrix["cos"].astype("float32").transpose()))
        sparse_shape = (65536, 128520)
        sparse_indices = torch.tensor(indice.transpose())
        sparse_values = torch.tensor(data)
        self.register_buffer("A_Matrix", torch.sparse_coo_tensor(sparse_indices, sparse_values, sparse_shape).coalesce())
        self.out_shape = (256, 256)
        fbp_filter_weight = torch.tensor(raw_matrix["filt"].astype("float32"))
        self.fbp_filter = nn.Conv2d(1, 1, kernel_size=(713, 1), stride=(1, 1), padding="same")
        self.fbp_filter.weight.data = fbp_filter_weight.reshape(1, 1, 1, -1)
        self.fbp_filter.bias.data = torch.tensor([0.0])

    def forward(self, sin_fan):
        sin_fan_unsqueezed = sin_fan.unsqueeze(1)
        sin_size = sin_fan_unsqueezed.shape[1] * sin_fan_unsqueezed.shape[2] * sin_fan_unsqueezed.shape[3]
        weighted_sinogram = sin_fan_unsqueezed * self.cos
        filtered_sinogram = self.fbp_filter(weighted_sinogram).permute(0, 2, 3, 1)
        filtered_sinogram_flat = torch.reshape(filtered_sinogram, [-1, sin_size]).transpose(1, 0)
        fbp_output = torch.sparse.mm(self.A_Matrix.float(), filtered_sinogram_flat.float()).transpose(1, 0)
        fbp_image = torch.reshape(fbp_output, [-1, self.out_shape[0], self.out_shape[1], 1])
        return fbp_image.clamp(0, 1)


class ForwardProjectionLayer(nn.Module):
    def __init__(self, index_path, data_path):
        super().__init__()
        fp_index = np.load(index_path).transpose((1, 0))
        fp_data = np.load(data_path)
        sparse_shape = (128520, 65536)
        fp_matrix = torch.sparse_coo_tensor(torch.tensor(fp_index), torch.tensor(fp_data), sparse_shape).coalesce()
        self.register_buffer("A", fp_matrix)

    def forward(self, image):
        batch_size, _, _, _ = image.shape
        image_flat = image.reshape(-1, 65536).permute(1, 0)
        sinogram = torch.sparse.mm(self.A.float(), image_flat.float()).reshape(360, 357, batch_size).permute(2, 0, 1)
        return sinogram


class GMLPSineFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.con1 = Attention(357, num_heads=1)
        self.con2 = Attention(357, num_heads=1)
        self.con3 = Attention(357, num_heads=1)

    def forward(self, feedback_sinogram, utr_sinogram):
        feedback_2d = feedback_sinogram.squeeze(1)
        utr_2d = utr_sinogram.squeeze(1)
        fused_sinogram = self.con1(feedback_2d)
        fused_sinogram = self.con2(fused_sinogram)
        fused_sinogram = self.con3(fused_sinogram)
        return (fused_sinogram + utr_2d).unsqueeze(1)


class SimpleSineFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, feedback_sinogram, utr_sinogram):
        fused_input = torch.cat((feedback_sinogram, utr_sinogram), dim=1)
        return self.conv(fused_input)


class ConvCTFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, ct_nd, ct_mid):
        return self.conv(torch.cat((ct_nd, ct_mid), dim=1))


def build_nafnet(config):
    nafnet_config = config["nafnet"]
    return NAFNet(
        img_channel=nafnet_config["img_channel"],
        width=nafnet_config["width"],
        middle_blk_num=nafnet_config["middle_blk_num"],
        enc_blk_nums=nafnet_config["enc_blk_nums"],
        dec_blk_nums=nafnet_config["dec_blk_nums"],
    )


class CascadeModel(nn.Module):
    def __init__(self, sparse_factor, config):
        super().__init__()
        angle = int(360 / int(sparse_factor))
        self.sin = sin_angle(num_sensor=357, angle=angle, num_heads=1)
        self.fbp = FbpLayer(resolve_path(config["fbp_matrix"]))
        self.ct = build_nafnet(config)

    def forward(self, sparse_sinogram):
        utr_sinogram = self.sin(sparse_sinogram)
        sparse_fbp_nhwc = self.fbp(sparse_sinogram)
        fbp_image = self.fbp(utr_sinogram).permute(0, 3, 1, 2)
        ct_nd = self.ct(fbp_image)
        return ct_nd, {"sparse_fbp": sparse_fbp_nhwc.permute(0, 3, 1, 2), "utr_sinogram": utr_sinogram}


class DDFModel(nn.Module):
    def __init__(self, sparse_factor, config, sine_fusion="gmlp", ct_fusion="cgb"):
        super().__init__()
        angle = int(360 / int(sparse_factor))
        self.sin = sin_angle(num_sensor=357, angle=angle, num_heads=1)
        self.fbp = FbpLayer(resolve_path(config["fbp_matrix"]))
        self.fp = ForwardProjectionLayer(resolve_path(config["fp_index"]), resolve_path(config["fp_data"]))
        self.ct = build_nafnet(config)
        self.sine_fusion = GMLPSineFusion() if sine_fusion == "gmlp" else SimpleSineFusion()
        self.ct_fusion_name = ct_fusion
        if ct_fusion == "cgb":
            self.ct_fusion = CrossGatingBlock()
        elif ct_fusion == "conv":
            self.ct_fusion = ConvCTFusion()
        else:
            raise ValueError(f"unknown ct_fusion: {ct_fusion}")

    def forward(self, sparse_sinogram):
        utr_sinogram = self.sin(sparse_sinogram)
        sparse_fbp_nhwc = self.fbp(sparse_sinogram)
        fbp1 = self.fbp(utr_sinogram).permute(0, 3, 1, 2)
        ct_nd = self.ct(fbp1)
        feedback_sinogram = self.fp(ct_nd).unsqueeze(1)
        fused_sinogram = self.sine_fusion(feedback_sinogram, utr_sinogram.unsqueeze(1))
        fbp2 = self.fbp(fused_sinogram.squeeze(1)).permute(0, 3, 1, 2)
        if self.ct_fusion_name == "cgb":
            ct_pre, _ = self.ct_fusion(ct_nd, fbp2)
        else:
            ct_pre = self.ct_fusion(ct_nd, fbp2)
        return ct_pre, {
            "sparse_fbp": sparse_fbp_nhwc.permute(0, 3, 1, 2),
            "cascade": ct_nd,
            "fused_sinogram": fused_sinogram,
        }


def build_model(method, sparse_factor, config):
    if method == "cascade":
        return CascadeModel(sparse_factor, config)
    if method == "ddf":
        return DDFModel(sparse_factor, config, sine_fusion="gmlp", ct_fusion="cgb")
    if method == "ddf_no_sine_fusion":
        return DDFModel(sparse_factor, config, sine_fusion="simple", ct_fusion="cgb")
    if method == "ddf_ct_conv_fusion":
        return DDFModel(sparse_factor, config, sine_fusion="gmlp", ct_fusion="conv")
    raise ValueError(f"unknown method: {method}")


def freeze_projection_layers(model):
    for module_name in ("fbp", "fp"):
        module = getattr(model, module_name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = False


def psnr_batch(prediction, target):
    prediction_cpu = prediction.detach()
    target_cpu = target.detach()
    scores = []
    for item_index in range(prediction_cpu.shape[0]):
        pred_item = prediction_cpu[item_index]
        target_item = target_cpu[item_index]
        data_range = torch.max(target_item) - torch.min(target_item)
        mse_value = torch.mean((pred_item - target_item) ** 2)
        if mse_value.item() <= 0:
            scores.append(float("inf"))
        else:
            scores.append((10.0 * torch.log10((data_range ** 2) / mse_value)).item())
    return float(np.mean(scores))


def ssim_batch(prediction, target, window_size=11):
    channel_count = prediction.size(1)
    window = create_ssim_window(window_size, channel_count).to(prediction.device).type_as(prediction)
    return float(_ssim(prediction, target, window, window_size, channel_count).detach().item())


def create_ssim_window(window_size, channel):
    sigma = 1.5
    values = [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    gaussian_1d = torch.tensor(values, dtype=torch.float32)
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    window_2d = gaussian_1d.unsqueeze(1).mm(gaussian_1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def get_loader(data_path, sparse_factor, batch_size, shuffle=False):
    dataset = SinogramCTDataset(data_path, sparse_factor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


def check_required_files(config, method):
    required_keys = ["test_data", "fbp_matrix"]
    if method != "cascade":
        required_keys.extend(["fp_index", "fp_data"])
    missing = [str(resolve_path(config[key])) for key in required_keys if not resolve_path(config[key]).exists()]
    return missing


def resolve_checkpoint(checkpoint_arg, config, method_key, sparse_factor):
    if checkpoint_arg:
        checkpoint_path = Path(
            checkpoint_arg.format(S=sparse_factor, sparse_factor=sparse_factor)
        )
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        return checkpoint_path if checkpoint_path.exists() else None
    for pattern in config["checkpoint_patterns"].get(method_key, []):
        candidate = resolve_path(pattern.format(S=sparse_factor, sparse_factor=sparse_factor))
        if candidate.exists():
            return candidate
    return None


def evaluate_checkpoint(method, method_key, sparse_factor, config, checkpoint, batch_size, device):
    model = build_model(method, sparse_factor, config).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    test_loader = get_loader(resolve_path(config["test_data"]), sparse_factor, batch_size, shuffle=False)
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        for sparse_sinogram, target_ct in test_loader:
            sparse_sinogram_device = sparse_sinogram.to(device=device, dtype=torch.float32)
            target_ct_device = target_ct.to(device=device, dtype=torch.float32)
            prediction_ct, _ = model(sparse_sinogram_device)
            prediction_ct_clamped = prediction_ct.clamp(0, 1)
            psnr_values.append(psnr_batch(prediction_ct_clamped, target_ct_device))
            ssim_values.append(ssim_batch(prediction_ct_clamped, target_ct_device))
    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


def train_model(method, sparse_factor, config, output_checkpoint, batch_size, device):
    model = build_model(method, sparse_factor, config).to(device)
    freeze_projection_layers(model)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config["train"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["train"]["step_size"], gamma=config["train"]["gamma"]
    )
    train_loader = get_loader(resolve_path(config["train_data"]), sparse_factor, batch_size, shuffle=False)
    val_loader = get_loader(resolve_path(config["test_data"]), sparse_factor, batch_size, shuffle=False)
    best_psnr = -float("inf")
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(int(config["train"]["epochs"])):
        model.train()
        for sparse_sinogram, target_ct in train_loader:
            sparse_sinogram_device = sparse_sinogram.to(device=device, dtype=torch.float32)
            target_ct_device = target_ct.to(device=device, dtype=torch.float32)
            prediction_ct, _ = model(sparse_sinogram_device)
            loss = nn.MSELoss()(prediction_ct, target_ct_device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}: {loss.item()}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        psnr_value, ssim_value = evaluate_model(model, val_loader, device)
        print(f"epoch={epoch}, sparse_factor={sparse_factor}, psnr={psnr_value:.6f}, ssim={ssim_value:.6f}")
        if psnr_value > best_psnr:
            best_psnr = psnr_value
            torch.save(model.state_dict(), output_checkpoint)
    return output_checkpoint


def evaluate_model(model, data_loader, device):
    model.eval()
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        for sparse_sinogram, target_ct in data_loader:
            sparse_sinogram_device = sparse_sinogram.to(device=device, dtype=torch.float32)
            target_ct_device = target_ct.to(device=device, dtype=torch.float32)
            prediction_ct, _ = model(sparse_sinogram_device)
            prediction_ct_clamped = prediction_ct.clamp(0, 1)
            psnr_values.append(psnr_batch(prediction_ct_clamped, target_ct_device))
            ssim_values.append(ssim_batch(prediction_ct_clamped, target_ct_device))
    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


def append_csv(csv_path, rows, fieldnames):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_factors(sparse_factor):
    if sparse_factor:
        return [int(value.strip()) for value in str(sparse_factor).split(",") if value.strip()]
    return list(DEFAULT_FACTORS)


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--sparse_factor", default=None, help="Sparse factor, e.g. 2 or 2,4,8,12. Defaults to all.")
    parser.add_argument("--config", default=None, help="JSON config path. Defaults to experiments/config_default.json.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path or template with {S}.")
    parser.add_argument("--output_dir", default="results", help="Directory for csv or figures.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=("eval", "train"), default="eval")
    return parser


def run_fixed_factor_experiment(args, method, method_key, display_name, csv_name):
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = torch.device(args.device)
    missing_files = check_required_files(config, method)
    if missing_files:
        for missing_file in missing_files:
            print(f"required file not found: {missing_file}")
        print("TODO: update experiments/config_default.json or pass --config with the correct paths.")
        return
    rows = []
    for sparse_factor in parse_factors(args.sparse_factor):
        checkpoint_path = resolve_checkpoint(args.checkpoint, config, method_key, sparse_factor)
        if args.mode == "train":
            if checkpoint_path is None:
                checkpoint_path = ROOT / "weights" / f"{method_key}_S{sparse_factor}.pth"
            print(f"training {display_name}, S={sparse_factor}, checkpoint={checkpoint_path}")
            train_model(method, sparse_factor, config, checkpoint_path, args.batch_size, device)
        checkpoint_path = resolve_checkpoint(str(checkpoint_path) if checkpoint_path else args.checkpoint, config, method_key, sparse_factor)
        if checkpoint_path is None:
            print(f"checkpoint for S={sparse_factor} not found, please train first.")
            continue
        psnr_value, ssim_value = evaluate_checkpoint(
            method, method_key, sparse_factor, config, checkpoint_path, args.batch_size, device
        )
        print(f"{display_name}, S={sparse_factor}, PSNR={psnr_value:.6f}, SSIM={ssim_value:.6f}")
        rows.append(
            {
                "method": display_name,
                "sparse_factor": sparse_factor,
                "psnr": f"{psnr_value:.6f}",
                "ssim": f"{ssim_value:.6f}",
                "checkpoint": str(checkpoint_path),
                "config": str(args.config or Path(__file__).with_name("config_default.json")),
            }
        )
    if rows:
        append_csv(Path(args.output_dir) / csv_name, rows, ["method", "sparse_factor", "psnr", "ssim", "checkpoint", "config"])
