import os
import json
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import time
from scipy.stats import pearsonr

from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------- #
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)
IN_CHANNELS = 7
OUT_CHANNELS = 4
DEPTH = 64
# ---------------------------------------------------------------------------- #
# 2) 统计函数：计算目标字段全局均值与标准差
# ---------------------------------------------------------------------------- #
def compute_target_stats_layerwise(keys, target_folders, depth=64):
    """
    每个目标变量按高度层计算均值和标准差。
    返回 dict: means[var] 和 stds[var] 为 shape=(depth,) 的列表
    """
    means = {}
    stds = {}
    for tgt, folder in target_folders.items():
        folder = Path(folder)
        sum_layer = np.zeros(depth)
        sumsq_layer = np.zeros(depth)
        count_layer = 0
        for key in tqdm(keys, desc=f"Target stats (layerwise) - {tgt}"):
            arr3d = np.load(folder / f"{key}.npy", mmap_mode='r')[:, :, :depth]
            flat = arr3d.reshape(-1, depth)
            sum_layer += flat.sum(axis=0)
            sumsq_layer += (flat ** 2).sum(axis=0)
            count_layer += flat.shape[0]
        μ = sum_layer / count_layer
        σ = np.sqrt(sumsq_layer / count_layer - μ ** 2)
        stds[tgt] = np.where(σ < 1e-3, 1e-3, σ).astype(np.float32).tolist()
        means[tgt] = μ.astype(np.float32).tolist()
    return means, stds

# ---------------------------------------------------------------------------- #
# 统计函数：计算输入字段的全局均值与标准差（按高度分层）
# ---------------------------------------------------------------------------- #
def compute_input_stats(keys, input_folders, depth=DEPTH):
    """
    计算训练集 input 各变量、各高度层的均值与 std，用于标准化。
    返回 dict: means[var]: shape(depth,), stds[var]: shape(depth,)
    """
    sums = {var: np.zeros(depth, dtype=np.float64) for var in input_folders}
    sumsqs = {var: np.zeros(depth, dtype=np.float64) for var in input_folders}
    counts = {var: 0 for var in input_folders}
    for var, folder in input_folders.items():
        folder = Path(folder)
        for key in tqdm(keys, desc=f"Input stats {var}"):
            arr3d = np.load(folder / f"{key}.npy", mmap_mode='r').astype(np.float32)  # (H,W,51)
            arr = arr3d[:, :, :depth]  # (100,100,depth)
            arr_flat = arr.reshape(-1, depth)  # (100*100, depth)
            sums[var] += arr_flat.sum(axis=0)
            sumsqs[var] += (arr_flat**2).sum(axis=0)
            counts[var] += arr_flat.shape[0]
    means = {}
    stds = {}
    for var in input_folders:
        μ = sums[var] / counts[var]  # shape (depth,)
        σ = np.sqrt(sumsqs[var] / counts[var] - μ**2)
        σ = np.where(σ < 1e-3, 1e-3, σ)
        means[var] = μ.astype(np.float32).tolist()
        stds[var] = σ.astype(np.float32).tolist()
    return means, stds

# ---------------------------------------------------------------------------- #
# 3) Dataset 定义：加载 memmap，加上全局归一化
# ---------------------------------------------------------------------------- #
class VerticalProfileDataset(Dataset):
    def __init__(self, keys, input_folders, target_folders,
                 terrain_path, std_path,
                 input_means, input_stds,
                 target_means, target_stds,
                 max_samples=100000,
                 height=None, width=None):
        """
        input_means/stds: dict var -> list of length DEPTH
        target_means/stds: dict tgt -> float
        """
        self.keys = keys
        self.in_folders = {k: Path(v) for k, v in input_folders.items()}
        self.tg_folders = {k: Path(v) for k, v in target_folders.items()}
        self.height = height
        self.width = width

        # 预加载 input 和 target 数据（mmap 节省内存）
        self.input_data = {var: {} for var in input_folders}
        self.target_data = {tgt: {} for tgt in target_folders}

        for var in input_folders:
            folder = self.in_folders[var]
            for key in keys:
                path = folder / f"{key}.npy"
                try:
                    arr = np.load(path, mmap_mode='r').astype(np.float32)  # shape: (H, W, D)
                except Exception as e:
                    raise RuntimeError(f"Load input npy failed: {path}") from e
                self.input_data[var][key] = arr

        for tgt in target_folders:
            folder = self.tg_folders[tgt]
            for key in keys:
                path = folder / f"{key}.npy"
                try:
                    arr = np.load(path, mmap_mode='r').astype(np.float32)  # shape: (H, W, D)
                except Exception as e:
                    raise RuntimeError(f"Load target npy failed: {path}") from e
                self.target_data[tgt][key] = arr

        # 加载地形数据并归一化
        terrain = np.load(terrain_path).astype(np.float32)  # (H,W)
        terrain_std = np.load(std_path).astype(np.float32)  # (H,W)
        t_mean = float(np.mean(terrain))
        t_std = max(float(np.std(terrain)), 1e-3)
        ts_mean = float(np.mean(terrain_std))
        ts_std = max(float(np.std(terrain_std)), 1e-3)
        self.terrain = (terrain - t_mean) / t_std
        self.terrain_std = (terrain_std - ts_mean) / ts_std

        # 保存标准化参数
        self.input_means = {var: np.array(input_means[var], dtype=np.float32) for var in input_means}
        self.input_stds = {var: np.array(input_stds[var], dtype=np.float32) for var in input_stds}
        self.target_means = {var: np.array(target_means[var], dtype=np.float32) for var in target_means}
        self.target_stds = {var: np.array(target_stds[var], dtype=np.float32) for var in target_stds}

        # 生成所有可能的位置索引 (key, i, j)
        self.locations = []
        for key in keys:
            for i in range(self.height):
                for j in range(self.width):
                    self.locations.append((key, i, j))

        # 随机采样
        random.shuffle(self.locations)
        self.locations = self.locations[:max_samples]
        print(f"Created dataset with {len(self.locations)} vertical profiles")

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        key, i, j = self.locations[index]

        # 拼接 input 通道
        input_channels = []
        for var in sorted(self.input_data.keys()):
            profile = self.input_data[var][key][i, j, :]  # shape (D,)
            mean = self.input_means[var]
            std = self.input_stds[var]
            input_channels.append((profile - mean) / std)

        # 加入地形特征
        terrain_feat = np.array([
            self.terrain[i, j],
            self.terrain_std[i, j]
        ], dtype=np.float32)  # shape (2,)

        x = np.concatenate(input_channels + [terrain_feat[:, None]], axis=0)  # shape (C, D)

        # 拼接 target 通道
        y_channels = []
        for tgt in sorted(self.target_data.keys()):
            profile = self.target_data[tgt][key][i, j, :]  # shape (D,)
            mean = self.target_means[tgt]
            std = self.target_stds[tgt]
            y_channels.append((profile - mean) / std)

        y = np.stack(y_channels, axis=0)  # shape (C, D)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
# ---------------------------------------------------------------------------- #
# 4) 1D U-Net 模块
# ---------------------------------------------------------------------------- #
class DoubleConv1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(in_ch, out_ch)
        )

    def forward(self, x):
        return self.seq(x)

class Up1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv1D(out_ch * 2, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.size(2) - x1.size(2)
        if diff > 0:
            pad_left = diff // 2
            pad_right = diff - pad_left
            x1 = nn.functional.pad(x1, [pad_left, pad_right])
        elif diff < 0:
            crop = -diff
            crop_left = crop // 2
            crop_right = crop - crop_left
            x1 = x1[:, :, crop_left: x1.size(2)-crop_right]
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)

class UNet1D(nn.Module):
    def __init__(self, in_c, out_c, base=64):
        super().__init__()
        self.inc = DoubleConv1D(in_c, base)
        self.down1 = Down1D(base, base * 2)
        self.down2 = Down1D(base * 2, base * 4)
        self.down3 = Down1D(base * 4, base * 8)

        self.bottleneck = DoubleConv1D(base * 8, base * 16)

        self.up3 = Up1D(base * 16, base * 8)
        self.up2 = Up1D(base * 8, base * 4)
        self.up1 = Up1D(base * 4, base * 2)
        self.up0 = Up1D(base * 2, base)

        self.outc = OutConv1D(base, out_c)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up3(x5, x4)
        x = self.up2(x, x3)
        x = self.up1(x, x2)
        x = self.up0(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------------------- #
# 5) 损失函数：加权MSE + 梯度损失
# ---------------------------------------------------------------------------- #
def compute_rmse_per_channel(pred, target):
    """
    pred, target: torch.Tensor, shape (B, C, L)
    返回 numpy array, shape (C,), per-channel RMSE averaged over batch和长度
    """
    with torch.no_grad():
        diff2 = (pred - target).cpu().numpy() ** 2
    sum_sq = diff2.sum(axis=(0, 2))
    B, C, L = diff2.shape
    rmse = np.sqrt(sum_sq / (B * L))
    return rmse

def weighted_huber_loss(pred, target, weight, delta=1.0):
    """
    pred, target: shape (B, C, L)
    weight: shape (C,)
    delta: float, Huber loss 的阈值参数
    """
    abs_diff = torch.abs(pred - target)
    quadratic = torch.minimum(abs_diff, torch.tensor(delta, device=pred.device))
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    loss = loss * weight.view(1, -1, 1)  # 每个通道加权
    return loss.mean()

# ---------------------------------------------------------------------------- #
# 6) 训练与验证函数
# ---------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, device, loss_weights):
    model.train()
    running_loss = 0.0
    running_value_loss = 0.0
    for x_batch, y_batch in tqdm(loader, desc="Training", leave=False):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        value_loss = weighted_huber_loss(outputs, y_batch, loss_weights, delta=1.0)
        loss = value_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x_batch.size(0)
        running_value_loss += value_loss.item() * x_batch.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_value = running_value_loss / len(loader.dataset)
    return epoch_loss, epoch_value

def compute_target_stats_layerwise(keys, target_folders, depth=64):
    """
    每个目标变量按高度层计算均值和标准差。
    返回 dict: means[var] 和 stds[var] 为 shape=(depth,) 的列表
    """
    means = {}
    stds = {}
    for tgt, folder in target_folders.items():
        folder = Path(folder)
        sum_layer = np.zeros(depth)
        sumsq_layer = np.zeros(depth)
        count_layer = 0
        for key in tqdm(keys, desc=f"Target stats (layerwise) - {tgt}"):
            arr3d = np.load(folder / f"{key}.npy", mmap_mode='r')[:, :, :depth]
            flat = arr3d.reshape(-1, depth)
            sum_layer += flat.sum(axis=0)
            sumsq_layer += (flat ** 2).sum(axis=0)
            count_layer += flat.shape[0]
        μ = sum_layer / count_layer
        σ = np.sqrt(sumsq_layer / count_layer - μ ** 2)
        stds[tgt] = np.where(σ < 1e-3, 1e-3, σ).astype(np.float32).tolist()
        means[tgt] = μ.astype(np.float32).tolist()
    return means, stds

# ---------------------------------------------------------------------------- #
# 统计函数：计算输入字段的全局均值与标准差（按高度分层）
# ---------------------------------------------------------------------------- #
def compute_input_stats(keys, input_folders, depth=DEPTH):
    """
    计算训练集 input 各变量、各高度层的均值与 std，用于标准化。
    返回 dict: means[var]: shape(depth,), stds[var]: shape(depth,)
    """
    sums = {var: np.zeros(depth, dtype=np.float64) for var in input_folders}
    sumsqs = {var: np.zeros(depth, dtype=np.float64) for var in input_folders}
    counts = {var: 0 for var in input_folders}
    for var, folder in input_folders.items():
        folder = Path(folder)
        for key in tqdm(keys, desc=f"Input stats {var}"):
            arr3d = np.load(folder / f"{key}.npy", mmap_mode='r').astype(np.float32)  # (H,W,51)
            arr = arr3d[:, :, :depth]  # (100,100,depth)
            arr_flat = arr.reshape(-1, depth)  # (100*100, depth)
            sums[var] += arr_flat.sum(axis=0)
            sumsqs[var] += (arr_flat**2).sum(axis=0)
            counts[var] += arr_flat.shape[0]
    means = {}
    stds = {}
    for var in input_folders:
        μ = sums[var] / counts[var]  # shape (depth,)
        σ = np.sqrt(sumsqs[var] / counts[var] - μ**2)
        σ = np.where(σ < 1e-3, 1e-3, σ)
        means[var] = μ.astype(np.float32).tolist()
        stds[var] = σ.astype(np.float32).tolist()
    return means, stds

