#!/usr/bin/env python3
"""
PI-xLSTM trajectory training.

IMPORTANT: A **folder** `./xlstm/` here (e.g. git clone of NX-AI/xlstm) or a file
`xlstm.py` shadows the pip package: imports resolve to that path first and you may get
  ImportError: cannot import name 'FeedForwardConfig' from 'xlstm' (.../xLSTM/xlstm/__init__.py)
This script demotes the script directory on sys.path when `./xlstm/` exists so `pip install xlstm`
wins. Prefer not adding `__init__.py` at the **clone root** (non-standard); the real package is
`xlstm/xlstm/` inside the repo.

Install the official package (NX-AI): https://github.com/NX-AI/xlstm
  pip install xlstm

Then:
  cd xLSTM && python train_pi_xlstm.py --mat_path /path/to/PPTR_TrainDataset_1000.mat

Checkpoint / resume:
  python train_pi_xlstm.py ... --resume ./run/<timestamp>_pid…/weights/latest.pt

每次训练在 run/ 下新建目录（时间戳+可选标签+pid），内含 config.json、metrics.jsonl、viz/、weights/。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_script_dir = Path(__file__).resolve().parent
if (_script_dir / "xlstm").is_dir():
    _sd = str(_script_dir)
    while sys.path and Path(sys.path[0]).resolve() == _script_dir:
        sys.path.pop(0)
    if _sd not in sys.path:
        sys.path.append(_sd)

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

# 基线模型（bilstm / bigru / transformer / tcn）
from baselines import build_baseline


def _h5_to_ntd(arr: np.ndarray, *, feat_dims: tuple[int, ...], name: str) -> np.ndarray:
    """将 HDF5 数组统一为 [N, T, F]（兼容 MATLAB 列优先存盘与 Python 直接存盘）。"""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 3:
        raise ValueError(f"{name} 应为 3 维，得到 {a.shape}")
    if a.shape[-1] in feat_dims:
        return a
    if a.shape[0] in feat_dims:
        return a.transpose(2, 1, 0)
    raise ValueError(
        f"{name} 无法识别布局 {a.shape}，末维或首维应为 {feat_dims} 之一"
    )


# Official NeurIPS stack API: https://github.com/NX-AI/xlstm (pip package `xlstm`)
from xlstm import (
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
)


class SARDataset(Dataset):
    """仿真数据集：所有样本都有标签（has_label=True），所有 RCM 参考点均有效（rcm_weight 全 1）。

    seq_stride > 1 时对时序维度做均匀下采样（每 seq_stride 点取一），
    用于将 T=4096 的数据集降为 T=2048，避免 mLSTM C 矩阵 O(S²) 显存爆炸。

    旧 mat：__getitem__ 返回 9 元组。
    v4_phase mat（Phase_rel / R_ref）：返回 11 元组。
    旧 v4 cascade mat（Delta_coarse / Phase_sin / Phase_cos）：返回 12 元组。
    """

    def __init__(self, mat_file_path: str, seq_stride: int = 1) -> None:
        super().__init__()
        self.seq_stride = max(1, int(seq_stride))
        print(f"正在加载数据: {mat_file_path} ...")
        with h5py.File(mat_file_path, "r") as f:
            self.feat = _h5_to_ntd(np.array(f["Feat_All"]), feat_dims=(6, 9, 12), name="Feat_All")
            self.p_raw = _h5_to_ntd(np.array(f["P_raw"]), feat_dims=(3,), name="P_raw")
            self.p_true = _h5_to_ntd(np.array(f["P_true"]), feat_dims=(3,), name="P_true")
            N = self.feat.shape[0]

            def _load_pos(key: str) -> np.ndarray:
                arr = np.array(f[key]).astype(np.float32)
                if arr.shape == (3, N) or arr.shape == (N, 3):
                    return arr.T if arr.shape == (3, N) else arr
                return np.tile(arr.flatten(), (N, 1))

            self.pos_a = _load_pos("Pos_A")
            self.pos_b = _load_pos("Pos_B")
            self.pos_c = _load_pos("Pos_C")

            self.is_v4_phase = "Phase_rel" in f
            self.is_v4_legacy = ("Delta_coarse" in f) and not self.is_v4_phase
            self.is_v4 = self.is_v4_phase or self.is_v4_legacy
            if self.is_v4_phase:
                self.phase_rel = _h5_to_ntd(
                    np.array(f["Phase_rel"]), feat_dims=(3,), name="Phase_rel"
                )
                rref = np.array(f["R_ref"]).astype(np.float32)
                if rref.shape == (3, N):
                    self.r_ref = rref.T
                elif rref.shape == (N, 3):
                    self.r_ref = rref
                else:
                    raise ValueError(f"R_ref 应为 [N,3] 或 [3,N]，得到 {rref.shape}")
                self.delta_coarse = None
                self.phase_sin = None
                self.phase_cos = None
            elif self.is_v4_legacy:
                self.delta_coarse = _h5_to_ntd(
                    np.array(f["Delta_coarse"]), feat_dims=(3,), name="Delta_coarse"
                )
                self.phase_sin = _h5_to_ntd(
                    np.array(f["Phase_sin"]), feat_dims=(3,), name="Phase_sin"
                )
                self.phase_cos = _h5_to_ntd(
                    np.array(f["Phase_cos"]), feat_dims=(3,), name="Phase_cos"
                )
                self.phase_rel = None
                self.r_ref = None
            else:
                self.delta_coarse = None
                self.phase_sin = None
                self.phase_cos = None
                self.phase_rel = None
                self.r_ref = None
            if "fc" in f:
                fc_arr = np.array(f["fc"]).astype(np.float32).flatten()
                self.fc = float(fc_arr[0])
            else:
                self.fc = 9.5e9

        T_eff = self.feat.shape[1] // self.seq_stride
        if self.is_v4_phase:
            v4_tag = " [v4_phase: Phase_rel+R_ref]"
        elif self.is_v4_legacy:
            v4_tag = " [v4_legacy: Delta_coarse+Phase_sin/cos]"
        else:
            v4_tag = ""
        print(
            f"数据加载完成! {self.feat.shape}{v4_tag}"
            + (f"  → 下采样 stride={self.seq_stride} 后 T={T_eff}" if self.seq_stride > 1 else "")
        )

    def __len__(self) -> int:
        return self.feat.shape[0]

    def _base_item(self, idx: int):
        s = self.seq_stride
        feat = self.feat[idx][::s]
        p_raw = self.p_raw[idx][::s]
        p_true = self.p_true[idx][::s]
        T, F = feat.shape
        feat_mask = np.ones((T, F), dtype=np.float32)
        has_label = np.array(True)
        rcm_weight = np.ones(3, dtype=np.float32)
        return feat, feat_mask, p_raw, p_true, has_label, rcm_weight

    def __getitem__(self, idx: int):
        feat, feat_mask, p_raw, p_true, has_label, rcm_weight = self._base_item(idx)
        base = (
            feat,
            feat_mask,
            p_raw,
            p_true,
            has_label,
            rcm_weight,
            self.pos_a[idx],
            self.pos_b[idx],
            self.pos_c[idx],
        )
        if not self.is_v4:
            return base
        s = self.seq_stride
        if self.is_v4_phase:
            assert self.phase_rel is not None and self.r_ref is not None
            phase_rel = self.phase_rel[idx][::s]
            r_ref = self.r_ref[idx]
            return base + (phase_rel, r_ref)
        assert self.is_v4_legacy
        delta_coarse = self.delta_coarse[idx][::s]
        phase_sin = self.phase_sin[idx][::s]
        phase_cos = self.phase_cos[idx][::s]
        return base + (delta_coarse, phase_sin, phase_cos)


class RealSARDataset(Dataset):
    """实测数据集：无 P_true 标签（has_label=False），参考点数量可变（0~3）。

    期望的 .mat 文件格式（v7.3）：
        P_raw    : [N, T, 3]  原始 GPS/IMU 轨迹（必须）
        Pos_Ref  : [3, 3, N]  最多 3 个参考点坐标，[坐标维, 参考点索引, 样本索引]
                              缺失的参考点用 NaN 填充
        RCM_obs  : [N, T, 3]  SAR 实测斜距，缺失的参考点列用 NaN 填充

    当 Pos_Ref / RCM_obs 不存在时，视为 0 个参考点。

    __getitem__ 返回与 SARDataset 相同的 9 元组接口，以便 DataLoader 合并批次：
        (feat, feat_mask, p_raw, p_true_zeros, has_label=False, rcm_weight, pos_a, pos_b, pos_c)

    feat 构建规则：
        feat[:, 0:3]  = P_raw 高斯平滑（始终可用）
        feat[:, 3:6]  = V_raw 数值微分（始终可用）
        feat[:, 6:9]  = RCM_obs（有参考点时填入，否则 0）
        feat[:, 9:12] = 导航-RCM 残差（有参考点且坐标已知时填入，否则 0）
    feat_mask 对应维度：有效=1，缺失=0
    """

    _GAUSS_WIN = 20

    def __init__(self, mat_file_path: str, seq_stride: int = 1) -> None:
        super().__init__()
        self.seq_stride = max(1, int(seq_stride))
        print(f"正在加载实测数据: {mat_file_path} ...")
        with h5py.File(mat_file_path, "r") as f:
            p_raw_raw = np.array(f["P_raw"]).transpose(2, 1, 0).astype(np.float64)  # [N, T, 3]
            N, T, _ = p_raw_raw.shape

            # --- 参考点坐标（可选）---
            if "Pos_Ref" in f:
                # 期望存储形状 [3, n_pts, N]，取前 3 个参考点
                pos_ref_raw = np.array(f["Pos_Ref"]).astype(np.float64)
                # 统一转为 [N, 3, 3]：[样本, 参考点索引, 坐标维]
                if pos_ref_raw.ndim == 3 and pos_ref_raw.shape[0] == 3:
                    pos_ref = pos_ref_raw.transpose(2, 1, 0)  # [N, n_pts, 3]
                elif pos_ref_raw.ndim == 3 and pos_ref_raw.shape[-1] == 3:
                    pos_ref = pos_ref_raw  # [N, n_pts, 3]
                else:
                    pos_ref = np.full((N, 3, 3), np.nan, dtype=np.float64)
                # 补齐到恰好 3 个参考点
                n_pts = pos_ref.shape[1]
                if n_pts < 3:
                    pad = np.full((N, 3 - n_pts, 3), np.nan, dtype=np.float64)
                    pos_ref = np.concatenate([pos_ref, pad], axis=1)
                else:
                    pos_ref = pos_ref[:, :3, :]
            else:
                pos_ref = np.full((N, 3, 3), np.nan, dtype=np.float64)

            # --- SAR 测量斜距（可选）---
            if "RCM_obs" in f:
                rcm_obs_raw = np.array(f["RCM_obs"]).astype(np.float64)
                # 期望 [N, T, n_pts] 或 [n_pts, T, N]
                if rcm_obs_raw.shape == (N, T, 3):
                    rcm_obs = rcm_obs_raw
                elif rcm_obs_raw.ndim == 3 and rcm_obs_raw.shape[-1] == N:
                    rcm_obs = rcm_obs_raw.transpose(2, 1, 0)  # [N, T, 3]
                else:
                    rcm_obs = np.full((N, T, 3), np.nan, dtype=np.float64)
                n_pts_obs = rcm_obs.shape[2] if rcm_obs.ndim == 3 else 0
                if n_pts_obs < 3:
                    pad = np.full((N, T, 3 - n_pts_obs), np.nan, dtype=np.float64)
                    rcm_obs = np.concatenate([rcm_obs, pad], axis=2)
            else:
                rcm_obs = np.full((N, T, 3), np.nan, dtype=np.float64)

        # --- 构建 feat [N, T, 12] 与 feat_mask [N, T, 12] ---
        from scipy.ndimage import uniform_filter1d  # 轻量高斯平滑替代
        feat = np.zeros((N, T, 12), dtype=np.float32)
        feat_mask = np.zeros((N, T, 12), dtype=np.float32)

        # 高斯平滑辅助函数（沿时间轴）
        def _smooth(x: np.ndarray, win: int) -> np.ndarray:
            from scipy.ndimage import gaussian_filter1d
            sigma = win / 6.0
            return gaussian_filter1d(x.astype(np.float64), sigma=sigma, axis=-1).astype(np.float32)

        PRF = 1000.0
        for i in range(N):
            # feat[0:3]: P_raw 平滑
            pr = p_raw_raw[i].T.copy()  # [3, T]
            pr[0] = _smooth(pr[0], self._GAUSS_WIN)
            pr[1] = _smooth(pr[1], self._GAUSS_WIN)
            pr[2] = _smooth(pr[2], self._GAUSS_WIN)
            feat[i, :, 0:3] = pr.T
            feat_mask[i, :, 0:3] = 1.0

            # feat[3:6]: V_raw
            vr = np.zeros_like(pr)
            vr[:, 1:] = (pr[:, 1:] - pr[:, :-1]) * PRF
            vr[:, 0] = vr[:, 1]
            vr[0] = _smooth(vr[0], self._GAUSS_WIN * 2)
            vr[1] = _smooth(vr[1], self._GAUSS_WIN * 2)
            vr[2] = _smooth(vr[2], self._GAUSS_WIN * 2)
            feat[i, :, 3:6] = vr.T
            feat_mask[i, :, 3:6] = 1.0

            # feat[6:9]: RCM 实测斜距
            for j in range(3):
                col = rcm_obs[i, :, j]
                valid = ~np.isnan(col)
                if valid.all():
                    feat[i, :, 6 + j] = col.astype(np.float32)
                    feat_mask[i, :, 6 + j] = 1.0

            # feat[9:12]: 导航-RCM 残差（需要参考点坐标且对应 RCM 有效）
            for j in range(3):
                pj = pos_ref[i, j]           # [3]
                if np.any(np.isnan(pj)):
                    continue
                if feat_mask[i, 0, 6 + j] < 0.5:
                    continue
                d_nav = np.sqrt(np.sum((pr.T - pj[None, :]) ** 2, axis=-1))  # [T]
                feat[i, :, 9 + j] = (d_nav - feat[i, :, 6 + j]).astype(np.float32)
                feat_mask[i, :, 9 + j] = 1.0

        # rcm_weight [N, 3]：该参考点的两个条件都满足（坐标 + 测距）时为 1
        rcm_weight = np.zeros((N, 3), dtype=np.float32)
        for j in range(3):
            has_pos = ~np.any(np.isnan(pos_ref[:, j, :]), axis=-1)  # [N]
            has_rcm = ~np.any(np.isnan(rcm_obs[:, :, j]), axis=-1)  # [N]
            rcm_weight[:, j] = (has_pos & has_rcm).astype(np.float32)

        # 参考点坐标（NaN 替换为 0，不参与 RCM loss 的项由 rcm_weight 屏蔽）
        pos_ref_clean = np.nan_to_num(pos_ref, nan=0.0).astype(np.float32)  # [N, 3, 3]

        s = self.seq_stride
        self.feat      = feat[:, ::s, :]                           # [N, T_eff, 12]
        self.feat_mask = feat_mask[:, ::s, :]                      # [N, T_eff, 12]
        self.p_raw = np.array([
            p_raw_raw[i][::s].astype(np.float32) for i in range(N)
        ])  # [N, T_eff, 3]
        self.rcm_weight = rcm_weight                               # [N, 3]
        self.pos_a = pos_ref_clean[:, 0, :]                        # [N, 3]
        self.pos_b = pos_ref_clean[:, 1, :]                        # [N, 3]
        self.pos_c = pos_ref_clean[:, 2, :]                        # [N, 3]
        self._T = T // s

        n_ref_counts = rcm_weight.sum(axis=1).astype(int)
        print(f"实测数据加载完成! 样本数={N}, 时序长={T}")
        print(f"  参考点统计: 0点={np.sum(n_ref_counts==0)} | "
              f"1点={np.sum(n_ref_counts==1)} | "
              f"2点={np.sum(n_ref_counts==2)} | "
              f"3点={np.sum(n_ref_counts==3)}")

    def __len__(self) -> int:
        return self.feat.shape[0]

    def __getitem__(self, idx: int):
        p_true_dummy = np.zeros((self._T, 3), dtype=np.float32)   # 无标签，占位
        has_label = np.array(False)
        return (
            self.feat[idx],      # [T, F]
            self.feat_mask[idx], # [T, F]
            self.p_raw[idx],     # [T, 3]
            p_true_dummy,        # [T, 3]  占位，不参与损失
            has_label,           # scalar bool False
            self.rcm_weight[idx],# [3]
            self.pos_a[idx],     # [3]
            self.pos_b[idx],     # [3]
            self.pos_c[idx],     # [3]
        )


class RefPointMaskedDataset(Dataset):
    """包装任何返回 9-tuple 的数据集，在训练时随机丢弃参考点，使模型鲁棒于推理时仅有 1 个点的场景。

    策略：以 p_1pt / p_2pt / (1-p_1pt-p_2pt) 的概率随机保留 1 / 2 / 3 个参考点。
    被丢弃点对应的 feat 维度（RCM 斜距 + 残差）在归一化之前就置零，feat_mask 也置 0，
    rcm_weight 置 0，确保它们对 MSE / RCM loss 均无贡献。

    关键：feat_mask 传入归一化步骤后应再乘回（`feat_norm *= feat_mask`），
    使被丢弃通道在归一化空间恰好为 0（即"均值"位置），避免误导网络。
    """

    def __init__(
        self,
        dataset: Dataset,
        p_1pt: float = 0.20,   # 保留 1 个点的概率（推理最常见场景）
        p_2pt: float = 0.30,   # 保留 2 个点的概率
        # 保留 3 个点的概率 = 1 - p_1pt - p_2pt = 0.30
    ) -> None:
        super().__init__()
        if p_1pt + p_2pt > 1.0:
            raise ValueError(f"p_1pt + p_2pt = {p_1pt + p_2pt:.3f} > 1.0")
        self.dataset = dataset
        self._probs = [p_1pt, p_2pt, 1.0 - p_1pt - p_2pt]  # [P(1pt), P(2pt), P(3pt)]

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        n_item = len(item)
        is_v4_phase = n_item == 11
        is_v4_legacy = n_item == 12
        if is_v4_phase:
            (
                feat, feat_mask, p_raw, p_true, has_label, rcm_weight,
                pos_a, pos_b, pos_c, phase_rel, r_ref,
            ) = item
            delta_coarse = phase_sin = phase_cos = None
        elif is_v4_legacy:
            (
                feat, feat_mask, p_raw, p_true, has_label, rcm_weight,
                pos_a, pos_b, pos_c, delta_coarse, phase_sin, phase_cos,
            ) = item
            phase_rel = r_ref = None
        else:
            feat, feat_mask, p_raw, p_true, has_label, rcm_weight, pos_a, pos_b, pos_c = item
            delta_coarse = phase_sin = phase_cos = phase_rel = r_ref = None

        n_keep = int(np.random.choice([1, 2, 3], p=self._probs))
        if n_keep >= 3:
            return item

        keep_set = set(np.random.choice(3, size=n_keep, replace=False).tolist())

        new_feat_mask = feat_mask.copy() if isinstance(feat_mask, np.ndarray) else feat_mask.clone()
        new_rcm_weight = rcm_weight.copy() if isinstance(rcm_weight, np.ndarray) else rcm_weight.clone()
        new_feat = feat.copy() if isinstance(feat, np.ndarray) else feat.clone()
        n_feat = new_feat.shape[-1]

        for j in range(3):
            if j not in keep_set:
                if n_feat > 6 + j:
                    new_feat[:, 6 + j] = 0.0
                    new_feat_mask[:, 6 + j] = 0.0
                if n_feat > 9 + j:
                    new_feat[:, 9 + j] = 0.0
                    new_feat_mask[:, 9 + j] = 0.0
                new_rcm_weight[j] = 0.0

        if is_v4_phase:
            assert phase_rel is not None
            new_phase_rel = phase_rel.copy()
            for j in range(3):
                if j not in keep_set:
                    new_phase_rel[:, j] = 0.0
            return (
                new_feat, new_feat_mask, p_raw, p_true, has_label, new_rcm_weight,
                pos_a, pos_b, pos_c, new_phase_rel, r_ref,
            )

        if is_v4_legacy:
            new_phase_sin = phase_sin.copy()
            new_phase_cos = phase_cos.copy()
            for j in range(3):
                if j not in keep_set:
                    new_phase_sin[:, j] = 0.0
                    new_phase_cos[:, j] = 0.0
            return (
                new_feat, new_feat_mask, p_raw, p_true, has_label, new_rcm_weight,
                pos_a, pos_b, pos_c, delta_coarse, new_phase_sin, new_phase_cos,
            )

        return (
            new_feat, new_feat_mask, p_raw, p_true,
            has_label, new_rcm_weight, pos_a, pos_b, pos_c,
        )


class DataNormalizer(nn.Module):
    """以 buffer 形式持有 feat 与 error 的均值/方差，可随 .to(device) 迁移、随 state_dict 持久化。

    - feat 形状 [N,T,9]，输出统计量形状 [9]
    - error = p_raw - p_true 形状 [N,T,3]，输出统计量形状 [3]

    设计目的：xLSTM 是带指数门控的自回归网络，直接吃未归一化的物理绝对值会数值崩塌。
    我们在送入网络前 z-score 归一化 feat，并以归一化空间预测 error；
    经反归一化后回到物理量参与 RCM 损失计算。
    """

    def __init__(
        self,
        feat_mean: torch.Tensor,
        feat_std: torch.Tensor,
        err_mean: torch.Tensor,
        err_std: torch.Tensor,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if feat_mean.shape != feat_std.shape:
            raise ValueError(f"feat_mean/feat_std 形状不一致: {feat_mean.shape} vs {feat_std.shape}")
        if err_mean.shape != err_std.shape:
            raise ValueError(f"err_mean/err_std 形状不一致: {err_mean.shape} vs {err_std.shape}")
        self.eps = float(eps)
        self.register_buffer("feat_mean", feat_mean.to(torch.float32).contiguous())
        self.register_buffer("feat_std", feat_std.clamp_min(self.eps).to(torch.float32).contiguous())
        self.register_buffer("err_mean", err_mean.to(torch.float32).contiguous())
        self.register_buffer("err_std", err_std.clamp_min(self.eps).to(torch.float32).contiguous())

    @classmethod
    def fit(
        cls,
        dataset: "SARDataset",
        indices: Optional[np.ndarray] = None,
        eps: float = 1e-6,
    ) -> "DataNormalizer":
        """仅用训练集样本拟合统计量，传入 indices 可避免验证集数据泄漏。"""
        if indices is not None:
            feat = dataset.feat[indices].astype(np.float64)
            err = (dataset.p_raw[indices] - dataset.p_true[indices]).astype(np.float64)
        else:
            feat = dataset.feat.astype(np.float64)
            err = (dataset.p_raw - dataset.p_true).astype(np.float64)
        feat_mean_np = feat.mean(axis=(0, 1))
        feat_std_np = feat.std(axis=(0, 1))
        err_mean_np = err.mean(axis=(0, 1))
        err_std_np = err.std(axis=(0, 1))
        return cls(
            feat_mean=torch.from_numpy(feat_mean_np.astype(np.float32)),
            feat_std=torch.from_numpy(feat_std_np.astype(np.float32)),
            err_mean=torch.from_numpy(err_mean_np.astype(np.float32)),
            err_std=torch.from_numpy(err_std_np.astype(np.float32)),
            eps=eps,
        )

    def norm_feat(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feat_mean) / self.feat_std

    def norm_err(self, e: torch.Tensor) -> torch.Tensor:
        return (e - self.err_mean) / self.err_std

    def denorm_err(self, e_norm: torch.Tensor) -> torch.Tensor:
        return e_norm * self.err_std + self.err_mean

    def summary(self) -> str:
        with torch.no_grad():
            return (
                f"feat μ={self.feat_mean.tolist()} σ={self.feat_std.tolist()} | "
                f"err μ={self.err_mean.tolist()} σ={self.err_std.tolist()}"
            )


class CascadeDataNormalizer(DataNormalizer):
    """v4 cascade 训练：在 feat/err 之外增加 delta_coarse 与 phase(sin/cos) 统计量。"""

    def __init__(
        self,
        feat_mean: torch.Tensor,
        feat_std: torch.Tensor,
        err_mean: torch.Tensor,
        err_std: torch.Tensor,
        delta_c_mean: torch.Tensor,
        delta_c_std: torch.Tensor,
        phase_mean: torch.Tensor,
        phase_std: torch.Tensor,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(feat_mean, feat_std, err_mean, err_std, eps=eps)
        self.register_buffer("delta_c_mean", delta_c_mean.to(torch.float32).contiguous())
        self.register_buffer("delta_c_std", delta_c_std.clamp_min(self.eps).to(torch.float32).contiguous())
        self.register_buffer("phase_mean", phase_mean.to(torch.float32).contiguous())
        self.register_buffer("phase_std", phase_std.clamp_min(self.eps).to(torch.float32).contiguous())

    @classmethod
    def fit_cascade(
        cls,
        dataset: SARDataset,
        indices: Optional[np.ndarray] = None,
        eps: float = 1e-6,
    ) -> "CascadeDataNormalizer":
        if not dataset.is_v4:
            raise ValueError("CascadeDataNormalizer.fit_cascade 需要 v4 mat（含 Delta_coarse / Phase_sin / Phase_cos）")
        if indices is not None:
            feat = dataset.feat[indices].astype(np.float64)
            err = (dataset.p_raw[indices] - dataset.p_true[indices]).astype(np.float64)
            dc = dataset.delta_coarse[indices].astype(np.float64)
            ph_sin = dataset.phase_sin[indices].astype(np.float64)
            ph_cos = dataset.phase_cos[indices].astype(np.float64)
        else:
            feat = dataset.feat.astype(np.float64)
            err = (dataset.p_raw - dataset.p_true).astype(np.float64)
            dc = dataset.delta_coarse.astype(np.float64)
            ph_sin = dataset.phase_sin.astype(np.float64)
            ph_cos = dataset.phase_cos.astype(np.float64)
        phase = np.concatenate([ph_sin, ph_cos], axis=-1)
        base = DataNormalizer.fit(dataset, indices=indices, eps=eps)
        return cls(
            feat_mean=base.feat_mean,
            feat_std=base.feat_std,
            err_mean=base.err_mean,
            err_std=base.err_std,
            delta_c_mean=torch.from_numpy(dc.mean(axis=(0, 1)).astype(np.float32)),
            delta_c_std=torch.from_numpy(dc.std(axis=(0, 1)).astype(np.float32)),
            phase_mean=torch.from_numpy(phase.mean(axis=(0, 1)).astype(np.float32)),
            phase_std=torch.from_numpy(phase.std(axis=(0, 1)).astype(np.float32)),
            eps=eps,
        )

    def norm_delta_c(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.delta_c_mean) / self.delta_c_std

    def denorm_delta_c(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.delta_c_std + self.delta_c_mean

    def norm_phase(self, ph_sin: torch.Tensor, ph_cos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.cat([ph_sin, ph_cos], dim=-1)
        normed = (stacked - self.phase_mean) / self.phase_std
        return normed[..., :3], normed[..., 3:]

    def norm_p_coarse(self, p_raw: torch.Tensor, delta_c_phys: torch.Tensor) -> torch.Tensor:
        """P_coarse 用与 err 相同的 per-axis 统计做归一化（粗轨迹仍在米制邻域）。"""
        return (p_raw - delta_c_phys - self.err_mean) / self.err_std


class PI_xLSTM_Tracker(nn.Module):
    """双向共享权重 xLSTM 轨迹误差预测器。

    use_input_mask=True 时启用输入掩码支持（半监督模式）：
      - 模型接受 [B, T, feat_dim * 2] 的拼接张量（特征 + 掩码位）
      - input_dim 参数仍为原始特征维数（如 12），内部自动 *2
      - 掩码位（0/1）让模型学会忽略缺失特征，不依赖填充值的语义
    use_input_mask=False（默认）时与旧版完全兼容。
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        output_dim: int = 3,
        seq_len: int = 16384,
        slstm_backend: Optional[str] = None,
        num_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_input_mask: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        backend = slstm_backend or ("cuda" if torch.cuda.is_available() else "vanilla")

        # 启用掩码时，输入维度翻倍（feat || mask）
        self.use_input_mask = use_input_mask
        effective_input_dim = input_dim * 2 if use_input_mask else input_dim

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=num_heads,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=backend,
                    num_heads=num_heads,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=seq_len,
            num_blocks=num_blocks,
            embedding_dim=hidden_dim,
            slstm_at=[1],
        )
        self.input_mapping = nn.Linear(effective_input_dim, hidden_dim)
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.dropout = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor, feat_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:         [B, T, feat_dim]  归一化特征
            feat_mask: [B, T, feat_dim]  有效掩码（0/1），use_input_mask=True 时必须传入
        """
        if self.use_input_mask:
            if feat_mask is None:
                feat_mask = torch.ones_like(x)
            x = torch.cat([x, feat_mask], dim=-1)          # [B, T, feat_dim*2]

        x_mapped = self.input_mapping(x)
        out_forward = self.xlstm_stack(x_mapped)
        x_reversed = torch.flip(x_mapped, dims=[1])
        out_backward_rev = self.xlstm_stack(x_reversed)
        out_backward = torch.flip(out_backward_rev, dims=[1])
        out_combined = torch.cat([out_forward, out_backward], dim=-1)
        return self.output_layer(self.dropout(out_combined))


class PI_xLSTM_Cascade(nn.Module):
    """双头 Bi-xLSTM：δ_c（粗修）+ δ_f（精修）。需显式 --model cascade。"""

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        output_dim: int = 3,
        seq_len: int = 16384,
        slstm_backend: Optional[str] = None,
        num_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_input_mask: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        backend = slstm_backend or ("cuda" if torch.cuda.is_available() else "vanilla")
        self.use_input_mask = use_input_mask
        effective_input_dim = input_dim * 2 if use_input_mask else input_dim
        self.hidden_dim = hidden_dim
        self._fine_in_dim = hidden_dim * 2 + 3 + 6

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=num_heads,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=backend,
                    num_heads=num_heads,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=seq_len,
            num_blocks=num_blocks,
            embedding_dim=hidden_dim,
            slstm_at=[1],
        )
        self.input_mapping = nn.Linear(effective_input_dim, hidden_dim)
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.dropout = nn.Dropout(p=dropout)
        self.head_coarse = nn.Linear(hidden_dim * 2, output_dim)
        self.head_fine = nn.Sequential(
            nn.Linear(self._fine_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.head_fine[-1].weight)
        nn.init.zeros_(self.head_fine[-1].bias)

    def _encode(self, x: torch.Tensor, feat_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.use_input_mask:
            if feat_mask is None:
                feat_mask = torch.ones_like(x)
            x = torch.cat([x, feat_mask], dim=-1)
        x_mapped = self.input_mapping(x)
        out_forward = self.xlstm_stack(x_mapped)
        x_reversed = torch.flip(x_mapped, dims=[1])
        out_backward_rev = self.xlstm_stack(x_reversed)
        out_backward = torch.flip(out_backward_rev, dims=[1])
        return torch.cat([out_forward, out_backward], dim=-1)

    def forward(
        self,
        feat_norm: torch.Tensor,
        phase_sin: torch.Tensor,
        phase_cos: torch.Tensor,
        p_raw: torch.Tensor,
        normalizer: "CascadeDataNormalizer",
        feat_mask: Optional[torch.Tensor] = None,
        detach_coarse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._encode(feat_norm, feat_mask)
        delta_c_norm = self.head_coarse(self.dropout(h))
        delta_c_phys = normalizer.denorm_delta_c(delta_c_norm)
        if detach_coarse:
            delta_c_phys = delta_c_phys.detach()
        p_coarse_norm = normalizer.norm_p_coarse(p_raw, delta_c_phys)
        ph_sin_n, ph_cos_n = normalizer.norm_phase(phase_sin, phase_cos)
        fine_in = torch.cat([h, p_coarse_norm, ph_sin_n, ph_cos_n], dim=-1)
        delta_f_norm = self.head_fine(fine_in)
        return delta_c_norm, delta_f_norm


class RadarPhysicsLoss(nn.Module):
    """支持归一化训练 + 物理一致性的复合损失（v2：半监督扩展版）。

    新增能力：
      - has_label [B]：True 的样本才参与 MSE 监督；False（实测）仅用物理约束。
      - rcm_weight [B, 3]：每个参考点的有效标志（0/1），用于对可变数量的参考点
        做加权平均，0 个有效点时 l_rcm=0。
      - 物理空间 MSE 仅在 has_label 样本上计算（detach，仅日志）。

    损失项坐标系：
      - l_mse_norm：归一化空间，无量纲，O(0.1~1)。
      - l_smooth ：归一化输出空间的差分平方，无量纲。
      - l_rcm    ：**已归一化到无量纲**（除以 err_std²），与 l_mse_norm 同坐标系，
                   λ_rcm=1.0 即两项同重；超参选择与数据集 err 量级解耦。

    接口向后兼容：can_label / rcm_weight 不传则退化为全 True / 全 1（纯仿真模式）。
    """

    def __init__(
        self,
        pos_A: torch.Tensor,
        pos_B: torch.Tensor,
        pos_C: torch.Tensor,
        normalizer: "DataNormalizer",
        lambda_smooth: float = 0.1,
        lambda_rcm: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_A", pos_A)
        self.register_buffer("pos_B", pos_B)
        self.register_buffer("pos_C", pos_C)
        self._normalizer = normalizer
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_rcm = float(lambda_rcm)

    @property
    def normalizer(self) -> "DataNormalizer":
        return self._normalizer

    def forward(
        self,
        delta_pred_norm,
        p_raw,
        p_true,
        feat,
        has_label=None,
        rcm_weight=None,
        pos_a=None,
        pos_b=None,
        pos_c=None,
    ):
        norm = self._normalizer
        B = delta_pred_norm.shape[0]
        device = delta_pred_norm.device

        # 默认值：全部有标签、全部 RCM 有效（纯仿真模式向后兼容）
        if has_label is None:
            has_label = torch.ones(B, dtype=torch.bool, device=device)
        else:
            has_label = has_label.bool()
        if rcm_weight is None:
            rcm_weight = torch.ones(B, 3, device=device)

        # 反归一化到物理量（全 batch）
        delta_pred_phys = delta_pred_norm * norm.err_std + norm.err_mean

        # ── MSE 监督（仅有标签的样本）──────────────────────────────────────
        n_labeled = has_label.sum().item()
        if n_labeled > 0:
            dp_labeled = delta_pred_norm[has_label]          # [n_lab, T, 3]
            error_true_phys = p_raw[has_label] - p_true[has_label]
            error_true_norm = (error_true_phys - norm.err_mean) / norm.err_std
            l_mse_norm = nn.functional.mse_loss(dp_labeled, error_true_norm)
            with torch.no_grad():
                l_mse_phys = nn.functional.mse_loss(
                    delta_pred_phys[has_label], error_true_phys
                )
        else:
            # 无标签批次：梯度为 0 的占位损失（不产生 NaN）
            l_mse_norm = (delta_pred_norm * 0.0).sum()
            l_mse_phys = torch.tensor(float("nan"), device=device)

        # ── 平滑约束（全 batch，无标签样本也参与）────────────────────────
        delta_diff = torch.diff(delta_pred_norm, n=1, dim=1)   # [B, T-1, 3]
        l_smooth = torch.mean(delta_diff ** 2)

        # ── RCM 物理约束（按 rcm_weight 加权平均，支持 0~3 个有效点）────
        p_fixed = p_raw - delta_pred_phys
        pA = pos_a.unsqueeze(1) if pos_a is not None else self.pos_A.view(1, 1, 3)
        pB = pos_b.unsqueeze(1) if pos_b is not None else self.pos_B.view(1, 1, 3)
        pC = pos_c.unsqueeze(1) if pos_c is not None else self.pos_C.view(1, 1, 3)
        # 安全范数：eps=1.0（1米）避免轨迹经过参考点时 sqrt 梯度趋于无穷
        # 原 eps=1e-8 在距离<0.1mm 时梯度放大 1e4 倍，是梯度爆炸的根本原因
        # 用 sqrt(sum²+eps) 代替 torch.norm，避免距离趋零时梯度 = x/||x|| → ∞ 的奇点
        dist_A_pred = torch.sqrt(((p_fixed - pA) ** 2).sum(dim=-1) + 1e-8)
        dist_B_pred = torch.sqrt(((p_fixed - pB) ** 2).sum(dim=-1) + 1e-8)
        dist_C_pred = torch.sqrt(((p_fixed - pC) ** 2).sum(dim=-1) + 1e-8)
        dist_A_obs = feat[:, :, 6]
        dist_B_obs = feat[:, :, 7]
        dist_C_obs = feat[:, :, 8]

        # 每个参考点的 per-sample MSE，再按 rcm_weight 加权求和
        def _weighted_rcm(d_pred, d_obs, w):
            # d_pred/d_obs: [B, T];  w: [B]  →  加权后的标量 loss
            per_sample = ((d_pred - d_obs) ** 2).mean(dim=1)   # [B]
            total_w = w.sum() + 1e-8
            return (per_sample * w).sum() / total_w

        w_A = rcm_weight[:, 0]
        w_B = rcm_weight[:, 1]
        w_C = rcm_weight[:, 2]
        total_valid = (w_A + w_B + w_C).sum() + 1e-8
        l_rcm_raw = (
            _weighted_rcm(dist_A_pred, dist_A_obs, w_A) * w_A.sum()
            + _weighted_rcm(dist_B_pred, dist_B_obs, w_B) * w_B.sum()
            + _weighted_rcm(dist_C_pred, dist_C_obs, w_C) * w_C.sum()
        ) / total_valid

        # 把物理空间的 RCM 残差 (m²) 归一化到与 l_mse_norm 同坐标系（无量纲）。
        # 除以 err_std² 的均值 ≈ 1.18e-3 m² → λ_rcm=1.0 即 "RCM 项与 MSE 项同重"。
        # 注意（向后不兼容）：metrics.jsonl 中 'rcm' 列由 m² 变为无量纲，量级从 ~4e-4 升到 O(0.1~1)。
        rcm_scale = (norm.err_std ** 2).mean().clamp_min(1e-12)
        l_rcm = l_rcm_raw / rcm_scale

        l_total = l_mse_norm + self.lambda_smooth * l_smooth + self.lambda_rcm * l_rcm
        return l_total, l_mse_norm, l_mse_phys, l_smooth, l_rcm


class CascadePhysicsLoss(nn.Module):
    """双头 cascade 损失：粗修 MSE/RCM + 精修 MSE/相位 + 联合平滑与总误差。"""

    _C_LIGHT = 299792458.0

    def __init__(
        self,
        pos_A: torch.Tensor,
        pos_B: torch.Tensor,
        pos_C: torch.Tensor,
        normalizer: "CascadeDataNormalizer",
        fc_hz: float = 9.5e9,
        lambda_smooth: float = 0.1,
        lambda_rcm: float = 0.0,
        lambda_c: float = 1.0,
        lambda_f: float = 0.0,
        lambda_phase: float = 0.0,
        lambda_total: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_A", pos_A)
        self.register_buffer("pos_B", pos_B)
        self.register_buffer("pos_C", pos_C)
        self._normalizer = normalizer
        self.register_buffer("wavelength", torch.tensor(self._C_LIGHT / float(fc_hz), dtype=torch.float32))
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_rcm = float(lambda_rcm)
        self.lambda_c = float(lambda_c)
        self.lambda_f = float(lambda_f)
        self.lambda_phase = float(lambda_phase)
        self.lambda_total = float(lambda_total)

    @property
    def normalizer(self) -> "CascadeDataNormalizer":
        return self._normalizer

    def forward(
        self,
        delta_c_norm: torch.Tensor,
        delta_f_norm: torch.Tensor,
        p_raw: torch.Tensor,
        p_true: torch.Tensor,
        feat: torch.Tensor,
        delta_coarse_phys: torch.Tensor,
        delta_fine_phys: torch.Tensor,
        phase_sin: torch.Tensor,
        phase_cos: torch.Tensor,
        has_label=None,
        rcm_weight=None,
        pos_a=None,
        pos_b=None,
        pos_c=None,
    ):
        norm = self._normalizer
        B = delta_c_norm.shape[0]
        device = delta_c_norm.device
        if has_label is None:
            has_label = torch.ones(B, dtype=torch.bool, device=device)
        else:
            has_label = has_label.bool()
        if rcm_weight is None:
            rcm_weight = torch.ones(B, 3, device=device)

        delta_c_phys = norm.denorm_delta_c(delta_c_norm)
        delta_f_phys = norm.denorm_err(delta_f_norm)
        delta_sum_norm = delta_c_norm + delta_f_norm

        n_labeled = has_label.sum().item()
        if n_labeled > 0:
            dc_tgt_norm = norm.norm_delta_c(delta_coarse_phys[has_label])
            l_mse_c = nn.functional.mse_loss(delta_c_norm[has_label], dc_tgt_norm)
            df_tgt_norm = norm.norm_err(delta_fine_phys[has_label])
            l_mse_f = nn.functional.mse_loss(delta_f_norm[has_label], df_tgt_norm)
            err_total_norm = norm.norm_err(p_raw[has_label] - p_true[has_label])
            l_mse_total = nn.functional.mse_loss(delta_sum_norm[has_label], err_total_norm)
            with torch.no_grad():
                l_mse_phys = nn.functional.mse_loss(
                    delta_c_phys[has_label] + delta_f_phys[has_label],
                    p_raw[has_label] - p_true[has_label],
                )
        else:
            z = (delta_c_norm * 0.0).sum()
            l_mse_c = l_mse_f = l_mse_total = z
            l_mse_phys = torch.tensor(float("nan"), device=device)

        delta_diff = torch.diff(delta_sum_norm, n=1, dim=1)
        l_smooth = torch.mean(delta_diff ** 2)

        p_coarse = p_raw - delta_c_phys
        pA = pos_a.unsqueeze(1) if pos_a is not None else self.pos_A.view(1, 1, 3)
        pB = pos_b.unsqueeze(1) if pos_b is not None else self.pos_B.view(1, 1, 3)
        pC = pos_c.unsqueeze(1) if pos_c is not None else self.pos_C.view(1, 1, 3)
        dist_A_pred = torch.sqrt(((p_coarse - pA) ** 2).sum(dim=-1) + 1e-8)
        dist_B_pred = torch.sqrt(((p_coarse - pB) ** 2).sum(dim=-1) + 1e-8)
        dist_C_pred = torch.sqrt(((p_coarse - pC) ** 2).sum(dim=-1) + 1e-8)
        dist_A_obs = feat[:, :, 6]
        dist_B_obs = feat[:, :, 7]
        dist_C_obs = feat[:, :, 8]

        def _weighted_rcm(d_pred, d_obs, w):
            per_sample = ((d_pred - d_obs) ** 2).mean(dim=1)
            total_w = w.sum() + 1e-8
            return (per_sample * w).sum() / total_w

        w_A, w_B, w_C = rcm_weight[:, 0], rcm_weight[:, 1], rcm_weight[:, 2]
        total_valid = (w_A + w_B + w_C).sum() + 1e-8
        l_rcm_raw = (
            _weighted_rcm(dist_A_pred, dist_A_obs, w_A) * w_A.sum()
            + _weighted_rcm(dist_B_pred, dist_B_obs, w_B) * w_B.sum()
            + _weighted_rcm(dist_C_pred, dist_C_obs, w_C) * w_C.sum()
        ) / total_valid
        rcm_scale = (norm.err_std ** 2).mean().clamp_min(1e-12)
        l_rcm = l_rcm_raw / rcm_scale

        p_fix = p_raw - delta_c_phys - delta_f_phys
        lam = self.wavelength
        phi_obs = torch.atan2(phase_sin, phase_cos)
        ref_positions = (pA, pB, pC)
        phase_terms = []
        for j, p_ref in enumerate(ref_positions):
            w_j = rcm_weight[:, j]
            if w_j.sum() < 1e-8:
                continue
            r_pred = torch.sqrt(((p_fix - p_ref) ** 2).sum(dim=-1) + 1e-8)
            phi_ideal = -4.0 * math.pi * r_pred / lam
            phase_err = 1.0 - torch.cos(phi_obs[:, :, j] - phi_ideal)
            per_sample = phase_err.mean(dim=1)
            phase_terms.append((per_sample * w_j).sum() / (w_j.sum() + 1e-8))
        l_phase = sum(phase_terms) / max(len(phase_terms), 1) if phase_terms else (delta_c_norm * 0.0).sum()

        l_total = (
            self.lambda_c * l_mse_c
            + self.lambda_f * l_mse_f
            + self.lambda_rcm * l_rcm
            + self.lambda_phase * l_phase
            + self.lambda_smooth * l_smooth
            + self.lambda_total * l_mse_total
        )
        return l_total, l_mse_c, l_mse_f, l_rcm, l_phase, l_smooth, l_mse_total, l_mse_phys


class PhasePhysicsLoss(nn.Module):
    """v4_phase：单头轨迹修正 + 相对相位监督（Phase_rel 与 R_ref 成对）。"""

    _C_LIGHT = 299792458.0

    def __init__(
        self,
        pos_A: torch.Tensor,
        pos_B: torch.Tensor,
        pos_C: torch.Tensor,
        normalizer: "DataNormalizer",
        fc_hz: float = 9.5e9,
        lambda_smooth: float = 0.1,
        lambda_phase: float = 1.0,
        lambda_mse: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_A", pos_A)
        self.register_buffer("pos_B", pos_B)
        self.register_buffer("pos_C", pos_C)
        self._normalizer = normalizer
        self.register_buffer("wavelength", torch.tensor(self._C_LIGHT / float(fc_hz), dtype=torch.float32))
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_phase = float(lambda_phase)
        self.lambda_mse = float(lambda_mse)

    @property
    def normalizer(self) -> "DataNormalizer":
        return self._normalizer

    def forward(
        self,
        delta_pred_norm: torch.Tensor,
        p_raw: torch.Tensor,
        p_true: torch.Tensor,
        phase_rel: torch.Tensor,
        r_ref: torch.Tensor,
        has_label=None,
        rcm_weight=None,
        pos_a=None,
        pos_b=None,
        pos_c=None,
    ):
        norm = self._normalizer
        B = delta_pred_norm.shape[0]
        device = delta_pred_norm.device
        if has_label is None:
            has_label = torch.ones(B, dtype=torch.bool, device=device)
        else:
            has_label = has_label.bool()
        if rcm_weight is None:
            rcm_weight = torch.ones(B, 3, device=device)

        delta_pred_phys = norm.denorm_err(delta_pred_norm)

        n_labeled = has_label.sum().item()
        if n_labeled > 0 and self.lambda_mse > 0:
            dp_labeled = delta_pred_norm[has_label]
            error_true_phys = p_raw[has_label] - p_true[has_label]
            error_true_norm = (error_true_phys - norm.err_mean) / norm.err_std
            l_mse_norm = nn.functional.mse_loss(dp_labeled, error_true_norm)
            with torch.no_grad():
                l_mse_phys = nn.functional.mse_loss(
                    delta_pred_phys[has_label], error_true_phys
                )
        else:
            l_mse_norm = (delta_pred_norm * 0.0).sum()
            l_mse_phys = torch.tensor(float("nan"), device=device)

        delta_diff = torch.diff(delta_pred_norm, n=1, dim=1)
        l_smooth = torch.mean(delta_diff ** 2)

        p_fixed = p_raw - delta_pred_phys
        lam = self.wavelength
        pA = pos_a.unsqueeze(1) if pos_a is not None else self.pos_A.view(1, 1, 3)
        pB = pos_b.unsqueeze(1) if pos_b is not None else self.pos_B.view(1, 1, 3)
        pC = pos_c.unsqueeze(1) if pos_c is not None else self.pos_C.view(1, 1, 3)
        ref_positions = (pA, pB, pC)
        r_ref_exp = r_ref.unsqueeze(1)
        phase_terms = []
        for j, p_ref in enumerate(ref_positions):
            w_j = rcm_weight[:, j]
            if w_j.sum() < 1e-8:
                continue
            r_pred = torch.sqrt(((p_fixed - p_ref) ** 2).sum(dim=-1) + 1e-8)
            phi_ideal = -4.0 * math.pi * (r_pred - r_ref_exp[:, :, j]) / lam
            phi_obs = phase_rel[:, :, j]
            dphi = phi_obs - phi_ideal
            dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
            phase_err = 1.0 - torch.cos(dphi)
            per_sample = phase_err.mean(dim=1)
            phase_terms.append((per_sample * w_j).sum() / (w_j.sum() + 1e-8))
        l_phase = sum(phase_terms) / max(len(phase_terms), 1) if phase_terms else (delta_pred_norm * 0.0).sum()

        l_rcm = (delta_pred_norm * 0.0).sum()
        l_total = (
            self.lambda_mse * l_mse_norm
            + self.lambda_phase * l_phase
            + self.lambda_smooth * l_smooth
        )
        return l_total, l_mse_norm, l_mse_phys, l_smooth, l_rcm, l_phase


class _NullContext:
    """autocast 关闭时使用的空上下文管理器。"""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def compute_lambda_warmup(
    epoch_idx: int,
    *,
    warmup_start: int,
    warmup_end: int,
    max_lambda: float,
) -> float:
    """课程学习：epoch_idx 是 0-based 的当前 epoch 序号。

    - epoch_idx <  warmup_start：返回 0.0（先纯 MSE 学到合理输出再上 RCM）
    - warmup_start ≤ epoch_idx < warmup_end：从 0 线性升到 max_lambda
    - epoch_idx ≥ warmup_end：保持 max_lambda
    """
    if epoch_idx < warmup_start:
        return 0.0
    if epoch_idx >= warmup_end:
        return float(max_lambda)
    span = max(1, warmup_end - warmup_start)
    return float(max_lambda) * (epoch_idx - warmup_start) / span


def compute_lambda_rcm(epoch_idx: int, *, warmup_start: int, warmup_end: int, max_lambda: float) -> float:
    return compute_lambda_warmup(
        epoch_idx, warmup_start=warmup_start, warmup_end=warmup_end, max_lambda=max_lambda
    )


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN
        return "—"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m}m{s}s"
    if m > 0:
        return f"{m}m{s}s"
    return f"{s}s"


def build_train_meta(
    *,
    mat_path: Path,
    seq_len: int,
    input_dim: int,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    slstm_backend: str,
    num_blocks: int,
    num_heads: int,
    lambda_smooth: float,
    lambda_rcm_max: float,
    rcm_warmup_start: int,
    rcm_warmup_end: int,
    scheduler_eta_min: float,
    scheduler_tmax: int,
    early_stop_patience: int,
    val_ratio: float,
    val_seed: int,
    real_mat_path: str = "",
    real_mix_ratio: float = 0.0,
    use_input_mask: bool = False,
    random_ref_mask: bool = True,
    seq_stride: int = 1,
) -> dict[str, Any]:
    return {
        "mat_path": str(mat_path),
        "seq_len": seq_len,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "slstm_backend": slstm_backend,
        "num_blocks": num_blocks,
        "num_heads": num_heads,
        "lambda_smooth": lambda_smooth,
        "lambda_rcm_max": lambda_rcm_max,
        "rcm_warmup_start": rcm_warmup_start,
        "rcm_warmup_end": rcm_warmup_end,
        "scheduler": "CosineAnnealingLR",
        "scheduler_eta_min": scheduler_eta_min,
        "scheduler_tmax": scheduler_tmax,
        "early_stop_patience": early_stop_patience,
        "val_ratio": val_ratio,
        "val_seed": val_seed,
        # 半监督扩展参数
        "real_mat_path": real_mat_path,
        "real_mix_ratio": real_mix_ratio,
        "use_input_mask": use_input_mask,
        "random_ref_mask": random_ref_mask,
        # 序列下采样
        "seq_stride": seq_stride,
        # 架构标识
        "bidirectional": True,
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    normalizer: "DataNormalizer",
    next_epoch_idx: int,
    train_meta: dict[str, Any],
    run_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        # normalizer 的 buffer（feat/err 的 mean/std）必须随权重一起持久化，
        # 否则推理端无法把网络输出反归一化到米制。
        "normalizer_state_dict": normalizer.state_dict(),
        "next_epoch_idx": next_epoch_idx,
        "train_meta": train_meta,
        "run_dir": str(run_dir.resolve()),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    normalizer: "DataNormalizer",
    expected_meta: dict[str, Any],
) -> tuple[int, Path | None]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    meta = ckpt.get("train_meta", {})
    keys = ("mat_path", "seq_len", "input_dim", "hidden_dim", "batch_size", "num_blocks", "num_heads")
    for k in keys:
        if meta.get(k) != expected_meta.get(k):
            raise ValueError(
                f"Checkpoint '{path}' 与当前训练设置不一致: {k}={meta.get(k)!r} "
                f"(checkpoint) vs {expected_meta.get(k)!r} (当前)。请使用匹配的 --mat_path / 维度 / batch。"
            )
    if not meta.get("bidirectional", False):
        raise ValueError(
            f"Checkpoint '{path}' 来自单向 PI_xLSTM_Tracker（旧架构），与当前双向模型不兼容；"
            f"请重新训练或使用旧版本脚本恢复。"
        )
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    norm_sd = ckpt.get("normalizer_state_dict")
    if not isinstance(norm_sd, dict):
        raise ValueError(
            f"Checkpoint '{path}' 缺少 normalizer_state_dict；该 checkpoint 来自归一化前的训练流程，"
            f"请用本脚本重新训练。"
        )
    normalizer.load_state_dict(norm_sd)
    # 一致性校验：和当前训练数据拟合的统计量差异 > 1e-3 时报错（数据集变了）
    fitted_state: dict[str, torch.Tensor] = {k: v.detach().cpu() for k, v in normalizer.state_dict().items()}
    for k, v in norm_sd.items():
        if k not in fitted_state:
            continue
        diff = (fitted_state[k] - v.detach().cpu()).abs().max().item()
        if diff > 1e-3:
            print(
                f"    [warn] normalizer.{k} 与 checkpoint 不一致（max abs diff={diff:.3e}），"
                f"已使用 checkpoint 中的统计量；若数据集已替换请重新训练。"
            )
    next_epoch_idx = int(ckpt["next_epoch_idx"])
    run_dir_raw = ckpt.get("run_dir")
    run_dir: Path | None = Path(run_dir_raw).resolve() if run_dir_raw else None
    return next_epoch_idx, run_dir


def resolve_run_dir_from_checkpoint(resume_path: Path) -> Path:
    """若 checkpoint 内无 run_dir（旧文件），从路径推断本次运行根目录。"""
    if resume_path.parent.name == "weights":
        return resume_path.parent.parent.resolve()
    return resume_path.parent.resolve()


def namespace_to_json_dict(ns: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in vars(ns).items():
        if isinstance(v, Path):
            out[k] = str(v.resolve())
        else:
            out[k] = v
    return out


def write_run_config(
    run_dir: Path,
    *,
    args_ns: argparse.Namespace,
    train_meta: dict[str, Any],
    device: torch.device,
    seq_len: int,
    resume_from: str,
) -> None:
    payload = {
        "run_dir": str(run_dir.resolve()),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "seq_len_from_mat": seq_len,
        "resume_from": resume_from or None,
        "args": namespace_to_json_dict(args_ns),
        "train_meta": train_meta,
    }
    path = run_dir / "config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")


def append_metric_jsonl(run_dir: Path, record: dict[str, Any]) -> None:
    path = run_dir / "metrics.jsonl"
    with path.open("a", encoding="utf8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_metrics_history(run_dir: Path) -> dict[str, list[float]]:
    path = run_dir / "metrics.jsonl"
    keys = (
        "epoch",
        "total_loss",
        "mse_norm",
        "mse_phys",
        "smooth",
        "rcm",
        "lambda_rcm",
        "grad_norm",
        "lr",
        "epoch_time_sec",
        "val_total_loss",
        "val_mse_norm",
        "val_pos_rmse",
    )
    hist: dict[str, list[float]] = {k: [] for k in keys}
    if not path.is_file():
        return hist
    with path.open(encoding="utf8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for k in keys:
                if k in row:
                    hist[k].append(float(row[k]))
                elif k == "mse_norm" and "mse" in row:
                    hist[k].append(float(row["mse"]))
                else:
                    hist[k].append(float("nan"))
    return hist


def save_training_figures(history: dict[str, list[float]], viz_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    [viz] 未安装 matplotlib，跳过曲线图保存（可 pip install matplotlib）。")
        return
    viz_dir.mkdir(parents=True, exist_ok=True)
    epochs = history["epoch"]
    if not epochs:
        return
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    ax0, ax1, ax2 = axes

    ax0.plot(epochs, history["total_loss"], label="train total", color="C0")
    ax0.plot(epochs, history["mse_norm"], label="train mse(norm)", color="C1", alpha=0.85)
    ax0.plot(epochs, history["rcm"], label="rcm", color="C2", alpha=0.85)
    ax0.plot(epochs, history["smooth"], label="smooth", color="C4", alpha=0.7)
    # val 曲线（虚线）：仅在存在有效值时绘制
    val_total = history.get("val_total_loss", [])
    val_mse = history.get("val_mse_norm", [])
    import math
    if val_total and not all(math.isnan(v) for v in val_total):
        ax0.plot(epochs, val_total, label="val total", color="C0", linestyle="--", alpha=0.9)
    if val_mse and not all(math.isnan(v) for v in val_mse):
        ax0.plot(epochs, val_mse, label="val mse(norm)", color="C1", linestyle="--", alpha=0.75)
    ax0.set_ylabel("loss (norm/used)")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.3)
    ax0.set_title("PI-xLSTM training losses")

    ax1.plot(epochs, history["mse_phys"], label="train mse(physical)", color="C5")
    val_rmse = history.get("val_pos_rmse", [])
    if val_rmse and not all(math.isnan(v) for v in val_rmse):
        ax1.plot(epochs, val_rmse, label="val pos RMSE [m]", color="C5", linestyle="--", alpha=0.85)
    ax1.set_ylabel("mse [m^2] / rmse [m]")
    ax1.set_yscale("log")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3, which="both")

    ax2.plot(epochs, history["lr"], label="lr", color="C3")
    ax2.set_ylabel("learning rate")
    ax2.grid(True, alpha=0.3)
    ax2_lambda = ax2.twinx()
    ax2_lambda.plot(epochs, history["lambda_rcm"], label="λ_rcm", color="C6", linestyle="--")
    ax2_lambda.set_ylabel("λ_rcm")
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_lambda.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)
    ax2.set_xlabel("epoch")

    fig.tight_layout()
    out_path = viz_dir / "training_curves.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def print_training_params_report(
    *,
    device: torch.device,
    train_meta: dict[str, Any],
    completed_epochs: int,
    total_epochs: int,
    epoch_times_sec: list[float],
    train_wall_elapsed_sec: float,
) -> None:
    remaining = max(0, total_epochs - completed_epochs)
    window = epoch_times_sec[-min(10, len(epoch_times_sec)) :]
    avg_epoch = sum(window) / len(window) if window else float("nan")
    eta_sec = avg_epoch * remaining if window and remaining else float("nan")
    print("    ---------- 训练参数快照 ----------")
    print(f"    device: {device} | mat: {train_meta['mat_path']}")
    print(
        f"    seq_len={train_meta['seq_len']} | batch_size={train_meta['batch_size']} | "
        f"hidden_dim={train_meta['hidden_dim']} | blocks={train_meta['num_blocks']} | "
        f"heads={train_meta['num_heads']} | sLSTM backend={train_meta['slstm_backend']}"
    )
    sched_name = train_meta.get("scheduler", "CosineAnnealingLR")
    if "Cosine" in sched_name:
        sched_info = (
            f"CosineAnnealingLR(T_max={train_meta.get('scheduler_tmax', 'auto')},"
            f" eta_min={train_meta.get('scheduler_eta_min', 1e-6):.0e})"
        )
    else:
        sched_info = f"StepLR(step={train_meta.get('scheduler_step_size','?')}, gamma={train_meta.get('scheduler_gamma','?')})"
    print(
        f"    lr(initial)={train_meta['lr']:.6g} | weight_decay={train_meta['weight_decay']} | "
        f"{sched_info}"
    )
    print(
        f"    loss: lambda_smooth={train_meta['lambda_smooth']} | "
        f"lambda_rcm_max={train_meta['lambda_rcm_max']} | "
        f"warmup=[{train_meta['rcm_warmup_start']},{train_meta['rcm_warmup_end']})"
    )
    print(
        f"    进度: {completed_epochs}/{total_epochs} epoch | 本轮以来已用墙钟: {format_duration(train_wall_elapsed_sec)}"
    )
    print(
        f"    近 {len(window)} epoch 平均耗时: {format_duration(avg_epoch)} | "
        f"预计剩余: {format_duration(eta_sec)}"
    )
    print("    --------------------------------")


def main() -> None:
    parser = argparse.ArgumentParser(description="PI-xLSTM SAR trajectory training.")
    parser.add_argument(
        "--mat_path",
        type=str,
        default="PPTR_Dataset/PPTR_TrainDataset_1000.mat",
        help="Path to v7.3 .mat with Feat_All, P_raw, P_true",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="梯度累积步数，effective_batch = batch_size * grad_accum_steps * world_size")
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="总训练轮数。课程学习需要足够长的尾段让 RCM 充分发挥（默认 100）。",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument(
        "--run_root",
        type=str,
        default=str(_script_dir / "run"),
        help="Root folder; each训练新建 run/<timestamp>_pid…/",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="可选标签，出现在运行文件夹名称中",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="覆盖权重保存目录；默认使用 <run_dir>/weights",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to .pt checkpoint to continue training (next_epoch_idx restored)",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=50,
        help="Save checkpoint every N completed epochs (also saves latest.pt)",
    )
    parser.add_argument(
        "--param_report_every",
        type=int,
        default=10,
        help="Print hyperparameter snapshot + ETA every N epochs",
    )
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout 概率（默认 0.1）。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xlstm",
        choices=["xlstm", "phase", "cascade", "bilstm", "bigru", "transformer", "tcn"],
        help=(
            "模型主干选择（默认 xlstm）：\n"
            "  xlstm       — PI-xLSTM 单头（本文方法）\n"
            "  phase       — v4_phase 单头 + Phase_rel/R_ref 相位损失（需 v4_phase mat）\n"
            "  cascade     — v4 双头粗修+精修（需旧 v4 mat，显式指定）\n"
            "  bilstm      — 双向 LSTM 基线\n"
            "  bigru       — 双向 GRU 基线\n"
            "  transformer — 双向 Transformer Encoder 基线\n"
            "  tcn         — 空洞因果 TCN 基线\n"
            "非 xlstm/cascade 时仍使用相同的数据管线和物理损失函数，仅替换模型主干。"
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader 进程数。数据已全量驻留内存，2 个 worker 通常够用；设 0 关闭多进程。",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="启用 bfloat16 自动混合精度（CUDA 上默认开），加速外层 Linear / loss。",
    )
    parser.add_argument(
        "--no_amp",
        dest="amp",
        action="store_false",
        help="关闭 AMP（CPU 或调试时用）。",
    )
    parser.add_argument(
        "--scheduler_eta_min",
        type=float,
        default=1e-6,
        help="CosineAnnealingLR 的最小学习率下限（默认 1e-6）。",
    )
    parser.add_argument(
        "--scheduler_tmax",
        type=int,
        default=0,
        help="CosineAnnealingLR 的 T_max（epoch 数）；设为 0 则自动取 epochs//2，"
             "即前半段衰减到 eta_min 后维持，避免后期过拟合时 LR 仍偏高（默认 0）。",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="早停耐心轮数：val_pos_rmse（米）连续 N 轮不改善则停止训练；设为 0 关闭早停（默认 0）。"
             "判据用物理位置 RMSE 而非 val_total，避免 RCM 课程学习 warmup 期间的乘子膨胀误警。",
    )
    parser.add_argument(
        "--shutdown_on_finish",
        action="store_true",
        help="训练正常结束后（含早停触发）请求关机；仅 rank0/单卡主进程执行，需 root 或 shutdown 权限。",
    )
    parser.add_argument(
        "--shutdown_delay_sec",
        type=int,
        default=60,
        help="与 --shutdown_on_finish 配合：执行 shutdown 前等待的秒数（默认 60，便于取消）；0 为立即。",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="从数据集中划分验证集的比例（默认 0.1，即 10%%）；设为 0 可关闭验证集。",
    )
    parser.add_argument(
        "--val_seed",
        type=int,
        default=42,
        help="训练/验证集划分的随机种子（默认 42，保持每次运行一致）。",
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--lambda_smooth",
        type=float,
        default=2.0,
        help="P_fix 加速度平滑惩罚权重（物理空间，默认 2.0；原 5.0 过大会放大差分梯度）。",
    )
    parser.add_argument(
        "--lambda_mse",
        type=float,
        default=0.0,
        help="phase 模式：归一化空间 MSE，监督 δ≈(P_raw-P_true)（仿真有标签时建议 1.0；"
             "与相位项联用时宜 0.3~1.0）。",
    )
    parser.add_argument(
        "--lambda_rcm_max",
        type=float,
        default=1.0,
        help="课程学习目标值：RCM 损失权重在 warmup 结束后保持的水平（默认 1.0）。"
             "RCM 已归一化到与 mse_norm 同坐标系，λ=1.0 即 'RCM 与 MSE 同重'；"
             "若需让 RCM 主导优化（推理仅靠物理一致性时建议），可设 2.0~5.0。",
    )
    parser.add_argument(
        "--rcm_warmup_start",
        type=int,
        default=0,
        help="0-based epoch 索引；之前 lambda_rcm=0（默认 0，即从第 1 epoch 开始引入 RCM）。",
    )
    parser.add_argument(
        "--rcm_warmup_end",
        type=int,
        default=80,
        help="0-based epoch 索引；从 warmup_start 到该值线性升到 lambda_rcm_max（默认 80；延迟接入避免过拟合阶段梯度爆炸）。",
    )
    parser.add_argument(
        "--lambda_rcm",
        type=float,
        default=None,
        help="[DEPRECATED] 旧参数，等价于 --lambda_rcm_max（若同时给出，--lambda_rcm_max 优先）。",
    )
    # ── 半监督扩展参数 ──────────────────────────────────────────────────────
    parser.add_argument(
        "--real_mat_path",
        type=str,
        default="",
        help="实测数据 .mat 文件路径（可选）。留空则仅使用仿真数据（纯监督模式）。"
             "文件格式：P_raw [N,T,3]，可选 Pos_Ref [3,3,N] 和 RCM_obs [N,T,3]（NaN 填充缺失点）。",
    )
    parser.add_argument(
        "--real_mix_ratio",
        type=float,
        default=0.2,
        help="混合训练时实测样本占每 epoch 采样量的比例（0~1，默认 0.2）。"
             "仅在 --real_mat_path 有效时生效。",
    )
    parser.add_argument(
        "--use_input_mask",
        action="store_true",
        default=False,
        help="启用输入掩码模式：将 12-bit 有效掩码与特征拼接为 24-dim 输入，"
             "让模型显式区分缺失特征（仅在混合半监督训练时推荐开启）。",
    )
    parser.add_argument(
        "--random_ref_mask",
        action="store_true",
        default=True,
        help="训练时随机丢弃参考点（默认开启）：以 40%%/30%%/30%% 的概率随机保留 1/2/3 个参考点，"
             "使模型适应推理时仅有 1 个参考点的场景。",
    )
    parser.add_argument(
        "--no_random_ref_mask",
        dest="random_ref_mask",
        action="store_false",
        help="关闭随机参考点丢弃（调试或基线对比时使用）。",
    )
    parser.add_argument(
        "--seq_stride",
        type=int,
        default=1,
        help="时序下采样步长（默认 1=不下采样）。"
             "设为 2 将 T=4096 降为 T=2048，mLSTM C 矩阵从 S²=16M 降为 4M，"
             "显存减少 4 倍，是 T=4096 数据集 OOM 的首选修复方案。",
    )
    # ── cascade（v4 双头）课程学习参数 ───────────────────────────────────────
    parser.add_argument("--lambda_c_max", type=float, default=1.0)
    parser.add_argument("--c_warmup_start", type=int, default=0)
    parser.add_argument("--c_warmup_end", type=int, default=0, help="0 表示从第 1 epoch 即满权重")
    parser.add_argument("--lambda_f_max", type=float, default=1.0)
    parser.add_argument("--f_warmup_start", type=int, default=40)
    parser.add_argument("--f_warmup_end", type=int, default=100)
    parser.add_argument("--lambda_phase_max", type=float, default=1.0)
    parser.add_argument("--phase_warmup_start", type=int, default=40)
    parser.add_argument("--phase_warmup_end", type=int, default=100)
    parser.add_argument("--lambda_total_max", type=float, default=0.3)
    parser.add_argument("--total_warmup_start", type=int, default=100)
    parser.add_argument("--total_warmup_end", type=int, default=160)
    parser.add_argument("--freeze_fine_until_epoch", type=int, default=40)
    parser.add_argument("--detach_coarse_until_epoch", type=int, default=60)
    args = parser.parse_args()

    if args.lambda_rcm is not None:
        # 向后兼容：用户传了旧参数则覆盖默认 lambda_rcm_max（除非他们显式给了新参数）。
        # 这里我们采用 "旧参数仅在新参数仍为默认时生效" 的策略，但 argparse 无法区分
        # "默认 0.5" 和 "用户显式 0.5"，所以打印一行 deprecation 提示，并把值同步过去。
        print(
            f"    [deprecation] --lambda_rcm 已废弃，请改用 --lambda_rcm_max；"
            f"本次按 {args.lambda_rcm} 覆写 lambda_rcm_max。"
        )
        args.lambda_rcm_max = float(args.lambda_rcm)
    if args.rcm_warmup_start < 0 or args.rcm_warmup_end < args.rcm_warmup_start:
        raise ValueError(
            f"非法 warmup 区间：[{args.rcm_warmup_start},{args.rcm_warmup_end})。"
            f"要求 0 ≤ rcm_warmup_start ≤ rcm_warmup_end。"
        )

    mat_path = Path(args.mat_path).expanduser().resolve()
    if not mat_path.is_file():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    # ── DDP 初始化 ─────────────────────────────────────────────────────────
    # 由 torchrun 注入 LOCAL_RANK 环境变量时自动启用 DDP；单进程启动时退化为普通模式。
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_ddp = local_rank >= 0
    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = resolve_device()
        rank = 0
        world_size = 1
    is_main = rank == 0  # 只有主进程负责 I/O（打印、保存权重、写指标等）

    # 性能开关：matmul 走 TF32（4080 Super 等 Ampere+ 显著加速），cudnn benchmark
    # 让 conv1d 自动挑最快算法。这里集中开启，避免散落到处。
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Infer seq_len and input_dim from file
    seq_stride = max(1, int(args.seq_stride))
    with h5py.File(mat_path, "r") as f:
        feat_nt9 = _h5_to_ntd(np.array(f["Feat_All"]), feat_dims=(6, 9, 12), name="Feat_All")
        if feat_nt9.ndim != 3 or feat_nt9.shape[-1] not in (6, 9, 12):
            raise ValueError(
                f"Feat_All must be [N,T,6|9|12], got {feat_nt9.shape}"
            )
        _, seq_len_raw, input_dim_f = feat_nt9.shape
    # 自动 stride：T > 2048 且用户未显式设置时，自动降为 2 避免 mLSTM OOM
    # （mLSTM C 矩阵 [B,NH,S,S] 在 S=4096 时单次需要 B×4×4096²×2B = B×1GiB 显存）
    if seq_stride == 1 and seq_len_raw > 2048:
        seq_stride = seq_len_raw // 2048
        if is_main:
            print(f"    [自动 seq_stride={seq_stride}] 检测到 T={seq_len_raw} > 2048，"
                  f"自动下采样为 T={seq_len_raw // seq_stride} 以防止 mLSTM OOM。"
                  f"（可用 --seq_stride 1 强制关闭）")
    # 下采样后的有效序列长度（供模型 context_length 使用）
    seq_len_f = seq_len_raw // seq_stride
    if is_main and seq_stride > 1:
        print(f"    [seq_stride={seq_stride}] 原始 T={seq_len_raw} → 有效 T={seq_len_f}"
              f"（mLSTM C矩阵 {seq_len_raw}²={seq_len_raw**2//10**6}M → {seq_len_f}²={seq_len_f**2//10**6}M 元素）")

    slstm_backend = "cuda" if device.type == "cuda" else "vanilla"
    real_mat_path_str = args.real_mat_path.strip() if args.real_mat_path else ""
    train_meta = build_train_meta(
        mat_path=mat_path,
        seq_len=int(seq_len_f),
        input_dim=int(input_dim_f),
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        slstm_backend=slstm_backend,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        lambda_smooth=args.lambda_smooth,
        lambda_rcm_max=args.lambda_rcm_max,
        rcm_warmup_start=args.rcm_warmup_start,
        rcm_warmup_end=args.rcm_warmup_end,
        scheduler_eta_min=args.scheduler_eta_min,
        scheduler_tmax=args.scheduler_tmax,
        early_stop_patience=args.early_stop_patience,
        val_ratio=args.val_ratio,
        val_seed=args.val_seed,
        real_mat_path=real_mat_path_str,
        real_mix_ratio=float(args.real_mix_ratio),
        use_input_mask=bool(args.use_input_mask),
        random_ref_mask=bool(args.random_ref_mask),
        seq_stride=seq_stride,
    )

    resume_path = Path(args.resume).expanduser().resolve() if args.resume.strip() else None
    run_dir: Path | None = None
    _run_dir_str = ""
    if resume_path is None:
        if is_main:
            run_root = Path(args.run_root).expanduser().resolve()
            run_root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            name_tag = f"_{args.run_name.strip()}" if args.run_name.strip() else ""
            run_dir = run_root / f"{stamp}{name_tag}_pid{os.getpid()}"
            run_dir.mkdir(parents=False)
            (run_dir / "viz").mkdir(parents=True)
            (run_dir / "weights").mkdir(parents=True)
            write_run_config(
                run_dir,
                args_ns=args,
                train_meta=train_meta,
                device=device,
                seq_len=int(seq_len_f),
                resume_from="",
            )
            _run_dir_str = str(run_dir)
        # 广播 run_dir 路径给所有进程（非 DDP 时 broadcast 是空操作）
        if is_ddp:
            _bcast = [_run_dir_str]
            dist.broadcast_object_list(_bcast, src=0)
            _run_dir_str = _bcast[0]
        run_dir = Path(_run_dir_str)

    if is_main:
        ddp_info = f" | DDP world_size={world_size}" if is_ddp else ""
        print("=====================================")
        print("    启动 PI-xLSTM 轨迹重建训练系统")
        print(f"    当前运行设备: {device}{ddp_info}")
        print(f"    seq_len (from mat): {seq_len_f}")
        if run_dir is not None:
            print(f"    本次运行目录 run_dir: {run_dir}")
        if resume_path is not None:
            print(f"    将继续训练，读取: {resume_path}")
        print("=====================================")

    # ── 仿真数据集加载 + 训练/验证集划分 ─────────────────────────────────
    from torch.utils.data import ConcatDataset, WeightedRandomSampler

    sim_dataset = SARDataset(str(mat_path), seq_stride=seq_stride)
    _model_name = args.model.lower().strip()
    is_cascade = _model_name == "cascade"
    is_phase = _model_name == "phase"
    if is_cascade and not sim_dataset.is_v4_legacy:
        raise ValueError(
            "--model cascade 需要旧版 v4 mat（含 Delta_coarse / Phase_sin / Phase_cos）；"
            f"当前文件缺少这些字段: {mat_path}"
        )
    if is_phase and not sim_dataset.is_v4_phase:
        raise ValueError(
            "--model phase 需要 v4_phase mat（含 Phase_rel / R_ref）；"
            f"当前文件缺少这些字段: {mat_path}"
        )
    if (is_cascade or is_phase) and real_mat_path_str:
        raise ValueError("cascade / phase 模式暂不支持 --real_mat_path 半监督混合训练。")
    n_total = len(sim_dataset)

    val_ratio = float(args.val_ratio)
    n_val = max(1, int(n_total * val_ratio)) if val_ratio > 0 else 0
    n_train = n_total - n_val
    rng_split = np.random.default_rng(int(args.val_seed))
    all_idx = rng_split.permutation(n_total).astype(np.int64)
    val_idx   = all_idx[:n_val]
    train_idx = all_idx[n_val:]
    _sim_train_raw   = Subset(sim_dataset, train_idx.tolist())
    val_subset       = Subset(sim_dataset, val_idx.tolist()) if n_val > 0 else None

    # 训练集：按需套上随机参考点丢弃包装（推理适配）
    # 验证集不做随机丢弃，始终用完整 3 点保证评估一致性
    if args.random_ref_mask:
        sim_train_subset = RefPointMaskedDataset(_sim_train_raw, p_1pt=0.40, p_2pt=0.30)
    else:
        sim_train_subset = _sim_train_raw

    if is_main:
        mask_str = "ON（p_1pt=40%, p_2pt=30%, p_3pt=30%）" if args.random_ref_mask else "OFF"
        print(f"    [仿真数据划分] 总样本={n_total}  训练={n_train}  验证={n_val}  "
              f"（val_ratio={val_ratio:.2f}, seed={args.val_seed}）")
        print(f"    [random_ref_mask] {mask_str}")

    # ── 可选：加载实测数据（半监督扩展）──────────────────────────────────
    # 归一化统计量始终只在仿真训练集上拟合（不受实测数据影响）。
    # 实测数据只参与训练，不进入验证集。
    real_dataset: Optional[RealSARDataset] = None
    if real_mat_path_str:
        real_mat_path = Path(real_mat_path_str).expanduser().resolve()
        if not real_mat_path.is_file():
            raise FileNotFoundError(f"实测数据文件未找到: {real_mat_path}")
        if is_main:
            print(f"    [半监督] 加载实测数据: {real_mat_path}")
        real_dataset = RealSARDataset(str(real_mat_path), seq_stride=seq_stride)
        if is_main:
            print(f"    [半监督] 实测样本数={len(real_dataset)}，"
                  f"混合比例 real_mix_ratio={args.real_mix_ratio:.2f}")

    # ── DataLoader 构建 ───────────────────────────────────────────────────
    pin_mem = bool(torch.cuda.is_available())
    n_workers = int(args.num_workers)

    if real_dataset is not None and len(real_dataset) > 0:
        # 半监督混合：仿真（有标签）+ 实测（无标签）
        # 使用 WeightedRandomSampler 控制实测占比，按 real_mix_ratio 定权重
        mix_ratio = float(args.real_mix_ratio)
        mix_ratio = max(0.0, min(mix_ratio, 1.0))
        n_sim = len(sim_train_subset)
        n_real = len(real_dataset)
        # 目标：在 epoch 内 mix_ratio 比例的 batch 来自实测数据
        # 每次采样总量 = n_sim（保持与纯仿真模式同规模）
        w_sim  = (1.0 - mix_ratio) / max(n_sim, 1)
        w_real = mix_ratio / max(n_real, 1)
        sample_weights = [w_sim] * n_sim + [w_real] * n_real
        train_combined: ConcatDataset = ConcatDataset([sim_train_subset, real_dataset])
        wrs = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=n_sim,      # 每 epoch 采样 n_sim 个样本
            replacement=True,
        )
        train_sampler = None        # DDP 与 WeightedRandomSampler 暂不同时支持
        dataloader = DataLoader(
            train_combined,
            batch_size=args.batch_size,
            sampler=wrs,
            num_workers=n_workers,
            pin_memory=pin_mem,
            persistent_workers=(n_workers > 0),
            drop_last=True,
        )
        if is_ddp and is_main:
            print("    [警告] 混合半监督训练不支持 DDP + WeightedRandomSampler 同时使用；"
                  "已退化为单进程 WeightedRandomSampler 模式。")
    elif is_ddp:
        train_sampler: Optional[DistributedSampler] = DistributedSampler(
            sim_train_subset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        dataloader = DataLoader(
            sim_train_subset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=n_workers,
            pin_memory=pin_mem,
            persistent_workers=(n_workers > 0),
        )
    else:
        train_sampler = None
        dataloader = DataLoader(
            sim_train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=pin_mem,
            persistent_workers=(n_workers > 0),
            drop_last=True,
        )

    # 验证集 DataLoader（只在 rank 0 上运行，无需 DDP sampler）
    val_loader: Optional[DataLoader] = None
    if is_main and val_subset is not None:
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_mem,
        )

    # Z-Score 归一化：只在仿真训练集上拟合统计量；统计量随 checkpoint 持久化。
    if is_main:
        if is_cascade:
            tag = "feat/err/delta_c/phase"
        elif is_phase:
            tag = "feat/error (phase 监督在物理弧度空间)"
        else:
            tag = "feat/error"
        print(f"正在统计仿真训练集 {tag} 的 mean / std ...")
    if is_cascade:
        normalizer = CascadeDataNormalizer.fit_cascade(sim_dataset, indices=train_idx).to(device)
        fc_hz = float(sim_dataset.fc)
    else:
        normalizer = DataNormalizer.fit(sim_dataset, indices=train_idx).to(device)
        fc_hz = float(sim_dataset.fc) if is_phase else 9.5e9
    if is_main:
        print(f"    [normalizer] {normalizer.summary()}")

    pos_A = torch.tensor([387.0, 25.0, 0.0], dtype=torch.float32)
    pos_B = torch.tensor([417.0, 56.0, 0.0], dtype=torch.float32)
    pos_C = torch.tensor([403.0, 78.0, 0.0], dtype=torch.float32)

    _model_kwargs = dict(
        input_dim=int(input_dim_f),
        hidden_dim=args.hidden_dim,
        output_dim=3,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_input_mask=bool(args.use_input_mask),
    )
    if _model_name == "xlstm":
        model = PI_xLSTM_Tracker(
            seq_len=int(seq_len_f),
            slstm_backend=slstm_backend,
            **_model_kwargs,
        ).to(device)
    elif is_cascade:
        model = PI_xLSTM_Cascade(
            seq_len=int(seq_len_f),
            slstm_backend=slstm_backend,
            **_model_kwargs,
        ).to(device)
    elif is_phase:
        model = PI_xLSTM_Tracker(
            seq_len=int(seq_len_f),
            slstm_backend=slstm_backend,
            **_model_kwargs,
        ).to(device)
    else:
        model = build_baseline(_model_name, num_layers=args.num_blocks, **_model_kwargs).to(device)
        if is_main:
            print(f"    [baseline] 使用 {_model_name.upper()} 模型主干（物理损失函数不变）")
    if is_main:
        train_meta["model_arch"] = _model_name
        if is_phase:
            train_meta.update(
                {
                    "fc_hz": fc_hz,
                    "lambda_phase_max": args.lambda_phase_max,
                    "phase_warmup_start": args.phase_warmup_start,
                    "phase_warmup_end": args.phase_warmup_end,
                    "lambda_mse": args.lambda_mse,
                }
            )
        if is_cascade:
            train_meta.update(
                {
                    "fc_hz": fc_hz,
                    "fine_input_dim": args.hidden_dim * 2 + 9,
                    "lambda_c_max": args.lambda_c_max,
                    "c_warmup_start": args.c_warmup_start,
                    "c_warmup_end": args.c_warmup_end,
                    "lambda_f_max": args.lambda_f_max,
                    "f_warmup_start": args.f_warmup_start,
                    "f_warmup_end": args.f_warmup_end,
                    "lambda_phase_max": args.lambda_phase_max,
                    "phase_warmup_start": args.phase_warmup_start,
                    "phase_warmup_end": args.phase_warmup_end,
                    "lambda_total_max": args.lambda_total_max,
                    "total_warmup_start": args.total_warmup_start,
                    "total_warmup_end": args.total_warmup_end,
                    "freeze_fine_until_epoch": args.freeze_fine_until_epoch,
                    "detach_coarse_until_epoch": args.detach_coarse_until_epoch,
                }
            )
    if is_cascade:
        criterion = CascadePhysicsLoss(
            pos_A,
            pos_B,
            pos_C,
            normalizer=normalizer,
            fc_hz=fc_hz,
            lambda_smooth=args.lambda_smooth,
            lambda_rcm=0.0,
            lambda_c=0.0,
            lambda_f=0.0,
            lambda_phase=0.0,
            lambda_total=0.0,
        ).to(device)
    elif is_phase:
        criterion = PhasePhysicsLoss(
            pos_A,
            pos_B,
            pos_C,
            normalizer=normalizer,
            fc_hz=fc_hz,
            lambda_smooth=args.lambda_smooth,
            lambda_phase=0.0,
            lambda_mse=args.lambda_mse,
        ).to(device)
    else:
        criterion = RadarPhysicsLoss(
            pos_A,
            pos_B,
            pos_C,
            normalizer=normalizer,
            lambda_smooth=args.lambda_smooth,
            lambda_rcm=0.0,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # CosineAnnealingLR：LR 在 T_max 个 epoch 内余弦衰减至 eta_min，之后维持 eta_min。
    # T_max 默认为 epochs（全程单调衰减），避免后半段 LR 回升引发不稳定。
    # 若需要 warm-restart 行为，可通过 --scheduler_tmax 显式设置（如 epochs//2）。
    t_max = args.scheduler_tmax if args.scheduler_tmax > 0 else max(1, args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_max,
        eta_min=args.scheduler_eta_min,
    )

    start_epoch_idx = 0
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch_idx, run_dir_ckpt = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            normalizer=normalizer,
            expected_meta=train_meta,
        )
        model.to(device)
        normalizer.to(device)
        run_dir = run_dir_ckpt if run_dir_ckpt is not None else resolve_run_dir_from_checkpoint(resume_path)
        run_dir = run_dir.resolve()
        if is_main:
            (run_dir / "viz").mkdir(parents=True, exist_ok=True)
            (run_dir / "weights").mkdir(parents=True, exist_ok=True)
            print(f"已从 checkpoint 恢复，将从 epoch {start_epoch_idx + 1} 继续（共 {args.epochs} epoch）。")
            print(f"    续跑目录 run_dir: {run_dir}")
        # 同步 run_dir 路径给所有 DDP 进程
        if is_ddp:
            _bcast2 = [str(run_dir)]
            dist.broadcast_object_list(_bcast2, src=0)
            run_dir = Path(_bcast2[0])

    # ── DDP 包裹模型 ────────────────────────────────────────────────────────
    # cascade：前期 freeze_fine / λ_f=0 时 head_fine 可能不参与反传，需 find_unused_parameters。
    if is_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(is_cascade),
        )
        if is_main and is_cascade:
            print("    [DDP] cascade: find_unused_parameters=True（兼容 freeze_fine 课程阶段）")
    raw_model: nn.Module = model.module if is_ddp else model  # type: ignore[assignment]

    assert run_dir is not None

    weights_dir = (
        Path(args.checkpoint_dir).expanduser().resolve()
        if args.checkpoint_dir.strip()
        else run_dir / "weights"
    )
    weights_dir.mkdir(parents=True, exist_ok=True)

    metrics_hist = load_metrics_history(run_dir)

    if start_epoch_idx >= args.epochs:
        if is_main:
            print("checkpoint 进度已达到或超过 --epochs，无需继续训练。")
        if is_ddp:
            dist.destroy_process_group()
        return

    epoch_times_sec: list[float] = []
    train_wall_start = time.perf_counter()

    # AMP：bfloat16 autocast（无需 GradScaler，bf16 不会溢出）。仅 CUDA 上启用。
    use_amp = bool(args.amp) and device.type == "cuda"
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else _NullContext()
    )
    if is_main:
        if use_amp:
            print("    [perf] AMP bfloat16 autocast: ON")
        else:
            print("    [perf] AMP autocast: OFF (fp32)")

    # best.pt 与早停：单头用 val_pos_rmse；cascade 用 val_rmse_total（‖P_fix−P_true‖）。
    best_val_metric = float("inf")
    early_stop_counter = 0
    should_stop = False

    for epoch in range(start_epoch_idx, args.epochs):
        # DDP 同步上一轮的早停决定：所有 rank 一起 break，避免 rank0 退出后
        # rank1 在 backward 同步处等待超时（NCCL watchdog 600s timeout）。
        if is_ddp:
            stop_tensor = torch.tensor(
                [1.0 if should_stop else 0.0], device=device, dtype=torch.float32
            )
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item() > 0.5:
                break
        elif should_stop:
            break

        epoch_wall_start = time.perf_counter()
        model.train()
        # DDP：每 epoch 告知 sampler 当前轮次，保证各进程打乱不重叠
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        current_lambda_rcm = compute_lambda_rcm(
            epoch,
            warmup_start=args.rcm_warmup_start,
            warmup_end=args.rcm_warmup_end,
            max_lambda=args.lambda_rcm_max,
        )
        if hasattr(criterion, "lambda_rcm"):
            criterion.lambda_rcm = current_lambda_rcm
        if is_cascade:
            criterion.lambda_c = compute_lambda_warmup(
                epoch,
                warmup_start=args.c_warmup_start,
                warmup_end=args.c_warmup_end,
                max_lambda=args.lambda_c_max,
            )
            criterion.lambda_f = compute_lambda_warmup(
                epoch,
                warmup_start=args.f_warmup_start,
                warmup_end=args.f_warmup_end,
                max_lambda=args.lambda_f_max,
            )
            criterion.lambda_phase = compute_lambda_warmup(
                epoch,
                warmup_start=args.phase_warmup_start,
                warmup_end=args.phase_warmup_end,
                max_lambda=args.lambda_phase_max,
            )
            criterion.lambda_total = compute_lambda_warmup(
                epoch,
                warmup_start=args.total_warmup_start,
                warmup_end=args.total_warmup_end,
                max_lambda=args.lambda_total_max,
            )
            detach_coarse = epoch < args.detach_coarse_until_epoch
            freeze_fine = epoch < args.freeze_fine_until_epoch
            for _p in raw_model.head_fine.parameters():
                _p.requires_grad = not freeze_fine
        elif is_phase:
            detach_coarse = False
            freeze_fine = False
            criterion.lambda_phase = compute_lambda_warmup(
                epoch,
                warmup_start=args.phase_warmup_start,
                warmup_end=args.phase_warmup_end,
                max_lambda=args.lambda_phase_max,
            )
        else:
            detach_coarse = False
            freeze_fine = False
        if is_cascade:
            current_lambda_c = float(criterion.lambda_c)
            current_lambda_f = float(criterion.lambda_f)
            current_lambda_phase = float(criterion.lambda_phase)
            current_lambda_total = float(criterion.lambda_total)
        elif is_phase:
            current_lambda_c = current_lambda_f = current_lambda_total = float("nan")
            current_lambda_phase = float(criterion.lambda_phase)
        else:
            current_lambda_c = current_lambda_f = current_lambda_phase = current_lambda_total = float("nan")

        total_loss_epoch = 0.0
        total_mse_norm_epoch = 0.0
        total_mse_phys_epoch = 0.0
        total_smooth_epoch = 0.0
        total_rcm_epoch = 0.0
        total_mse_c_epoch = 0.0
        total_mse_f_epoch = 0.0
        total_phase_epoch = 0.0
        max_grad_norm_epoch = 0.0
        grad_clip_max = 1.0

        diag_gnorm_mse = float("nan")
        diag_gnorm_smooth = float("nan")
        diag_gnorm_rcm = float("nan")

        accum_steps = args.grad_accum_steps
        use_mask = bool(args.use_input_mask)
        optimizer.zero_grad(set_to_none=True)
        for _batch_idx, batch in enumerate(dataloader):
            if is_cascade:
                (
                    feat, feat_mask, p_raw, p_true, has_label, rcm_weight,
                    pos_a, pos_b, pos_c, dc, ph_sin, ph_cos,
                ) = batch
                phase_rel = r_ref = None
            elif is_phase:
                (
                    feat, feat_mask, p_raw, p_true, has_label, rcm_weight,
                    pos_a, pos_b, pos_c, phase_rel, r_ref,
                ) = batch
                dc = ph_sin = ph_cos = None
            else:
                feat, feat_mask, p_raw, p_true, has_label, rcm_weight, pos_a, pos_b, pos_c = batch
                dc = ph_sin = ph_cos = phase_rel = r_ref = None
            feat = feat.to(device, non_blocking=True)
            feat_mask = feat_mask.to(device, non_blocking=True)
            p_raw = p_raw.to(device, non_blocking=True)
            p_true = p_true.to(device, non_blocking=True)
            has_label = has_label.to(device, non_blocking=True)
            rcm_weight = rcm_weight.to(device, non_blocking=True)
            pos_a = pos_a.to(device, non_blocking=True)
            pos_b = pos_b.to(device, non_blocking=True)
            pos_c = pos_c.to(device, non_blocking=True)

            feat_normalized = normalizer.norm_feat(feat) * feat_mask

            is_last_in_accum = ((_batch_idx + 1) % accum_steps == 0) or (
                _batch_idx + 1 == len(dataloader)
            )
            with autocast_ctx:
                if is_cascade:
                    assert dc is not None and ph_sin is not None and ph_cos is not None
                    dc = dc.to(device, non_blocking=True)
                    ph_sin = ph_sin.to(device, non_blocking=True)
                    ph_cos = ph_cos.to(device, non_blocking=True)
                    ph_sin = ph_sin * rcm_weight.unsqueeze(1)
                    ph_cos = ph_cos * rcm_weight.unsqueeze(1)
                    delta_fine = (p_raw - p_true) - dc
                    delta_c_norm, delta_f_norm = model(
                        feat_normalized,
                        ph_sin,
                        ph_cos,
                        p_raw,
                        normalizer,
                        feat_mask if use_mask else None,
                        detach_coarse=detach_coarse,
                    )
                    loss, l_mse_c, l_mse_f, l_rcm, l_phase, l_smooth, l_mse_total, l_mse_phys = criterion(
                        delta_c_norm,
                        delta_f_norm,
                        p_raw,
                        p_true,
                        feat,
                        dc,
                        delta_fine,
                        ph_sin,
                        ph_cos,
                        has_label,
                        rcm_weight,
                        pos_a,
                        pos_b,
                        pos_c,
                    )
                    l_mse_norm = l_mse_total
                elif is_phase:
                    assert phase_rel is not None and r_ref is not None
                    phase_rel = phase_rel.to(device, non_blocking=True)
                    r_ref = r_ref.to(device, non_blocking=True)
                    phase_rel = phase_rel * rcm_weight.unsqueeze(1)
                    delta_pred_norm = model(
                        feat_normalized,
                        feat_mask if use_mask else None,
                    )
                    loss, l_mse_norm, l_mse_phys, l_smooth, l_rcm, l_phase = criterion(
                        delta_pred_norm,
                        p_raw,
                        p_true,
                        phase_rel,
                        r_ref,
                        has_label,
                        rcm_weight,
                        pos_a,
                        pos_b,
                        pos_c,
                    )
                    l_mse_c = l_mse_f = l_mse_total = None
                else:
                    delta_pred_norm = model(
                        feat_normalized,
                        feat_mask if use_mask else None,
                    )
                    loss, l_mse_norm, l_mse_phys, l_smooth, l_rcm = criterion(
                        delta_pred_norm,
                        p_raw,
                        p_true,
                        feat,
                        has_label,
                        rcm_weight,
                        pos_a,
                        pos_b,
                        pos_c,
                    )
                    l_mse_c = l_mse_f = l_phase = l_mse_total = None

            if is_main and _batch_idx == 0 and not is_cascade and not is_phase:
                _diag_params = [p for p in model.parameters() if p.requires_grad]

                def _grad_l2(_term):
                    if _term is None:
                        return float("nan")
                    try:
                        if torch.isnan(_term).any() or torch.isinf(_term).any():
                            return float("nan")
                    except RuntimeError:
                        return float("nan")
                    try:
                        _grads = torch.autograd.grad(
                            _term,
                            _diag_params,
                            retain_graph=True,
                            allow_unused=True,
                            create_graph=False,
                        )
                    except RuntimeError:
                        return float("nan")
                    _sq = 0.0
                    for _g in _grads:
                        if _g is not None:
                            _sq += _g.detach().float().pow(2).sum().item()
                    return _sq ** 0.5

                diag_gnorm_mse = _grad_l2(l_mse_norm)
                _eff_lam_sm = float(criterion.lambda_smooth)
                diag_gnorm_smooth = _grad_l2(_eff_lam_sm * l_smooth) if _eff_lam_sm > 0 else 0.0
                _eff_lam_rcm = float(criterion.lambda_rcm)
                diag_gnorm_rcm = _grad_l2(_eff_lam_rcm * l_rcm) if _eff_lam_rcm > 0 else 0.0

            (loss / accum_steps).backward()

            if is_last_in_accum:
                raw_gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max)
                if raw_gnorm.item() > max_grad_norm_epoch:
                    max_grad_norm_epoch = raw_gnorm.item()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss_epoch += loss.item()
            total_mse_norm_epoch += l_mse_norm.item()
            if not (l_mse_phys != l_mse_phys):
                total_mse_phys_epoch += l_mse_phys.item()
            total_smooth_epoch += l_smooth.item()
            total_rcm_epoch += l_rcm.item()
            if is_cascade:
                total_mse_c_epoch += l_mse_c.item()
                total_mse_f_epoch += l_mse_f.item()
                total_phase_epoch += l_phase.item()
            elif is_phase and l_phase is not None:
                total_phase_epoch += l_phase.item()

        scheduler.step()
        n = len(dataloader)
        epoch_duration = time.perf_counter() - epoch_wall_start

        # ── 以下仅 rank 0 执行（打印、写指标、保存权重） ──────────────────
        if not is_main:
            continue

        epoch_times_sec.append(epoch_duration)
        completed_epochs = epoch + 1
        wall_elapsed = time.perf_counter() - train_wall_start
        avg_total = total_loss_epoch / n
        avg_mse_norm = total_mse_norm_epoch / n
        avg_mse_phys = total_mse_phys_epoch / n
        avg_smooth = total_smooth_epoch / n
        avg_rcm = total_rcm_epoch / n
        lr_now = scheduler.get_last_lr()[0]

        # ── 验证集评估（rank 0 独立运行，无梯度） ─────────────────────────
        avg_val_total = float("nan")
        avg_val_mse_norm = float("nan")
        avg_val_pos_rmse = float("nan")
        avg_val_rmse_coarse = float("nan")
        avg_val_rmse_total = float("nan")
        if val_loader is not None:
            raw_model.eval()
            val_total_loss = val_mse_norm_loss = val_sq_pos = val_sq_coarse = 0.0
            val_n_batch = val_n_ts = 0
            with torch.no_grad():
                for v_batch in val_loader:
                    if is_cascade:
                        (
                            v_feat, v_feat_mask, v_p_raw, v_p_true, v_has_label, v_rcm_w,
                            v_pos_a, v_pos_b, v_pos_c, v_dc, v_ph_sin, v_ph_cos,
                        ) = v_batch
                        v_phase_rel = v_r_ref = None
                    elif is_phase:
                        (
                            v_feat, v_feat_mask, v_p_raw, v_p_true, v_has_label, v_rcm_w,
                            v_pos_a, v_pos_b, v_pos_c, v_phase_rel, v_r_ref,
                        ) = v_batch
                        v_dc = v_ph_sin = v_ph_cos = None
                    else:
                        v_feat, v_feat_mask, v_p_raw, v_p_true, v_has_label, v_rcm_w, v_pos_a, v_pos_b, v_pos_c = v_batch
                        v_dc = v_ph_sin = v_ph_cos = v_phase_rel = v_r_ref = None
                    v_feat = v_feat.to(device, non_blocking=True)
                    v_feat_mask = v_feat_mask.to(device, non_blocking=True)
                    v_p_raw = v_p_raw.to(device, non_blocking=True)
                    v_p_true = v_p_true.to(device, non_blocking=True)
                    v_has_label = v_has_label.to(device, non_blocking=True)
                    v_rcm_w = v_rcm_w.to(device, non_blocking=True)
                    v_pos_a = v_pos_a.to(device, non_blocking=True)
                    v_pos_b = v_pos_b.to(device, non_blocking=True)
                    v_pos_c = v_pos_c.to(device, non_blocking=True)
                    v_feat_norm = normalizer.norm_feat(v_feat) * v_feat_mask
                    if is_cascade:
                        assert v_dc is not None
                        v_dc = v_dc.to(device, non_blocking=True)
                        v_ph_sin = v_ph_sin.to(device, non_blocking=True)
                        v_ph_cos = v_ph_cos.to(device, non_blocking=True)
                        v_ph_sin = v_ph_sin * v_rcm_w.unsqueeze(1)
                        v_ph_cos = v_ph_cos * v_rcm_w.unsqueeze(1)
                        v_delta_fine = (v_p_raw - v_p_true) - v_dc
                        v_dc_norm, v_df_norm = raw_model(
                            v_feat_norm,
                            v_ph_sin,
                            v_ph_cos,
                            v_p_raw,
                            normalizer,
                            v_feat_mask if use_mask else None,
                            detach_coarse=False,
                        )
                        v_loss, v_mse_norm, *_ = criterion(
                            v_dc_norm,
                            v_df_norm,
                            v_p_raw,
                            v_p_true,
                            v_feat,
                            v_dc,
                            v_delta_fine,
                            v_ph_sin,
                            v_ph_cos,
                            v_has_label,
                            v_rcm_w,
                            v_pos_a,
                            v_pos_b,
                            v_pos_c,
                        )
                        v_dc_phys = normalizer.denorm_delta_c(v_dc_norm)
                        v_df_phys = normalizer.denorm_err(v_df_norm)
                        v_p_coarse = v_p_raw - v_dc_phys
                        v_p_fix = v_p_raw - v_dc_phys - v_df_phys
                        val_sq_coarse += ((v_p_coarse - v_p_true) ** 2).sum(dim=-1).sum().item()
                    elif is_phase:
                        assert v_phase_rel is not None and v_r_ref is not None
                        v_phase_rel = v_phase_rel.to(device, non_blocking=True)
                        v_r_ref = v_r_ref.to(device, non_blocking=True)
                        v_phase_rel = v_phase_rel * v_rcm_w.unsqueeze(1)
                        v_delta_norm = raw_model(
                            v_feat_norm,
                            v_feat_mask if use_mask else None,
                        )
                        v_loss, v_mse_norm, *_ = criterion(
                            v_delta_norm,
                            v_p_raw,
                            v_p_true,
                            v_phase_rel,
                            v_r_ref,
                            v_has_label,
                            v_rcm_w,
                            v_pos_a,
                            v_pos_b,
                            v_pos_c,
                        )
                        v_delta = normalizer.denorm_err(v_delta_norm)
                        v_p_fix = v_p_raw - v_delta
                    else:
                        v_delta_norm = raw_model(
                            v_feat_norm,
                            v_feat_mask if use_mask else None,
                        )
                        v_loss, v_mse_norm, *_ = criterion(
                            v_delta_norm,
                            v_p_raw,
                            v_p_true,
                            v_feat,
                            v_has_label,
                            v_rcm_w,
                            v_pos_a,
                            v_pos_b,
                            v_pos_c,
                        )
                        v_delta = normalizer.denorm_err(v_delta_norm)
                        v_p_fix = v_p_raw - v_delta
                    val_total_loss += v_loss.item()
                    val_mse_norm_loss += v_mse_norm.item()
                    val_sq_pos += ((v_p_fix - v_p_true) ** 2).sum(dim=-1).sum().item()
                    val_n_ts += v_p_fix.shape[0] * v_p_fix.shape[1]
                    val_n_batch += 1
            raw_model.train()
            if val_n_batch > 0:
                avg_val_total = val_total_loss / val_n_batch
                avg_val_mse_norm = val_mse_norm_loss / val_n_batch
                avg_val_pos_rmse = (val_sq_pos / max(val_n_ts, 1)) ** 0.5
                avg_val_rmse_total = avg_val_pos_rmse
                if is_cascade:
                    avg_val_rmse_coarse = (val_sq_coarse / max(val_n_ts, 1)) ** 0.5

        val_metric = avg_val_rmse_total if is_cascade else avg_val_pos_rmse
        if is_cascade:
            val_str = (
                f" | ValRMSE_total: {avg_val_rmse_total:.4f}m"
                f"  coarse: {avg_val_rmse_coarse:.4f}m"
                if not (avg_val_rmse_total != avg_val_rmse_total)
                else ""
            )
        else:
            val_str = (
                f" | Val: {avg_val_total:.5f}  ValMSE: {avg_val_mse_norm:.5f}"
                f"  ValRMSE: {avg_val_pos_rmse:.4f}m"
                if not (avg_val_total != avg_val_total)
                else ""
            )
        if is_cascade:
            avg_mse_c = total_mse_c_epoch / n
            avg_mse_f = total_mse_f_epoch / n
            avg_phase = total_phase_epoch / n
            print(
                f"Epoch [{completed_epochs:03d}/{args.epochs:03d}] | "
                f"Total: {avg_total:.5f} | "
                f"mse_c: {avg_mse_c:.5f} | mse_f: {avg_mse_f:.5f} | "
                f"phase: {avg_phase:.4f} | RCM: {avg_rcm:.4f} | "
                f"λ_c/f/φ/rcm: {current_lambda_c:.2f}/{current_lambda_f:.2f}/"
                f"{current_lambda_phase:.2f}/{current_lambda_rcm:.2f} | "
                f"GradN: {max_grad_norm_epoch:.3f} | LR: {lr_now:.6f} | "
                f"用时: {format_duration(epoch_duration)}{val_str}"
            )
        elif is_phase:
            avg_phase = total_phase_epoch / n
            print(
                f"Epoch [{completed_epochs:03d}/{args.epochs:03d}] | "
                f"Total: {avg_total:.5f} | "
                f"MSE(norm): {avg_mse_norm:.5f} | "
                f"phase: {avg_phase:.4f} | "
                f"λ_φ: {current_lambda_phase:.2f} | "
                f"GradN: {max_grad_norm_epoch:.3f} | LR: {lr_now:.6f} | "
                f"用时: {format_duration(epoch_duration)}{val_str}"
            )
        else:
            print(
                f"Epoch [{completed_epochs:03d}/{args.epochs:03d}] | "
                f"Total: {avg_total:.5f} | "
                f"MSE(norm): {avg_mse_norm:.5f} | "
                f"MSE(phys): {avg_mse_phys:.4f} | "
                f"RCM: {avg_rcm:.4f} | "
                f"λ_rcm: {current_lambda_rcm:.3f} | "
                f"GradN: {max_grad_norm_epoch:.3f} | "
                f"LR: {lr_now:.6f} | "
                f"用时: {format_duration(epoch_duration)}{val_str}"
            )
        # GradN 为「裁剪前」范数，大于 grad_clip_max 是常态；仅当异常大时提示（避免每 epoch 误报）
        grad_norm_warn_preclip = 200.0
        if max_grad_norm_epoch > grad_norm_warn_preclip:
            print(
                f"    ⚠ 裁剪前梯度范数异常: {max_grad_norm_epoch:.1f}（>"
                f"{grad_norm_warn_preclip:.0f}），裁剪阈值为 {grad_clip_max}；请留意稳定性或调小 lr。"
            )
        # 分项 grad_norm 诊断输出（每 epoch 首 batch 采样的快照）
        _rcm_lam_print = float(getattr(criterion, "lambda_rcm", 0.0))
        print(
            f"    [GradN 分项@batch0] "
            f"MSE: {diag_gnorm_mse:.3f} | "
            f"Smooth(×{criterion.lambda_smooth:.2f}): {diag_gnorm_smooth:.3f} | "
            f"RCM(×{_rcm_lam_print:.2f}): {diag_gnorm_rcm:.3f}"
        )

        metric_row: dict[str, float] = {
            "epoch": completed_epochs,
            "total_loss": avg_total,
            "mse_norm": avg_mse_norm,
            "mse_phys": avg_mse_phys,
            "smooth": avg_smooth,
            "rcm": avg_rcm,
            "lambda_rcm": current_lambda_rcm,
            "grad_norm": max_grad_norm_epoch,
            "gnorm_mse": diag_gnorm_mse,
            "gnorm_smooth": diag_gnorm_smooth,
            "gnorm_rcm": diag_gnorm_rcm,
            "lr": lr_now,
            "epoch_time_sec": epoch_duration,
            "val_total_loss": avg_val_total,
            "val_mse_norm": avg_val_mse_norm,
            "val_pos_rmse": avg_val_pos_rmse,
        }
        if is_cascade:
            metric_row.update(
                {
                    "mse_c": total_mse_c_epoch / n,
                    "mse_f": total_mse_f_epoch / n,
                    "phase": total_phase_epoch / n,
                    "lambda_c": current_lambda_c,
                    "lambda_f": current_lambda_f,
                    "lambda_phase": current_lambda_phase,
                    "lambda_total": current_lambda_total,
                    "val_rmse_coarse": avg_val_rmse_coarse,
                    "val_rmse_total": avg_val_rmse_total,
                }
            )
        elif is_phase:
            metric_row.update(
                {
                    "phase": total_phase_epoch / n,
                    "lambda_phase": current_lambda_phase,
                }
            )
        append_metric_jsonl(run_dir, metric_row)
        metrics_hist["epoch"].append(float(completed_epochs))
        metrics_hist["total_loss"].append(float(avg_total))
        metrics_hist["mse_norm"].append(float(avg_mse_norm))
        metrics_hist["mse_phys"].append(float(avg_mse_phys))
        metrics_hist["smooth"].append(float(avg_smooth))
        metrics_hist["rcm"].append(float(avg_rcm))
        metrics_hist["lambda_rcm"].append(float(current_lambda_rcm))
        metrics_hist["grad_norm"].append(float(max_grad_norm_epoch))
        metrics_hist["lr"].append(float(lr_now))
        metrics_hist["epoch_time_sec"].append(float(epoch_duration))
        metrics_hist["val_total_loss"].append(float(avg_val_total))
        metrics_hist["val_mse_norm"].append(float(avg_val_mse_norm))
        metrics_hist["val_pos_rmse"].append(float(avg_val_pos_rmse))
        if is_cascade:
            for k, v in (
                ("mse_c", total_mse_c_epoch / n),
                ("mse_f", total_mse_f_epoch / n),
                ("phase", total_phase_epoch / n),
                ("lambda_c", current_lambda_c),
                ("lambda_f", current_lambda_f),
                ("lambda_phase", current_lambda_phase),
                ("val_rmse_coarse", avg_val_rmse_coarse),
                ("val_rmse_total", avg_val_rmse_total),
            ):
                metrics_hist.setdefault(k, []).append(float(v))
        elif is_phase:
            for k, v in (
                ("phase", total_phase_epoch / n),
                ("lambda_phase", current_lambda_phase),
            ):
                metrics_hist.setdefault(k, []).append(float(v))
        save_training_figures(metrics_hist, run_dir / "viz")

        if args.param_report_every > 0 and completed_epochs % args.param_report_every == 0:
            print_training_params_report(
                device=device,
                train_meta=train_meta,
                completed_epochs=completed_epochs,
                total_epochs=args.epochs,
                epoch_times_sec=epoch_times_sec,
                train_wall_elapsed_sec=wall_elapsed,
            )

        # ── 保存物理位置 RMSE 最优的权重 best.pt ─────────────────────────────
        # 判据：val_pos_rmse（米）。它与 λ_rcm 课程无关，warmup 期间不会被乘子膨胀假性抬高。
        is_best = (
            is_main
            and val_loader is not None
            and not (val_metric != val_metric)
            and val_metric < best_val_metric
        )
        if is_best:
            best_val_metric = val_metric
            early_stop_counter = 0
            best_path = weights_dir / "best.pt"
            save_checkpoint(
                best_path,
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                normalizer=normalizer,
                next_epoch_idx=completed_epochs,
                train_meta=train_meta,
                run_dir=run_dir,
            )
            metric_label = "val_rmse_total" if is_cascade else "val_pos_rmse"
            print(
                f"    [最佳] epoch {completed_epochs:03d} {metric_label}={val_metric:.4f}m"
                f"  → 已更新 {best_path}"
            )
        elif val_loader is not None and not (val_metric != val_metric):
            early_stop_counter += 1
            patience = args.early_stop_patience
            if patience > 0:
                metric_label = "val_rmse_total" if is_cascade else "val_pos_rmse"
                print(f"    [早停] {metric_label} 未改善 {early_stop_counter}/{patience} 轮")
                if early_stop_counter >= patience:
                    print(
                        f"  ⚠ 早停触发：{metric_label} 连续 {patience} 轮无改善，"
                        f"将在下一 epoch 顶部同步退出。"
                    )
                    should_stop = True

        next_epoch_idx = completed_epochs
        should_save_periodic = args.checkpoint_every > 0 and completed_epochs % args.checkpoint_every == 0
        should_save_final = completed_epochs == args.epochs
        if should_save_periodic or should_save_final:
            stem = f"checkpoint_epoch_{completed_epochs:05d}.pt"
            ckpt_path = weights_dir / stem
            save_checkpoint(
                ckpt_path,
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                normalizer=normalizer,
                next_epoch_idx=next_epoch_idx,
                train_meta=train_meta,
                run_dir=run_dir,
            )
            latest_path = weights_dir / "latest.pt"
            save_checkpoint(
                latest_path,
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                normalizer=normalizer,
                next_epoch_idx=next_epoch_idx,
                train_meta=train_meta,
                run_dir=run_dir,
            )
            tag = "收尾" if should_save_final and not should_save_periodic else "定期"
            print(f"    [{tag}] 已保存权重: {ckpt_path} （并已更新 {latest_path}）")

    if is_main:
        print(f"训练完成！运行目录: {run_dir}")
        best_pt = weights_dir / "best.pt"
        if best_pt.exists():
            bl = "val_rmse_total" if is_cascade else "val_pos_rmse"
            print(f"  - 最优权重: {best_pt}  ({bl}={best_val_metric:.4f}m)")
        print(f"  - 权重与恢复: {weights_dir}/latest.pt / checkpoint_epoch_*.pt")
        print(f"  - 指标: {run_dir / 'metrics.jsonl'} | 配置: {run_dir / 'config.json'}")
        print(f"  - 曲线图: {run_dir / 'viz' / 'training_curves.png'}")

    if is_ddp:
        dist.destroy_process_group()

    if is_main and args.shutdown_on_finish:
        delay = max(0, int(args.shutdown_delay_sec))
        shutdown_bin = shutil.which("shutdown")
        if not shutdown_bin:
            print("    [shutdown_on_finish] 未在 PATH 中找到 shutdown，跳过关机。")
        else:
            if delay > 0:
                print(
                    f"    [shutdown_on_finish] {delay}s 后将执行关机；"
                    f"若仍在本终端前台运行，可 Ctrl+C 终止脚本以中止关机流程。"
                )
                time.sleep(delay)
            print(f"    [shutdown_on_finish] 调用 {shutdown_bin} -h now …")
            rc = subprocess.run([shutdown_bin, "-h", "now"], check=False).returncode
            if rc != 0:
                print(
                    f"    [shutdown_on_finish] shutdown 退出码 {rc}（常见原因：非 root / 无 polkit 权限）；"
                    f"可改用: sudo {shutdown_bin} -h now"
                )


if __name__ == "__main__":
    main()
