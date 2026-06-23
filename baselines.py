"""
baselines.py — 对比实验基线模型

所有基线与 PI_xLSTM_Tracker 保持完全相同的输入/输出接口：
    forward(x: [B,T,F], feat_mask: Optional[B,T,F]) -> [B,T,3]

支持的模型：
    bilstm       双向 LSTM（4 层）
    bigru        双向 GRU（4 层）
    transformer  双向 Transformer Encoder（非因果，全局注意力）
    tcn          Temporal Convolutional Network（空洞因果卷积）

通过 build_model(name, **kwargs) 统一创建。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Bi-LSTM
# ──────────────────────────────────────────────────────────────────────────────

class BiLSTM_Tracker(nn.Module):
    """双向 LSTM 轨迹误差预测器（对比基线）。"""

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_input_mask: bool = False,
        **kwargs,                          # 忽略 xlstm 专用参数
    ) -> None:
        super().__init__()
        self.use_input_mask = use_input_mask
        eff_dim = input_dim * 2 if use_input_mask else input_dim

        self.input_proj = nn.Linear(eff_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, feat_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_input_mask:
            feat_mask = feat_mask if feat_mask is not None else torch.ones_like(x)
            x = torch.cat([x, feat_mask], dim=-1)
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        return self.head(self.dropout(x))


# ──────────────────────────────────────────────────────────────────────────────
# Bi-GRU
# ──────────────────────────────────────────────────────────────────────────────

class BiGRU_Tracker(nn.Module):
    """双向 GRU 轨迹误差预测器（对比基线）。"""

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_input_mask: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_input_mask = use_input_mask
        eff_dim = input_dim * 2 if use_input_mask else input_dim

        self.input_proj = nn.Linear(eff_dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, feat_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_input_mask:
            feat_mask = feat_mask if feat_mask is not None else torch.ones_like(x)
            x = torch.cat([x, feat_mask], dim=-1)
        x = self.input_proj(x)
        x, _ = self.gru(x)
        return self.head(self.dropout(x))


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Encoder（双向 / 非因果）
# ──────────────────────────────────────────────────────────────────────────────

class _SinPosEnc(nn.Module):
    """标准正弦余弦位置编码。"""

    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Transformer_Tracker(nn.Module):
    """双向（非因果）Transformer Encoder 轨迹误差预测器（对比基线）。

    注：序列长度 T=2048 时全局自注意力的内存复杂度为 O(T²)≈4M，
    在 32GB GPU 上可运行（约 256MB/batch），但明显慢于 xLSTM。
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_input_mask: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_input_mask = use_input_mask
        eff_dim = input_dim * 2 if use_input_mask else input_dim

        self.input_proj = nn.Linear(eff_dim, hidden_dim)
        self.pos_enc = _SinPosEnc(hidden_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,               # Pre-LN：训练更稳定
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, feat_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_input_mask:
            feat_mask = feat_mask if feat_mask is not None else torch.ones_like(x)
            x = torch.cat([x, feat_mask], dim=-1)
        x = self.pos_enc(self.input_proj(x))
        x = self.transformer(x)
        return self.head(self.dropout(x))


# ──────────────────────────────────────────────────────────────────────────────
# TCN（Temporal Convolutional Network，空洞因果卷积）
# ──────────────────────────────────────────────────────────────────────────────

class _TCNBlock(nn.Module):
    """单个 TCN 残差块（空洞因果卷积 + WeightNorm）。"""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.utils.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.utils.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pad = pad

    def _chomp(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.pad].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._chomp(self.net[0](x))
        out = self.net[1](out)
        out = self.net[2](out)
        out = self._chomp(self.net[3](out))
        out = self.net[4](out)
        out = self.net[5](out)
        return F.gelu(out + x)


class TCN_Tracker(nn.Module):
    """TCN 轨迹误差预测器（对比基线）。

    使用 8 个空洞因果卷积块，感受野 = kernel_size × (2^0+...+2^7) = 3×255 = 765 步。
    对于 T=2048，感受野约 37%，长程依赖不如 xLSTM，但计算效率极高。
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_input_mask: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_input_mask = use_input_mask
        eff_dim = input_dim * 2 if use_input_mask else input_dim

        self.input_proj = nn.Conv1d(eff_dim, hidden_dim, 1)
        blocks = []
        for i in range(num_layers):
            blocks.append(_TCNBlock(hidden_dim, kernel_size, dilation=2**i, dropout=dropout))
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, feat_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_input_mask:
            feat_mask = feat_mask if feat_mask is not None else torch.ones_like(x)
            x = torch.cat([x, feat_mask], dim=-1)
        x = self.input_proj(x.transpose(1, 2))   # [B, C, T]
        x = self.network(x)
        return self.head(x.transpose(1, 2))       # [B, T, 3]


# ──────────────────────────────────────────────────────────────────────────────
# 统一工厂函数
# ──────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type] = {
    "bilstm":      BiLSTM_Tracker,
    "bigru":       BiGRU_Tracker,
    "transformer": Transformer_Tracker,
    "tcn":         TCN_Tracker,
}


def build_baseline(name: str, **kwargs) -> nn.Module:
    """按名称构建基线模型。

    Args:
        name:    模型名，支持 bilstm / bigru / transformer / tcn
        **kwargs: 传给对应 __init__（input_dim, hidden_dim, output_dim, ...）
    Returns:
        nn.Module
    """
    name = name.lower().strip()
    if name not in _REGISTRY:
        raise ValueError(f"未知模型名 '{name}'，可选: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](**kwargs)
