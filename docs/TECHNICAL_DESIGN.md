# PI-xLSTM 轨迹误差校正系统 — 技术方案

> 版本: v2.0 | 日期: 2026-06-22 | 状态: 设计中 / 部分实现

---

## 1. 问题定义

### 1.1 背景

六旋翼无人机搭载 SAR（合成孔径雷达，f=9.6 GHz，v≈10 m/s，h≈400 m）执行对地成像任务时，导航系统（GPS/IMU）输出的飞行轨迹 $P_{raw}$ 含有积累性误差，导致 SAR 图像散焦。

**核心难点**：实测场景中**不存在真实轨迹 $P_{true}$ 作为监督标签**，仅有 SAR 回波中提取的 RCM（Range Cell Migration）斜距观测和相位观测作为绝对约束。

### 1.2 任务公式

$$P_{fix} = P_{raw} - \hat{\delta},\quad \hat{\delta} = f_\theta(X)$$

其中 $f_\theta$ 是 Bi-xLSTM 网络，$X$ 是 12 维输入特征序列。

### 1.3 坐标系

| 轴 | 方向 | 物理含义 |
|----|------|----------|
| Y ($X_1$) | 侧向 | 距离向（cross-track） |
| X ($X_2$) | 方位向 | 飞行方向（along-track） |
| Z ($X_3$) | 高度向 | 竖直（height） |

---

## 2. 网络架构

### 2.1 整体结构

```
输入 X [B, T, 12] (+ feat_mask [B, T, 12] 可选)
    │
    ▼ Concat [B, T, 12 or 24]     ← use_input_mask 时拼接 mask
    │
    ▼ Linear(12 or 24 → hidden_dim)
    │
    ├──────────────────────────┐
    │  正向 xLSTM Stack         │  翻转序列
    ▼                          ▼
[B, T, H]             [B, T, H] (反转后 flip 回来)
    │                          │
    └────── Concat ────────────┘
                  │
                  ▼ Dropout(0.1)
                  │
                  ▼ Linear(2H → 3)
                  │
             [B, T, 3]  = δ_pred (归一化空间)
```

**关键设计决策**：
- **单头输出**（δ_pred），而非 Cascade 双头（δ_c + δ_f）。v20 实验验证表明精修头从未带来可测的位置增益（ValRMSE_coarse ≈ ValRMSE_total ≈ 4.6 cm），双头共享 backbone 还导致多损失冲突。
- **权重共享双向**：同一组 xLSTM 参数处理正向和反向序列，梯度叠加 ≈ 2×，由 Dropout + 梯度裁剪缓解。

### 2.2 xLSTM Block Stack

使用 NX-AI 官方 `xlstm` 库，默认 `num_blocks=4`：

```
Block 0 : mLSTM  → FFN(GELU)
Block 1 : sLSTM  → FFN(GELU)   ← slstm_at=[1]，仅此位放 sLSTM
Block 2 : mLSTM  → FFN(GELU)
Block 3 : mLSTM  → FFN(GELU)
```

### 2.3 mLSTM（矩阵 LSTM）

```
Q, K, V = Linear(x)
i_gate = Linear([Q,K,V])       # 输入门 (log 空间)
f_gate = Linear([Q,K,V])       # 遗忘门 (logsigmoid)

log_D  = cumsum(f_gate) + i_gate   # [B, NH, S, S] 门控衰减
D      = exp(log_D - max(log_D))   # 数值稳定

C = (Q @ K^T / √DH) * D           # 组合矩阵
C_norm = C / (|sum(C)| + ε)        # 行归一化
h = C_norm @ V                     # 输出

内存复杂度: O(B × NH × S²) — S=2048 时 ≈ 4M 元素/head
```

### 2.4 sLSTM（标量 LSTM）

传统 LSTM 扩展，指数门控 + CUDA kernel 加速：
- `backend = "cuda"`（编译好的 CUDA kernel）
- `bias_init = "powerlaw_blockdependent"`（遗忘门偏置初始化）
- `conv1d_kernel_size = 4`（输入端时序卷积预处理）

### 2.5 推荐超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_dim | 128 | 嵌入维度 |
| num_blocks | 4 | xLSTM block 数 |
| num_heads | 4 | 注意力头数 |
| dropout | 0.1 | 输出层前 Dropout |
| seq_len | 2048 | 有效序列长度 |
| input_dim | 12（自动读取） | 特征维度 |

### 2.6 基线模型（baselines.py）

统一接口 `build_baseline(name, **kwargs)` — 输入输出形状与 PI_xLSTM_Tracker 完全一致：

| 模型 | 命令 | 特点 |
|------|------|------|
| Bi-LSTM | `--model bilstm` | 4 层双向 LSTM |
| Bi-GRU | `--model bigru` | 4 层双向 GRU |
| Transformer | `--model transformer` | Pre-LN，正弦位置编码，O(S²) 注意力 |
| TCN | `--model tcn` | 8 层空洞因果卷积，感受野 ≈ 765 步 |

---

## 3. 输入特征设计

### 3.1 12 维特征

| 索引 | 内容 | 符号 | 来源 | 缺失时 |
|------|------|------|------|--------|
| 0–2 | 导航轨迹（高斯平滑） | $P_{raw}^{smooth}$ | GPS/IMU | 永不缺失 |
| 3–5 | 数值微分速度（高斯平滑） | $V_{raw}$ | diff($P_{raw}$)×PRF | 永不缺失 |
| 6–8 | SAR 实测斜距 | $R_{obs}$ | SAR 回波峰搜 | 0, mask=0 |
| 9–11 | 导航-RCM 残差 | $\|P_{raw}-Pos_{ref}\| - R_{obs}$ | 计算 | 0, mask=0 |

**V_raw 保留理由**：
1. GPS 中断段：位置漂移但 IMU 速度短期仍可靠 → 两种传感器互补
2. 旋翼振动（20-50 Hz）在速度域 SNR 远高于位置域 → 更好的振动指纹
3. 代价可忽略：Linear(9→128) vs Linear(12→128) 仅差 384 参数（≈ 0.02% 总参数量）

### 3.2 输入掩码（use_input_mask）

当启用 `use_input_mask=True` 时：
- 输入变为 **24 维**：`[feat_12 | feat_mask_12]`
- feat_mask 标记每维是否有效（1=有效，0=缺失）
- 让网络显式区分"值为 0 是因为缺失"还是"值确实是 0"
- 仅在混合半监督训练（仿真+实测）时推荐开启

### 3.3 归一化

```python
# Z-Score 归一化，统计量仅在仿真训练集上拟合（防止数据泄漏 + 实测污染）
feat_norm = (feat - feat_mean) / feat_std  # [N,T,F]

# 输出在归一化误差空间预测，反归一化后参与物理损失
delta_phys = delta_norm * err_std + err_mean
```

---

## 4. 损失函数

### 4.1 RadarPhysicsLoss（仿真预训练 + 实测半监督）

**主损失（v2：支持半监督，has_label / rcm_weight）**：

$$\mathcal{L}_{total} = \mathcal{L}_{MSE}^{norm} + \lambda_{smooth} \cdot \mathcal{L}_{smooth} + \lambda_{rcm} \cdot \mathcal{L}_{RCM}^{norm}$$

| 项 | 作用 | 坐标空间 | 有效条件 |
|----|------|----------|----------|
| $\mathcal{L}_{MSE}^{norm}$ | 监督 δ 预测对齐真实误差 | 归一化空间 | has_label=True |
| $\mathcal{L}_{smooth}$ | 预测 δ 时序平滑约束 | 归一化空间 | 始终 |
| $\mathcal{L}_{RCM}^{norm}$ | 斜距一致性（物理锚定） | 归一化到与 MSE 同坐标 | rcm_weight > 0 |

**RCM 归一化**：`l_rcm_norm = l_rcm_raw(m²) / mean(err_std²)`，使 λ_rcm=1.0 表示"RCM 与 MSE 同重"，权重选择与数据量级解耦。

**半监督行为**：
- `has_label=True`（仿真样本）：MSE + RCM + Smooth 全部激活
- `has_label=False`（实测样本）：**仅 RCM + Smooth**，MSE 梯度为 0

### 4.2 PhasePhysicsLoss（v4_phase 数据专用）

当数据集中包含 `Phase_rel`（真实解缠相位，相对于 $R_{ref}$）时使用：

$$\mathcal{L}_{phase} = \frac{1}{N_{ref}} \sum_{j} \left\langle 1 - \cos(\phi_{obs} - \phi_{ideal}) \right\rangle$$

其中：
- $\phi_{ideal} = -4\pi \cdot (r(P_{fix}, Pos_j) - R_{ref,j}) / \lambda$
- $\phi_{obs}$ = 数据集中的 `Phase_rel`（已解缠相位）
- 相位差先 wrap 到 [-π, π] 再计算 1-cos 损失
- **仅在物理弧度空间计算**，不再归一化

$$\mathcal{L}_{total} = \lambda_{mse} \cdot \mathcal{L}_{MSE}^{norm} + \lambda_{phase} \cdot \mathcal{L}_{phase} + \lambda_{smooth} \cdot \mathcal{L}_{smooth}$$

### 4.3 损失函数选择指南

| 数据情况 | 推荐损失 | --model | 说明 |
|----------|----------|---------|------|
| 纯仿真（旧版 v4 mat） | RadarPhysicsLoss | xlstm | MSE + RCM + Smooth |
| 仿真 v4_phase（Phase_rel + R_ref） | PhasePhysicsLoss | phase | Phase + MSE + Smooth |
| 仿真 + 实测混合 | RadarPhysicsLoss | xlstm | 半监督，has_label 控制 |
| 纯实测（无 P_true） | RadarPhysicsLoss | xlstm | 仅 RCM + Smooth，无 MSE |

### 4.4 默认超参数

| 参数 | 仿真预训练 | 半监督实测 | 说明 |
|------|-----------|-----------|------|
| λ_smooth | 0.1–2.0 | 0.1–2.0 | 时序平滑约束 |
| λ_rcm_max | 1.0 | 3.0–5.0 | RCM 权重（归一化坐标系） |
| rcm_warmup_start | 0 | 0 | RCM 从第几个 epoch 开始 |
| rcm_warmup_end | 80 | — | RCM 线性增加到 max |
| λ_phase_max | 1.0 | 3.0–5.0 | Phase 权重（仅 v4_phase） |
| λ_mse | 1.0 | 0.3–1.0 | MSE 权重（仅 v4_phase） |

---

## 5. 训练策略

### 5.1 数据生成与仿真升级

**当前仿真参数（v4）**：

| 参数 | 当前值 | 实测目标 | 需要修改 |
|------|--------|----------|----------|
| 飞行高度 | 60–140 m | ~400 m | ✅ 改为 380–420 m |
| 载频 | 未显式设定（默认 9.5 GHz） | 9.6 GHz | ✅ 配置 fc=9.6e9 |
| 飞行速度 | 7–13 m/s | ~10 m/s | ✅ 范围已覆盖 |
| RCM 仿真方式 | P_true 几何斜距 + 3mm 白噪声 | 模拟 SAR 测量过程 | ✅ 见下文 |

**SAR RCM 仿真升级**（从几何仿真 → 测量仿真）：
- 多点散射体回波叠加（模拟相干斑）
- 多径效应（地面二次反射引入的延迟伪影）
- 距离向采样抖动（定时误差 ±0.1 个采样单元）
- 随机野值（2% 概率，±2 个距离单元）
- 阴影效应（低掠射角时 SNR 退化）

### 5.2 训练阶段划分

```
第一阶段：仿真预训练（Stage 1）
├── 数据: 5000~10000 条仿真样本（全标签）
├── 模型: PI_xLSTM_Tracker, use_input_mask=False
├── 损失: RadarPhysicsLoss or PhasePhysicsLoss
├── 目的: 学习从 P_raw → δ 的通用映射
├── 指标: ValRMSE < 2cm（仿真验证集）
└── 输出: best.pt

第二阶段：半监督微调（Stage 2）
├── 数据: 仿真（80%）+ 实测（20%）混合
├── 模型: 加载 Stage 1 best.pt 权重
├── 参数: use_input_mask=True（实测有缺失通道）
├── 损失: 仿真有 MSE + RCM + Smooth；实测仅 RCM + Smooth
├── random_ref_mask: 开启（40%/30%/30% 概率保留 1/2/3 点）
├── λ_rcm: 提升到 3.0–5.0（RCM 成为实测样本的绝对锚定）
├── 目的: 让模型适应实测数据分布，同时不遗忘仿真学的轨迹修正能力
└── 输出: best_semi.pt

第三阶段：可选纯实测自监督（Stage 3）
├── 数据: 100% 实测（无 P_true）
├── 损失: 仅 RCM + Smooth + Phase
├── λ_rcm: 5.0（最高权重）
├── 目的: 进一步适应实测场景的特定误差模式
├── 风险: 无标签，可能漂移；建议配合多次 checkpoint 对比 SAR 成像质量
└── 验证: SAR 图像 sharpness / entropy / contrast（不依赖 P_true）
```

### 5.3 优化器与调度

```python
# AdamW
lr = 5e-4（Stage 1）→ 1e-4（Stage 2）→ 5e-5（Stage 3）
weight_decay = 5e-4

# CosineAnnealingLR
T_max = epochs // 2
eta_min = 1e-6

# 梯度累积
batch_size=8 × grad_accum=2 × DDP_2卡 = effective 32

# 梯度裁剪
max_norm = 0.5

# 混合精度
BF16 AMP（无需 GradScaler）
```

### 5.4 课程学习（λ 调度）

```
λ_rcm(t):
  epoch < warmup_start: 0.0
  warmup_start ≤ epoch < warmup_end: 线性 0 → λ_rcm_max
  epoch ≥ warmup_end: λ_rcm_max

推荐设置:
  Stage 1: warmup_start=0, warmup_end=80, λ_rcm_max=1.0
  Stage 2: warmup_start=0, warmup_end=0, λ_rcm_max=3.0~5.0
```

### 5.5 随机参考点丢弃（Random Ref Mask）

训练时以 40%/30%/30% 概率随机保留 1/2/3 个参考点：
- 被丢弃点的 feat[6+j] 和 feat[9+j] 置零，feat_mask 置 0，rcm_weight 置 0
- **推理时最常见场景是仅 1 个参考点**（实测中布置多个特显点成本高）
- 默认开启；`--no_random_ref_mask` 关闭

---

## 6. 实测数据处理管线

### 6.1 预处理流程

```
实测数据输入
│
├── GPS/IMU 轨迹 → P_raw [T, 3]
│   └── 高斯平滑（σ = 20/6 帧）
│
├── SAR 回波数据 → RCM 提取
│   ├── 脉冲压缩 → 峰值搜索 → 二次插值（子格点精度）
│   ├── SNR 门限筛选（< 15 dB 的峰值标记为 NaN）
│   └── 输出: R_obs [T, 3]（每个参考点一路距离序列）
│
├── SAR 相位数据 → 距离辅助解缠
│   ├── 干涉相位 φ_wrapped [T, 3]
│   ├── 用 R_obs 的帧间差分确定 2π 整数：
│   │   ΔR = R_obs[t] - R_obs[t-1]
│   │   k = round(ΔR / (λ/2) - (φ_wrapped[t] - φ_wrapped[t-1]) / (2π))
│   │   φ_unwrapped[t] = φ_wrapped[t] + 2π·k_cumulative
│   └── 输出: φ_unwrapped [T, 3]（解缠相位）
│
├── 组装 12 维特征
│   ├── feat[0:3]  = P_raw 平滑
│   ├── feat[3:6]  = V_raw 平滑（diff × PRF）
│   ├── feat[6:9]  = R_obs（NaN 填 0）
│   └── feat[9:12] = ||P_raw - Pos_ref|| - R_obs（NaN 填 0）
│
└── 构建 feat_mask [T, 12]（有效 = 1, 缺失 = 0）
```

### 6.2 推理与成像验证

```
实测 12-dim feat + mask
    │
    ▼ 网络前向（use_input_mask=True）
    │
    ▼ δ_pred [T, 3]
    │
    ▼ P_fix = P_raw - δ_pred
    │
    ▼ SAR 后向投影（Back Projection）成像
    │
    ▼ 图像质量评估（无需 P_true）
        ├── Sharpness = mean(|∇I|)               越高越好
        ├── Entropy   = -Σ p·log(p)              越低越好
        ├── Contrast  = std(I) / mean(I)         越高越好
        └── 与原始 P_raw 成像对比 → 验证修正有效性
```

### 6.3 SAR 图像质量对比策略

由于没有 P_true，最终验证依赖图像质量指标：
1. 用 $P_{raw}$ 成像 → 基准指标
2. 用 $P_{fix}$ 成像 → 修正后指标
3. Delta = $P_{fix}$ 指标 − $P_{raw}$ 指标，期望 Sharpness ↑, Entropy ↓, Contrast ↑
4. 多组实测数据交叉验证，确保一致性改善

---

## 7. 从 v20 Cascade 到当前方案的关键教训

### 7.1 为什么放弃 Cascade

| 问题 | 现象 | 根因 |
|------|------|------|
| 精修无增益 | ValRMSE_total ≈ ValRMSE_coarse ≈ 4.6 cm | 粗修已接近"完美标签"天花板 |
| 精修标签 mismatch | delta_fine 用 GT 粗修计算，推理用预测 δ_c | train/infer 不一致 |
| 多损失冲突 | epoch 41+ 后 RCM 上升、粗修退化 | Phase 梯度反传修改了 δ_c，但无保护机制 |
| 共享 backbone 的表征漂移 | 精修/相位开启后间接影响粗修输出 | 多任务梯度方向不一致 |

### 7.2 当前方案的核心原则

1. **单头 > 双头**：单一 δ 预测，单一优化目标，无多任务冲突
2. **相位 + RCM 互补**：RCM 距离提供绝对尺度锚定（~15 cm），相位提供亚毫米精度；两者失败模式正交
3. **仿真预训练 → 半监督微调**：仿真建立通用映射，实测用 RCM 物理约束自适应
4. **V_raw 保留**：代价可忽略，实测中提供 GPS 中断段的 IMU 速度信息和更好的振动指纹

---

## 8. 训练命令示例

### 8.1 Stage 1：仿真预训练（v4_phase 数据）

```bash
cd /root/xLSTM
torchrun --standalone --nproc_per_node=2 train_pi_xlstm.py \
  --mat_path /root/xLSTM/gene4/dataset1/PPTR_TrainDataset_v4_phase_5000.mat \
  --model phase \
  --batch_size 8 \
  --grad_accum_steps 2 \
  --epochs 200 \
  --lr 5e-4 \
  --hidden_dim 128 \
  --num_blocks 4 \
  --num_heads 4 \
  --lambda_mse 1.0 \
  --lambda_phase_max 1.0 \
  --phase_warmup_start 0 \
  --phase_warmup_end 80 \
  --lambda_smooth 2.0 \
  --random_ref_mask \
  --early_stop_patience 15 \
  --run_name stage1_phase
```

### 8.2 Stage 1：仿真预训练（RadarPhysicsLoss）

```bash
torchrun --standalone --nproc_per_node=2 train_pi_xlstm.py \
  --mat_path /root/xLSTM/dataset/PPTR_TrainDataset_5000.mat \
  --model xlstm \
  --batch_size 8 \
  --grad_accum_steps 2 \
  --epochs 200 \
  --lr 5e-4 \
  --hidden_dim 128 \
  --num_blocks 4 \
  --num_heads 4 \
  --lambda_rcm_max 1.0 \
  --rcm_warmup_start 0 \
  --rcm_warmup_end 80 \
  --lambda_smooth 2.0 \
  --random_ref_mask \
  --early_stop_patience 15 \
  --run_name stage1_rcm
```

### 8.3 Stage 2：半监督微调

```bash
torchrun --standalone --nproc_per_node=2 train_pi_xlstm.py \
  --mat_path /root/xLSTM/dataset/PPTR_TrainDataset_5000.mat \
  --real_mat_path /root/xLSTM/real_data/flight_001_processed.mat \
  --real_mix_ratio 0.2 \
  --use_input_mask \
  --random_ref_mask \
  --model xlstm \
  --batch_size 8 \
  --grad_accum_steps 2 \
  --epochs 100 \
  --lr 1e-4 \
  --hidden_dim 128 \
  --num_blocks 4 \
  --num_heads 4 \
  --lambda_rcm_max 5.0 \
  --rcm_warmup_start 0 \
  --rcm_warmup_end 0 \
  --lambda_smooth 2.0 \
  --resume /root/xLSTM/run/stage1_best/weights/best.pt \
  --run_name stage2_semi
```

---

## 9. 评估命令

```bash
BEST=/root/xLSTM/run/<run_dir>/weights/best.pt

# 仿真验证集评估
python eval_pi_xlstm.py \
  --checkpoint $BEST \
  --mat_path /root/xLSTM/dataset/PPTR_TrainDataset_5000.mat \
  --batch_size 8 \
  --save_p_fix run/p_fix_result.mat \
  --also_save_delta \
  --json_out run/eval_result.json

# 实测数据推理（无 P_true，仅输出 P_fix）
python eval_pi_xlstm.py \
  --checkpoint $BEST \
  --mat_path /root/xLSTM/real_data/flight_001_processed.mat \
  --real_data \
  --batch_size 4 \
  --save_p_fix run/p_fix_real.mat \
  --also_save_delta
```

---

## 10. 文件结构

```
xLSTM/
├── train_pi_xlstm.py           # 主训练脚本（2929 行）
│   ├── SARDataset               #   仿真数据加载（v4 / v4_phase / v4_legacy）
│   ├── RealSARDataset           #   实测数据加载（P_raw + 可选 Pos_Ref + RCM_obs）
│   ├── RefPointMaskedDataset    #   训练时随机丢弃参考点
│   ├── DataNormalizer           #   Z-Score 归一化 + feat_mask 支持
│   ├── PI_xLSTM_Tracker         #   单头 Bi-xLSTM（当前主架构）
│   ├── PI_xLSTM_Cascade         #   双头 Cascade（已弃用，保留用于对比）
│   ├── RadarPhysicsLoss         #   MSE + RCM + Smooth（仿真+半监督）
│   ├── CascadePhysicsLoss       #   双头损失（已弃用）
│   ├── PhasePhysicsLoss         #   Phase + MSE + Smooth（v4_phase 数据）
│   └── 训练 / DDP / 课程学习 / 日志
│
├── eval_pi_xlstm.py             # 评估/推理脚本
├── baselines.py                 # BiLSTM / BiGRU / Transformer / TCN 基线
├── summarize_results.py         # 训练结果汇总
│
├── data_generate.m              # MATLAB 数据生成（旧版）
├── data_generate_v2.m           #    v2：加入风场
├── data_generate_v3.m           #    v3：增强误差模型
├── data_generate_v4.m           #    v4：RCM 几何仿真 + Phase 支持
├── main.m                       #    MATLAB 主控脚本
├── sar.m                        #    SAR 反投影成像
├── sar_batch_validate_with_Pfix.m  # 批量 SAR 成像验证
├── test_rcm_geom_vs_extract.m   #    几何 RCM vs 提取 RCM 对比测试
│
├── xlstm/                       # NX-AI/xlstm 源码（pip install xlstm）
├── dataset/                     # 仿真训练数据集
├── gene4/dataset1/              # v4 系列数据
├── run/                         # 训练输出目录
│   └── <timestamp>_<name>_pid<N>/
│       ├── config.json
│       ├── metrics.jsonl
│       ├── viz/training_curves.png
│       └── weights/{best,latest,checkpoint_epoch_XXXXX}.pt
└── docs/
    ├── TECHNICAL_DESIGN.md      # ← 本文档
    └── v20_cascade_training_summary.md  # v20 Cascade 实验分析
```

---

## 11. 待办事项

### P0（仿真对齐实测）
- [ ] 更新 `data_generate_v4.m`：飞行高度从 60–140 m → 380–420 m
- [ ] 配置载频 fc=9.6 GHz（当前默认 9.5 GHz）
- [ ] 升级 RCM 仿真：多点散射叠加 + 多径 + 抖动 + 野值 + 阴影（替代 3 mm 白噪声）
- [ ] 加入 GPS 中断段仿真（IMU-only 段，位置漂移但速度短期可靠）

### P1（数据增强，防过拟合）
- [ ] 随机侧向偏置（±100 m）—— 打破坐标记忆
- [ ] 随机航向角（±15°）—— 打破固定坐标系依赖
- [ ] 阶跃跳变误差事件 —— 丰富误差模式
- [ ] RCM 噪声增大（5–30 mm）—— 防止单步精确求逆
- [ ] 近距离参考点（50–150 m）—— 提升 Z 轴可观测性
- [ ] err_scale 分档（弱/中/强）—— 消除误差量级单一性
- [ ] S 形机动轨迹 —— 丰富航迹形状

### P2（实测管线）
- [ ] 实现 SAR 回波 RCM 提取（脉冲压缩 + 峰搜 + 二次插值 + SNR 门限）
- [ ] 实现距离辅助相位解缠（利用 R_obs 帧间差分确定 2π 整数）
- [ ] 实现 SAR 图像质量评估（Sharpness / Entropy / Contrast）
- [ ] 端到端实测推理 + 成像验证管线

### P3（训练策略完善）
- [ ] Stage 1 仿真预训练完成并达到 ValRMSE < 2 cm
- [ ] Stage 2 半监督微调验证（仿真 ValRMSE 不退化 + 实测成像质量改善）
- [ ] Stage 3 纯实测自监督可行性评估
- [ ] λ_phase / λ_rcm 在实测场景的最优值搜索
- [ ] 多损失权重 AutoML（如 Optuna 超参搜索）

---

## 12. 关键设计决策速览

| 决策 | 选择 | 理由 |
|------|------|------|
| 网络架构 | 单头 Bi-xLSTM | Cascade 精修无增益，多损失冲突 |
| 输入维度 | 12 dim（含 V_raw） | V_raw 代价可忽略，实测 GPS 中断时有互补价值 |
| 相位表示 | 真实解缠相位 | ∂φ/∂R = const，学习近线性（vs sin/cos 梯度位置依赖） |
| 相位与 RCM | 两者保留 | RCM 距离提供绝对锚定，相位提供亚毫米精度；失败模式互补 |
| use_input_mask | Stage 2 开启 | 实测有缺失通道（0~3 个参考点），网络需区分 0 值 vs 缺失 |
| random_ref_mask | 训练时开启 | 推理时常见仅 1 个特显点，训练时需覆盖此场景 |
| ddp find_unused | True | 兼容半监督时部分样本无 MSE 计算路径 |
| 梯度裁剪 | 0.5 | Bi-xLSTM 双路梯度叠加 + mLSTM C 矩阵归一化容错 |
| 归一化统计量 | 仅仿真训练集拟合 | 防止实测数据污染统计量 + 防止验证集数据泄漏 |
| SAR 成像验证 | Sharpness/Entropy/Contrast | 实测无 P_true，图像质量是唯一不依赖标签的指标 |

---

*文档维护：随设计方案演进持续更新。最后更新：2026-06-22*
