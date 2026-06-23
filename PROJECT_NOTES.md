# PI-xLSTM 轨迹精化系统 技术文档

> 最后更新：2026-05-11

---

## 1. 问题定义

### 背景

无人机搭载 SAR（合成孔径雷达）执行对地成像任务时，导航系统（IMU/GPS）输出的飞行轨迹 $P_{raw}$ 包含积累性误差，导致 SAR 图像模糊散焦。**PPTR（Physics-informed Precise Trajectory Reconstruction）**的目标是利用深度学习从含噪观测中恢复真实轨迹 $P_{true}$。

### 任务公式

$$P_{fix} = P_{raw} - \hat{\delta}$$

其中 $\hat{\delta} = f_\theta(X)$ 是网络预测的误差向量，$X$ 是 12 维输入特征序列。

### 坐标系

```
[Y_lat, X_fwd, Z_up]
  Y_lat : 侧向（距离向）
  X_fwd : 方位向（飞行方向）
  Z_up  : 高度向
```

---

## 2. 数据集构建

### 2.1 特征维度说明（12维）

| 索引 | 内容 | 说明 |
|------|------|------|
| feat[0:3] | $P_{raw}$ | 导航轨迹（经高斯平滑，20点窗口） |
| feat[3:6] | $V_{raw}$ | 数值微分速度（经高斯平滑，40点窗口） |
| feat[6:9] | RCM 斜距 | 从 $P_{true}$ 计算 + 3mm 测量噪声 |
| feat[9:12] | 导航-RCM残差 | $\|P_{raw}-Pos_X\| - RCM_X$，误差在距离方向的投影 |

**标签**：$\delta = P_{raw} - P_{true}$，形状 `[T, 3]`

### 2.2 数据格式

```
Feat_All  : [N, T, 12]   # N=样本数, T=序列长度(2048), 12=特征维
P_true    : [N, T, 3]    # 真实轨迹
P_raw     : [N, T, 3]    # 含噪导航轨迹
Pos_A/B/C : [3, N]       # 每条样本对应的3个SAR特显点坐标（per-sample）
```

### 2.3 物理仿真流程

```
随机化物理参数
  → 6-DOF动力学仿真 (风场+PID控制) → P_true
  → 叠加复合误差模型 → P_raw
  → 构建12维特征张量 + 计算RCM
  → 保存 .mat 文件
```

### 2.4 误差模型（4类叠加）

```
E_total = E_trend + E_rw + E_sine + E_white

E_trend : 多项式发散趋势（偏置 + 速度漂移 + 加速度漂移）
E_rw    : 随机游走（累积高斯噪声）
E_sine  : 多频周期震荡（低频0.05~0.5Hz / 中频0.3~1.5Hz / 高频1.5~3Hz）
E_white : 高频白噪声

err_scale : 整体幅度缩放因子，范围 0.5~2.0
```

### 2.5 随机化参数范围

| 参数 | 范围 |
|------|------|
| 飞行速度 | 7~13 m/s |
| 飞行高度 | 60~140 m |
| 无人机质量 | 8.0 ± 15% kg |
| PID 增益 | 标称值 ± 15% |
| 风场振幅 | 0~1.5 m/s |
| 风场主频 | 0.05~0.25 Hz |
| 特显点侧向距离 | 300~500 m |
| 特显点方位向 | 航迹长度 10%~90% 处 |

---

## 3. 网络架构

### 3.1 整体结构

```
输入 [B, T, 12]
    │
    ▼ Linear(12 → hidden_dim)          # input_mapping
    │
    ├──────────────────────┐
    │  正向 xLSTM Stack     │  翻转序列
    ▼                      ▼
[B,T,H]             [B,T,H] (反向后翻转回来)
    │                      │
    └──────── Concat ───────┘          # [B, T, 2H]
                   │
                   ▼ Dropout(0.1)
                   │
                   ▼ Linear(2H → 3)   # output_layer
                   │
              [B, T, 3]               # δ_pred (归一化空间)
```

### 3.2 xLSTM Block Stack

使用 NX-AI 官方 `xlstm` 库，`num_blocks=4` 个 Block 按如下规则排列：

```
Block 0 : mLSTM Block
Block 1 : sLSTM Block  ← slstm_at=[1]
Block 2 : mLSTM Block
Block 3 : mLSTM Block
```

**每个 Block 的内部结构：**

```
输入 x
  │
  ├─ LayerNorm → xLSTM层 → 残差连接
  │
  └─ (若含FFN) LayerNorm → FeedForward(proj_factor=1.3, GELU) → 残差连接
```

### 3.3 mLSTM（矩阵LSTM）

核心计算：

```
Q, K, V = 输入的线性投影
i_gate  = Linear(concat[Q,K,V]) → 输入门（log空间）
f_gate  = Linear(concat[Q,K,V]) → 遗忘门（logsigmoid）

log_D = cumsum(f_gate) + i_gate      # [B, NH, S, S] 门控衰减矩阵
D     = exp(log_D - max(log_D))      # 数值稳定化

C = (Q @ K^T / √DH) * D             # 组合矩阵
C_norm = C / (|sum(C)| + ε)          # 行归一化
h = C_norm @ V                        # 输出

# 内存复杂度: O(B × NH × S²) —— 长序列时显存瓶颈
```

### 3.4 sLSTM（标量LSTM）

传统 LSTM 扩展，带指数门控，支持 CUDA 核函数加速：

```
backend = "cuda"       # 使用编译好的 CUDA kernel（SLSTM_BATCH_SIZE=8）
bias_init = "powerlaw_blockdependent"   # 遗忘门偏置特殊初始化
conv1d_kernel_size = 4  # 输入端 1D 卷积预处理
```

### 3.5 双向处理（Bi-xLSTM）

使用**共享权重**的双向策略：

```python
x_mapped = input_mapping(x)
out_forward  = xlstm_stack(x_mapped)               # 正向
out_backward = flip(xlstm_stack(flip(x_mapped)))   # 反向（翻转输入，翻转输出）
out = concat([out_forward, out_backward], dim=-1)  # [B, T, 2H]
```

- 优点：双向上下文，适合轨迹平滑任务
- 注意：同一组参数被调用两次，反传时梯度叠加（≈2×放大），由 Dropout + 梯度裁剪缓解

### 3.6 当前推荐超参数

| 参数 | 值 |
|------|-----|
| hidden_dim | 128 |
| num_blocks | 4 |
| num_heads | 4 |
| dropout | 0.1 |
| seq_len | 2048 |
| input_dim | 12（自动读取） |

---

## 4. 损失函数

### 4.1 复合损失（RadarPhysicsLoss）

$$\mathcal{L}_{total} = \mathcal{L}_{MSE}^{norm} + \lambda_{smooth} \cdot \mathcal{L}_{smooth} + \lambda_{rcm} \cdot \mathcal{L}_{RCM}$$

### 4.2 各项详解

**① MSE 损失（归一化空间，反传主信号）**

$$\mathcal{L}_{MSE}^{norm} = \frac{1}{BT} \sum \|\hat{\delta}_{norm} - \delta^*_{norm}\|^2$$

在 Z-score 归一化后的空间计算，梯度方差 ≈ 1，数值最稳定。

**② 平滑性约束（时序差分惩罚）**

$$\mathcal{L}_{smooth} = \frac{1}{B(T-1)} \sum_{t=1}^{T-1} \|\hat{\delta}_{norm}^{t} - \hat{\delta}_{norm}^{t-1}\|^2$$

惩罚预测误差的快速变化（轨迹误差在物理上应该缓变），在归一化空间计算。默认权重 $\lambda_{smooth}=2.0$。

**③ RCM 物理约束（课程学习，延迟接入）**

$$\mathcal{L}_{RCM} = \frac{1}{3} \sum_{X \in \{A,B,C\}} MSE\left(\|P_{fix} - Pos_X\|_{safe},\ d_X^{obs}\right)$$

其中 $P_{fix} = P_{raw} - \hat{\delta}_{phys}$，$\|\cdot\|_{safe} = \sqrt{\sum x_i^2 + \epsilon}$（避免距离趋零时梯度奇点）。

**物理含义**：预测修正后的轨迹到3个已知地面目标的斜距，必须与SAR测量值一致。

### 4.3 课程学习（λ_rcm 调度）

```
epoch < rcm_warmup_start(10)  : λ_rcm = 0
epoch in [10, 80)              : λ_rcm 线性增大到 1.0
epoch ≥ 80                    : λ_rcm = 1.0
```

策略：先让模型用MSE学会基本趋势，再逐步引入物理约束，避免初期梯度爆炸。

### 4.4 监控指标

| 指标 | 说明 | 是否参与反传 |
|------|------|------------|
| MSE(norm) | 归一化空间均方误差 | ✅ 是（主信号） |
| MSE(phys) | 物理空间均方误差（米²） | ❌ 仅监控 |
| Smooth | 时序平滑损失 | ✅ 是 |
| RCM | 斜距一致性损失 | ✅ 是（课程） |
| ValRMSE | 验证集位置RMSE（米） | ❌ 仅监控 |
| GradN | 梯度裁剪前的L2范数 | ❌ 仅监控 |

---

## 5. 训练策略

### 5.1 数据归一化

```python
# 仅在训练集上拟合统计量（防止验证集数据泄漏）
normalizer = DataNormalizer.fit(dataset, indices=train_idx)

# feat: Z-score 归一化 → 送入网络
feat_norm = (feat - feat_mean) / feat_std

# 预测输出: 在归一化 error 空间
# 反归一化: delta_phys = delta_norm * err_std + err_mean
```

### 5.2 优化器（AdamW）

```python
optimizer = AdamW(
    model.parameters(),
    lr = 1e-3,
    weight_decay = 5e-4    # L2正则，抑制过拟合
)
```

### 5.3 学习率调度（CosineAnnealingLR）

```
LR 从 1e-3 余弦衰减至 eta_min=1e-6
T_max = epochs // 2 = 100  （前半段衰减，后半段维持低LR）

目的：避免后期过拟合状态下LR仍偏高导致梯度不稳定
```

![LR曲线示意](assets/lr_cosine.png)

### 5.4 梯度累积 + 梯度裁剪

```python
# 每 accum_steps 个 mini-batch 才做一次 optimizer.step()
(loss / accum_steps).backward()

if is_last_in_accum:
    grad_norm = clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()

# effective_batch = batch_size × accum_steps × world_size
# 推荐: 4 × 2 × 2卡 = 16（或 8 × 2 × 2卡 = 32）
```

### 5.5 早停（Early Stopping）

```python
# val_loss 连续 patience 轮未改善则停止训练
--early_stop_patience 15
```

### 5.6 最优模型保存

每轮验证后，若 `val_total_loss` 创历史最优，自动保存 `weights/best.pt`。

### 5.7 分布式训练（DDP）

```bash
torchrun --standalone --nproc_per_node=2 train_pi_xlstm.py ...
```

- 使用 PyTorch DistributedDataParallel + NCCL 后端
- 梯度通过 All-Reduce 在 GPU 间同步
- 只有 rank=0 进程负责写指标、保存权重

### 5.8 混合精度（BF16 AMP）

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    delta_pred_norm = model(feat_normalized)
    loss = criterion(...)
```

BF16 不会溢出（无需 GradScaler），加速 Linear / attention 计算。

---

## 6. 推荐训练命令

```bash
cd /root/xLSTM
torchrun --standalone --nproc_per_node=2 train_pi_xlstm.py \
  --mat_path /root/xLSTM/dataset/PPTR_TrainDataset_5000.mat \
  --batch_size 8 \
  --grad_accum_steps 2 \
  --epochs 200 \
  --lr 1e-3 \
  --hidden_dim 128 \
  --num_blocks 4 \
  --num_heads 4 \
  --lambda_rcm_max 1.0 \
  --rcm_warmup_start 10 \
  --scheduler_eta_min 1e-5 \
  --val_ratio 0.1 \
  --early_stop_patience 15 \
  --checkpoint_every 10 \
  --param_report_every 10 \
  --run_name v11_fix_grad
# 以下参数取新默认值，无需显式指定：
# --lambda_smooth 2.0  --rcm_warmup_end 80  --dropout 0.1  --weight_decay 5e-4
```

---

## 7. 已知问题与当前状态

### 7.1 梯度爆炸（v11 已修复）

| 根因 | 修复方案 |
|------|---------|
| mLSTM C_matrix 归一化失效（正负抵消→分母趋零） | 梯度裁剪 1.0→0.5，监控 GradN |
| Bi-xLSTM 双路梯度叠加（≈2×放大） | output_layer 前加 Dropout(0.1) |
| RCM `torch.norm` 距离趋零时梯度奇点 | 替换为 `sqrt(sum²+1e-8)` |
| RCM 课程学习突然接入 + 过拟合 | warmup_end 延迟到 epoch 80 |
| lambda_smooth=5.0 放大差分梯度 | 降为 2.0 |

### 7.2 过拟合（持续存在）

- v11 最优 ValRMSE **1.56cm**（epoch 13），之后 Val/Train 比值升至 ≈2×
- 根本原因：**数据集同质性**，feat[9:12] 直接线性可逆为误差，模型不需要时序推理

### 7.3 RCM 损失未激活（持续存在）

- rcm 值始终 ≈ 1e-5，MSE 已经给了答案，RCM 无额外学习价值
- 需从数据集设计层面解决（增大测距噪声、改善参考点几何）

---

## 8. 数据集增强方向（待实施）

| 优先级 | 改进项 | 解决的问题 |
|--------|--------|-----------|
| P1 | 随机侧向偏置（±100m） | 坐标记忆 |
| P1 | 随机航向角（±15°） | 坐标记忆 |
| P1 | 阶跃跳变误差事件 | 误差模式单一 |
| P2 | RCM 噪声增大（5~30mm） | 单步精确求逆 |
| P2 | 近距离参考点（50~150m） | Z轴不可观测 |
| P2 | err_scale 分三档 | 误差量级单一 |
| P3 | S形机动轨迹 | 航迹形状单一 |

---

## 9. 文件结构

```
xLSTM/
├── train_pi_xlstm.py      # 主训练脚本
├── eval_pi_xlstm.py       # 评估/推理脚本
├── data_generate_v2.m     # MATLAB 数据生成脚本（当前版本 v4）
├── sar.m                  # SAR 成像脚本（反投影算法）
├── xlstm/                 # NX-AI/xlstm 源码（本地克隆）
├── dataset/               # 训练数据集目录
│   └── PPTR_TrainDataset_5000.mat
└── run/                   # 训练输出目录
    └── <timestamp>_<name>_pid<N>/
        ├── config.json        # 训练超参数快照
        ├── metrics.jsonl      # 每轮指标（JSON Lines 格式）
        ├── viz/
        │   └── training_curves.png
        └── weights/
            ├── best.pt        # 验证集最优权重
            ├── latest.pt      # 最新权重
            └── checkpoint_epoch_XXXXX.pt
```

---

## 10. 评估

```bash
BEST=/root/xLSTM/run/<run_dir>/weights/best.pt

python eval_pi_xlstm.py \
  --checkpoint $BEST \
  --mat_path /root/xLSTM/dataset/PPTR_TrainDataset_5000.mat \
  --batch_size 8 \
  --save_p_fix run/p_fix_result.mat \
  --also_save_delta
```

关键评估指标：
- **ValRMSE**：验证集位置 RMSE（米），目标 < 1cm
- **MSE(phys)**：物理空间均方误差（米²）
- **RCM**：斜距一致性残差（越小越好）
