# PI-xLSTM v4 Cascade 训练总结（v20 / geom_5000）

> 本文档用于与更强模型讨论：当前训练任务、现象、根因分析与待决方案。  
> 项目路径：`/root/xLSTM`  
> 主训练脚本：`train_pi_xlstm.py`  
> 数据生成：`data_generate_v4.m`  
> 评估：`eval_pi_xlstm.py`

---

## 1. 任务目标

在 SAR 轨迹校正场景下，用 **PI-xLSTM Cascade（双头）** 从带误差的导航轨迹 `P_raw` 预测修正量，使校正后轨迹 **`P_fix ≈ P_true`**：

```
P_coarse = P_raw - δ_c          # 粗修
P_fix    = P_raw - δ_c - δ_f    # 粗修 + 精修
```

- **`P_true`**：仿真真值航迹（ground truth）
- **`P_raw`**：带误差的导航航迹（真值 + 误差），**不是**真值
- 训练监督：粗修对齐 `Delta_coarse`；精修对齐 `delta_fine = (P_raw - P_true) - Delta_coarse_GT`（在 loader 中现算）；另有 RCM 斜距约束与相位约束

---

## 2. 数据（v4 geom）

**文件**：`/root/xLSTM/gene4/dataset1/PPTR_TrainDataset_v4_geom_5000.mat`（5000 样本；另有 `geom_10.mat` 用于快速 eval）

| 变量 | 形状/含义 |
|------|-----------|
| `Feat_All` | `[N, 2048, 12]` |
| feat 1–6 | `P_raw` 及速度分量 |
| feat 7–9 | **RCM_true**：`P_true` 到参考点 A/B/C 的几何斜距（米，未归一化） |
| feat 10–12 | 导航相对真值的斜距残差 |
| `P_raw`, `P_true` | `[N, 3, T]`（HDF5 读入后 transpose） |
| `Delta_coarse` | 粗修标签 |
| `Phase_sin`, `Phase_cos` | 由 **P_true** 几何斜距得到 φ=-4πR/λ，unwrap + 线性去趋势 |
| `RCM_true/recorded/err` | 存盘可选；**训练未直接加载** |

**说明**：当前为 **几何仿真** pipeline（`data_generate_v4.m`），已删除真实回波脉压/峰搜 RCM 提取路径；观测斜距来自 `P_true`，非实测 SAR。

---

## 3. 模型与损失（Cascade）

**结构**：`PI_xLSTM_Cascade` — 共享 xLSTM backbone + `head_coarse` + `head_fine`

**`CascadePhysicsLoss` 主要项**：

| 项 | 作用对象 | 说明 |
|----|----------|------|
| `l_mse_c` | δ_c | 对齐 `Delta_coarse`（归一化 MSE） |
| `l_mse_f` | δ_f | 对齐 `delta_fine` GT |
| `l_rcm` | **仅 P_coarse** | 预测斜距 vs feat 7–9；除以 `mean(err_std²)` 无量纲化 |
| `l_phase` | **P_fix** | `phi_obs=atan2(Phase_sin,cos)` vs `phi_ideal` from **P_fix** 几何斜距；`1-cos(Δφ)` |
| `l_smooth` | δ_c + δ_f | 修正量时间差分平滑 |
| `l_mse_total` | δ_c+δ_f | 对齐整段 `(P_raw-P_true)`（课程后期才加权） |

**关键代码位置**（`train_pi_xlstm.py`）：

- RCM：`p_coarse = p_raw - delta_c_phys`，与 `feat[:,:,6:8]` 比较
- Phase：`p_fix = p_raw - delta_c_phys - delta_f_phys`，`phi_ideal = -4π·r(P_fix)/λ`
- `detach_coarse`（until epoch 60）：仅在 **精修头输入** 上对 `delta_c_phys.detach()`，**不**阻止 phase/mse_f 对 δ_c 的梯度

---

## 4. 当前训练 Run

| 项 | 值 |
|----|-----|
| Run 目录 | `/root/xLSTM/run/20260518_104331_912254_v20_cascade_geom_5000~_pid90519/` |
| 启动 | `torchrun --standalone --nproc_per_node=2 train_pi_xlstm.py` |
| 数据 | `PPTR_TrainDataset_v4_geom_5000.mat` |
| batch | 8 × grad_accum 2（有效 16） |
| epochs | 200 |
| lr | 5e-4，CosineAnnealing |
| hidden | 128，4 blocks，4 heads |

**课程学习参数**：

```text
freeze_fine_until_epoch:     40    # epoch<40 冻结 head_fine
detach_coarse_until_epoch:     60    # epoch<60 精修输入 detach δ_c
f_warmup / phase_warmup:       40 → 100   (λ_f, λ_φ 线性 0→1)
total_warmup:                  100 → 160  (λ_total 0→0.3)
rcm_warmup:                    0 → 80     (λ_rcm 0→1)
lambda_smooth:                 2.0
lambda_c_max / f / phase:      1.0 / 1.0 / 1.0
early_stop_patience:           0
val_ratio:                     0.1
```

**DDP**：`find_unused_parameters=True`（兼容 freeze_fine）；epoch 40 后出现 “unused parameters” 警告，可忽略。

---

## 5. 训练现象（至约 epoch 50）

### 5.1 验证指标（主要 KPI）

- **`ValRMSE_total`**：稳定在 **~0.0459–0.0464 m（4.6 cm）**
- **`ValRMSE_coarse`**：与 total **几乎相同**（精修未带来可测位置增益）
- **`best.pt`**：多在 **epoch 32 前后**（val_rmse_total ≈ 4.59 cm）

### 5.2 分阶段

| 阶段 | Epoch（打印） | 现象 |
|------|----------------|------|
| 粗修期 | 1–40 | `λ_f=λ_φ=0`，精修 frozen；`mse_c` 很小；ValRMSE 快速到 ~4.6 cm 平台 |
| 精修/相位开启 | 41+ | `head_fine` 解冻；`λ_f, λ_φ` 从 0 线性增加 |

### 5.3 epoch 41+ 训练 loss 变化

| 指标 | 趋势 | 说明 |
|------|------|------|
| **Total** | 0.004 → 0.2+ | 主要由 `λ_f·mse_f + λ_φ·phase` 打开导致，**不代表验证变差** |
| **mse_f** | ~0.69–0.71 | 归一化空间，一直很大 |
| **phase** | 0.97 → 0.67 | **相位 loss 在下降**（相位监督生效） |
| **RCM (l_rcm)** | 0.001 → 0.011 | **粗修斜距拟合变差**（见下节） |
| **mse_c** | 略升 | 粗修标签 MSE 从 ~0.0006 到 ~0.004 |
| **GradN** | 41 轮 spike ~35 后回落 | 解冻精修所致；总 GradN 有效 |

### 5.4 日志噪音

- `[GradN 分项] MSE/Smooth/RCM: nan`：**cascade 未实现分项诊断**（`if not is_cascade`），可忽略；旁路 `GradN` 总数有效。

---

## 6. 核心问题：多损失冲突 + 早期粗修可能被后期冲掉

### 6.1 现象

epoch 41 后：

1. **phase 在学**（`l_phase` 下降）
2. **RCM (`l_rcm`) 缓慢上升**
3. **ValRMSE 几乎不变**，total ≈ coarse

### 6.2 根因（机制）

1. **RCM 只约束 `P_coarse`**，与 feat 7–9（P_true 几何斜距）一致。
2. **Phase 约束 `P_fix`**，梯度同时更新 **δ_c 和 δ_f**（`p_fix` 未对 δ_c detach）。
3. **粗修同时受**：`mse_c`（对齐 Delta_coarse）、`RCM`、`phase`（经 P_fix 拉 δ_c）— 三者不完全等价。
4. **共享 backbone**：精修/相位开启后，表征漂移，**间接影响粗修输出**。
5. **`detach_coarse`** 只切断精修 **输入路径**，**不**切断 phase/mse_f/smooth 对 δ_c 的梯度。

### 6.3 用户关切

> 前面 40 轮学好的 RCM/粗修，会不会被后面改坏？

**答：会。** 没有独立的 “RCM 权重”；`backbone + head_coarse` 持续更新。早期斜距拟合可能被 phase/精修梯度部分覆盖。若需保留粗修状态，应 **freeze coarse** 或 **phase 对 δ_c stop-grad**，或使用 **epoch 40 checkpoint**。

### 6.4 与 pipeline 设计相关的结构性限制

（前期分析结论，供讨论参考）

| 点 | 说明 |
|----|------|
| 粗修天花板 | δ_c 主要 Y 向；与 feat10–12 相关性 r≈-0.9996；验证 RMSE ~4.6 cm 已接近“完美粗修标签”水平 |
| 精修监督 | `delta_fine` 用 **GT 粗修** 计算，推理用 **预测 δ_c**，存在 train/infer mismatch |
| geom 数据 | 观测来自 P_true 几何，非真实 SAR RCM；泛化到实测待验证 |

---

## 7. 已讨论的解决方案（未全部实现）

| 方案 | 做法 | 预期效果 |
|------|------|----------|
| **A. 分阶段冻结粗修** | epoch≥40 后 `head_coarse.requires_grad=False`，只训精修 | 保住粗修/RCM；精修不能改 δ_c |
| **B. phase 断梯度** | `p_fix_phase = p_raw - delta_c.detach() - delta_f` 仅用于 `l_phase` | phase 不再改 δ_c，减轻与 RCM 冲突 |
| **C. 推迟 phase** | e.g. `phase_warmup_start=80` | 先让 mse_f 学位移，后加相位 |
| **D. 提前拉满 λ_rcm** | `rcm_warmup_end=40` | 加强斜距约束权重 |
| **E. 多 checkpoint 对比** | eval `epoch_40`、`best`、`latest` | 选粗修/全链路最优部署点 |

**建议下一轮优先**：B + C；若 RCM 仍恶化再加 A。

**待加代码开关（用户曾表示可加）**：

- `--freeze_coarse_from_epoch`
- `--phase_detach_coarse`

---

## 8. 评估命令示例

```bash
cd /root/xLSTM
RUN=/root/xLSTM/run/20260518_104331_912254_v20_cascade_geom_5000~_pid90519
LATEST=$RUN/weights/best.pt   # 或 checkpoint_epoch_00040.pt

python eval_pi_xlstm.py \
  --checkpoint "$LATEST" \
  --mat_path /root/xLSTM/gene4/dataset1/PPTR_TrainDataset_v4_geom_5000.mat \
  --batch_size 8 \
  --json_out "$RUN/eval_best.json"

python eval_pi_xlstm.py \
  --checkpoint "$LATEST" \
  --mat_path /root/xLSTM/gene4/dataset1/PPTR_TrainDataset_v4_geom_10.mat \
  --batch_size 4 \
  --json_out "$RUN/eval_geom10.json" \
  --save_p_fix "$RUN/test/p_fix_geom10.mat" \
  --also_save_delta
```

---

## 9. 希望更强模型帮助回答的问题

1. **课程设计**：`freeze_fine=40` 与 `f/phase_warmup=40` 同时开启是否必然导致粗修被冲？推荐的时间表？
2. **phase loss 是否应对 δ_c detach**？对 SAR 物理一致性 vs 位置 RMSE 的权衡？
3. **精修标签**是否应改为基于 **预测 δ_c** 的 `delta_fine`，以消除 train/infer gap？
4. **feat 7–9** 是否应改为 `RCM_recorded`（P_raw 几何）而非 `RCM_true`，更贴近实测？
5. 在 **ValRMSE≈coarse** 前提下，是否应放弃精修/相位，或改为 **单头 + RCM** 更简单 pipeline？
6. **λ 权重 / warmup** 是否有更优默认（避免 Total 暴涨但 Val 不动）？

---

## 10. 关键文件索引

| 文件 | 用途 |
|------|------|
| `train_pi_xlstm.py` | 训练、CascadePhysicsLoss、课程学习 |
| `eval_pi_xlstm.py` | 验证、导出 P_fix |
| `data_generate_v4.m` | v4 geom 数据 |
| `mingling` | 命令备忘 |
| `run/.../metrics.jsonl` | 逐 epoch 指标 |
| `run/.../viz/training_curves.png` | 曲线 |
| `run/.../weights/best.pt` | 按 val_rmse_total 保存 |

---

## 11. 一句话结论

**v20 cascade 在 geom_5000 上粗修约 40 epoch 即达到 ~4.6 cm 验证平台；41 epoch 后精修与相位开始训练，相位 loss 下降但 RCM 项上升、ValRMSE 未改善，疑为多损失共享网络且 phase 反传修改 δ_c 所致。需讨论是否冻结粗修、phase 断梯度、推迟相位或修正精修/RCM 监督定义。**

---

*文档生成上下文：2026-05-18，训练仍在进行（目标 200 epoch）。*
