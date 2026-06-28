# P0 仿真对齐实测 — 改动清单

> 日期: 2026-06-28 | 分支: feature/p0-sim-realign | 状态: ✅ 已实现并推送

---

## 改动总览

| 仓库 | 文件 | 改动类型 | 提交 |
|------|------|----------|------|
| ✅ | `data_generate_v4.m` | 参数 + 布局 + RCM 升级 + GPS 中断 | d9494d6 |
| ✅ | `train_pi_xlstm.py` | 默认超参数对齐 | bc331bf |
| ✅ | `sar_batch_validate_with_Pfix.m` | fc 同步 9.6 GHz | e36bfbb |
| ✅ | `test_rcm_geom_vs_extract.m` | fc fallback 同步 9.6 GHz | e36bfbb |

---

## 1. `data_generate_v4.m` — 核心仿真脚本

### 1.1 飞行参数 (当前行 78, 85, 143)

```
当前:
  H_target = 60 + 80*rand();        % 60~140 m
  Z_des = H_target * ones(1, Ny);

改为:
  H_target = 280 + 40*rand();       % 280~320 m
  Z_des = H_target * ones(1, Ny);
```

### 1.2 载频 (当前行 29-31)

```
当前:
  radar.fc = 9.5e9;

改为:
  radar.fc = 9.6e9;
```

### 1.3 参考点坐标生成 (当前行 89)

**当前逻辑**（v4）:
```matlab
gen_pos = @() single([350 + 40*rand();           % Y: [350, 390]
                       total_x*(0.1+0.8*rand());  % X: 航迹 10%~90%
                       Z_max*rand()]);             % Z: [0, Z_max]
```

**改为**（100m×100m 场景，地面距中心 ~954 m，斜距 ~1000 m）:
```matlab
d_center = 954;
Pos_A = single([d_center + 0;   total_x * 0.5 + 0;   0]);  % 中心   R≈1000m
Pos_B = single([d_center + 40;  total_x * 0.5 + 40;  0]);  % 远边   +40m Y/X
Pos_C = single([d_center - 40;  total_x * 0.5 - 40;  0]);  % 近边   -40m Y/X
```

**改动要点**：
- Y 坐标从 `[350, 390]`（40m 窄带）改为以 954m 为中心、±40m 的 80m 范围
- X 坐标从一个区间 `[10%, 90%]` 改为中心 ±40m 的确定值，保证沿航迹拉开 ¥80m
- 三个点固定布局（A 中心 / B 远边 / C 近边），不再随机
- Z 高度从 `[0, Z_max]` 改为固定 0（地面点）

### 1.4 参考点约束检查 (当前行 99-131)

由于改为固定坐标布局，原有的随机约束循环可以简化或移除。保留最小检查：三点 pairwise XY 距离 ≥ 5m（已验证满足）。

### 1.5 RCM 仿真升级 (当前行 255-258)

**当前**（v4_phase）: 纯几何斜距，无噪声，直接存相位
```matlab
R_slant = vecnorm(p - p_tgt, 2, 1).';
r_ref_j = min(R_slant);
phase = -4.0 * pi * fc * (double(R_slant) - r_ref_j) / c;
```

**改为**: 在 `compute_phase_rel_rref` 函数内添加 RCM 测量噪声仿真，并返回 `R_obs_nt3`
```matlab
dr = c / (2 * 200e6);       % 距离分辨率 ≈0.75m (B≈200MHz)

% 1. 多点散射叠加（相干斑模拟）
scatter_amp = 0.1 * randn(3, 1);
scatter_dr  = dr * (rand(3, 1) - 0.5);
R_equiv = R_slant + sum(scatter_amp .* scatter_dr) / (sum(abs(scatter_amp)) + eps);

% 2. 多径效应（5% 概率）
if rand() < 0.05
    mp_delay = dr * (0.5 + 1.5 * rand());
    mp_amp   = 0.05 + 0.1 * rand();
    R_equiv = R_equiv + mp_amp * mp_delay;
end

% 3. 距离向采样抖动（±0.1 距离单元）
R_jitter = R_equiv + dr * 0.2 * (rand() - 0.5);

% 4. 随机野值（2% 概率，±2 距离单元）
R_outlier = R_jitter;
outlier_mask = rand(Ny, 1) < 0.02;
R_outlier(outlier_mask) = R_jitter(outlier_mask) + dr * 4 * (rand(sum(outlier_mask), 1) - 0.5);

% 5. 阴影效应（低掠射角时 SNR 退化）
if abs(p(3,1) - p_tgt(3)) / mean(R_slant) < 0.15
    shadow_scale = 3.0;
else
    shadow_scale = 1.0;
end
R_obs = R_outlier + shadow_scale * 0.003 * randn(size(R_outlier));

% R_ref 和相位从 R_obs 计算
r_ref_j = min(R_obs);
phase = -4.0 * pi * fc * (double(R_obs) - r_ref_j) / c;
```

### 1.6 GPS 中断段仿真 (新增)

**目标**: 模拟 GPS 信号丢失时 IMU-only 的位置漂移
```matlab
% === GPS 中断段仿真 ===
% 在 10% 的样本中插入 GPS 中断
gps_outage_prob = 0.1;
if rand() < gps_outage_prob
    % 随机选 1~3 个中断段
    n_outages = randi([1, 3]);
    for oi = 1:n_outages
        % 中断持续 1~5 秒（PRF 通常 256~1024 Hz，对应 256~5120 帧）
        outage_start = randi([Ny*0.2, Ny*0.7]);        % 航迹 20%~70% 区间
        outage_len   = randi([256, 5120]);              % 1~5 秒
        outage_end   = min(outage_start + outage_len, Ny);
        
        % IMU 速度短期仍可靠，但位置漂移逐渐增大
        % 用累积随机游走模拟位置漂移
        drift_rate = 0.01 + 0.04*rand();                % m/s，漂移率
        t_outage = (0:(outage_end-outage_start)) / PRF;
        drift_y = drift_rate * cumsum(randn(1, length(t_outage))) * sqrt(1/PRF);
        drift_x = drift_rate * cumsum(randn(1, length(t_outage))) * sqrt(1/PRF);
        drift_z = drift_rate * 0.3 * cumsum(randn(1, length(t_outage))) * sqrt(1/PRF);
        
        p_raw(:, outage_start:outage_end) = p_raw(:, outage_start:outage_end) ...
            + [drift_y; drift_x; drift_z];
    end
end
```

### 1.7 保存字段更新

新增/修改保存到 `.mat` 的字段：
- `R_obs` (新): 仿真 RCM 斜距 [T, 3]（替代直接存 Phase 的做法——Phase 仍存，但 R_obs 作为独立字段）
- `fc` (改): 9.6e9
- `H_target_all` (改): 范围 280~320

---

## 2. `train_pi_xlstm.py` — 训练脚本

### 2.1 默认超参数对齐

```python
# 已修改的默认值:
--hidden_dim   64 → 128        # 对齐 §2.5
--num_blocks    2 → 4           # 对齐 §2.5
--batch_size    4 → 8           # 对齐 §5.3
--epochs      100 → 200         # Stage 1 满训练
```

### 2.2 SARDataset 数据加载（待完成）

新增的 `R_obs` 字段尚未接入训练脚本的数据加载器。当前 `SARDataset` 在 v4_phase 模式下仅读 `Feat_All[:,:,:6]`（6 维）+ `Phase_rel`。后续需要：
- 在 `SARDataset` 中加载 `R_obs` [B, T, 3]
- 将 6 维 feat 扩展为 12 维：`[P_raw, V_raw, R_obs, ‖P_raw−Pos‖−R_obs]`
- 更新 `DataNormalizer` 以支持 12 维

### 2.3 RealSARDataset（实测数据加载，待完成）

实测 `.mat` 的字段名需要与仿真对齐：`R_obs` 作为独立字段。

---

## 3. `sar_batch_validate_with_Pfix.m` — SAR 成像验证

### 3.1 载频 (当前行 77)

```matlab
fc = 9.5e9;  →  fc = 9.6e9;
```

### 3.2 场景中心 (当前行 91)

```matlab
H_target = 300;  % 已正确，无需改
```

---

## 4. `test_rcm_geom_vs_extract.m` — RCM 对比测试

### 4.1 载频 (当前行 59-61)

```matlab
fc = double(S.fc(1));  % 优先从 mat 读取，无需硬编码
% 仅 fallback 时：fc = 9.5e9 → fc = 9.6e9
```

---

## 后续工作

### 待完成（当前分支未包含）
- **SARDataset 12 维特征接入**：将 `R_obs` 字段加载并组装为完整的 12 维输入 + feat_mask
- **本地 smoke test**：MATLAB 生成小批量（10 条）→ 验证 train_pi_xlstm.py 能加载并跑通
- **基线对比实验**：用旧版 v4 数据与新数据分别训练，对比 ValRMSE

---

## 不需要改的文件

| 文件 | 原因 |
|------|------|
| `data_generate.m`, `data_generate_v2.m`, `data_generate_v3.m` | 旧版，保留用于对比 |
| `main.m` | MATLAB 主控，不改架构只调 `data_generate_v4` |
| `sar.m` | 实测 BP 成像，fc 从加载数据读取，不硬编码 |
| `baselines.py` | 基线模型，输入输出接口不变 |
| `eval_pi_xlstm.py` | 评估脚本，接口不变 |
| `summarize_results.py` | 结果汇总，接口不变 |
