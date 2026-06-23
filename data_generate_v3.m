% =========================================================================
% SAR 轨迹精化 (PPTR) 深度学习批量数据集生成脚本 (泛化增强版 v6.3)
% 物理模型: 随机直线航迹 -> 叠加风场扰动与飞控震荡 (P_true) -> 叠加连续综合误差 (P_raw)
%
% v6.3 相比 v6.2 关键改动（用户要求小幅缩窄位置偏置）：
%   ◆ 保留"随机航向旋转 ±15°"机制（沿用 v5 设计，刚性变换不影响 RCM 几何）
%       - p_true / p_raw / V_raw / Pos_A/B/C 同步绕 Z_up 旋转
%       - 打破模型对"飞行方向 = +X"的绝对坐标记忆
%   ◆ 侧向偏置 ±10m（v6.3 由原 ±100m 缩小）
%       - 仅 Y_lat 平移，不影响 RCM 几何
%       - 飞机起点紧贴 Y=0，可视化更紧凑，但仍打破"起点 Y=0"记忆
%
% v6.2 相比 v6.1 关键改动（用户要求加大误差到消费级 GPS 漂移量级）：
%   ◆ err_scale 三档整体放大 2×：
%       小档 [0.40, 1.00]  ←  原 [0.20, 0.50]
%       中档 [1.00, 3.00]  ←  原 [0.50, 1.50]
%       大档 [3.00, 5.00]  ←  原 [1.50, 2.50]
%   ◆ 数据集 err_std 由约 34 mm/轴 升至约 70 mm/轴（×2.06）
%   ◆ 单条样本 max|P_raw - P_true| 99% 分位由 ~14 cm 升至 ~28 cm
%   ◆ 训练目标变化：模型需要纠正更大的误差，对应消费级 GPS 短时漂移场景
%   ◆ 参考点几何约束加强（消除三点共线/极扁三角形病态）：
%       原 v6.1: 仅要求两两间距 ≥ 10m
%       新 v6.2: 两两间距 ≥ 10m & 三角形最小高 ≥ 20m & 最小内角 ≥ 5°
%       → 保证 RCM 三方程的线性独立性，避免反求飞机位置时矩阵退化
%
% v6.1 相比 v6 关键改动（航迹完全连续，移除所有非平稳跳变事件）：
%   ◆ 完全禁用 E_step 阶跃跳变（原 0~5 次"GPS 跳变"事件已置零）
%   ◆ 完全禁用 0.2% 突发侧向阵风（原瞬时 2-3 m/s 附加风已注释关闭）
%   ◆ P_raw 现在只包含连续误差成分：E_trend + E_rw + E_sine + E_white
%   ◆ P_true 现在完全由 6-DOF 动力学 + 平滑风场（无突发事件）演化
%   → P_true 与 P_raw 均为时间的连续函数，无任何阶跃 / 突跳
%
% v6 相比 v5 主要改动（控制极端尾部样本，避免 sLSTM 梯度爆炸）：
%   1. err_scale 大档上限收窄（v6 → 2.5；v6.2 由于整体 ×2 后单独评估）
%   2. 参考点高度随机化 [0, 20]m（原固定 0m）
%      → 覆盖地面 / 植被 / 低层建筑等真实地物高度，提升几何泛化
%   3. 统计并报告每条样本的 max|E_total|，便于离线评估极端样本占比
%   4. （新增）`v6_clip_meta` 字段保存到 .mat，记录本次生成的元数据
%
% 注：v6.1 起 E_step 框架置零，元数据 'E_step_enabled=0' 记录此状态。
%
% 与训练侧（train_pi_xlstm.py 的 SARDataset）对接 100% 兼容：
%   - 字段名：Feat_All / P_raw / P_true / Pos_A / Pos_B / Pos_C 全部保留
%   - 维度顺序：Feat_All [N,T,12]、P_*[N,T,3]、Pos_*[3,N] 不变
%   - 坐标系：[Y_lat, X_fwd, Z_up]，旋转 + 偏置后统一
%
% 特征说明 (Feat_All 12维):
%   feat[:, 1:3]  = P_raw (平滑后的导航轨迹，已含航向旋转+侧向偏置)
%   feat[:, 4:6]  = V_raw (P_raw 数值微分速度，平滑，已含旋转)
%   feat[:, 7:9]  = RCM 斜距测量 (从 P_true 计算 + 随机测量噪声)
%   feat[:, 10:12]= 导航-RCM 残差 (||P_raw-Pos_X|| - RCM，导航误差投影)
%
% 输出格式: Feat_All [N, T, 12], P_true [N, T, 3], P_raw [N, T, 3]
%           Pos_A_all, Pos_B_all, Pos_C_all [3, N]（每条样本各自的特显点，含旋转+偏置）
% =========================================================================
clear; clc; close all;

%% 0. 批量生成参数设置
num_samples = 5000;
output_dir  = 'PPTR_Dataset_5';   % v6 输出到新目录，避免覆盖 v5 的 PPTR_Dataset_4
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% 1. 固定仿真基础参数
Ny  = 2048;   % 单条航迹采样点数 (4096)
PRF = 500;
dt  = 1 / PRF;    % 仿真步长 0.001s

% v6 新增：E_step 累积幅度 clip 阈值（单方向，米）
% 经离线分析：原 v5 在 err_scale=3 + n_jumps=5 时可产生 0.75 m 的 Y_lat 偏移
% （是数据集 err_std≈0.034m 的 22 倍），触发 sLSTM exp 门控数值溢出
E_step_clip = [0.050; 0.030; 0.020];   % [Y_lat; X_fwd; Z_up]

% RCM 测距噪声：v5 起每条样本独立随机化（5~30mm），在循环内赋值

%% 2. 预分配深度学习训练集张量
Feat_All  = zeros(num_samples, Ny, 12);
P_true_all = zeros(num_samples, Ny, 3);
P_raw_all  = zeros(num_samples, Ny, 3);

% 每条样本特显点也需保存（训练/评估时要用）
Pos_A_all = zeros(3, num_samples);
Pos_B_all = zeros(3, num_samples);
Pos_C_all = zeros(3, num_samples);

% 每条样本的飞行参数（成像脚本恢复 R_c 时必须）
H_target_all = zeros(num_samples,1);
V_target_all = zeros(num_samples,1);

% v6 新增：诊断信息（每条样本的 max|E_total| 与 err_scale）
err_scale_all   = zeros(num_samples,1);
max_err_all     = zeros(num_samples,1);   % max|P_raw - P_true| over time, 3D 范数
n_jumps_all     = zeros(num_samples,1);

% v6.2 新增：每条样本的三角形几何指标（旋转/偏置前的"原始"参考点坐标）
ref_min_height_all = zeros(num_samples,1);  % 三角形最长边上的高 (m)
ref_min_angle_all  = zeros(num_samples,1);  % 三角形最小内角 (deg)
ref_resample_all   = zeros(num_samples,1);  % 几何重采样次数（高表示几何约束较紧）

%% 3. 主生成循环
disp(['开始生成 ', num2str(num_samples), ' 组泛化增强样本数据 (v6)...']);
tic;

for iter = 1:num_samples
    if mod(iter, 50) == 0 || iter == 1
        disp(['当前进度: ', num2str(iter), ' / ', num2str(num_samples), ...
              ' (耗时: ', num2str(toc,'%.1f'), ' 秒)']);
    end

    % =================================================================
    % --- A. 每条样本随机化物理参数 ---
    % =================================================================

    m   = 8.0  * (1 + 0.15 * (2*rand()-1));
    g   = 9.81;
    Ixx = 0.08 * (1 + 0.15 * (2*rand()-1));
    Iyy = 0.08 * (1 + 0.15 * (2*rand()-1));
    Izz = 0.15 * (1 + 0.15 * (2*rand()-1));

    Kp_pos = 0.8 * (1 + 0.15*(2*rand()-1));
    Kd_pos = 1.2 * (1 + 0.15*(2*rand()-1));
    Kp_z   = 2.5 * (1 + 0.15*(2*rand()-1));
    Kd_z   = 1.5 * (1 + 0.15*(2*rand()-1));
    Kp_att = 5.0 * (1 + 0.15*(2*rand()-1));
    Kd_att = 0.8 * (1 + 0.15*(2*rand()-1));
    Kp_yaw = 2.0 * (1 + 0.15*(2*rand()-1));
    Kd_yaw = 0.5 * (1 + 0.15*(2*rand()-1));

    v_target = 7  + 6 * rand();       % 7~13 m/s
    H_target = 60 + 80 * rand();      % 60~140 m
    H_target_all(iter) = H_target;
    V_target_all(iter) = v_target;

    time   = (0:Ny-1) * dt;
    X_des  = v_target * time;
    Y_des  = zeros(1, Ny);
    Z_des  = H_target * ones(1, Ny);

    % =================================================================
    % --- B. 随机化特显点位置 ---
    %     坐标系: Pos_X = [Y_lat; X_fwd; Z]，与 p_true 一致
    %     地距向 (Y_lat, 第1分量): 均匀分布 [300, 500]
    %     方位向 (X_fwd, 第2分量): 沿飞行轨迹 10%~90% 区间
    %     高度向 (Z_up,  第3分量): 均匀分布 [0, 20]m（v6 新增）
    %         覆盖地面 / 低矮植被 / 1-6 层建筑物等真实地物高度，
    %         每个参考点独立采样，与飞行高度 H_target∈[60,140] 仍有
    %         至少 40m 的垂直间隔，几何不会病态。
    % =================================================================
    total_x = v_target * (Ny-1) * dt;   % 飞行总 X 距离
    Z_max   = 20.0;                     % 地面目标最大高度（米）
    gen_pos = @() [300 + 200*rand(); total_x*(0.1 + 0.8*rand()); Z_max*rand()];

    Pos_A = gen_pos();
    Pos_B = gen_pos();
    Pos_C = gen_pos();

    % v6.2 几何约束（保证三点构成"良好" RCM 几何）：
    %   (1) 两两间距 ≥ 10m         → 避免重合
    %   (2) 三角形最小高 ≥ 20m     → 避免共线（线性求解会退化）
    %   (3) 最小内角 ≥ 5°           → 避免极扁三角形（数值条件数差）
    % 仅用 XY 平面（[Y_lat, X_fwd]）判定，Z 维量级太小不参与
    geom_ok = false;
    resample_count = 0;
    min_height = 0; min_angle_deg = 0;   % 在 while 外部声明，便于循环后保存
    while ~geom_ok
        AB = norm(Pos_A(1:2) - Pos_B(1:2));
        BC = norm(Pos_B(1:2) - Pos_C(1:2));
        CA = norm(Pos_A(1:2) - Pos_C(1:2));
        if AB < 10 || BC < 10 || CA < 10
            Pos_A = gen_pos(); Pos_B = gen_pos(); Pos_C = gen_pos();
            resample_count = resample_count + 1;
            continue;
        end
        v1 = Pos_B(1:2) - Pos_A(1:2);
        v2 = Pos_C(1:2) - Pos_A(1:2);
        tri_area = 0.5 * abs(v1(1)*v2(2) - v1(2)*v2(1));
        max_edge = max([AB, BC, CA]);
        min_height = 2 * tri_area / max_edge;        % 最长边上的高
        cos_A = (AB^2 + CA^2 - BC^2) / (2*AB*CA);
        cos_B = (AB^2 + BC^2 - CA^2) / (2*AB*BC);
        cos_C = (BC^2 + CA^2 - AB^2) / (2*BC*CA);
        min_angle_deg = min([acosd(cos_A), acosd(cos_B), acosd(cos_C)]);
        if min_height < 20 || min_angle_deg < 5
            Pos_A = gen_pos(); Pos_B = gen_pos(); Pos_C = gen_pos();
            resample_count = resample_count + 1;
            continue;
        end
        geom_ok = true;
    end
    ref_min_height_all(iter) = min_height;
    ref_min_angle_all(iter)  = min_angle_deg;
    ref_resample_all(iter)   = resample_count;

    % =================================================================
    % --- C. 随机化风场模型 ---
    % =================================================================
    wind_amp  = rand(3,1) * 1.5;
    wind_freq = 0.05 + 0.2*rand(3,1);
    wind_phase = 2 * pi * rand(3, 1);
    wind_noise_std = 0.02 + 0.05*rand();

    % =================================================================
    % --- D. 6-DOF 物理动力学仿真（生成 P_true）---
    % =================================================================
    Pos   = zeros(3, Ny);  Pos(:,1)   = [0; 0; H_target];
    Vel   = zeros(3, Ny);  Vel(:,1)   = [v_target; 0; 0];
    Att   = zeros(3, Ny);
    Omega = zeros(3, Ny);

    for k = 1:Ny-1
        wind_x = wind_amp(1)*sin(2*pi*wind_freq(1)*time(k)+wind_phase(1)) + wind_noise_std*randn();
        wind_y = wind_amp(2)*sin(2*pi*wind_freq(2)*time(k)+wind_phase(2)) + wind_noise_std*randn();
        wind_z = wind_amp(3)*sin(2*pi*wind_freq(3)*time(k)+wind_phase(3)) + wind_noise_std*0.5*randn();

        % v6.1: 已禁用突发侧向阵风（用户要求航迹完全连续，避免动力学颠簸）
        % 原 0.2% 概率注入 2-3 m/s 瞬时阵风，经 PID 滤波后位置颠簸 < 5mm，
        % 严格意义上仍是连续函数（被控制系统平滑），但为彻底消除"非平稳事件"故关闭。
        % if rand() < 0.002
        %     wind_y = wind_y + (2.0 + rand());
        % end
        V_wind = [wind_x; wind_y; wind_z];

        e_p = [X_des(k); Y_des(k); Z_des(k)] - Pos(:,k);
        e_v = [v_target; 0; 0] - Vel(:,k);
        U1 = m*g + Kp_z*e_p(3) + Kd_z*e_v(3);

        phi_des   = -(Kp_pos*e_p(2) + Kd_pos*e_v(2)) / g;
        theta_des =  (Kp_pos*e_p(1) + Kd_pos*e_v(1)) / g;
        psi_des   = 0;

        phi_des   = max(min(phi_des,   0.3), -0.3);
        theta_des = max(min(theta_des, 0.3), -0.3);

        U2 = Kp_att*(phi_des   - Att(1,k)) + Kd_att*(0 - Omega(1,k));
        U3 = Kp_att*(theta_des - Att(2,k)) + Kd_att*(0 - Omega(2,k));
        U4 = Kp_yaw*(psi_des   - Att(3,k)) + Kd_yaw*(0 - Omega(3,k));

        p_dot = (U2 - (Izz-Iyy)*Omega(2,k)*Omega(3,k)) / Ixx;
        q_dot = (U3 - (Ixx-Izz)*Omega(1,k)*Omega(3,k)) / Iyy;
        r_dot = (U4 - (Iyy-Ixx)*Omega(1,k)*Omega(2,k)) / Izz;

        Omega(:,k+1) = Omega(:,k) + [p_dot; q_dot; r_dot]*dt;
        Att(:,k+1)   = Att(:,k)   + Omega(:,k)*dt;

        phi=Att(1,k+1); theta=Att(2,k+1); psi=Att(3,k+1);
        R_x = [1 0 0; 0 cos(phi) -sin(phi); 0 sin(phi) cos(phi)];
        R_y = [cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)];
        R_z = [cos(psi) -sin(psi) 0; sin(psi) cos(psi) 0; 0 0 1];
        R   = R_z * R_y * R_x;

        Acc = (R*[0;0;U1] - [0;0;m*g]) / m;
        Vel(:,k+1) = Vel(:,k) + Acc*dt + V_wind*dt*0.5;
        Pos(:,k+1) = Pos(:,k) + Vel(:,k)*dt;
    end

    L_arm   = [0; 0; -0.2];
    Pos_SAR = zeros(3, Ny);
    for k = 1:Ny
        phi=Att(1,k); theta=Att(2,k); psi=Att(3,k);
        R_x=[1 0 0;0 cos(phi) -sin(phi);0 sin(phi) cos(phi)];
        R_y=[cos(theta) 0 sin(theta);0 1 0;-sin(theta) 0 cos(theta)];
        R_z=[cos(psi) -sin(psi) 0;sin(psi) cos(psi) 0;0 0 1];
        Pos_SAR(:,k) = Pos(:,k) + R_z*R_y*R_x * L_arm;
    end
    % 坐标轴调整: [X_fwd, Y_lat, Z_up] -> [Y_lat, X_fwd, Z_up]
    p_true = [Pos_SAR(2,:); Pos_SAR(1,:); Pos_SAR(3,:)];

    % =================================================================
    % --- E. 复杂综合误差注入（v6 强化版：削峰 + 跳变-误差互斥）---
    % =================================================================

    % v6.2: err_scale 三档采样，整体放大 2×（用户要求加大误差到消费级 GPS 量级）
    %   小: 0.40~1.00 (×2 原 0.20~0.50)
    %   中: 1.00~3.00 (×2 原 0.50~1.50)
    %   大: 3.00~5.00 (×2 原 1.50~2.50)
    % 期望 err_scale 平均 ≈ 2.0，数据集 err_std 由 ~34mm 升至 ~70mm（每轴）
    tier_r = rand();
    if tier_r < 0.30
        err_scale = 0.40 + 0.60 * rand();   % 小误差: 0.40~1.00
    elseif tier_r < 0.80
        err_scale = 1.00 + 2.00 * rand();   % 中误差: 1.00~3.00
    else
        err_scale = 3.00 + 2.00 * rand();   % 大误差: 3.00~5.00
    end
    err_scale_all(iter) = err_scale;

    % 1. 多项式发散趋势
    E_bias  = err_scale * (rand(3,1)-0.5) * 0.02;
    v_drift = err_scale * randn(3,1) * 0.002;
    a_drift = err_scale * randn(3,1) * 0.0001;
    E_trend = E_bias + v_drift*time + 0.5*a_drift*(time.^2);

    % 2. 随机游走
    E_rw = err_scale * cumsum(0.0005*randn(3,Ny), 2) * sqrt(dt);

    % 3. 多频周期震荡
    freq1 = 0.05 + 0.45*rand(3,1);
    freq2 = 0.30 + 1.20*rand(3,1);
    freq3 = 1.50 + 1.50*rand(3,1);
    amp1  = err_scale * rand(3,1) * 0.025;
    amp2  = err_scale * rand(3,1) * 0.015;
    amp3  = err_scale * rand(3,1) * 0.003;
    E_sine = amp1 .* sin(2*pi*freq1*time + 2*pi*rand(3,1)) + ...
             amp2 .* sin(2*pi*freq2*time + 2*pi*rand(3,1)) + ...
             amp3 .* sin(2*pi*freq3*time + 2*pi*rand(3,1));

    % 4. 高频白噪声
    E_white = err_scale * 0.00015 * randn(3, Ny);

    % =================================================================
    % 5. v6.1: 阶跃跳变误差已完全禁用（用户要求航迹连续）
    %    P_raw 现在只包含连续误差成分：
    %      - E_trend (多项式漂移, 平滑)
    %      - E_rw    (随机游走, C0 连续)
    %      - E_sine  (多频正弦叠加, C∞ 连续)
    %      - E_white (高斯白噪声, 离散但无阶跃；幅度 < 1mm)
    %    → 不再存在"GPS 跳变/失锁"类阶跃跳变事件
    %    保留 E_step 框架为 0 矩阵，便于未来按需启用
    % =================================================================
    E_step = zeros(3, Ny);   % 强制为 0，不注入任何阶跃
    n_jumps = 0;             % 跳变次数固定为 0
    n_jumps_all(iter) = n_jumps;

    E_total = E_trend + E_rw + E_sine + E_white + E_step;
    p_raw   = p_true + E_total;

    % v6 诊断：记录该样本的 max 3D 误差范数
    max_err_all(iter) = max(sqrt(sum(E_total.^2, 1)));

    % =================================================================
    % --- F. 构建 12 维特征张量 ---
    % =================================================================
    feat = zeros(Ny, 12);
    window_size = 20;   % 20点 @ 1000Hz = 20ms

    % feat[:,1:3]: P_raw 高斯平滑
    p_raw(1,:) = smoothdata(p_raw(1,:), 'gaussian', window_size);
    p_raw(2,:) = smoothdata(p_raw(2,:), 'gaussian', window_size);
    p_raw(3,:) = smoothdata(p_raw(3,:), 'gaussian', window_size);
    feat(:,1:3) = p_raw';

    % feat[:,4:6]: V_raw 平滑速度
    V_raw = zeros(3, Ny);
    V_raw(:,2:end) = (p_raw(:,2:end) - p_raw(:,1:end-1)) / dt;
    V_raw(:,1) = V_raw(:,2);
    V_raw(1,:) = smoothdata(V_raw(1,:), 'gaussian', window_size*2);
    V_raw(2,:) = smoothdata(V_raw(2,:), 'gaussian', window_size*2);
    V_raw(3,:) = smoothdata(V_raw(3,:), 'gaussian', window_size*2);
    feat(:,4:6) = V_raw';

    % =================================================================
    % --- 随机航向旋转 + 侧向偏置（沿用 v5 的设计）---
    % =================================================================

    % 1. 随机航向旋转 ±15°（绕 Z_up 轴，在 [Y_lat, X_fwd] 平面旋转）
    psi_offset = (rand()-0.5) * (30*pi/180);
    cos_psi = cos(psi_offset);
    sin_psi = sin(psi_offset);

    pt_y = p_true(1,:); pt_x = p_true(2,:);
    p_true(1,:) = cos_psi*pt_y - sin_psi*pt_x;
    p_true(2,:) = sin_psi*pt_y + cos_psi*pt_x;

    pr_y = p_raw(1,:); pr_x = p_raw(2,:);
    p_raw(1,:) = cos_psi*pr_y - sin_psi*pr_x;
    p_raw(2,:) = sin_psi*pr_y + cos_psi*pr_x;
    feat(:,1:3) = p_raw';

    vr_y = V_raw(1,:); vr_x = V_raw(2,:);
    V_raw(1,:) = cos_psi*vr_y - sin_psi*vr_x;
    V_raw(2,:) = sin_psi*vr_y + cos_psi*vr_x;
    feat(:,4:6) = V_raw';

    rot_pos2d = @(p) [cos_psi*p(1)-sin_psi*p(2); sin_psi*p(1)+cos_psi*p(2); p(3)];
    Pos_A = rot_pos2d(Pos_A);
    Pos_B = rot_pos2d(Pos_B);
    Pos_C = rot_pos2d(Pos_C);

    % 2. 随机侧向偏置 ±10m（v6.3 由 ±100m 缩小，仅 Y_lat 分量；平移不影响 RCM 几何）
    Y_offset = (rand()-0.5) * 20;
    p_true(1,:) = p_true(1,:) + Y_offset;
    p_raw(1,:)  = p_raw(1,:)  + Y_offset;
    feat(:,1)   = feat(:,1)   + Y_offset;
    Pos_A(1)    = Pos_A(1) + Y_offset;
    Pos_B(1)    = Pos_B(1) + Y_offset;
    Pos_C(1)    = Pos_C(1) + Y_offset;

    Pos_A_all(:, iter) = Pos_A;
    Pos_B_all(:, iter) = Pos_B;
    Pos_C_all(:, iter) = Pos_C;

    % =================================================================
    % --- feat[:,7:9]: RCM 真实斜距（随机化测量噪声 5~30mm）---
    % =================================================================
    rcm_noise_std = 0.005 + 0.025 * rand();

    for k = 1:Ny
        feat(k,7) = norm(p_true(:,k) - Pos_A) + rcm_noise_std*randn();
        feat(k,8) = norm(p_true(:,k) - Pos_B) + rcm_noise_std*randn();
        feat(k,9) = norm(p_true(:,k) - Pos_C) + rcm_noise_std*randn();
    end

    % feat[:,10:12]: 导航-RCM 斜距残差
    for k = 1:Ny
        feat(k,10) = norm(p_raw(:,k) - Pos_A) - feat(k,7);
        feat(k,11) = norm(p_raw(:,k) - Pos_B) - feat(k,8);
        feat(k,12) = norm(p_raw(:,k) - Pos_C) - feat(k,9);
    end

    Feat_All(iter,:,:)  = feat;
    P_raw_all(iter,:,:) = p_raw';
    P_true_all(iter,:,:) = p_true';
end

%% 3.5 v6.3 诊断报告：误差分布统计
disp(' ');
disp('========== v6.3 数据诊断报告 (连续航迹 + 误差×2 + 无旋转) ==========');
disp(['err_scale 统计:  小档[0.4,1.0) = ', num2str(sum(err_scale_all<1.0)), ...
      ' | 中档[1.0,3.0) = ', num2str(sum(err_scale_all>=1.0 & err_scale_all<3.0)), ...
      ' | 大档[3.0,5.0] = ', num2str(sum(err_scale_all>=3.0))]);
disp(['err_scale 平均 / 最大: ', num2str(mean(err_scale_all),'%.2f'), ...
      ' / ', num2str(max(err_scale_all),'%.2f')]);
disp(['阶跃跳变: 已禁用 (n_jumps 恒为 0)']);
disp(['max|P_raw - P_true| 3D范数:']);
disp(['   均值/中位数: ', num2str(mean(max_err_all),'%.4f'), ' m / ', ...
      num2str(median(max_err_all),'%.4f'), ' m']);
disp(['   95% / 99% / 100% 分位: ', ...
      num2str(quantile(max_err_all, 0.95),'%.4f'), ' / ', ...
      num2str(quantile(max_err_all, 0.99),'%.4f'), ' / ', ...
      num2str(max(max_err_all),'%.4f'), ' m']);
n_extreme = sum(max_err_all > 0.50);
disp(['极端样本(>50cm): ', num2str(n_extreme), ' / ', num2str(num_samples), ...
      ' (', num2str(100*n_extreme/num_samples,'%.2f'), '%)']);
ref_z_all = [Pos_A_all(3,:), Pos_B_all(3,:), Pos_C_all(3,:)];
disp(['参考点高度 Z[m]: min=', num2str(min(ref_z_all),'%.2f'), ...
      ' | mean=', num2str(mean(ref_z_all),'%.2f'), ...
      ' | max=', num2str(max(ref_z_all),'%.2f'), ...
      ' （目标范围 [0, ', num2str(Z_max), ']m）']);
disp(['参考点三角形最小高 [m]: 5/50/95 分位 = ', ...
      num2str(quantile(ref_min_height_all, 0.05),'%.2f'), ' / ', ...
      num2str(quantile(ref_min_height_all, 0.50),'%.2f'), ' / ', ...
      num2str(quantile(ref_min_height_all, 0.95),'%.2f'), ...
      ' （阈值 ≥ 20）']);
disp(['参考点三角形最小内角 [deg]: 5/50/95 分位 = ', ...
      num2str(quantile(ref_min_angle_all, 0.05),'%.2f'), ' / ', ...
      num2str(quantile(ref_min_angle_all, 0.50),'%.2f'), ' / ', ...
      num2str(quantile(ref_min_angle_all, 0.95),'%.2f'), ...
      ' （阈值 ≥ 5）']);
disp(['几何重采样次数: 平均 = ', num2str(mean(ref_resample_all),'%.2f'), ...
      ' | 最大 = ', num2str(max(ref_resample_all)), ...
      ' （越高说明几何约束越紧）']);
disp('====================================');

%% 4. 保存数据集
disp('正在打包保存...');
save_path = fullfile(output_dir, ['PPTR_TrainDataset_', num2str(num_samples), '.mat']);

Feat_All  = Feat_All;   %#ok
P_true    = P_true_all; %#ok
P_raw     = P_raw_all;  %#ok

Pos_A = Pos_A_all;
Pos_B = Pos_B_all;
Pos_C = Pos_C_all;

% v6.3 元数据与诊断字段（供训练侧分析使用，SARDataset 不强制读取）
v6_clip_meta = struct( ...
    'version', 'v6.3', ...
    'E_step_enabled', 0, ...           % 0=已禁用; 1=启用并按 E_step_clip 截断
    'E_step_clip', E_step_clip, ...    % 保留供未来启用时使用
    'gust_burst_enabled', 0, ...       % 0=突发阵风已禁用
    'heading_rotation_enabled', 1, ... % v6.3 恢复航向旋转 ±15°（沿用 v5 设计）
    'lateral_offset_enabled', 1, ...   % 仍保留侧向偏置（仅 Y 平移）
    'lateral_offset_range_m', 10.0, ...% v6.3: ±10m（原 v6.2 是 ±100m）
    'err_scale_max', 5.0, ...          % v6.2 放大 2×（原 2.5 → 5.0）
    'err_scale_amplification', 2.0, ...% 相对 v6.1 的放大倍率
    'ref_pos_z_max', Z_max, ...
    'ref_geom_pair_dist_min', 10.0, ...   % 两两间距下限 (m)
    'ref_geom_min_height_min', 20.0, ...  % 三角形最小高下限 (m)
    'ref_geom_min_angle_min', 5.0, ...    % 最小内角下限 (deg)
    'trajectory_continuity', 'C0_continuous_no_step_events');

save(save_path, 'Feat_All', 'P_true', 'P_raw', ...
     'Pos_A', 'Pos_B', 'Pos_C', ...
     'v6_clip_meta', 'err_scale_all', 'max_err_all', 'n_jumps_all', ...
     'ref_min_height_all', 'ref_min_angle_all', 'ref_resample_all', '-v7.3');

% 飞行参数单独保存（成像脚本加载，不参与深度学习训练）
flight_params_path = fullfile(output_dir, ['PPTR_FlightParams_', num2str(num_samples), '.mat']);

H_target = H_target_all;
V_target = V_target_all;
save(flight_params_path, 'H_target', 'V_target','-v7.3');
disp(['【飞行参数】已保存至: ', flight_params_path]);

disp(['【生成完毕】 ', num2str(num_samples), ' 组数据已保存至: ', save_path]);
disp('特征维度说明 (v6):');
disp('  Feat_All [N,T,12]: feat[:,1:3]=P_raw(旋转+偏置) | feat[:,4:6]=V_raw(旋转)');
disp('                     feat[:,7:9]=RCM斜距(随机噪声5~30mm) | feat[:,10:12]=导航-RCM残差');
disp('  P_true  [N,T,3]:  真实轨迹（旋转+偏置后坐标系）');
disp('  P_raw   [N,T,3]:  含噪观测轨迹（旋转+偏置后坐标系）');
disp('  Pos_A/B/C_all [3,N]: 每条样本特显点（旋转+偏置后坐标系，与feat一致）');
disp('v6.3 改动: 完全禁用阶跃跳变 / 突发阵风 | err_scale整体×2 | 参考点高度[0,20]m');
disp('         航迹完全连续 (P_true / P_raw 均为时间的 C0 连续函数)');
disp('         保留随机航向旋转 ±15°（沿用 v5）；侧向偏置缩小为 ±10m（原 ±100m）');
disp('         数据集 err_std 由 v6.1 的 ~34mm 升至 ~70mm（消费级 GPS 漂移量级）');
