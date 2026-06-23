% =========================================================================
% SAR 轨迹精化 (PPTR) 深度学习批量数据集生成脚本 (强化误差版 v2)
% 物理模型: 理想直线 -> 叠加风场/阵风与飞控震荡 (P_true) -> 叠加复杂综合误差 (P_raw)
% 误差模型: 多项式发散 + 随机游走 + 多频震荡 + 高频白噪声
%
% 特征说明 (Feat_All 12维):
%   feat[:, 1:3]  = P_raw (平滑后的导航轨迹)
%   feat[:, 4:6]  = V_raw (P_raw 数值微分速度，平滑)
%   feat[:, 7:9]  = RCM 斜距测量 (从 P_true 计算 + 测量噪声，模拟真实雷达回波)
%   feat[:, 10:12]= 导航-RCM 残差 (||P_raw-Pos_X|| - RCM，直接反映导航误差投影)
%
% 输出格式: Feat_All [N, T, 12], P_true [N, T, 3], P_raw [N, T, 3]
% =========================================================================
clear; clc; close all;

%% 0. 批量生成参数设置
num_samples = 1000;     % 需要生成的样本总数
output_dir = 'PPTR_Dataset';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% 1. 无人机物理参数 & 仿真环境设置
m = 8.0;              % 质量 (kg)
g = 9.81;             % 重力加速度
L = 0.4;              % 轴距
Ixx = 0.08;
Iyy = 0.08;
Izz = 0.15;

Ny = 2048*2;          % 单条航迹采样点数 (4096)
PRF = 1000;
dt = 1/PRF;           % 仿真步长 (0.001s = 1000Hz)
time = (0:Ny-1) * dt;

%% 2. 目标航迹设定 & 控制器参数
v_target = 10;
H_target = 100;
X_des = v_target * time;          % X轴匀速前进
Y_des = zeros(1, Ny);             % Y轴保持直线
Z_des = H_target * ones(1, Ny);   % Z轴保持定高

% PID 控制器参数
Kp_pos = 0.8;  Kd_pos = 1.2;
Kp_z   = 2.5;  Kd_z   = 1.5;
Kp_att = 5.0;  Kd_att = 0.8;
Kp_yaw = 2.0;  Kd_yaw = 0.5;

% 特显点参考坐标 (SAR 场景中已知的强散射目标，用于 RCM 约束)
Pos_A = [387; 25; 0];
Pos_B = [417; 56; 0];
Pos_C = [403; 78; 0];

% RCM 测距噪声标准差 (模拟 SAR 脉冲压缩后的测距精度, 约 3mm)
rcm_noise_std = 0.003;

%% 3. 预分配深度学习训练集张量 (feat 扩展为 12 维)
Feat_All = zeros(num_samples, Ny, 12);
P_true   = zeros(num_samples, Ny, 3);
P_raw    = zeros(num_samples, Ny, 3);

%% 4. 主生成循环
disp(['开始生成 ', num2str(num_samples), ' 组高逼真样本数据...']);
tic;

for iter = 1:num_samples
    if mod(iter, 50) == 0 || iter == 1
        disp(['当前进度: ', num2str(iter), ' / ', num2str(num_samples), ...
              ' (耗时: ', num2str(toc,'%.1f'), ' 秒)']);
    end

    % --- 状态变量初始化 ---
    Pos   = zeros(3, Ny);  Pos(:,1)   = [0; 0; H_target];
    Vel   = zeros(3, Ny);  Vel(:,1)   = [v_target; 0; 0];
    Att   = zeros(3, Ny);
    Omega = zeros(3, Ny);

    wind_phase = 2 * pi * rand(3, 1); % 风场初相随机化

    % =====================================================================
    % --- 第一阶段：6-DOF 物理动力学仿真 (生成带物理高频震荡的真值) ---
    % =====================================================================
    for k = 1:Ny-1
        % 基础风场 + 突发阵风模型
        wind_x = 0.5 * sin(2*pi*0.1*time(k)  + wind_phase(1)) + 0.05 * randn();
        wind_y = 0.8 * sin(2*pi*0.15*time(k) + wind_phase(2)) + 0.05 * randn();
        wind_z = 0.2 * sin(2*pi*0.05*time(k) + wind_phase(3)) + 0.02 * randn();

        % 0.2% 概率遇到突发侧向阵风
        if rand() < 0.002
            wind_y = wind_y + (2.0 + rand());
        end
        V_wind = [wind_x; wind_y; wind_z];

        % 飞控与动力学演算
        e_p = [X_des(k); Y_des(k); Z_des(k)] - Pos(:,k);
        e_v = [v_target; 0; 0] - Vel(:,k);
        U1 = m * g + Kp_z * e_p(3) + Kd_z * e_v(3);

        phi_des   = -(Kp_pos * e_p(2) + Kd_pos * e_v(2)) / g;
        theta_des =  (Kp_pos * e_p(1) + Kd_pos * e_v(1)) / g;
        psi_des   = 0;

        phi_des   = max(min(phi_des,   0.3), -0.3);
        theta_des = max(min(theta_des, 0.3), -0.3);

        U2 = Kp_att * (phi_des   - Att(1,k)) + Kd_att * (0 - Omega(1,k));
        U3 = Kp_att * (theta_des - Att(2,k)) + Kd_att * (0 - Omega(2,k));
        U4 = Kp_yaw * (psi_des   - Att(3,k)) + Kd_yaw * (0 - Omega(3,k));

        p_dot = (U2 - (Izz - Iyy) * Omega(2,k) * Omega(3,k)) / Ixx;
        q_dot = (U3 - (Ixx - Izz) * Omega(1,k) * Omega(3,k)) / Iyy;
        r_dot = (U4 - (Iyy - Ixx) * Omega(1,k) * Omega(2,k)) / Izz;

        Omega(:, k+1) = Omega(:, k) + [p_dot; q_dot; r_dot] * dt;
        Att(:, k+1)   = Att(:, k)   + Omega(:, k) * dt;

        phi = Att(1,k+1); theta = Att(2,k+1); psi = Att(3,k+1);
        R_x = [1 0 0; 0 cos(phi) -sin(phi); 0 sin(phi) cos(phi)];
        R_y = [cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)];
        R_z = [cos(psi) -sin(psi) 0; sin(psi) cos(psi) 0; 0 0 1];
        R = R_z * R_y * R_x;

        Acc = (R * [0; 0; U1] - [0; 0; m*g]) / m;
        Vel(:, k+1) = Vel(:, k) + Acc * dt + V_wind * dt * 0.5;
        Pos(:, k+1) = Pos(:, k) + Vel(:, k) * dt;
    end

    % --- 提取雷达天线相位中心 (APC) 得到物理真值航迹 (P_true) ---
    L_arm   = [0; 0; -0.2];
    Pos_SAR = zeros(3, Ny);
    for k = 1:Ny
        phi = Att(1,k); theta = Att(2,k); psi = Att(3,k);
        R_x = [1 0 0; 0 cos(phi) -sin(phi); 0 sin(phi) cos(phi)];
        R_y = [cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)];
        R_z = [cos(psi) -sin(psi) 0; sin(psi) cos(psi) 0; 0 0 1];
        R = R_z * R_y * R_x;
        Pos_SAR(:, k) = Pos(:, k) + R * L_arm;
    end
    % 坐标轴调整: 仿真坐标 [X_fwd, Y_lat, Z_up] -> 输出 [Y_lat, X_fwd, Z_up]
    p_true = [Pos_SAR(2,:); Pos_SAR(1,:); Pos_SAR(3,:)];

    % =====================================================================
    % --- 第二阶段：复杂综合误差注入 (模拟真实导航系统的复合误差) ---
    % =====================================================================

    % 1. 多项式发散趋势 (常数零偏 + 线性速度漂移 + 二次加速度零偏)
    E_bias  = (rand(3,1) - 0.5) * 0.02;       % +/- 1cm 初始偏差
    v_drift = randn(3,1) * 0.002;              % 速度随机漂移率
    a_drift = randn(3,1) * 0.0001;             % 加速度随机漂移率
    E_trend = E_bias + v_drift * time + 0.5 * a_drift * (time.^2);

    % 2. 随机游走误差 (传感器白噪声的积分累积)
    E_rw = cumsum(0.0005 * randn(3, Ny), 2) * sqrt(dt);  % 略微降低，增强可学性

    % 3. 多频周期震荡 (环境变化、机体共振等多源低频干扰)
    freq1 = 0.1 + 0.1 * rand(3,1);  % 极低频 (0.1~0.2 Hz)
    freq2 = 0.5 + 0.2 * rand(3,1);  % 中低频 (0.5~0.7 Hz)
    E_sine1 = (rand(3,1)*0.02) .* sin(2*pi * freq1 * time + 2*pi*rand(3,1));
    E_sine2 = (rand(3,1)*0.01) .* sin(2*pi * freq2 * time + 2*pi*rand(3,1));
    E_multi_sine = E_sine1 + E_sine2;

    % 4. 传感器高频观测白噪声
    E_white = 0.00015 * randn(3, Ny);

    % 组合所有误差成分，注入得到原始观测轨迹
    E_total = E_trend + E_rw + E_multi_sine + E_white;
    p_raw   = p_true + E_total;

    % =====================================================================
    % --- 第三阶段：构建深度学习特征张量 (12 维) ---
    % =====================================================================
    feat = zeros(Ny, 12);

    % 平滑窗口 (去除高频毛刺，保留低频漂移与震荡)
    window_size = 20;  % 20点 @ 1000Hz = 20ms

    % --- feat[:, 1:3]: P_raw 高斯平滑后的导航轨迹 ---
    p_raw(1,:) = smoothdata(p_raw(1,:), 'gaussian', window_size);
    p_raw(2,:) = smoothdata(p_raw(2,:), 'gaussian', window_size);
    p_raw(3,:) = smoothdata(p_raw(3,:), 'gaussian', window_size);
    feat(:, 1:3) = p_raw';

    % --- feat[:, 4:6]: V_raw 平滑速度 (P_raw 数值微分) ---
    V_raw = zeros(3, Ny);
    V_raw(:, 2:end) = (p_raw(:, 2:end) - p_raw(:, 1:end-1)) / dt;
    V_raw(:, 1) = V_raw(:, 2);
    V_raw(1,:) = smoothdata(V_raw(1,:), 'gaussian', window_size * 2);
    V_raw(2,:) = smoothdata(V_raw(2,:), 'gaussian', window_size * 2);
    V_raw(3,:) = smoothdata(V_raw(3,:), 'gaussian', window_size * 2);
    feat(:, 4:6) = V_raw';

    % --- feat[:, 7:9]: RCM 真实斜距测量 ---
    % 物理本质: SAR 回波脉冲压缩后提取特显点 RCM 曲线，独立于导航系统
    % 从 P_true 计算 (真实电磁测距)，叠加 3mm 测量噪声
    for k = 1:Ny
        feat(k, 7) = norm(p_true(:, k) - Pos_A) + rcm_noise_std * randn();
        feat(k, 8) = norm(p_true(:, k) - Pos_B) + rcm_noise_std * randn();
        feat(k, 9) = norm(p_true(:, k) - Pos_C) + rcm_noise_std * randn();
    end

    % --- feat[:, 10:12]: 导航-RCM 斜距残差 ---
    % 含义: ||P_raw - Pos_X|| - RCM_measured
    % 当 P_raw 无误差时残差 ≈ 0；导航误差越大，残差越大
    % 这是导航误差在各斜距方向上的直接可观测投影，是最接近标签的输入特征
    for k = 1:Ny
        feat(k, 10) = norm(p_raw(:, k) - Pos_A) - feat(k, 7);
        feat(k, 11) = norm(p_raw(:, k) - Pos_B) - feat(k, 8);
        feat(k, 12) = norm(p_raw(:, k) - Pos_C) - feat(k, 9);
    end

    % --- 写入总张量 ---
    Feat_All(iter, :, :) = feat;
    P_raw(iter, :, :)    = p_raw';
    P_true(iter, :, :)   = p_true';
end

%% 5. 保存并打包
disp('正在打包保存 DL 特征数据集 (用于 Python / PyTorch)...');
save_path = fullfile(output_dir, ['PPTR_TrainDataset_', num2str(num_samples), '.mat']);
save(save_path, 'Feat_All', 'P_true', 'P_raw', 'Pos_A', 'Pos_B', 'Pos_C', '-v7.3');
disp(['【生成完毕】 ', num2str(num_samples), ' 组数据已保存至: ', save_path]);
disp('特征维度说明:');
disp('  feat[:, 1:3]  = P_raw (导航轨迹)');
disp('  feat[:, 4:6]  = V_raw (导航速度)');
disp('  feat[:, 7:9]  = RCM 斜距 (从 P_true 计算, 独立约束)');
disp('  feat[:, 10:12]= 导航-RCM 残差 (导航误差的直接代理)');
