% =========================================================================
% SAR 轨迹精化 (PPTR) 深度学习批量数据集生成脚本 (泛化增强版 v5)
% 物理模型: 随机直线航迹 -> 叠加风场/阵风与飞控震荡 (P_true) -> 叠加复杂综合误差 (P_raw)
%
% v5 相比 v4 主要改动（进一步打破过拟合，引入半监督训练支持）：
%   1. err_scale 分三档采样（小:0.2~0.5 / 中:0.5~1.5 / 大:1.5~3.0）
%      → 覆盖更宽的误差量级，避免模型只适应均匀分布
%   2. 阶跃跳变误差事件（每条样本 0~5 次随机跳变）
%      → 增加非平稳误差模式，提升时序泛化能力
%   3. 随机航向旋转 ±15°（旋转 feat[1:6] 中的 X/Y 分量及参考点坐标）
%      → 打破模型对绝对坐标方向的记忆（飞行方向不再固定为 X 轴）
%   4. 随机侧向偏置 ±100m（平移 P_raw/P_true/参考点的 Y_lat 分量）
%      → 打破绝对位置记忆，提升空间泛化能力
%   5. RCM 测距噪声随机化（5~30mm，原固定 3mm）
%      → 增加 feat[7:9] 的噪声多样性，减弱 feat[10:12] 的线性可逆性
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
output_dir = 'PPTR_Dataset_4';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% 1. 固定仿真基础参数
Ny  = 2048 * 2;   % 单条航迹采样点数 (4096)
PRF = 1000;
dt  = 1 / PRF;    % 仿真步长 0.001s

% RCM 测距噪声：v5 改为每条样本独立随机化（5~30mm），在循环内赋值
% 原固定值 rcm_noise_std = 0.003 已移至循环内

%% 2. 预分配深度学习训练集张量
Feat_All  = zeros(num_samples, Ny, 12);
P_true_all = zeros(num_samples, Ny, 3);
P_raw_all  = zeros(num_samples, Ny, 3);

% 每条样本特显点也需保存（训练/评估时要用）
Pos_A_all = zeros(3, num_samples);
Pos_B_all = zeros(3, num_samples);
Pos_C_all = zeros(3, num_samples);

% 每条样本的飞行参数（成像脚本恢复 R_c 时必须）
% 第1行: H_target (飞行高度, m)
% 第2行: v_target (飞行速度, m/s)
H_target_all = zeros(num_samples,1);
V_target_all = zeros(num_samples,1);

%% 3. 主生成循环
disp(['开始生成 ', num2str(num_samples), ' 组泛化增强样本数据...']);
tic;

for iter = 1:num_samples
    if mod(iter, 50) == 0 || iter == 1
        disp(['当前进度: ', num2str(iter), ' / ', num2str(num_samples), ...
              ' (耗时: ', num2str(toc,'%.1f'), ' 秒)']);
    end

    % =================================================================
    % --- A. 每条样本随机化物理参数 ---
    % =================================================================

    % 无人机质量与惯量 ±15% 随机扰动
    m   = 8.0  * (1 + 0.15 * (2*rand()-1));
    g   = 9.81;
    Ixx = 0.08 * (1 + 0.15 * (2*rand()-1));
    Iyy = 0.08 * (1 + 0.15 * (2*rand()-1));
    Izz = 0.15 * (1 + 0.15 * (2*rand()-1));

    % PID 控制器增益 ±15% 随机扰动
    Kp_pos = 0.8 * (1 + 0.15*(2*rand()-1));
    Kd_pos = 1.2 * (1 + 0.15*(2*rand()-1));
    Kp_z   = 2.5 * (1 + 0.15*(2*rand()-1));
    Kd_z   = 1.5 * (1 + 0.15*(2*rand()-1));
    Kp_att = 5.0 * (1 + 0.15*(2*rand()-1));
    Kd_att = 0.8 * (1 + 0.15*(2*rand()-1));
    Kp_yaw = 2.0 * (1 + 0.15*(2*rand()-1));
    Kd_yaw = 0.5 * (1 + 0.15*(2*rand()-1));

    % 飞行速度与高度随机化（速度范围收窄，降低绝对坐标变化幅度）
    v_target = 7  + 6 * rand();       % 7~13 m/s（原 5~15，3倍变化导致 X_fwd 绝对坐标相差3倍）
    H_target = 60 + 80 * rand();      % 60~140 m（原 50~150，适度收窄）
    H_target_all(iter) = H_target;
    V_target_all(iter) = v_target;

    time   = (0:Ny-1) * dt;
    X_des  = v_target * time;
    Y_des  = zeros(1, Ny);
    Z_des  = H_target * ones(1, Ny);

    % =================================================================
    % --- B. 随机化特显点位置 ---
    %     坐标系: Pos_X = [Y_lat; X_fwd; Z]，与 p_true 一致
    %     地距向 (Y_lat, 第1分量): 400 ± 100 m  →  均匀分布 [300, 500]
    %     方位向 (X_fwd, 第2分量): 100 ± 100 m  →  均匀分布 [0, 200]
    %     高度固定为 0（地面目标）
    % =================================================================
   % gen_pos = @() [300 + 200*rand(); 200*rand(); 0];
    total_x = v_target * (Ny-1) * dt;   % 飞行总 X 距离
    gen_pos = @() [300 + 200*rand(); total_x*(0.1 + 0.8*rand()); 0];

    Pos_A = gen_pos();
    Pos_B = gen_pos();
    Pos_C = gen_pos();

    % 确保三点两两间距 > 10m，避免 RCM 约束矩阵退化（几何相关）
    while norm(Pos_A(1:2) - Pos_B(1:2)) < 10 || ...
          norm(Pos_B(1:2) - Pos_C(1:2)) < 10 || ...
          norm(Pos_A(1:2) - Pos_C(1:2)) < 10
        Pos_A = gen_pos();
        Pos_B = gen_pos();
        Pos_C = gen_pos();
    end
    % 注意：Pos_X_all 的保存已移至航向旋转+侧向偏置之后（v5 新增，保证存储坐标与 feat 一致）

    % =================================================================
    % --- C. 随机化风场模型 ---
    % =================================================================
    wind_amp  = rand(3,1) * 1.5;          % 振幅 0~1.5 m/s（原先固定 0.5/0.8/0.2）
    wind_freq = 0.05 + 0.2*rand(3,1);     % 主频 0.05~0.25 Hz
    wind_phase = 2 * pi * rand(3, 1);
    wind_noise_std = 0.02 + 0.05*rand();  % 阵风白噪声强度

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

        % 0.2% 概率突发侧向阵风
        if rand() < 0.002
            wind_y = wind_y + (2.0 + rand());
        end
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

    % 提取 SAR 天线相位中心（APC）作为物理真值
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
    % --- E. 复杂综合误差注入（v5 增强版）---
    % =================================================================

    % v5: err_scale 分三档采样，覆盖大/中/小误差场景（原均匀 0.5~2 改为分层）
    % 各档权重：小(30%) / 中(50%) / 大(20%)，合计=100%
    tier_r = rand();
    if tier_r < 0.30
        err_scale = 0.20 + 0.30 * rand();   % 小误差: 0.20~0.50
    elseif tier_r < 0.80
        err_scale = 0.50 + 1.00 * rand();   % 中误差: 0.50~1.50
    else
        err_scale = 1.50 + 1.50 * rand();   % 大误差: 1.50~3.00
    end

    % 1. 多项式发散趋势
    E_bias  = err_scale * (rand(3,1)-0.5) * 0.02;
    v_drift = err_scale * randn(3,1) * 0.002;
    a_drift = err_scale * randn(3,1) * 0.0001;
    E_trend = E_bias + v_drift*time + 0.5*a_drift*(time.^2);

    % 2. 随机游走
    E_rw = err_scale * cumsum(0.0005*randn(3,Ny), 2) * sqrt(dt);

    % 3. 多频周期震荡
    freq1 = 0.05 + 0.45*rand(3,1);   % 0.05~0.50 Hz 低频趋势
    freq2 = 0.30 + 1.20*rand(3,1);   % 0.30~1.50 Hz 中频
    freq3 = 1.50 + 1.50*rand(3,1);   % 1.50~3.00 Hz 高频
    amp1  = err_scale * rand(3,1) * 0.025;
    amp2  = err_scale * rand(3,1) * 0.015;
    amp3  = err_scale * rand(3,1) * 0.003;
    E_sine = amp1 .* sin(2*pi*freq1*time + 2*pi*rand(3,1)) + ...
             amp2 .* sin(2*pi*freq2*time + 2*pi*rand(3,1)) + ...
             amp3 .* sin(2*pi*freq3*time + 2*pi*rand(3,1));

    % 4. 高频白噪声
    E_white = err_scale * 0.00015 * randn(3, Ny);

    % 5. v5 新增：阶跃跳变误差（模拟 GPS 信号跳变/失锁事件）
    % 每条样本随机 0~5 次跳变，跳变幅度在 [Y_lat, X_fwd, Z_up] 方向上独立
    E_step = zeros(3, Ny);
    n_jumps = randi([0, 5]);
    for jj = 1:n_jumps
        t_jump = randi([round(Ny*0.05), round(Ny*0.95)]);  % 避免首尾 5%
        % 跳变幅度：侧向最大 ±50mm，方位向 ±30mm，高度向 ±20mm
        jump_amp = err_scale * [0.050*(2*rand()-1); ...
                                0.030*(2*rand()-1); ...
                                0.020*(2*rand()-1)];
        E_step(:, t_jump:end) = E_step(:, t_jump:end) + jump_amp;
    end

    E_total = E_trend + E_rw + E_sine + E_white + E_step;
    p_raw   = p_true + E_total;

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
    % --- v5 新增：随机航向旋转 + 侧向偏置 ---
    % 在 feat[1:6] (P_raw, V_raw) 已构建、RCM 计算之前应用，
    % 使所有坐标（feat + p_true/p_raw + Pos_X）处于统一的旋转+偏置坐标系。
    % 旋转和偏置对距离特征（feat[7:12]）天然不变（旋转不变量 + 平移不变量），
    % 仅影响绝对位置/速度特征（feat[1:6]），从而打破坐标记忆。
    % =================================================================

    % 1. 随机航向旋转 ±15°（绕 Z_up 轴，在 [Y_lat, X_fwd] 平面旋转）
    psi_offset = (rand()-0.5) * (30*pi/180);   % ±15°
    cos_psi = cos(psi_offset);
    sin_psi = sin(psi_offset);

    % 旋转 p_true（[Y_lat; X_fwd; Z_up] 格式）
    pt_y = p_true(1,:); pt_x = p_true(2,:);
    p_true(1,:) = cos_psi*pt_y - sin_psi*pt_x;
    p_true(2,:) = sin_psi*pt_y + cos_psi*pt_x;

    % 旋转平滑后的 p_raw（feat[:,1:3] 基础）
    pr_y = p_raw(1,:); pr_x = p_raw(2,:);
    p_raw(1,:) = cos_psi*pr_y - sin_psi*pr_x;
    p_raw(2,:) = sin_psi*pr_y + cos_psi*pr_x;
    feat(:,1:3) = p_raw';                       % 更新 feat P_raw

    % 旋转 V_raw
    vr_y = V_raw(1,:); vr_x = V_raw(2,:);
    V_raw(1,:) = cos_psi*vr_y - sin_psi*vr_x;
    V_raw(2,:) = sin_psi*vr_y + cos_psi*vr_x;
    feat(:,4:6) = V_raw';                       % 更新 feat V_raw

    % 旋转参考点坐标（[Y_lat; X_fwd; Z] 格式）
    rot_pos2d = @(p) [cos_psi*p(1)-sin_psi*p(2); sin_psi*p(1)+cos_psi*p(2); p(3)];
    Pos_A = rot_pos2d(Pos_A);
    Pos_B = rot_pos2d(Pos_B);
    Pos_C = rot_pos2d(Pos_C);

    % 2. 随机侧向偏置 ±100m（仅 Y_lat 分量）
    Y_offset = (rand()-0.5) * 200;   % ±100m
    p_true(1,:) = p_true(1,:) + Y_offset;
    p_raw(1,:)  = p_raw(1,:)  + Y_offset;
    feat(:,1)   = feat(:,1)   + Y_offset;       % 更新 feat P_raw Y_lat
    Pos_A(1)    = Pos_A(1) + Y_offset;
    Pos_B(1)    = Pos_B(1) + Y_offset;
    Pos_C(1)    = Pos_C(1) + Y_offset;

    % 参考点保存（v5：移至旋转+偏置之后，坐标系与 feat 一致）
    Pos_A_all(:, iter) = Pos_A;
    Pos_B_all(:, iter) = Pos_B;
    Pos_C_all(:, iter) = Pos_C;

    % =================================================================
    % --- feat[:,7:9]: RCM 真实斜距（v5: 随机化测量噪声 5~30mm）---
    % =================================================================
    % rcm_noise_std 每条样本独立随机（原固定 3mm → 5~30mm 随机）
    % 增大噪声多样性使 feat[10:12] 与 δ 的线性关系更复杂，抑制过拟合
    rcm_noise_std = 0.005 + 0.025 * rand();  % 5~30mm

    % feat[:,7:9]: RCM 真实斜距（从旋转后的 P_true 计算，加随机测量噪声）
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

%% 4. 保存数据集
disp('正在打包保存...');
save_path = fullfile(output_dir, ['PPTR_TrainDataset_', num2str(num_samples), '.mat']);

% 注意：Pos_A/B/C 现在是 [3, N] 矩阵，每列对应一条样本
% Python 读取时需按样本索引取对应列
Feat_All  = Feat_All;   %#ok
P_true    = P_true_all; %#ok
P_raw     = P_raw_all;  %#ok

Pos_A = Pos_A_all;
Pos_B = Pos_B_all;
Pos_C = Pos_C_all;
save(save_path, 'Feat_All', 'P_true', 'P_raw', ...
     'Pos_A', 'Pos_B', 'Pos_C', '-v7.3');

% 飞行参数单独保存（成像脚本加载，不参与深度学习训练）
flight_params_path = fullfile(output_dir, ['PPTR_FlightParams_', num2str(num_samples), '.mat']);

H_target = H_target_all;
V_target = V_target_all;
save(flight_params_path, 'H_target', 'V_target','-v7.3');
disp(['【飞行参数】已保存至: ', flight_params_path]);

disp(['【生成完毕】 ', num2str(num_samples), ' 组数据已保存至: ', save_path]);
disp('特征维度说明 (v5):');
disp('  Feat_All [N,T,12]: feat[:,1:3]=P_raw(旋转+偏置) | feat[:,4:6]=V_raw(旋转)');
disp('                     feat[:,7:9]=RCM斜距(随机噪声5~30mm) | feat[:,10:12]=导航-RCM残差');
disp('  P_true  [N,T,3]:  真实轨迹（旋转+偏置后坐标系）');
disp('  P_raw   [N,T,3]:  含噪观测轨迹（旋转+偏置后坐标系）');
disp('  Pos_A/B/C_all [3,N]: 每条样本特显点（旋转+偏置后坐标系，与feat一致）');
disp('v5 新增: err_scale分三档 | 阶跃跳变误差 | 航向旋转±15° | 侧向偏置±100m | RCM噪声5~30mm');
