% =========================================================================
% PPTR 数据集构建 v4_phase：6-DOF 航迹仿真 + P_true 相对相位
%
% 单文件训练 mat（float32 / -v7.3）:
%   Feat_All [N,Ny,6], P_raw, P_true, Phase_rel, R_ref,
%   Pos_A, Pos_B, Pos_C, fc, v4_meta
%
%   Phase_rel: unwrap(-4*pi*fc*(R-R_ref)/c), R from P_true, radians
%   R_ref(j) = min_t ||P_true(t)-Pos_j||
%   Feat_All: [:,1:3]=P_raw, [:,4:6]=V_raw
%
% 飞行参数另存: PPTR_FlightParams_v4_*.mat
% =========================================================================
clear; clc;

%% 0. 批量参数
num_samples = 10;          % 正式训练改为 5000
output_dir  = fullfile(fileparts(mfilename('fullpath')), 'dataset');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% 1. 航迹与时间
Ny  = 2048;
PRF = 500;
dt  = 1 / PRF;

%% 2. 雷达参数
radar.fc = 9.6e9;
radar.c  = 3e8;
fc = single(radar.fc);

%% 3. 预分配
Feat_All     = zeros(num_samples, Ny, 6, 'single');
P_true_all   = zeros(num_samples, Ny, 3, 'single');
P_raw_all    = zeros(num_samples, Ny, 3, 'single');
Phase_rel_all = zeros(num_samples, Ny, 3, 'single');
R_ref_all    = zeros(num_samples, 3, 'single');
R_obs_all    = zeros(num_samples, Ny, 3, 'single');
Pos_A_all    = zeros(3, num_samples, 'single');
Pos_B_all    = zeros(3, num_samples, 'single');
Pos_C_all    = zeros(3, num_samples, 'single');
H_target_all = zeros(num_samples, 1, 'single');
V_target_all = zeros(num_samples, 1, 'single');

ref_min_height_all = zeros(num_samples, 1, 'single');
ref_min_angle_all  = zeros(num_samples, 1, 'single');
ref_resample_all   = zeros(num_samples, 1, 'single');

Z_max = 20.0;

%% 4. 主循环
disp(['[v4_phase] P_true 相对相位 | 样本 ', num2str(num_samples), ', Ny=', num2str(Ny)]);
tic;

for iter = 1:num_samples
    if mod(iter, max(1, floor(num_samples/20))) == 0 || iter == 1
        disp(['  进度 ', num2str(iter), '/', num2str(num_samples), ...
              '  elapsed=', num2str(toc,'%.1f'), 's']);
    end

    %% --- A. 随机物理参数 ---
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

    v_target = 7 + 6*rand();
    H_target = 280 + 40*rand();          % 280~320 m
    H_target_all(iter) = H_target;
    V_target_all(iter) = v_target;

    time  = (0:Ny-1) * dt;
    X_des = v_target * time;
    Y_des = zeros(1, Ny);
    Z_des = H_target * ones(1, Ny);

    %% --- B. 参考点（固定三点布局，场景 100m×100m） ---
    total_x = v_target * (Ny-1) * dt;
    d_center = 954;                         % 地面距 ≈ sqrt(1000²-300²)

    Pos_A = single([d_center + 0;   total_x * 0.5 + 0;   0]);  % 中心,  R≈1000m
    Pos_B = single([d_center + 40;  total_x * 0.5 + 40;  0]);  % 远边, +40m Y, +40m X
    Pos_C = single([d_center - 40;  total_x * 0.5 - 40;  0]);  % 近边, -40m Y, -40m X

    ref_min_height_all(iter) = 40;
    ref_min_angle_all(iter)  = 45;
    ref_resample_all(iter)   = 0;

    %% --- C. 6-DOF -> p_true ---
    wind_amp  = rand(3,1) * 1.5;
    wind_freq = 0.05 + 0.2*rand(3,1);
    wind_phase = 2 * pi * rand(3, 1);
    wind_noise_std = 0.02 + 0.05*rand();

    Pos = zeros(3, Ny); Pos(:,1) = [0; 0; H_target];
    Vel = zeros(3, Ny); Vel(:,1) = [v_target; 0; 0];
    Att = zeros(3, Ny);
    Omega = zeros(3, Ny);

    for k = 1:Ny-1
        wind_x = wind_amp(1)*sin(2*pi*wind_freq(1)*time(k)+wind_phase(1)) + wind_noise_std*randn();
        wind_y = wind_amp(2)*sin(2*pi*wind_freq(2)*time(k)+wind_phase(2)) + wind_noise_std*randn();
        wind_z = wind_amp(3)*sin(2*pi*wind_freq(3)*time(k)+wind_phase(3)) + wind_noise_std*0.5*randn();
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

    L_arm = [0; 0; -0.2];
    Pos_SAR = zeros(3, Ny);
    for k = 1:Ny
        phi=Att(1,k); theta=Att(2,k); psi=Att(3,k);
        R_x=[1 0 0;0 cos(phi) -sin(phi);0 sin(phi) cos(phi)];
        R_y=[cos(theta) 0 sin(theta);0 1 0;-sin(theta) 0 cos(theta)];
        R_z=[cos(psi) -sin(psi) 0;sin(psi) cos(psi) 0;0 0 1];
        Pos_SAR(:,k) = Pos(:,k) + R_z*R_y*R_x * L_arm;
    end
    p_true = single([Pos_SAR(2,:); Pos_SAR(1,:); Pos_SAR(3,:)]);

    %% --- D. 误差 -> p_raw ---
    tier_r = rand();
    if tier_r < 0.30
        err_scale = 0.40 + 0.60 * rand();
    elseif tier_r < 0.80
        err_scale = 1.00 + 2.00 * rand();
    else
        err_scale = 3.00 + 2.00 * rand();
    end

    E_bias  = err_scale * (rand(3,1)-0.5) * 0.02;
    v_drift = err_scale * randn(3,1) * 0.002;
    a_drift = err_scale * randn(3,1) * 0.0001;
    E_trend = E_bias + v_drift*time + 0.5*a_drift*(time.^2);
    E_rw = err_scale * cumsum(0.0005*randn(3,Ny), 2) * sqrt(dt);
    freq1 = 0.05 + 0.45*rand(3,1);
    freq2 = 0.30 + 1.20*rand(3,1);
    freq3 = 1.50 + 1.50*rand(3,1);
    amp1  = err_scale * rand(3,1) * 0.025;
    amp2  = err_scale * rand(3,1) * 0.015;
    amp3  = err_scale * rand(3,1) * 0.003;
    E_sine = amp1 .* sin(2*pi*freq1*time + 2*pi*rand(3,1)) + ...
             amp2 .* sin(2*pi*freq2*time + 2*pi*rand(3,1)) + ...
             amp3 .* sin(2*pi*freq3*time + 2*pi*rand(3,1));
    E_white = err_scale * 0.00015 * randn(3, Ny);
    E_total = E_trend + E_rw + E_sine + E_white;
    p_raw = single(p_true + E_total);

    %% --- D2. GPS 中断段仿真（IMU-only，位置漂移但速度短期可靠） ---
    if rand() < 0.10
        n_outages = randi([1, 3]);
        for oi = 1:n_outages
            o_start = randi([round(Ny*0.2), round(Ny*0.7)]);
            o_len   = randi([256, 5120]);
            o_end   = min(o_start + o_len, Ny);
            drift_rate = 0.01 + 0.04*rand();
            t_out = (0:(o_end - o_start)) * dt;
            drift = drift_rate * [cumsum(randn(1, length(t_out))) * sqrt(dt);
                                  cumsum(randn(1, length(t_out))) * sqrt(dt);
                                  0.3 * cumsum(randn(1, length(t_out))) * sqrt(dt)];
            p_raw(:, o_start:o_end) = p_raw(:, o_start:o_end) + drift;
        end
    end

    %% --- E. feat 1~6 ---
    feat = zeros(Ny, 6, 'single');
    window_size = 20;
    p_raw(1,:) = smoothdata(p_raw(1,:), 'gaussian', window_size);
    p_raw(2,:) = smoothdata(p_raw(2,:), 'gaussian', window_size);
    p_raw(3,:) = smoothdata(p_raw(3,:), 'gaussian', window_size);
    feat(:,1:3) = p_raw';

    V_raw = zeros(3, Ny, 'single');
    V_raw(:,2:end) = (p_raw(:,2:end) - p_raw(:,1:end-1)) / dt;
    V_raw(:,1) = V_raw(:,2);
    V_raw(1,:) = smoothdata(V_raw(1,:), 'gaussian', window_size*2);
    V_raw(2,:) = smoothdata(V_raw(2,:), 'gaussian', window_size*2);
    V_raw(3,:) = smoothdata(V_raw(3,:), 'gaussian', window_size*2);
    feat(:,4:6) = V_raw;

    Y_offset = (rand()-0.5) * 200;
    p_true(1,:) = p_true(1,:) + Y_offset;
    p_raw(1,:)  = p_raw(1,:)  + Y_offset;
    feat(:,1)   = feat(:,1) + Y_offset;
    Pos_A(1) = Pos_A(1) + Y_offset;
    Pos_B(1) = Pos_B(1) + Y_offset;
    Pos_C(1) = Pos_C(1) + Y_offset;

    Pos_A_all(:, iter) = Pos_A;
    Pos_B_all(:, iter) = Pos_B;
    Pos_C_all(:, iter) = Pos_C;

    targets = single([Pos_A, Pos_B, Pos_C]);

    %% --- F. P_true 相对相位 + R_obs RCM ---
    [phase_rel, r_ref, R_obs] = compute_phase_rel_rref(p_true, targets, radar);
    Phase_rel_all(iter, :, :) = phase_rel;
    R_ref_all(iter, :) = r_ref;
    R_obs_all(iter, :, :) = R_obs;

    Feat_All(iter, :, :)  = feat;
    P_raw_all(iter, :, :)  = p_raw';
    P_true_all(iter, :, :) = p_true';
end

disp(['[v4_phase] 样本循环完成, ', num2str(toc,'%.1f'), ' s']);

%% 5. 保存
ds_tag = ['PPTR_TrainDataset_v4_phase_', num2str(num_samples)];
save_name = fullfile(output_dir, [ds_tag, '.mat']);

Feat_All = Feat_All;
P_true = P_true_all;
P_raw = P_raw_all;
Phase_rel = Phase_rel_all;
R_ref = R_ref_all;
R_obs = R_obs_all;
Pos_A = Pos_A_all;
Pos_B = Pos_B_all;
Pos_C = Pos_C_all;

v4_meta = struct( ...
    'version', 'v4_phase', ...
    'phase_trajectory', 'P_true', ...
    'R_ref_def', 'min_t ||P_true-Pos_j||', ...
    'phase_def', 'unwrap(-4*pi*fc*(R-R_ref)/c), radians', ...
    'feat_dim', 6, ...
    'coord', 'Y_lat,X_fwd,Z_up', ...
    'Ny', Ny, 'PRF', PRF);

save(save_name, ...
    'Feat_All', 'P_raw', 'P_true', ...
    'Phase_rel', 'R_ref', ...
    'R_obs', ...
    'Pos_A', 'Pos_B', 'Pos_C', 'fc', ...
    'v4_meta', ...
    '-v7.3');

flight_path = fullfile(output_dir, ['PPTR_FlightParams_v4_', num2str(num_samples), '.mat']);
H_target = H_target_all;
V_target = V_target_all;
save(flight_path, 'H_target', 'V_target', '-v7.3');

disp(['[v4_phase] 训练集: ', save_name]);
disp(['[v4_phase] 飞行参数: ', flight_path]);

%% ========================================================================
function [phase_rel_nt3, r_ref_13, R_obs_nt3] = compute_phase_rel_rref(p, targets, radar)
% p: 3×Ny, targets: 3×3
%   返回: 相对相位 (Ny×3), R_ref (1×3), 含测量噪声的 R_obs (Ny×3)
    Ny = size(p, 2);
    fc = double(radar.fc);
    c  = double(radar.c);
    dr = c / (2 * 200e6);       % 距离分辨率 ≈0.75m (B≈200MHz)

    phase_rel_nt3 = zeros(Ny, 3, 'single');
    r_ref_13 = zeros(1, 3, 'single');
    R_obs_nt3 = zeros(Ny, 3, 'single');

    for j = 1:3
        p_tgt = targets(:, j);
        R_slant = vecnorm(p - p_tgt, 2, 1).';          % 几何斜距

        % ── RCM 测量仿真（从几何→测量）───────────────
        % 1. 多点散射叠加（模拟相干斑）
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
        if abs(p(3, 1) - p_tgt(3)) / mean(R_slant) < 0.15
            shadow_scale = 3.0;
        else
            shadow_scale = 1.0;
        end
        R_obs = R_outlier + shadow_scale * 0.003 * randn(size(R_outlier));   % 3mm 基础噪声
        % ── 测量仿真结束 ─────────────────────────────

        r_ref_j = min(R_obs);
        r_ref_13(j) = single(r_ref_j);
        phase = -4.0 * pi * fc * (double(R_obs) - r_ref_j) / c;
        phase_rel_nt3(:, j) = single(unwrap(phase));
        R_obs_nt3(:, j) = single(R_obs);
    end
end
