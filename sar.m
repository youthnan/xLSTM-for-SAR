% =========================================================================
% SAR 批量数据集 (1000组) 随机抽查测试与成像验证脚本 (含航迹可视化)
% =========================================================================
clear; clc; close all;

%% 1. 加载生成的批量样本数据
dataset_path = '/Users/nan/LSTM-reconstruction/物理生成航迹/PPTR_Dataset_4/PPTR_TrainDataset_10.mat'; % 根据实际路径调整
avg_path = '/Users/nan/LSTM-reconstruction/物理生成航迹/PPTR_Dataset_4/PPTR_FlightParams_10.mat'; % 根据实际路径调整


disp(['正在加载批量数据集: ', dataset_path, ' ...']);
load(dataset_path); 

disp(['正在加载批量数据集: ', avg_path, ' ...']);
load(avg_path); 
%%
% 随机抽取一个样本
num_samples = size(P_true, 1);
sample_idx = randi(num_samples); 
disp(['================================================']);
disp(['>> 成功抽取测试样本: 第 ', num2str(sample_idx), ' 个 / 共 ', num2str(num_samples), ' 个']);
disp(['================================================']);

% 提取当前随机样本的轨迹数据 (从 [N, T, 3] 转为 [3, T])
P_true = squeeze(P_true(sample_idx, :, :))'; 
P_raw  = squeeze(P_raw(sample_idx, :, :))';
H_target = H_target(sample_idx);

% ★ 新格式: Pos_A/B/C 为 [3, N]，必须提取对应样本列 → [3, 1]
% 若文件仍为旧格式 [3,1]，size(...,2)==1 时跳过提取
if size(Pos_A, 2) > 1
    Pos_A = Pos_A(:, sample_idx);
    Pos_B = Pos_B(:, sample_idx);
    Pos_C = Pos_C(:, sample_idx);
end

%% 2. 重建雷达与时间轴参数 
disp('重建雷达参数...');
fc = 9.5e9;              
c = 3e8;                
B = 2e9;              
Fs = 1.2 * B;           
SampTime = 1/Fs;
Tp = 1e-6;              
Kr = B / Tp;            
Ny = size(P_true, 2);
Nr = 2048*2;
PRF = 1000;
PRI = 1/PRF; % 飞控仿真步长
Ts = 10;
flag_Ran = (Nr*2* SampTime > Tp);
flag_Azi = (Ny * PRI > Ts);
% 重建快时间轴
% R_c = 参考斜距，取 Pos_B 的最短斜距（垂飞时：只有侧向距离 + 高度）
% 坐标系: p_true(1,:)=Y_lat(侧向), p_true(2,:)=X_fwd(方位向), p_true(3,:)=Z(高度)
% Pos_B(1)=Y_lat(侧向偏置), Pos_B(3)=0(地面目标)
% 飞机 Y_lat ≈ 0，最短斜距 ≈ sqrt(Pos_B(1)^2 + H_target^2)
R_c = sqrt(Pos_B(1)^2 + H_target^2);
tr = 2*R_c/c + SampTime*(-Nr/2:1:Nr/2-1);

%% 3. 在线生成当前随机样本的 SAR 回波并做脉压
disp('正在为当前抽取样本重新生成原始回波并进行距离向压缩...');
Echo_true = zeros(Ny, Nr); 
targets = [Pos_A, Pos_B, Pos_C];
sigma = [1, 1, 1]; 

for ii = 1:Ny
    for t_idx = 1:3
        Rt = norm(P_true(:, ii) - targets(:, t_idx));
        tau = 2*Rt/c;
        tf = tr - tau; 
        phase = -2*pi*fc*tau + pi*Kr*tf.^2;
        Echo_true(ii, :) = Echo_true(ii, :) + sigma(t_idx) * exp(1j*phase) .* (abs(tf) < Tp/2);
    end
end

signal_fft = fft(Echo_true, [], 2); 
t_ref = -Tp/2 : 1/Fs : Tp/2;
ref_base = [exp(1j*pi*Kr*t_ref.^2), zeros(1, Nr - length(t_ref))];
shift_val = -round((Tp/2) * Fs); 
RefSignal_Range = circshift(ref_base, shift_val);
RS_R_fft = fft(RefSignal_Range);
Srancom_true = ifft(signal_fft .* conj(RS_R_fft), [], 2);

%% ========================================================================
%% 3.5 航迹三维可视化与误差分析 (新增部分)
%% ========================================================================
% 1. 三维航迹对比
figure;
plot3(P_true(2,:), P_true(1,:), P_true(3,:), 'k--', 'LineWidth', 2); hold on;
plot3(P_raw(2,:),  P_raw(1,:),  P_raw(3,:),  'r',   'LineWidth', 1.5);
grid on; xlabel('X_{fwd}(m) 方位向'); ylabel('Y_{lat}(m) 距离向'); zlabel('Z(m) 高度');
title('3D SAR 航迹'); legend('理想直线航迹', '带物理误差的真实航迹');
view([-30, 30]);



disp('绘制三维航迹与轨迹误差对比图...');
figure('Color', 'w', 'Name', ['样本 ', num2str(sample_idx), ' 航迹与误差分析'], 'Position', [150, 200, 1000, 400]);

% 子图1：3D 航迹与目标点相对位置
subplot(1, 2, 1);
plot3(P_true(2,:), P_true(1,:), P_true(3,:), 'g',   'LineWidth', 2); hold on;
plot3(P_raw(2,:),  P_raw(1,:),  P_raw(3,:),  'r--', 'LineWidth', 1.5);

% 绘制并标注三个点目标
scatter3([Pos_A(1), Pos_B(1), Pos_C(1)], ...
         [Pos_A(2), Pos_B(2), Pos_C(2)], ...
         [Pos_A(3), Pos_B(3), Pos_C(3)], 60, 'k', '^', 'filled');
text(Pos_A(1), Pos_A(2)+15, Pos_A(3), ' Target A', 'FontSize', 10, 'FontWeight', 'bold');
text(Pos_B(1), Pos_B(2)+15, Pos_B(3), ' Target B', 'FontSize', 10, 'FontWeight', 'bold');
text(Pos_C(1), Pos_C(2)+15, Pos_C(3), ' Target C', 'FontSize', 10, 'FontWeight', 'bold');

grid on;
xlabel('X_{fwd} 方位向 (m)'); ylabel('Y_{lat} 距离向 (m)'); zlabel('Z 高度 (m)');
title('3D 物理真值航迹 vs 惯导测量航迹');
legend('P\_true (真值)', 'P\_raw (带漂移)', '地面点目标', 'Location', 'best');
view([-30, 30]); % 设定最佳的三维观测视角

% 子图2：三轴轨迹测量误差剖面
subplot(1, 2, 2);
trajectory_error = P_raw - P_true;
time_axis = (0:Ny-1) * PRI; 

plot(time_axis, trajectory_error(1,:) * 100, 'r', 'LineWidth', 1.5); hold on;
plot(time_axis, trajectory_error(2,:) * 100, 'b', 'LineWidth', 1.5);
plot(time_axis, trajectory_error(3,:) * 100, 'k', 'LineWidth', 1.5);

grid on;
xlabel('慢时间 (s)'); ylabel('误差幅度 (cm)');
title('原始测量航迹三轴低频漂移残差');
legend('X轴漂移 (距离向)', 'Y轴漂移 (方位向)', 'Z轴漂移 (高度向)', 'Location', 'best');

%% 4. 计算网格点坐标 (设定成像区域)
dx = 0.02; dy = 0.02;           
ImageNumX = 256;               
ImageNumY = 256;              

x0 = Pos_B(1);                
y0 = Pos_B(2);                
z0 = 0;                       

xp = (-ImageNumX/2 : ImageNumX/2-1) * dx + x0;
yp = (-ImageNumY/2 : ImageNumY/2-1) * dy + y0;
[xp2, yp2] = meshgrid(xp, yp); 
zp2 = z0 * ones(ImageNumY, ImageNumX);

%% 5. 执行 BP 成像
disp('开始使用真值航迹 (P_true) 进行 BP 成像...');
Img_true = SAR_BackProjection2D(Srancom_true, P_true, xp, yp, xp2, yp2, zp2, tr, fc, SampTime, c,4,'理想');
disp('开始使用带低频误差的航迹 (P_raw) 进行 BP 成像...');
Img_raw = SAR_BackProjection2D(Srancom_true, P_raw, xp, yp, xp2, yp2, zp2, tr, fc, SampTime, c,4,'真实');


%% ========================================================================
% %  所需核心函数区 
% %  ========================================================================
% function Img = SAR_BackProjection2D(Srancom, P_radar, xp, yp, xp2, yp2, zp2, tr, fc, SampTime, c)
%     [Ny, Nr] = size(Srancom);
%     Img = zeros(size(xp2));
%     h = waitbar(0, '正在执行 BP 成像...');
%     for k = 1:Ny
%         xr = P_radar(1, k); 
%         yr = P_radar(2, k); 
%         zr = P_radar(3, k);
% 
%         R = sqrt((xp2 - xr).^2 + (yp2 - yr).^2 + (zp2 - zr).^2);
%         tau = 2 * R / c;
% 
%         idx_float = (tau - tr(1)) / SampTime + 1;
%         idx_floor = floor(idx_float);
%         idx_ceil = idx_floor + 1;
% 
%         valid = (idx_floor >= 1) & (idx_ceil <= Nr);
%         idx_floor(~valid) = 1; 
%         idx_ceil(~valid) = 1;
% 
%         w_ceil = idx_float - idx_floor;
%         w_floor = 1 - w_ceil;
% 
%         S_k = Srancom(k, :);
%         S_interp = w_floor .* S_k(idx_floor) + w_ceil .* S_k(idx_ceil);
%         S_interp(~valid) = 0;
% 
%         phase_comp = exp(1j * 4 * pi * fc * R / c);
%         Img = Img + S_interp .* phase_comp;
% 
%         if mod(k, round(Ny/20)) == 0
%             waitbar(k/Ny, h);
%         end
%     end
%     close(h);
% end
% 
% function [Img_out, phi_total] = SAR_PGA(Img_in, num_iter)
%     [Ny_img, Nx_img] = size(Img_in); 
%     Img_out = Img_in;
%     phi_total = zeros(1, Ny_img); 
% 
%     for iter = 1:num_iter
%         [~, max_idx] = max(max(abs(Img_out), [], 1)); 
%         brightest_range_line = Img_out(:, max_idx).'; 
% 
%         [~, peak_az] = max(abs(brightest_range_line));
%         shifted_line = circshift(brightest_range_line, round(Ny_img/2) - peak_az, 2);
% 
%         win_width = max(4, round(Ny_img / (2^(iter+1)))); 
%         win = zeros(1, Ny_img);
%         win(round(Ny_img/2) - win_width : round(Ny_img/2) + win_width) = 1;
%         shifted_line = shifted_line .* win;
% 
%         g = ifft(shifted_line);
%         g_dot = diff(g);
%         delta_phi = imag(conj(g(1:end-1)) .* g_dot) ./ (abs(g(1:end-1)).^2 + eps);
%         delta_phi = [delta_phi, 0];
% 
%         phi_est = cumsum(delta_phi);
% 
%         x_axis = 1:Ny_img;
%         p = polyfit(x_axis, phi_est, 1);
%         phi_est = phi_est - polyval(p, x_axis);
% 
%         phi_total = phi_total + phi_est;
% 
%         Img_ph = ifft(Img_out, [], 1);
%         Img_ph = Img_ph .* exp(-1j * phi_est.');
%         Img_out = fft(Img_ph, [], 1);
%     end
% end