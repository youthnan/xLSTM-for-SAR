% =========================================================================
% SAR 批量数据集随机抽查 + P_fix 校正航迹成像验证（含三维航迹与 BP 对比）
% =========================================================================
% 依赖：
%   1) 原始数据集 .mat：含 P_true_All / P_raw_All（或见下方 HDF5 兼容分支）
%   2) Python eval_pi_xlstm.py 导出的 P_fix_output.mat，变量名 P_fix，尺寸 [N,T,3]
% 用法：修改下方 dataset_path、pfix_path，在 MATLAB 中运行本脚本。
% =========================================================================
clear; clc; close all;

%% 0. 路径与随机种子（固定 sample_idx 可与 Python 单条导出索引对齐）
dataset_path = '/Users/nan/LSTM-reconstruction/物理生成航迹/PPTR_Dataset/PPTR_TrainDataset_1000.mat';
pfix_path    = '/Users/nan/LSTM-reconstruction/LSTM-reconstruction/xLSTM/run/20260510_013520_352874_pid7911/P_fix_output.mat';

if ~exist(dataset_path, 'file')
    dataset_path = 'PPTR_TrainDataset_1000.mat';
end
if ~exist(pfix_path, 'file')
    pfix_path = 'P_fix_output.mat';
end

rng(42);  % 固定随机样本；若要与 Python 指定第 k 条一致，可改为 sample_idx = k;

%% 1. 加载批量样本（整文件读入；超大文件可改用 matfile 分块）
disp(['正在加载批量数据集: ', dataset_path, ' ...']);
S_data = load(dataset_path);

if isfield(S_data, 'P_true_All') && isfield(S_data, 'P_raw_All')
    P_true_All = S_data.P_true_All;
    P_raw_All  = S_data.P_raw_All;
elseif isfield(S_data, 'P_true') && isfield(S_data, 'P_raw')
    P_true_All = S_data.P_true;
    P_raw_All  = S_data.P_raw;
else
    fns = fieldnames(S_data);
    error('未找到 P_true_All/P_raw_All 或 P_true/P_raw。文件中字段: %s', strjoin(fns', ', '));
end

num_samples = size(P_true_All, 1);
Ny          = size(P_true_All, 2);
sample_idx  = randi(num_samples);

disp('================================================');
disp(['>> 测试样本: 第 ', num2str(sample_idx), ' 个 / 共 ', num2str(num_samples), ' 个']);
disp('================================================');

P_true = squeeze(P_true_All(sample_idx, :, :))';  % [3, Ny]
P_raw  = squeeze(P_raw_All(sample_idx, :, :))';

%% 1.5 加载 P_fix（与数据集同 N、同 T）
disp(['正在加载 P_fix: ', pfix_path, ' ...']);
S_pfix = load(pfix_path, 'P_fix');
if ~isfield(S_pfix, 'P_fix')
    error('P_fix_output.mat 中缺少变量 P_fix，请用 eval_pi_xlstm.py --save_p_fix 导出。');
end
P_fix_all = S_pfix.P_fix;
if size(P_fix_all, 1) ~= num_samples || size(P_fix_all, 2) ~= Ny
    error('P_fix 尺寸 [%d,%d,*] 与数据集 [%d,%d,*] 不一致，请对同一份 .mat 导出 P_fix。', ...
        size(P_fix_all,1), size(P_fix_all,2), num_samples, Ny);
end
P_fix = squeeze(P_fix_all(sample_idx, :, :))';  % [3, Ny]

%% 雷达位置（若 mat 中无则与 Python RadarPhysicsLoss 一致）
if isfield(S_data, 'Pos_A')
    Pos_A = S_data.Pos_A(:);
    Pos_B = S_data.Pos_B(:);
    Pos_C = S_data.Pos_C(:);
else
    Pos_A = [387; 25; 0];
    Pos_B = [417; 56; 0];
    Pos_C = [403; 78; 0];
end
clear S_data

%% 2. 雷达与时间轴
disp('重建雷达参数...');
fc = 9.5e9;
c  = 3e8;
B  = 2e9;
Fs = 1.2 * B;
SampTime = 1/Fs;
Tp = 1e-6;
Kr = B / Tp;
Nr = 2048*2;
PRF = 1000;
PRI = 1/PRF;
Ts = 10;
flag_Ran = (Nr*2* SampTime > Tp);
flag_Azi = (Ny * PRI > Ts);

H_target = 300;
R_c = norm(Pos_B - [0; 0; H_target]);
tr = 2*R_c/c + SampTime*(-Nr/2:1:Nr/2-1);

%% 3. 回波仿真 + 距离向脉压（仍用真值航迹 P_true）
disp('正在为当前样本生成回波并进行距离向压缩...');
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

%% 4. 三维航迹对比
figure('Color','w','Name','3D 航迹','Position',[80 120 560 420]);
plot3(P_true(1,:), P_true(2,:), P_true(3,:), 'k--', 'LineWidth', 2); hold on;
plot3(P_raw(1,:),  P_raw(2,:),  P_raw(3,:),  'r', 'LineWidth', 1.5);
plot3(P_fix(1,:),  P_fix(2,:),  P_fix(3,:),  'b-', 'LineWidth', 1.5);
grid on; xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(['样本 ', num2str(sample_idx), ' — 航迹']);
legend('P\_true','P\_raw','P\_fix','Location','best');
view([-30, 30]);

%% 5. 航迹与误差分析（双子图）
disp('绘制航迹与误差...');
figure('Color', 'w', 'Name', ['样本 ', num2str(sample_idx), ' 误差'], 'Position', [150, 200, 1000, 400]);

subplot(1, 2, 1);
plot3(P_true(1,:), P_true(2,:), P_true(3,:), 'g', 'LineWidth', 2); hold on;
plot3(P_raw(1,:), P_raw(2,:), P_raw(3,:), 'r--', 'LineWidth', 1.5);
plot3(P_fix(1,:), P_fix(2,:), P_fix(3,:), 'b-', 'LineWidth', 1.5);
scatter3([Pos_A(1),Pos_B(1),Pos_C(1)], [Pos_A(2),Pos_B(2),Pos_C(2)], [Pos_A(3),Pos_B(3),Pos_C(3)], 60, 'k', '^', 'filled');
text(Pos_A(1), Pos_A(2)+15, Pos_A(3), ' A', 'FontWeight','bold');
text(Pos_B(1), Pos_B(2)+15, Pos_B(3), ' B', 'FontWeight','bold');
text(Pos_C(1), Pos_C(2)+15, Pos_C(3), ' C', 'FontWeight','bold');
grid on; xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('真值 / 测量 / 校正');
legend('P\_true','P\_raw','P\_fix','目标','Location','best');
view([-30, 30]);

subplot(1, 2, 2);
time_axis = (0:Ny-1) * PRI;
err_raw = P_raw - P_true;
err_fix = P_fix - P_true;
plot(time_axis, err_raw(1,:)*100, 'r-', 'LineWidth', 1.2); hold on;
plot(time_axis, err_raw(2,:)*100, 'b-', 'LineWidth', 1.2);
plot(time_axis, err_raw(3,:)*100, 'k-', 'LineWidth', 1.2);
plot(time_axis, err_fix(1,:)*100, 'r--', 'LineWidth', 1);
plot(time_axis, err_fix(2,:)*100, 'b--', 'LineWidth', 1);
plot(time_axis, err_fix(3,:)*100, 'k--', 'LineWidth', 1);
grid on; xlabel('慢时间 (s)'); ylabel('误差 (cm)');
title('相对真值残差：实线 raw，虚线 fix');
legend('raw X','raw Y','raw Z','fix X','fix Y','fix Z','Location','best');

%% 6. 成像网格
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

%% 7. BP 三路成像（同一 Srancom_true）
disp('BP 成像：P_true ...');
Img_true = SAR_BackProjection2D(Srancom_true, P_true, xp, yp, xp2, yp2, zp2, tr, fc, SampTime, c, 4, 'true');

disp('BP 成像：P_raw ...');
Img_raw_bp = SAR_BackProjection2D(Srancom_true, P_raw, xp, yp, xp2, yp2, zp2, tr, fc, SampTime, c, 4, 'raw');

disp('BP 成像：P_fix ...');
Img_fix = SAR_BackProjection2D(Srancom_true, P_fix, xp, yp, xp2, yp2, zp2, tr, fc, SampTime, c, 4, 'fix');

figure('Name','BP 幅度对比','Position',[100 80 1200 360]);
subplot(1,3,1); imagesc(abs(Img_true)); axis image; title('|Img true|'); colorbar;
subplot(1,3,2); imagesc(abs(Img_raw_bp)); axis image; title('|Img raw|'); colorbar;
subplot(1,3,3); imagesc(abs(Img_fix)); axis image; title('|Img fix|'); colorbar;

disp('完成。');

% =========================================================================
% 本地函数：二维 BP（与常见脚本一致）
% =========================================================================
function Img = SAR_BackProjection2D(Srancom, P_radar, xp, yp, xp2, yp2, zp2, tr, fc, SampTime, c, ~, tagStr)
    if nargin < 12 || isempty(tagStr)
        tagStr = '';
    end
    [Ny, Nr] = size(Srancom);
    Img = zeros(size(xp2));
    h = waitbar(0, ['BP 成像 ', tagStr, ' ...']);
    for k = 1:Ny
        xr = P_radar(1, k);
        yr = P_radar(2, k);
        zr = P_radar(3, k);

        R = sqrt((xp2 - xr).^2 + (yp2 - yr).^2 + (zp2 - zr).^2);
        tau = 2 * R / c;

        idx_float = (tau - tr(1)) / SampTime + 1;
        idx_floor = floor(idx_float);
        idx_ceil = idx_floor + 1;

        valid = (idx_floor >= 1) & (idx_ceil <= Nr);
        idx_floor(~valid) = 1;
        idx_ceil(~valid) = 1;

        w_ceil = idx_float - idx_floor;
        w_floor = 1 - w_ceil;

        S_k = Srancom(k, :);
        S_interp = w_floor .* S_k(idx_floor) + w_ceil .* S_k(idx_ceil);
        S_interp(~valid) = 0;

        phase_comp = exp(1j * 4 * pi * fc * R / c);
        Img = Img + S_interp .* phase_comp;

        if mod(k, max(1, round(Ny/20))) == 0
            waitbar(k/Ny, h);
        end
    end
    close(h);
end
