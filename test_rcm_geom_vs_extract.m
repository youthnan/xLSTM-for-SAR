% =========================================================================
% test_rcm_geom_vs_extract.m
% 校验 data_generate_v4 输出与 P_true/P_raw 几何量是否一致。
%
% v4_phase mat:
%   Phase_rel = unwrap(-4*pi*fc*(R_true-R_ref)/c)
%   R_ref = min_t ||P_true-Pos||
%
% 旧版 v4 mat（含 Feat 12 维 / Phase_sin）仍走 legacy 分支。
% =========================================================================
function test_rcm_geom_vs_extract(varargin)
    p = inputParser;
    addParameter(p, 'mat_path', '', @(s) ischar(s) || isstring(s));
    addParameter(p, 'sample_idx', [], @(x) isempty(x) || (isnumeric(x) && isvector(x)));
    addParameter(p, 'num_mat_samples', 5, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'save_fig', true, @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'out_dir', '', @(s) ischar(s) || isstring(s));
    parse(p, varargin{:});
    opts = p.Results;

    root = fileparts(mfilename('fullpath'));
    mat_path = resolve_v4_mat_path(root, char(opts.mat_path));
    fprintf('[check] mat: %s\n', mat_path);

  info = whos('-file', mat_path);
  names = {info.name};
  if ismember('Phase_rel', names)
      test_v4_phase_mat(mat_path, opts);
  else
      test_v4_legacy_mat(mat_path, opts);
  end
end

function test_v4_phase_mat(mat_path, opts)
    root = fileparts(mfilename('fullpath'));
    if strlength(string(opts.out_dir)) == 0
        out_dir = fullfile(root, 'dataset', 'phase_compare_report');
    else
        out_dir = char(opts.out_dir);
    end
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    S = load(mat_path, 'P_raw', 'P_true', 'Pos_A', 'Pos_B', 'Pos_C', ...
        'Feat_All', 'Phase_rel', 'R_ref', 'fc', 'v4_meta');

    n_total = size(S.P_raw, 1);
    Ny = size(S.P_raw, 2);
    n_feat = size(S.Feat_All, 3);
    if n_feat ~= 6
        error('v4_phase 期望 Feat_All 为 [N,Ny,6]，得到维度 %d', n_feat);
    end

    if isfield(S, 'fc')
        fc = double(S.fc(1));
    else
        fc = 9.6e9;
    end
    c = 3e8;
    ref_names = {'A', 'B', 'C'};

    if isempty(opts.sample_idx)
        idx_list = 1:min(opts.num_mat_samples, n_total);
    else
        idx_list = opts.sample_idx(:).';
    end
    fprintf('[v4_phase] 样本 [%s] / N=%d, Ny=%d, feat_dim=%d\n', ...
        num2str(idx_list), n_total, Ny, n_feat);

    max_phase_err = 0;
    max_rref_err = 0;

    for idx = idx_list
        p_true = squeeze(single(S.P_true(idx, :, :))).';
        targets = single([S.Pos_A(:, idx), S.Pos_B(:, idx), S.Pos_C(:, idx)]);
        phase_mat = squeeze(single(S.Phase_rel(idx, :, :)));
        rref_mat = single(S.R_ref(idx, :));
        if numel(rref_mat) ~= 3
            rref_mat = rref_mat(:).';
        end

        for j = 1:3
            R_slant = vecnorm(p_true - targets(:, j), 2, 1).';
            rref_j = min(R_slant);
            phase_calc = unwrap(-4.0 * pi * fc * (double(R_slant) - double(rref_j)) / c);
            d_phase = phase_calc - double(phase_mat(:, j));
            d_rref = double(rref_j) - double(rref_mat(j));
            max_phase_err = max(max_phase_err, max(abs(d_phase)));
            max_rref_err = max(max_rref_err, abs(d_rref));
        end
    end

    fprintf('  max |Phase_rel 重算差| = %.6e rad\n', max_phase_err);
    fprintf('  max |R_ref 重算差|     = %.6e m\n', max_rref_err);
    if max_phase_err > 1e-4 || max_rref_err > 1e-4
        warning('v4_phase 自洽检查未通过阈值 1e-4');
    else
        fprintf('  OK: Phase_rel / R_ref 与 P_true 几何一致。\n');
    end
end

function test_v4_legacy_mat(mat_path, opts)
    root = fileparts(mfilename('fullpath'));
    if strlength(string(opts.out_dir)) == 0
        out_dir = fullfile(root, 'dataset', 'rcm_compare_report');
    else
        out_dir = char(opts.out_dir);
    end
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    S = load(mat_path, 'P_raw', 'P_true', 'Pos_A', 'Pos_B', 'Pos_C', ...
        'Feat_All', 'Phase_sin', 'Phase_cos', 'fc', 'v4_meta');

    if ~isfield(S, 'Feat_All')
        error('mat 缺少 Feat_All');
    end
    n_total = size(S.P_raw, 1);
    Ny = size(S.P_raw, 2);
    n_feat = size(S.Feat_All, 3);
    if n_feat < 9
        error('legacy 检查需要 Feat_All 至少 9 维，得到 %d', n_feat);
    end

    if isfield(S, 'fc')
        fc = double(S.fc(1));
    else
        fc = 9.5e9;
    end
    lambda = 3e8 / fc;
    ref_names = {'A', 'B', 'C'};

    if isempty(opts.sample_idx)
        idx_list = 1:min(opts.num_mat_samples, n_total);
    else
        idx_list = opts.sample_idx(:).';
    end
    fprintf('[legacy] 样本 [%s] / N=%d, Ny=%d\n', num2str(idx_list), n_total, Ny);

    for idx = idx_list
        p_true = squeeze(single(S.P_true(idx, :, :))).';
        feat = squeeze(single(S.Feat_All(idx, :, :)));
        if n_feat >= 9
            RCM_mat = feat(:, 7:9);
            targets = single([S.Pos_A(:, idx), S.Pos_B(:, idx), S.Pos_C(:, idx)]);
            RCM_true = zeros(Ny, 3, 'single');
            for j = 1:3
                RCM_true(:, j) = single(vecnorm(p_true - targets(:, j), 2, 1).');
            end
            dR = double(RCM_mat) - double(RCM_true);
            fprintf('  idx %d: RCM vs P_true RMSE = %.4f m\n', idx, sqrt(mean(dR(:).^2)));
        end
        if isfield(S, 'Phase_sin') && isfield(S, 'Phase_cos')
            ph_sin = squeeze(single(S.Phase_sin(idx, :, :)));
            ph_cos = squeeze(single(S.Phase_cos(idx, :, :)));
            phi_mat = atan2(double(ph_sin), double(ph_cos));
            fprintf('  idx %d: 存盘相位 (legacy sin/cos) 已加载\n', idx);
        end
    end
    fprintf('  legacy 检查完成（详见旧版脚本逻辑）。\n');
end

function mat_path = resolve_v4_mat_path(root, mat_path_user)
    if strlength(strtrim(string(mat_path_user))) > 0
        mat_path = char(mat_path_user);
        if ~isfile(mat_path)
            error('mat 不存在: %s', mat_path);
        end
        return;
    end
    ds_dir = fullfile(root, 'dataset');
    d_all = dir(fullfile(ds_dir, 'PPTR_TrainDataset_v4*.mat'));
    d_all = d_all(~[d_all.isdir]);
    if isempty(d_all)
        error('dataset/ 下未找到 PPTR_TrainDataset_v4*.mat，请先运行 data_generate_v4.m');
    end
    [~, ix] = max([d_all.datenum]);
    mat_path = fullfile(d_all(ix).folder, d_all(ix).name);
end
