#!/usr/bin/env bash
# run_experiments.sh — 批量跑对比/消融实验
#
# 用法：
#   bash run_experiments.sh [实验组]
#
# 实验组：
#   baselines  — 跑 BiLSTM / BiGRU / Transformer 基线（依次串行）
#   ablations  — 跑消融实验（无RCM / 无Smooth / 单向xLSTM）
#   phase      — v4_phase 单头 + Phase_rel/R_ref（需 PPTR_TrainDataset_v4_phase_*.mat）
#   cascade    — v4 双头 cascade（需旧版 PPTR_TrainDataset_v4_*.mat）
#   all        — 全部跑（默认，不含 cascade）
#
# 前提：PI-xLSTM (v13) 已在跑，本脚本串行启动其余实验。
# 每个实验约 8-9 小时，建议在 tmux 中运行本脚本。

set -euo pipefail

MAT_PATH="/root/xLSTM/dataset3/PPTR_TrainDataset_5000.mat"
COMMON_ARGS=(
  --mat_path "$MAT_PATH"
  --batch_size 8
  --grad_accum_steps 2
  --epochs 200
  --lr 1e-3
  --hidden_dim 128
  --num_blocks 4
  --num_heads 4
  --lambda_smooth 2.0
  --lambda_rcm_max 1.0
  --rcm_warmup_start 10
  --rcm_warmup_end 80
  --scheduler_eta_min 1e-5
  --val_ratio 0.1
  --early_stop_patience 15
  --checkpoint_every 10
  --param_report_every 10
)
TORCHRUN="torchrun --standalone --nproc_per_node=2"
GROUP="${1:-all}"

run_exp() {
    local run_name="$1"
    shift
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  启动实验: $run_name"
    echo "════════════════════════════════════════════════════════"
    $TORCHRUN train_pi_xlstm.py "${COMMON_ARGS[@]}" --run_name "$run_name" "$@"
    echo "  ✓ 完成: $run_name"
}

# ──────────────────────────────────────────────
# 基线对比实验
# ──────────────────────────────────────────────
run_baselines() {
    run_exp "v14_bilstm"       --model bilstm
    run_exp "v15_bigru"        --model bigru
    run_exp "v16_transformer"  --model transformer
}

# ──────────────────────────────────────────────
# 消融实验
# ──────────────────────────────────────────────
run_ablations() {
    # 消融1：去掉 RCM 物理约束（lambda_rcm_max=0）
    run_exp "v17_ablation_no_rcm" \
        --model xlstm \
        --lambda_rcm_max 0.0

    # 消融2：去掉平滑约束（lambda_smooth=0）
    run_exp "v18_ablation_no_smooth" \
        --model xlstm \
        --lambda_smooth 0.0

    # 消融3：关闭随机参考点掩码（测试 training-inference mismatch 的影响）
    run_exp "v19_ablation_no_refmask" \
        --model xlstm \
        --no_random_ref_mask
}

# ──────────────────────────────────────────────
# v4_phase 单头 + 相对相位监督
# ──────────────────────────────────────────────
run_phase_v4() {
    local v4_mat="/root/xLSTM/dataset/PPTR_TrainDataset_v4_phase_5000.mat"
    if [[ ! -f "$v4_mat" ]]; then
        echo "跳过 phase：未找到 $v4_mat（请先用 data_generate_v4.m 生成）"
        return 0
    fi
    run_exp "v21_phase_v4" \
        --mat_path "$v4_mat" \
        --model phase \
        --batch_size 8 \
        --grad_accum_steps 2 \
        --epochs 200 \
        --lambda_rcm_max 0.0 \
        --lambda_phase_max 1.0 \
        --phase_warmup_start 0 \
        --phase_warmup_end 40 \
        --lambda_smooth 2.0
}

# ──────────────────────────────────────────────
# v4 双头 cascade（旧版 mat）
# ──────────────────────────────────────────────
run_cascade_v4() {
    local v4_mat="/root/xLSTM/dataset/PPTR_TrainDataset_v4_5000.mat"
    if [[ ! -f "$v4_mat" ]]; then
        echo "跳过 cascade：未找到 $v4_mat（请先用 data_generate_v4.m 生成）"
        return 0
    fi
    run_exp "v20_cascade_v4" \
        --mat_path "$v4_mat" \
        --model cascade \
        --batch_size 4 \
        --grad_accum_steps 2 \
        --epochs 200 \
        --lambda_c_max 1.0 \
        --lambda_f_max 1.0 \
        --lambda_phase_max 1.0 \
        --lambda_total_max 0.3 \
        --f_warmup_start 40 \
        --f_warmup_end 100 \
        --phase_warmup_start 40 \
        --phase_warmup_end 100 \
        --total_warmup_start 100 \
        --total_warmup_end 160 \
        --freeze_fine_until_epoch 40 \
        --detach_coarse_until_epoch 60
}

# ──────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────
case "$GROUP" in
    baselines) run_baselines ;;
    ablations) run_ablations ;;
    phase)     run_phase_v4 ;;
    cascade)   run_cascade_v4 ;;
    all)
        run_baselines
        run_ablations
        ;;
    *)
        echo "未知实验组 '$GROUP'，可选: baselines / ablations / phase / cascade / all"
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════════════════════"
echo "  所有实验完成！运行 python summarize_results.py 查看汇总"
echo "════════════════════════════════════════════════════════"
