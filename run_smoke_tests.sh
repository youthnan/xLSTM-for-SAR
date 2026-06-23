#!/usr/bin/env bash
# run_smoke_tests.sh — PI-xLSTM v4_phase 流水线快速测试
#
# 用法:
#   cd /root/xLSTM
#   conda activate xlstm
#   bash run_smoke_tests.sh          # 全部步骤
#   bash run_smoke_tests.sh data     # 仅数据自检
#   bash run_smoke_tests.sh train    # 仅 2 epoch 烟雾训练
#   bash run_smoke_tests.sh eval     # 仅评估（需已有 RUN_DIR）
#
# 环境变量（可选）:
#   MAT_SMOKE   默认 dataset/PPTR_TrainDataset_v4_phase_10.mat
#   RUN_DIR     eval 步骤用的 run 目录（含 weights/best.pt）
#   NPROC       DDP 卡数，默认 2

set -euo pipefail
cd "$(dirname "$0")"

MAT_SMOKE="${MAT_SMOKE:-/root/xLSTM/dataset/PPTR_TrainDataset_v4_phase_10.mat}"
NPROC="${NPROC:-2}"
RUN_ROOT="/root/xLSTM/run"
STEP="${1:-all}"

die() { echo "[ERROR] $*" >&2; exit 1; }
need_file() { [[ -f "$1" ]] || die "缺少文件: $1"; }

# ── 0. 若无 smoke mat，尝试用 MATLAB/Octave 生成 ───────────────────────
ensure_smoke_mat() {
    if [[ -f "${MAT_SMOKE}" ]]; then
        return 0
    fi
    echo "[prep] 未找到 ${MAT_SMOKE}，尝试运行 data_generate_v4.m (num_samples=10) ..."
    if command -v matlab &>/dev/null; then
        matlab -batch "cd('$(pwd)'); data_generate_v4;"
    elif command -v octave &>/dev/null; then
        octave --eval "cd('$(pwd)'); data_generate_v4;"
    else
        die "请先运行 data_generate_v4.m 生成 ${MAT_SMOKE}"
    fi
    need_file "${MAT_SMOKE}"
}

# ── 1. Python 数据加载 + 字段检查 ─────────────────────────────────────
step_data_py() {
    ensure_smoke_mat
    echo "════════════════════════════════════════"
    echo " [1/4] Python 加载 v4_phase mat 字段检查"
    echo "════════════════════════════════════════"
    python3 << PY
import sys
import numpy as np
import h5py
from pathlib import Path

mat = Path("${MAT_SMOKE}")
required = ["Feat_All", "P_raw", "P_true", "Phase_rel", "R_ref",
            "Pos_A", "Pos_B", "Pos_C", "fc"]
optional = ["v4_meta"]

with h5py.File(mat, "r") as f:
    missing = [k for k in required if k not in f]
    if missing:
        sys.exit(f"缺少必需字段: {missing}")
    feat = np.array(f["Feat_All"])
    n, ny, c = feat.shape[0], feat.shape[1], feat.shape[2]
    assert c == 6, f"Feat_All 最后一维应为 6，得到 {c}"
    pr = np.array(f["P_raw"]).transpose(2, 1, 0)
    pt = np.array(f["P_true"]).transpose(2, 1, 0)
    ph = np.array(f["Phase_rel"]).transpose(2, 1, 0)
    rref = np.array(f["R_ref"]).astype(np.float32)
    if rref.shape == (3, n):
        rref = rref.T
    assert rref.shape == (n, 3), f"R_ref 应为 [N,3]，得到 {rref.shape}"
    err = np.linalg.norm(pr - pt, axis=-1)
    print(f"mat: {mat}")
    print(f"  N={n}, Ny={ny}, Feat C={c}")
    print(f"  |P_raw-P_true| RMSE [mm]: {np.sqrt((err**2).mean())*1000:.2f}")
    print(f"  Phase_rel range [rad]: [{ph.min():.4f}, {ph.max():.4f}]")
    print(f"  R_ref range [m]: [{rref.min():.1f}, {rref.max():.1f}]")
    present_opt = [k for k in optional if k in f]
    print(f"  可选字段: {present_opt or '(无)'}")
print("  OK")
PY
}

# ── 2. MATLAB 相位自洽（需 matlab/octave，可选）────────────────────────
step_data_matlab() {
    ensure_smoke_mat
    if command -v matlab &>/dev/null; then
        echo "════════════════════════════════════════"
        echo " [2/4] MATLAB test_rcm_geom_vs_extract"
        echo "════════════════════════════════════════"
        matlab -batch "cd('$(pwd)'); test_rcm_geom_vs_extract('mat_path', '${MAT_SMOKE}', 'num_mat_samples', 3, 'save_fig', false);"
    elif command -v octave &>/dev/null; then
        echo "[2/4] 使用 octave 运行 test_rcm_geom_vs_extract.m ..."
        octave --eval "test_rcm_geom_vs_extract('mat_path', '${MAT_SMOKE}', 'num_mat_samples', 3, 'save_fig', false);"
    else
        echo "[2/4] 跳过 MATLAB 检查（未安装 matlab/octave）"
    fi
}

# ── 3. phase 短训练（DDP 烟雾）──────────────────────────────────────────
step_train() {
    ensure_smoke_mat
    echo "════════════════════════════════════════"
    echo " [3/4] phase 2-epoch DDP 烟雾训练"
    echo "════════════════════════════════════════"
    torchrun --standalone --nproc_per_node="${NPROC}" train_pi_xlstm.py \
        --mat_path "${MAT_SMOKE}" \
        --model phase \
        --batch_size 4 \
        --grad_accum_steps 2 \
        --epochs 2 \
        --lr 5e-4 \
        --hidden_dim 128 \
        --num_blocks 4 \
        --num_heads 4 \
        --lambda_smooth 2.0 \
        --lambda_rcm_max 0.0 \
        --lambda_phase_max 1.0 \
        --phase_warmup_start 0 \
        --phase_warmup_end 2 \
        --early_stop_patience 0 \
        --checkpoint_every 1 \
        --val_ratio 0.1 \
        --run_name smoke_phase_v4

    RUN_DIR="$(ls -td ${RUN_ROOT}/*smoke_phase_v4* 2>/dev/null | head -1)"
    [[ -n "${RUN_DIR}" ]] || die "未找到 smoke run 目录"
    echo "SMOKE_RUN_DIR=${RUN_DIR}" > "${RUN_ROOT}/.last_smoke_run"
    echo "  → run: ${RUN_DIR}"
}

# ── 4. 评估 best.pt ─────────────────────────────────────────────────────
step_eval() {
    if [[ -z "${RUN_DIR:-}" ]]; then
        if [[ -f "${RUN_ROOT}/.last_smoke_run" ]]; then
            # shellcheck source=/dev/null
            source "${RUN_ROOT}/.last_smoke_run"
        fi
    fi
    RUN_DIR="${RUN_DIR:-}"
    [[ -n "${RUN_DIR}" ]] || die "请设置 RUN_DIR 或先执行: bash run_smoke_tests.sh train"
    CKPT="${RUN_DIR}/weights/best.pt"
    [[ -f "${CKPT}" ]] || CKPT="${RUN_DIR}/weights/latest.pt"
    need_file "${CKPT}"
    ensure_smoke_mat

    echo "════════════════════════════════════════"
    echo " [4/4] eval_pi_xlstm.py"
    echo "════════════════════════════════════════"
    JSON_OUT="${RUN_DIR}/eval_smoke.json"
    python3 eval_pi_xlstm.py \
        --checkpoint "${CKPT}" \
        --mat_path "${MAT_SMOKE}" \
        --batch_size 4 \
        --json_out "${JSON_OUT}"
    echo "  → 指标: ${JSON_OUT}"
}

case "${STEP}" in
    all)
        step_data_py
        step_data_matlab
        step_train
        step_eval
        ;;
    data)
        step_data_py
        step_data_matlab
        ;;
    train)
        step_train
        ;;
    eval)
        step_eval
        ;;
    *)
        echo "用法: bash run_smoke_tests.sh [all|data|train|eval]"
        exit 1
        ;;
esac

echo ""
echo "全部请求步骤完成。"
