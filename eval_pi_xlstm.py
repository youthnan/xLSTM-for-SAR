#!/usr/bin/env python3
"""
在指定 .mat 上加载训练得到的 checkpoint，做验证 / 测试（无梯度）。

示例:
  cd /root/xLSTM
  python eval_pi_xlstm.py \\
    --checkpoint ./run/<某次>/weights/latest.pt \\
    --mat_path ./PPTR_TestDataset.mat

说明:
  - 序列长度 T 必须与训练时 train_meta['seq_len'] 一致（与 Feat_All 形状一致）。
  - 指标与训练时 RadarPhysicsLoss 一致，并额外给出校正后位置 RMSE（相对 p_true）。
  - 可用 --save_p_fix 导出 P_fix = P_raw - Delta（形状 [N,T,3]），便于 MATLAB/Python 成像链路读取。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_script_dir = Path(__file__).resolve().parent
if (_script_dir / "xlstm").is_dir():
    _sd = str(_script_dir)
    while sys.path and Path(sys.path[0]).resolve() == _script_dir:
        sys.path.pop(0)
    if _sd not in sys.path:
        sys.path.append(_sd)

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_pi_xlstm import (
    CascadeDataNormalizer,
    CascadePhysicsLoss,
    DataNormalizer,
    PhasePhysicsLoss,
    PI_xLSTM_Cascade,
    PI_xLSTM_Tracker,
    RadarPhysicsLoss,
    SARDataset,
    resolve_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="PI-xLSTM 验证 / 测试")
    parser.add_argument("--checkpoint", type=str, required=True, help="latest.pt 或 checkpoint_epoch_*.pt")
    parser.add_argument("--mat_path", type=str, required=True, help="测试集 .mat（Feat_All, P_raw, P_true）")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="cuda / cpu / mps；默认自动选择",
    )
    parser.add_argument("--json_out", type=str, default="", help="可选，将指标写入该 JSON 路径")
    parser.add_argument(
        "--save_p_fix",
        type=str,
        default="",
        help="导出校正轨迹 P_fix [N,T,3]：路径以 .mat 结尾（需 scipy）或以 .npz 结尾",
    )
    parser.add_argument(
        "--also_save_delta",
        action="store_true",
        help="导出文件中同时写入 Delta_pred（网络输出，形状与 P_fix 相同）",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    mat_path = Path(args.mat_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")
    if not mat_path.is_file():
        raise FileNotFoundError(f"找不到数据: {mat_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    train_meta = ckpt.get("train_meta")
    if not isinstance(train_meta, dict):
        raise ValueError("checkpoint 中缺少 train_meta，无法用原配置重建模型。")

    expected_seq = int(train_meta["seq_len"])
    seq_stride   = int(train_meta.get("seq_stride", 1))
    from train_pi_xlstm import _h5_to_ntd

    with h5py.File(mat_path, "r") as f:
        feat_nt9 = _h5_to_ntd(np.array(f["Feat_All"]), feat_dims=(6, 9, 12), name="Feat_All")
        if feat_nt9.ndim != 3 or feat_nt9.shape[-1] not in (6, 9, 12):
            raise ValueError(f"Feat_All 应为 [N,T,6|9|12]，得到 {feat_nt9.shape}")
        _, seq_len_raw, input_dim_f = feat_nt9.shape
    seq_len_f = seq_len_raw // seq_stride
    if int(seq_len_f) != expected_seq:
        raise ValueError(
            f"测试集下采样后序列长度 T={seq_len_f} (原始 T={seq_len_raw}, stride={seq_stride})"
            f" 与训练 seq_len={expected_seq} 不一致，无法直接验证。"
        )

    if args.device.strip():
        device = torch.device(args.device.strip())
    else:
        device = resolve_device()

    model_arch = str(train_meta.get("model_arch", "xlstm")).lower()
    is_cascade = model_arch == "cascade"
    is_phase = model_arch == "phase"
    backend = train_meta.get("slstm_backend", "vanilla")
    if device.type != "cuda" and backend == "cuda":
        backend = "vanilla"

    _model_kw = dict(
        input_dim=int(train_meta.get("input_dim", input_dim_f)),
        hidden_dim=int(train_meta["hidden_dim"]),
        output_dim=3,
        seq_len=expected_seq,
        slstm_backend=backend,
        num_blocks=int(train_meta["num_blocks"]),
        num_heads=int(train_meta["num_heads"]),
        use_input_mask=bool(train_meta.get("use_input_mask", False)),
    )
    if is_cascade:
        model = PI_xLSTM_Cascade(**_model_kw)
    else:
        model = PI_xLSTM_Tracker(**_model_kw)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    norm_sd = ckpt.get("normalizer_state_dict")
    if not isinstance(norm_sd, dict):
        raise ValueError(
            "checkpoint 缺少 normalizer_state_dict；该 checkpoint 来自归一化前的训练流程，"
            "请用最新版本 train_pi_xlstm.py 重新训练后再做评估。"
        )
    if is_cascade:
        if "delta_c_mean" not in norm_sd:
            raise ValueError("cascade checkpoint 的 normalizer 缺少 delta_c/phase 统计量。")
        normalizer = CascadeDataNormalizer(
            feat_mean=norm_sd["feat_mean"].to(torch.float32),
            feat_std=norm_sd["feat_std"].to(torch.float32),
            err_mean=norm_sd["err_mean"].to(torch.float32),
            err_std=norm_sd["err_std"].to(torch.float32),
            delta_c_mean=norm_sd["delta_c_mean"].to(torch.float32),
            delta_c_std=norm_sd["delta_c_std"].to(torch.float32),
            phase_mean=norm_sd["phase_mean"].to(torch.float32),
            phase_std=norm_sd["phase_std"].to(torch.float32),
        ).to(device)
    else:
        normalizer = DataNormalizer(
            feat_mean=norm_sd["feat_mean"].to(torch.float32),
            feat_std=norm_sd["feat_std"].to(torch.float32),
            err_mean=norm_sd["err_mean"].to(torch.float32),
            err_std=norm_sd["err_std"].to(torch.float32),
        ).to(device)

    # 从 mat 读取特显点（新格式 [3,N]→[N,3]；旧格式 [1,3] 广播）
    with h5py.File(mat_path, "r") as _f:
        def _ep(key):
            arr = np.array(_f[key]).astype(np.float32)
            if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] > 3:
                return torch.from_numpy(arr.T)   # [N,3]
            return None   # 旧格式，退回固定值
        pos_A_data = _ep("Pos_A")
        pos_B_data = _ep("Pos_B")
        pos_C_data = _ep("Pos_C")
    # 固定值备用（旧格式或读取失败时）
    pos_A = torch.tensor([387.0, 25.0, 0.0], dtype=torch.float32)
    pos_B = torch.tensor([417.0, 56.0, 0.0], dtype=torch.float32)
    pos_C = torch.tensor([403.0, 78.0, 0.0], dtype=torch.float32)
    # 训练 meta 字段名兼容：新版用 lambda_rcm_max；旧版本用 lambda_rcm。
    fc_hz = float(train_meta.get("fc_hz", 9.5e9))
    dataset = SARDataset(str(mat_path), seq_stride=seq_stride)
    if is_cascade and not dataset.is_v4_legacy:
        raise ValueError("cascade checkpoint 需要旧版 v4 mat（含 Delta_coarse / Phase_sin/cos）")
    if is_phase and not dataset.is_v4_phase:
        raise ValueError("phase checkpoint 需要 v4_phase mat（含 Phase_rel / R_ref）")

    lambda_rcm_eval = float(train_meta.get("lambda_rcm_max", train_meta.get("lambda_rcm", 0.5)))
    if is_cascade:
        assert isinstance(normalizer, CascadeDataNormalizer)
        criterion = CascadePhysicsLoss(
            pos_A,
            pos_B,
            pos_C,
            normalizer=normalizer,
            fc_hz=fc_hz,
            lambda_smooth=float(train_meta["lambda_smooth"]),
            lambda_rcm=lambda_rcm_eval,
            lambda_c=float(train_meta.get("lambda_c_max", 1.0)),
            lambda_f=float(train_meta.get("lambda_f_max", 1.0)),
            lambda_phase=float(train_meta.get("lambda_phase_max", 1.0)),
            lambda_total=float(train_meta.get("lambda_total_max", 0.3)),
        ).to(device)
    elif is_phase:
        criterion = PhasePhysicsLoss(
            pos_A,
            pos_B,
            pos_C,
            normalizer=normalizer,
            fc_hz=fc_hz,
            lambda_smooth=float(train_meta["lambda_smooth"]),
            lambda_phase=float(train_meta.get("lambda_phase_max", 1.0)),
            lambda_mse=0.0,
        ).to(device)
    else:
        criterion = RadarPhysicsLoss(
            pos_A,
            pos_B,
            pos_C,
            normalizer=normalizer,
            lambda_smooth=float(train_meta["lambda_smooth"]),
            lambda_rcm=lambda_rcm_eval,
        ).to(device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    n_samples = len(dataset)
    save_fix = bool(args.save_p_fix.strip())
    out_fix_path = Path(args.save_p_fix).expanduser().resolve() if save_fix else None
    if save_fix:
        assert out_fix_path is not None
        suf = out_fix_path.suffix.lower()
        if suf not in (".mat", ".npz"):
            raise ValueError("--save_p_fix 路径须以 .mat 或 .npz 结尾")

    P_fix_all: Optional[np.ndarray] = (
        np.zeros((n_samples, expected_seq, 3), dtype=np.float32) if save_fix else None
    )
    Delta_all: Optional[np.ndarray] = (
        np.zeros((n_samples, expected_seq, 3), dtype=np.float32)
        if save_fix and args.also_save_delta and not is_cascade and not is_phase
        else None
    )
    Delta_c_all: Optional[np.ndarray] = (
        np.zeros((n_samples, expected_seq, 3), dtype=np.float32)
        if save_fix and args.also_save_delta and is_cascade
        else None
    )
    Delta_f_all: Optional[np.ndarray] = (
        np.zeros((n_samples, expected_seq, 3), dtype=np.float32)
        if save_fix and args.also_save_delta and is_cascade
        else None
    )

    sum_total = sum_mse_norm = sum_mse_phys = sum_smooth = sum_rcm = 0.0
    sum_mse_c = sum_mse_f = sum_phase = 0.0
    sum_sq_norm_pos = sum_sq_coarse = 0.0
    n_batches = 0
    n_timesteps = 0
    row_off = 0

    with torch.no_grad():
        for batch in loader:
            if is_cascade:
                feat, _feat_mask, p_raw, p_true, has_label, rcm_weight, pa, pb, pc, dc, ph_sin, ph_cos = batch
                phase_rel = r_ref = None
            elif is_phase:
                feat, _feat_mask, p_raw, p_true, has_label, rcm_weight, pa, pb, pc, phase_rel, r_ref = batch
                dc = ph_sin = ph_cos = None
            else:
                feat, _feat_mask, p_raw, p_true, has_label, rcm_weight, pa, pb, pc = batch
                dc = ph_sin = ph_cos = phase_rel = r_ref = None
            feat = feat.to(device)
            _feat_mask = _feat_mask.to(device)
            p_raw = p_raw.to(device)
            p_true = p_true.to(device)
            has_label = has_label.to(device)
            rcm_weight = rcm_weight.to(device)
            pa = pa.to(device)
            pb = pb.to(device)
            pc = pc.to(device)
            feat_norm = normalizer.norm_feat(feat) * _feat_mask
            if is_cascade:
                assert dc is not None and isinstance(normalizer, CascadeDataNormalizer)
                dc = dc.to(device)
                ph_sin = ph_sin.to(device) * rcm_weight.unsqueeze(1)
                ph_cos = ph_cos.to(device) * rcm_weight.unsqueeze(1)
                delta_fine = (p_raw - p_true) - dc
                dc_norm, df_norm = model(
                    feat_norm, ph_sin, ph_cos, p_raw, normalizer, _feat_mask, detach_coarse=False
                )
                dc_phys = normalizer.denorm_delta_c(dc_norm)
                df_phys = normalizer.denorm_err(df_norm)
                loss, l_mse_c, l_mse_f, l_rcm, l_phase, l_smooth, l_mse_total, l_mse_phys = criterion(
                    dc_norm, df_norm, p_raw, p_true, feat, dc, delta_fine,
                    ph_sin, ph_cos, has_label, rcm_weight, pa, pb, pc,
                )
                p_coarse = p_raw - dc_phys
                p_fixed = p_raw - dc_phys - df_phys
                l_mse_norm = l_mse_total
                sum_mse_c += l_mse_c.item()
                sum_mse_f += l_mse_f.item()
                sum_phase += l_phase.item()
                sum_sq_coarse += ((p_coarse - p_true) ** 2).sum(dim=-1).sum().item()
            elif is_phase:
                assert phase_rel is not None and r_ref is not None
                phase_rel = phase_rel.to(device) * rcm_weight.unsqueeze(1)
                r_ref = r_ref.to(device)
                delta_norm = model(feat_norm, _feat_mask if bool(train_meta.get("use_input_mask", False)) else None)
                delta = normalizer.denorm_err(delta_norm)
                loss, l_mse_norm, l_mse_phys, l_smooth, l_rcm, l_phase = criterion(
                    delta_norm, p_raw, p_true, phase_rel, r_ref,
                    has_label, rcm_weight, pa, pb, pc,
                )
                p_fixed = p_raw - delta
                sum_phase += l_phase.item()
            else:
                delta_norm = model(feat_norm)
                delta = normalizer.denorm_err(delta_norm)
                loss, l_mse_norm, l_mse_phys, l_smooth, l_rcm = criterion(
                    delta_norm, p_raw, p_true, feat, has_label, rcm_weight, pa, pb, pc,
                )
                p_fixed = p_raw - delta
            sq_norm = ((p_fixed - p_true) ** 2).sum(dim=-1)
            sum_sq_norm_pos += sq_norm.sum().item()
            n_timesteps += sq_norm.numel()
            sum_total += loss.item()
            sum_mse_norm += l_mse_norm.item()
            sum_mse_phys += l_mse_phys.item()
            sum_smooth += l_smooth.item()
            sum_rcm += l_rcm.item()
            n_batches += 1

            if save_fix and P_fix_all is not None:
                b = feat.size(0)
                P_fix_all[row_off : row_off + b] = p_fixed.detach().cpu().numpy().astype(np.float32)
                if Delta_all is not None:
                    Delta_all[row_off : row_off + b] = (p_raw - p_fixed).detach().cpu().numpy().astype(np.float32)
                if Delta_c_all is not None and Delta_f_all is not None:
                    Delta_c_all[row_off : row_off + b] = dc_phys.detach().cpu().numpy().astype(np.float32)
                    Delta_f_all[row_off : row_off + b] = df_phys.detach().cpu().numpy().astype(np.float32)
                row_off += b

    if save_fix:
        assert P_fix_all is not None and out_fix_path is not None
        if row_off != n_samples:
            raise RuntimeError(f"P_fix 行数不一致: 写入 {row_off}，期望 {n_samples}")

    avg = lambda s: s / max(n_batches, 1)
    pos_rmse = (sum_sq_norm_pos / max(n_timesteps, 1)) ** 0.5
    coarse_rmse = (sum_sq_coarse / max(n_timesteps, 1)) ** 0.5 if is_cascade else float("nan")

    report = {
        "checkpoint": str(ckpt_path),
        "mat_path": str(mat_path),
        "device": str(device),
        "model_arch": model_arch,
        "num_samples": len(dataset),
        "batch_size": args.batch_size,
        "seq_len": expected_seq,
        "loss_total_mean_batch": avg(sum_total),
        "loss_mse_norm_mean_batch": avg(sum_mse_norm),
        "loss_mse_phys_mean_batch": avg(sum_mse_phys),
        "loss_smooth_mean_batch": avg(sum_smooth),
        "loss_rcm_mean_batch": avg(sum_rcm),
        "loss_phase_mean_batch": avg(sum_phase) if is_phase else float("nan"),
        "lambda_rcm_used": lambda_rcm_eval,
        "lambda_phase_used": float(train_meta.get("lambda_phase_max", 1.0)) if is_phase else float("nan"),
        "position_rmse_after_correction": pos_rmse,
        "rmse_total": pos_rmse,
        "rmse_coarse": coarse_rmse,
        "p_fix_export": str(out_fix_path) if save_fix and out_fix_path else None,
        "note": "loss_* 为各 batch 标量 loss 再对 batch 取均值；mse_norm 是网络输出空间的损失，"
        "mse_phys 是反归一化回米制的位移误差；position_rmse = sqrt(mean_{b,t} ||p_fixed-p_true||^2)。"
        + (" P_fix=P_raw-Delta，形状 [N,T,3]，与输入 mat 中样本顺序一致（shuffle=False）。" if save_fix else ""),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if save_fix and out_fix_path is not None and P_fix_all is not None:
        out_fix_path.parent.mkdir(parents=True, exist_ok=True)
        if is_cascade and args.also_save_delta:
            print("cascade 模式：--also_save_delta 导出 Delta_coarse_pred 与 Delta_fine_pred")
        if out_fix_path.suffix.lower() == ".npz":
            np_payload: dict[str, np.ndarray] = {"P_fix": P_fix_all}
            if Delta_all is not None:
                np_payload["Delta_pred"] = Delta_all
            if Delta_c_all is not None:
                np_payload["Delta_coarse_pred"] = Delta_c_all
                np_payload["Delta_fine_pred"] = Delta_f_all
            np.savez_compressed(out_fix_path, **np_payload)
            print(f"已导出 NPZ: {out_fix_path} variables={list(np_payload.keys())}")
        else:
            try:
                from scipy.io import savemat
            except ImportError as exc:
                raise ImportError("导出 .mat 需要 scipy：pip install scipy") from exc
            mat_payload: dict[str, np.ndarray] = {"P_fix": P_fix_all}
            if Delta_all is not None:
                mat_payload["Delta_pred"] = Delta_all
            if Delta_c_all is not None:
                mat_payload["Delta_coarse_pred"] = Delta_c_all
                mat_payload["Delta_fine_pred"] = Delta_f_all
            savemat(str(out_fix_path), mat_payload, do_compression=True)
            print(f"已导出 .mat（MATLAB load）: {out_fix_path} 变量 {list(mat_payload.keys())}")

    if args.json_out.strip():
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
        print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
