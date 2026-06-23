#!/usr/bin/env python3
"""生成 v4_phase 格式冒烟 mat（无 MATLAB 时用）。"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    out = ROOT / "dataset" / "PPTR_TrainDataset_v4_phase_10.mat"
    out.parent.mkdir(parents=True, exist_ok=True)
    N, Ny = 10, 2048
    fc = np.float32(9.5e9)
    c = 3e8
    prf = 500
    dt = 1.0 / prf

    Feat_All = np.zeros((N, Ny, 6), np.float32)
    P_raw = np.zeros((N, Ny, 3), np.float32)
    P_true = np.zeros((N, Ny, 3), np.float32)
    Phase_rel = np.zeros((N, Ny, 3), np.float32)
    R_ref = np.zeros((N, 3), np.float32)
    Pos_A = np.zeros((3, N), np.float32)
    Pos_B = np.zeros((3, N), np.float32)
    Pos_C = np.zeros((3, N), np.float32)

    for i in range(N):
        t = np.arange(Ny) * dt
        v = 8.0 + i * 0.1
        h = 80.0
        pt = np.stack(
            [
                np.zeros(Ny, np.float32),
                v * t,
                h * np.ones(Ny, np.float32),
            ],
            axis=1,
        )
        err = 0.02 * np.sin(2 * np.pi * 0.1 * t)[:, None] * np.array([1.0, 0.5, 0.2], np.float32)
        pr = pt + err.astype(np.float32)
        P_true[i] = pt
        P_raw[i] = pr
        V = np.zeros_like(pr)
        V[1:] = (pr[1:] - pr[:-1]) / dt
        V[0] = V[1]
        Feat_All[i, :, :3] = pr
        Feat_All[i, :, 3:6] = V

        targets = np.array(
            [
                [350.0 + i, 200.0, 5.0],
                [380.0 + i, 400.0, 8.0],
                [360.0 + i, 600.0, 3.0],
            ],
            dtype=np.float32,
        ).T
        Pos_A[:, i] = targets[:, 0]
        Pos_B[:, i] = targets[:, 1]
        Pos_C[:, i] = targets[:, 2]

        for j in range(3):
            R = np.linalg.norm(pt - targets[j], axis=1)
            r0 = float(R.min())
            R_ref[i, j] = r0
            phase = -4.0 * np.pi * fc * (R - r0) / c
            Phase_rel[i, :, j] = np.unwrap(phase).astype(np.float32)

    v4_meta = np.array(
        [(b"v4_phase", b"P_true", b"min slant", b"unwrap phase", 6)],
        dtype=[("version", "S16"), ("phase_trajectory", "S16"), ("R_ref_def", "S32"), ("phase_def", "S32"), ("feat_dim", "i4")],
    )

    with h5py.File(out, "w") as f:
        f.create_dataset("Feat_All", data=Feat_All)
        f.create_dataset("P_raw", data=P_raw)
        f.create_dataset("P_true", data=P_true)
        f.create_dataset("Phase_rel", data=Phase_rel)
        f.create_dataset("R_ref", data=R_ref)
        f.create_dataset("Pos_A", data=Pos_A)
        f.create_dataset("Pos_B", data=Pos_B)
        f.create_dataset("Pos_C", data=Pos_C)
        f.create_dataset("fc", data=fc)
        f.create_dataset("v4_meta", data=v4_meta)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
