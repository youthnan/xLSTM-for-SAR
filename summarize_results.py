"""
summarize_results.py — 自动汇总所有 run/ 目录的实验结果

用法：
    python summarize_results.py [--run_root ./run] [--top_n 5]

输出：
    1. 控制台打印 Markdown 格式对比表（可直接粘贴进论文）
    2. 保存 results_summary.csv 到 run_root 目录
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# 读取单个 run 目录的最佳指标
# ──────────────────────────────────────────────────────────────────────────────

def load_best_metrics(run_dir: Path) -> dict[str, Any] | None:
    """读取 run 目录下的 metrics.jsonl，返回 val_pos_rmse 最低 epoch 的指标。"""
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None

    rows: list[dict] = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not rows:
        return None

    best = min(rows, key=lambda r: r.get("val_pos_rmse", float("inf")))

    # 读取 train_meta（模型名、超参数等）
    meta_path = run_dir / "train_meta.json"
    meta: dict = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    # 读取 run_name（从目录名中解析）
    run_name = run_dir.name
    # 目录名格式：YYYYMMDD_HHMMSS_xxxxxx_<run_name>_pid<pid>
    parts = run_name.split("_")
    display_name = "_".join(parts[3:-1]) if len(parts) > 4 else run_name

    # 计算误差下降率：需要知道 P_raw 的 RMSE（近似用 val_pos_rmse epoch1 估算）
    first_rmse = rows[0].get("val_pos_rmse", None) if rows else None

    return {
        "run_name":        display_name,
        "run_dir":         str(run_dir),
        "model_arch":      meta.get("model_arch", "xlstm"),
        "best_epoch":      best.get("epoch", "?"),
        "total_epochs":    len(rows),
        "best_val_rmse_m": best.get("val_pos_rmse", float("nan")),
        "best_val_rmse_mm":best.get("val_pos_rmse", float("nan")) * 1000,
        "best_val_loss":   best.get("val_total_loss", float("nan")),
        "best_mse_norm":   best.get("val_mse_norm", float("nan")),
        "lambda_rcm_max":  meta.get("lambda_rcm_max", "?"),
        "lambda_smooth":   meta.get("lambda_smooth", "?"),
        "random_ref_mask": meta.get("random_ref_mask", "?"),
        "hidden_dim":      meta.get("hidden_dim", "?"),
        "num_blocks":      meta.get("num_blocks", "?"),
        "seq_stride":      meta.get("seq_stride", 1),
        "first_val_rmse_mm": first_rmse * 1000 if first_rmse else float("nan"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 格式化 Markdown 表格
# ──────────────────────────────────────────────────────────────────────────────

def fmt(v: Any, decimals: int = 2) -> str:
    if isinstance(v, float):
        if v != v:   # nan
            return "—"
        return f"{v:.{decimals}f}"
    return str(v)


def print_markdown_table(rows: list[dict]) -> None:
    headers = [
        "Run Name", "Model", "Best Epoch / Total",
        "Val RMSE (mm)", "Error Reduction (%)",
        "λ_rcm", "λ_smooth", "Ref Mask",
    ]
    print("\n## Experiment Results Summary\n")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    # 计算基准（P_raw 的 RMSE 近似用第一个 epoch 的 val_pos_rmse，或用未校正 baseline）
    # 这里用各行的 first_val_rmse_mm 作为 P_raw 参考
    for r in rows:
        first_rmse = r.get("first_val_rmse_mm", float("nan"))
        best_rmse  = r.get("best_val_rmse_mm", float("nan"))
        if first_rmse == first_rmse and first_rmse > 0:
            reduction = (1.0 - best_rmse / first_rmse) * 100
        else:
            reduction = float("nan")

        cols = [
            r["run_name"],
            r["model_arch"].upper(),
            f"{r['best_epoch']} / {r['total_epochs']}",
            fmt(best_rmse, 2),
            fmt(reduction, 1),
            fmt(r["lambda_rcm_max"]),
            fmt(r["lambda_smooth"]),
            str(r["random_ref_mask"]),
        ]
        print("| " + " | ".join(cols) + " |")


# ──────────────────────────────────────────────────────────────────────────────
# 保存 CSV
# ──────────────────────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ 已保存: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 run/ 目录下所有实验结果")
    parser.add_argument("--run_root", type=str, default="./run",
                        help="run 目录根路径（默认 ./run）")
    parser.add_argument("--top_n", type=int, default=0,
                        help="只展示 val_pos_rmse 最低的前 N 个（0=全部）")
    parser.add_argument("--filter", type=str, default="",
                        help="按 run_name 关键词过滤（空=全部）")
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    if not run_root.exists():
        print(f"目录不存在: {run_root}")
        return

    results: list[dict] = []
    for entry in sorted(run_root.iterdir()):
        if not entry.is_dir():
            continue
        if args.filter and args.filter not in entry.name:
            continue
        m = load_best_metrics(entry)
        if m is not None:
            results.append(m)

    if not results:
        print("未找到有效的实验目录（需包含 metrics.jsonl）。")
        return

    # 按 best_val_rmse_mm 升序排列
    results.sort(key=lambda r: r.get("best_val_rmse_mm", float("inf")))

    if args.top_n > 0:
        results = results[:args.top_n]

    print_markdown_table(results)
    save_csv(results, run_root / "results_summary.csv")

    # 额外打印 LaTeX 表格代码（方便直接贴进论文）
    print("\n## LaTeX Table (for paper)\n")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Comparison of trajectory refinement methods}")
    print(r"\label{tab:results}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"Method & Best Epoch & Val RMSE (mm) & $\lambda_\text{rcm}$ & Ref Mask \\")
    print(r"\hline")
    for r in results:
        best_rmse = r.get("best_val_rmse_mm", float("nan"))
        arch = r["model_arch"].upper()
        if arch == "XLSTM":
            arch = r"PI-xLSTM (Ours)"
        print(
            f"{arch} & {r['best_epoch']} & {fmt(best_rmse, 2)} & "
            f"{fmt(r['lambda_rcm_max'])} & {r['random_ref_mask']} \\\\"
        )
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
