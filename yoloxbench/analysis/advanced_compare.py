"""
advanced_compare.py — best‑in‑class metric comparison for YOLOxBench
-------------------------------------------------------------------
Reads every run folder under runs/detect/, gracefully handles:
  • training runs  …/results.csv  → full mAP / P / R metrics
  • val‑only runs  (no csv)      → inserts NaN but still plots curves

Outputs
  - cmp_runs/compare_<timestamp>.csv   (flat metrics table)
  - cmp_runs/compare_<timestamp>/      (bar charts, PR curves, CM grids)
  - cmp_runs/report_<timestamp>.md     (rich Markdown report)

CLI hook:  `yox compare --logdir runs` (wired in cli.py)
"""
from __future__ import annotations
import json, shutil, time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from rich.progress import track

METRICS = [
    "metrics/mAP50", "metrics/mAP50-95", "metrics/precision", "metrics/recall"
]


def _find_runs(root: Path):
    return sorted((root / "detect").iterdir())


def _read_metrics(run: Path):
    csv = run / "results.csv"
    if not csv.exists():
        return {m: np.nan for m in METRICS}  # val‑only folder
    df = pd.read_csv(csv).iloc[-1]
    out = {}
    for m in METRICS:
        # tolerate new suffixes “(B)”
        col = next((c for c in df.index if c.endswith(m) or c == m), None)
        out[m] = float(df[col]) if col else np.nan
    return out


def _bar(df: pd.DataFrame, metric: str, out: Path):
    plt.figure(figsize=(10, 4))
    df.sort_values(metric, ascending=False).plot.bar(x="run", y=metric, legend=False)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def compare(logdir: Path = Path("runs")) -> Path:
    runs = _find_runs(logdir)
    rows = []
    for run in track(runs, description="Collecting metrics"):
        r = {"run": run.name}
        r.update(_read_metrics(run))
        rows.append(r)
    df = pd.DataFrame(rows)

    out_dir = Path("cmp_runs")
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"compare_{stamp}.csv"
    df.to_csv(csv_path, index=False)

    # leaderboards
    plot_dir = out_dir / f"compare_{stamp}"
    plot_dir.mkdir(exist_ok=True)
    for m in METRICS:
        _bar(df, m, plot_dir / f"bar_{m.replace('/', '_')}.png")

    # link PR curves & CMs
    for run in runs:
        for img in ("PR_curve.png", "F1_curve.png", "confusion_matrix.png"):
            src = run / img
            if src.exists():
                shutil.copy(src, plot_dir / f"{run.name}_{img}")

    # markdown summary
    md = [f"# 📋 YOLOxBench comparison report ({stamp})", "\n| run | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS)+1)]
    for r in rows:
        md.append("| " + r["run"] + " | " + " | ".join(f"{r[m]:.3f}" if not np.isnan(r[m]) else "—" for m in METRICS) + " |")
    (out_dir / f"report_{stamp}.md").write_text("\n".join(md), encoding="utf-8")
    return csv_path

if __name__ == "__main__":
    compare()
