"""Aggregate run folders or tester CSVs into comparison plots."""
from pathlib import Path
from typing import Iterable, Sequence
import pandas as pd
import matplotlib.pyplot as plt

from ..plotting.plots import save_bar
from ..utils import ensure_dir

DEFAULT_METRICS: Sequence[str] = (
    "metrics/mAP50", "metrics/mAP50-95", "metrics/precision",
)


def compare(run_dirs: Iterable[Path], out: Path = Path("cmp_out"),
            metrics: Sequence[str] = DEFAULT_METRICS,
            iou: float | None = None, conf: float | None = None):
    rows = []
    for p in run_dirs:
        p = Path(p)
        if p.is_dir() and (p / "results.csv").exists():
            df = pd.read_csv(p / "results.csv")
            df["run"] = p.name
            rows.append(df)
        elif p.suffix == ".csv":
            df = pd.read_csv(p)
            rows.append(df)

    if not rows:
        raise FileNotFoundError("No valid results.csv or tester CSV supplied")

    df_all = pd.concat(rows, ignore_index=True)

    if iou is not None and "iou" in df_all.columns:
        df_all = df_all[df_all["iou"] == iou]
    if conf is not None and "conf" in df_all.columns:
        df_all = df_all[df_all["conf"] == conf]

    out = Path(out)
    ensure_dir(out)
    save_bar(df_all, out, metrics)