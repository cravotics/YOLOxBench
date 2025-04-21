"""Aggregate run dirs or tester CSVs into bar plots."""
from pathlib import Path
from typing import Iterable, Sequence
import pandas as pd

from ..plotting.plots import save_bar
from ..utils import ensure_dir

DEFAULT_METRICS: Sequence[str] = (
    "metrics/mAP50", "metrics/mAP50-95", "metrics/precision",
)

def compare(run_dirs: Iterable[Path], out: Path = Path("cmp_out"), *,
            metrics: Sequence[str] = DEFAULT_METRICS,
            iou: float | None = None, conf: float | None = None):
    rows = []
    for p in map(Path, run_dirs):
        if p.is_dir() and (p/"results.csv").exists():
            df = pd.read_csv(p/"results.csv"); df["run"] = p.name; rows.append(df)
        elif p.suffix == ".csv":
            df = pd.read_csv(p); rows.append(df)
    if not rows:
        raise FileNotFoundError("No results found in given paths")
    df = pd.concat(rows, ignore_index=True)
    if iou is not None and "iou" in df.columns: df = df[df.iou == iou]
    if conf is not None and "conf" in df.columns: df = df[df.conf == conf]
    out = Path(out); ensure_dir(out)
    save_bar(df, out, metrics)