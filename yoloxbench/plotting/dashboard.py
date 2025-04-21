"""
# YOLOxBench Dashboard
Interactive dashboard to compare YOLOxBench runs.
Launch with:

    yox ui --logdir runs
"""
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.image as mpimg

# Configure the page
st.set_page_config(page_title="YOLOxBench Dashboard", layout="wide")

# ------------------------------------------------------------------ Helpers

def list_run_dirs(root: Path) -> list[Path]:
    """
    Return every subdirectory under `root/detect/` that contains a results.csv.
    """
    detect_dir = root / "detect"
    if not detect_dir.exists():
        return []
    return sorted(p.parent for p in detect_dir.glob("*/results.csv"))

def get_metric(row: pd.Series, key: str):
    """
    Lookup `row[key]`, or first column that endswith/startswith key
    (to handle Ultralytics suffixes like '(B)' or '_B').
    """
    if key in row:
        return row[key]
    for colname in row.index:
        if colname.endswith(key) or colname.startswith(key):
            return row[colname]
    return None

# ------------------------------------------------------------------ UI Start

# 1) Read `--logdir` from the Typer wrapper
logdir = Path(st.query_params.get("logdir", ["runs"])[0]).expanduser()

# 2) Allow manual override of run folders
custom = st.sidebar.text_area(
    "Custom run dirs (one per line, absolute or relative to logdir/detect)",
    value=""
).strip()

# 3) Gather run directories
if not custom:
    run_dirs = list_run_dirs(logdir)
else:
    run_dirs = []
    for line in custom.splitlines():
        p = Path(line.strip())
        if not p.is_absolute():
            p = logdir / "detect" / line.strip()
        if p.is_dir() and (p / "results.csv").exists():
            run_dirs.append(p)
    run_dirs = sorted(run_dirs)

if not run_dirs:
    st.warning(f"No runs with results.csv found under {logdir/'detect'} or in custom list.")
    st.stop()

run_names = [d.name for d in run_dirs]

# ------------------------------------------------------------------ Sidebar controls

st.sidebar.title("YOLOxBench runs")
selected = st.sidebar.multiselect(
    "Select runs to compare",
    options=run_names,
    default=[run_names[0]],
)

metric = st.sidebar.selectbox(
    "Metric",
    ["metrics/mAP50", "metrics/mAP50-95", "metrics/precision", "metrics/recall"],
)

# ------------------------------------------------------------------ Main display

cols = st.columns(len(selected) or 1)
rows = []

for col, run_name in zip(cols, selected):
    run_dir = logdir / "detect" / run_name if not custom else next(d for d in run_dirs if d.name == run_name)
    csv_path = run_dir / "results.csv"

    if not csv_path.exists():
        col.warning("No results.csv")
        continue

    df = pd.read_csv(csv_path)
    last = df.iloc[-1]
    value = get_metric(last, metric)

    rows.append({"run": run_name, metric: value})
    col.header(run_name)
    if value is not None and not pd.isna(value):
        col.metric(metric.split("/")[-1], f"{value:.3f}")
    else:
        col.info(f"{metric} not found")

    pr_curve = run_dir / "PR_curve.png"
    if pr_curve.exists():
        img = mpimg.imread(pr_curve)
        col.image(img, caption="PR Curve", use_container_width=True)

# ------------------------------------------------------------------ Tabular comparison

st.markdown("---")
st.subheader("Tabular comparison")

if not rows:
    st.info("No runs produced numeric metrics for the current selection.")
    st.stop()

table = pd.DataFrame(rows)
if "run" not in table.columns:
    st.warning("Internal error: no 'run' column present in metric rows.")
    st.stop()

table = table.set_index("run")
if metric not in table.columns or table[metric].isna().all():
    st.info("No numeric metrics available for the current selection.")
else:
    st.dataframe(table)
