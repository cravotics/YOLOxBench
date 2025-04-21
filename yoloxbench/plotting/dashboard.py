"""
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

# Read `--logdir` from the Typer wrapper
logdir = Path(st.query_params.get("logdir", ["runs"])[0]).expanduser()

# Gather all runs with a CSV
run_dirs = list_run_dirs(logdir)
if not run_dirs:
    st.warning(f"No runs with results.csv found under {logdir/'detect'}")
    st.stop()

run_names = [d.name for d in run_dirs]

# Sidebar controls
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

# Create one Streamlit column per selected run
cols = st.columns(len(selected) or 1)
rows = []

for col, run_name in zip(cols, selected):
    run_dir = logdir / "detect" / run_name
    csv_path = run_dir / "results.csv"

    # If there is no CSV, warn and continue
    if not csv_path.exists():
        col.warning("No results.csv")
        continue

    # Load the last row of metrics
    df = pd.read_csv(csv_path)
    last = df.iloc[-1]
    value = get_metric(last, metric)

    # Prepare table row and display metric
    rows.append({"run": run_name, metric: value})
    col.header(run_name)
    if value is not None and not pd.isna(value):
        col.metric(metric.split("/")[-1], f"{value:.3f}")
    else:
        col.info(f"{metric} not found")

    # Show the PR curve if available
    pr_curve = run_dir / "PR_curve.png"
    if pr_curve.exists():
        img = mpimg.imread(pr_curve)
        col.image(img, caption="PR Curve", use_container_width=True)

# ------------------------------------------------------------------ Tabular comparison

st.markdown("---")
st.subheader("Tabular comparison")

table = pd.DataFrame(rows).set_index("run")
if table.empty or table[metric].isna().all():
    st.info("No numeric metrics available for the current selection.")
else:
    st.dataframe(table)
