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

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="YOLOxBench Dashboard", layout="wide")

# ─── Helpers ────────────────────────────────────────────────────────────────────
def list_run_dirs(root: Path) -> list[Path]:
    """
    Return all subdirectories under root/detect/ that contain a results.csv.
    """
    detect_dir = root / "detect"
    if not detect_dir.exists():
        return []
    runs = []
    for d in detect_dir.iterdir():
        if d.is_dir() and (d / "results.csv").exists():
            runs.append(d)
    return sorted(runs)

def get_metric(row: pd.Series, key: str):
    """
    Lookup row[key], or first column that endswith/startswith key
    (to handle Ultralytics suffixes like '(B)' or '_B').
    """
    if key in row:
        return row[key]
    for col in row.index:
        if col.endswith(key) or col.startswith(key):
            return row[col]
    return None

# ─── Sidebar: select logdir & (optional) custom runs ────────────────────────────
logdir = Path(st.query_params.get("logdir", ["runs"])[0]).expanduser()

custom_txt = st.sidebar.text_area(
    "Custom run dirs (one per line)\n"
    "(absolute paths or relative to logdir/detect)", 
    value=""
).strip()

# ─── Gather run directories ─────────────────────────────────────────────────────
if not custom_txt:
    run_dirs = list_run_dirs(logdir)
else:
    run_dirs = []
    for line in custom_txt.splitlines():
        p = Path(line.strip())
        if not p.is_absolute():
            p = logdir / "detect" / line.strip()
        if p.is_dir() and (p / "results.csv").exists():
            run_dirs.append(p)
    run_dirs = sorted(run_dirs)

if not run_dirs:
    st.warning(f"No runs with results.csv under {logdir/'detect'} and no valid custom entries.")
    st.stop()

# build name→Path map
run_map = {d.name: d for d in run_dirs}
run_names = list(run_map.keys())

# ─── Sidebar: pick which runs & which metric ────────────────────────────────────
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

# ─── Per‑run display ─────────────────────────────────────────────────────────────
cols = st.columns(len(selected) or 1)
rows = []

for col, run_name in zip(cols, selected):
    run_dir = run_map[run_name]
    csv_path = run_dir / "results.csv"

    df = pd.read_csv(csv_path)
    last = df.iloc[-1]
    value = get_metric(last, metric)

    # collect for bottom table
    rows.append({"run": run_name, metric: value})

    col.header(run_name)
    if pd.notna(value):
        col.metric(metric.split("/")[-1], f"{value:.3f}")
    else:
        col.info(f"{metric} not found")

    pr_curve = run_dir / "PR_curve.png"
    if pr_curve.exists():
        col.image(mpimg.imread(pr_curve), caption="PR Curve", use_container_width=True)

# ─── Tabular comparison ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Tabular comparison")

if not rows:
    st.info("No numeric metrics available for the current selection.")
else:
    table = pd.DataFrame(rows).set_index("run")
    st.dataframe(table)
    st.download_button(
        "Download CSV",
        table.to_csv().encode("utf-8"),
        file_name="yoxbench_comparison.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download Markdown",
        table.to_markdown(index=True).encode("utf-8"),
        file_name="yoxbench_comparison.md",
        mime="text/markdown",
    )