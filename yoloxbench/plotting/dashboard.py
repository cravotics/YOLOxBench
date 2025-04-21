"""
Interactive dashboard to compare YOLOxBench runs.
Launch with:

    yox ui --logdir runs
"""
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.image as mpimg

st.set_page_config(page_title="YOLOxBench Dashboard", layout="wide")

# ------------------------------------------------------------------ helpers
def list_run_dirs(root: Path) -> list[Path]:
    """Return every directory under detect/* that contains a results.csv."""
    return sorted(p.parent for p in (root / "detect").glob("*/results.csv"))

def get_metric(row: pd.Series, key: str):
    """Return row[key] or the first column that ends‑with key (handles '(B)')."""
    if key in row:
        return row[key]
    for colname in row.index:
        if colname.endswith(key):
            return row[colname]
    return None
# ------------------------------------------------------------------ UI

logdir = Path(st.query_params.get("logdir", ["runs"])[0]).expanduser()

run_dirs = list_run_dirs(logdir)
if not run_dirs:
    st.warning(f"No runs with results.csv found under {logdir/'detect'}")
    st.stop()

run_names = [d.name for d in run_dirs]

st.sidebar.title("YOLOxBench runs")
selected = st.sidebar.multiselect(
    "Select runs to compare", options=run_names, default=[run_names[0]]
)

metric = st.sidebar.selectbox(
    "Metric", ["metrics/mAP50", "metrics/mAP50-95", "metrics/precision", "metrics/recall"]
)

cols = st.columns(len(selected) or 1)
rows = []

for col, run_name in zip(cols, selected):
    run_dir = logdir / "detect" / run_name
    csv = run_dir / "results.csv"
    if not csv.exists():
        col.warning("No results.csv")
        continue

    df = pd.read_csv(csv)
    last = df.iloc[-1]
    value = get_metric(last, metric)

    rows.append({"run": run_name, metric: value})
    col.header(run_name)
    if value is not None:
        col.metric(metric.split("/")[-1], f"{value:.3f}")
    else:
        col.info(f"{metric} not found")

    pr_curve = run_dir / "PR_curve.png"
    if pr_curve.exists():
        col.image(mpimg.imread(pr_curve), caption="PR Curve", use_column_width=True)

# ---------------- table ----------------
st.markdown("---")
st.subheader("Tabular comparison")

if not rows or pd.isna(pd.DataFrame(rows)[metric]).all():
    st.info("No numeric metrics available for the current selection.")
else:
    st.dataframe(pd.DataFrame(rows).set_index("run"))
