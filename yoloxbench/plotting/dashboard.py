"""Interactive dashboard to compare YOLOxBench runs.
Run with:  `yox ui --logdir runs`  (Typer already calls this script).
"""
import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.image as mpimg

st.set_page_config(page_title="YOLOxBench Dashboard", layout="wide")

logdir = Path(st.experimental_get_query_params().get("logdir", ["runs"])[0]).expanduser()

st.sidebar.title("YOLOxBench Runs")
runs = sorted(p for p in logdir.glob("detect/*/results.csv"))
if not runs:
    st.warning(f"No results.csv under {logdir}/detect.")
    st.stop()

selected = st.sidebar.multiselect(
    "Select runs to compare", options=[r.parent.name for r in runs], default=[runs[0].parent.name]
)

metric = st.sidebar.selectbox(
    "Metric", ["metrics/mAP50", "metrics/mAP50-95", "metrics/precision", "metrics/recall"]
)

cols = st.columns(len(selected))
rows = []
for col, run_name in zip(cols, selected):
    csv = logdir / "detect" / run_name / "results.csv"
    df = pd.read_csv(csv)
    last = df.iloc[-1]
    rows.append({"run": run_name, metric: last[metric]})
    col.header(run_name)
    col.metric(metric.split("/")[-1], f"{last[metric]:.3f}")
    img_path = logdir / "detect" / run_name / "PR_curve.png"
    if img_path.exists():
        col.image(mpimg.imread(img_path), caption="PR Curve", use_column_width=True)

st.markdown("---")
st.subheader("Tabular comparison")
st.dataframe(pd.DataFrame(rows).set_index("run"))

st.markdown(
    "***NOTE:*** Use this dashboard to *qualitatively* inspect PR/F1 curves and metric deltas between models after training/validation."
)