"""
# YOLOxBench Dashboard
Interactive dashboard to compare YOLOxBench runs.
Launch with:

    yox ui --logdir <path>
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.image as mpimg
import plotly.express as px

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLOxBench Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Helpers ───────────────────────────────────────────────────────────────────
def find_runs(root: Path) -> list[Path]:
    """
    If `root/results.csv` exists → treat `root` as a single run.
    Else if `root/detect/*/results.csv` exist → collect those folders.
    """
    if (root / "results.csv").exists():
        return [root]
    detect_dir = root / "detect"
    if detect_dir.is_dir():
        return sorted(p.parent for p in detect_dir.glob("*/results.csv"))
    return []

def get_metric(row: pd.Series, key: str):
    """
    Return row[key], or first column ending/starting with key
    (handles Ultralytics suffixes like '(B)' or '_B').
    """
    if key in row:
        return row[key]
    for c in row.index:
        if c.endswith(key) or c.startswith(key):
            return row[c]
    return None

# ─── Gather run directories ────────────────────────────────────────────────────
logdir = Path(st.experimental_get_query_params().get("logdir", ["runs"])[0]).expanduser()

# Sidebar: custom override
custom_txt = st.sidebar.text_area(
    "Custom run dirs (one per line)", 
    help="Absolute paths, or relative to `logdir`",
    height=120,
).strip()

if custom_txt:
    runs = []
    for line in custom_txt.splitlines():
        p = Path(line.strip())
        if not p.is_absolute():
            p = logdir / line.strip()
        if (p / "results.csv").exists():
            runs.append(p)
    runs = sorted(runs)
else:
    runs = find_runs(logdir)

if not runs:
    st.sidebar.error(f"No runs found at `{logdir}` or under `{logdir}/detect`.")
    st.stop()

run_names = [r.name for r in runs]

# ─── Sidebar controls ───────────────────────────────────────────────────────────
st.sidebar.title("YOLOxBench 🔍")
selected = st.sidebar.multiselect(
    "Select runs to compare", run_names, default=[run_names[0]]
)
metric = st.sidebar.selectbox(
    "Primary metric", ["metrics/mAP50", "metrics/mAP50-95", "metrics/precision", "metrics/recall"]
)
x_metric = st.sidebar.selectbox(
    "3D‑X axis", ["metrics/precision", "metrics/recall", "metrics/mAP50", "metrics/mAP50-95"], index=2
)
y_metric = st.sidebar.selectbox(
    "3D‑Y axis", ["metrics/precision", "metrics/recall", "metrics/mAP50", "metrics/mAP50-95"], index=0
)

# ─── Tabs ───────────────────────────────────────────────────────────────────────
tab_overview, tab_train, tab_table, tab_3d = st.tabs(
    ["📊 Overview", "📈 Training Curves", "📋 Table & Downloads", "🕹️ 3D Scatter"]
)

# ─── TAB 1: Overview ────────────────────────────────────────────────────────────
with tab_overview:
    st.markdown("### Run‑by‑run Summary")
    cols = st.columns(len(selected) or 1)
    for col, name in zip(cols, selected):
        run_dir = runs[run_names.index(name)]
        col.subheader(name)
        csv = run_dir / "results.csv"
        if not csv.exists():
            col.error("No `results.csv`")
            continue

        df = pd.read_csv(csv)
        last = df.iloc[-1]
        val = get_metric(last, metric)
        col.markdown(f"**{metric.split('/')[-1]}** on final epoch")
        if pd.notna(val):
            col.metric(metric.split("/")[-1], f"{val:.3f}")
        else:
            col.info("Not available")

        col.markdown("Static plots from Ultralytics validation:")
        for fn, cap in [
            ("PR_curve.png", "PR Curve"),
            ("F1_curve.png", "F1 Curve"),
            ("confusion_matrix.png", "Confusion Matrix"),
            ("confusion_matrix_normalized.png", "Confusion Matrix (Normalized)"),
        ]:
            imgp = run_dir / fn
            if imgp.exists():
                col.image(mpimg.imread(imgp), caption=cap, use_container_width=True)

# ─── TAB 2: Training Curves ─────────────────────────────────────────────────────
with tab_train:
    st.markdown("### Per‑epoch Training & Online Metrics")
    for name in selected:
        run_dir = runs[run_names.index(name)]
        st.markdown(f"#### `{name}`")
        csv = run_dir / "results.csv"
        if not csv.exists():
            st.warning("No `results.csv`")
            continue

        df = pd.read_csv(csv)
        if "epoch" not in df.columns:
            st.info("No epoch‑wise data in CSV")
            continue

        st.markdown("**Loss curves**")
        loss_df = df[["epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss"]].set_index("epoch")
        loss_df = loss_df.rename(columns={
            "train/box_loss": "Box", "train/cls_loss": "Cls", "train/dfl_loss": "DFL"
        })
        st.line_chart(loss_df, use_container_width=True)

        st.markdown("**Online validation metrics**")
        online = [c for c in df.columns if c.startswith("metrics/")]
        if online:
            online_df = df[["epoch"] + online].set_index("epoch")
            online_df.columns = [c.split("/",1)[1].replace("(B)", "") for c in online_df.columns]
            st.line_chart(online_df, use_container_width=True)
        else:
            st.info("No `metrics/...` columns present")

# ─── TAB 3: Table & Downloads ───────────────────────────────────────────────────
with tab_table:
    st.markdown("### Numeric Comparison Table")
    rows = []
    for name in selected:
        run_dir = runs[run_names.index(name)]
        csv = run_dir / "results.csv"
        if not csv.exists():
            continue
        last = pd.read_csv(csv).iloc[-1]
        rows.append({
            "run": name,
            metric.split("/")[-1]: get_metric(last, metric)
        })
    if not rows:
        st.info("No numeric data available")
    else:
        table = pd.DataFrame(rows).set_index("run")
        st.dataframe(table, use_container_width=True)

        st.markdown("**Download your comparison**")
        csv_bytes = table.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv_bytes, "yoloxbench_compare.csv", "text/csv")
        md_bytes = table.to_markdown().encode("utf-8")
        st.download_button("Download Markdown", md_bytes, "yoloxbench_compare.md", "text/markdown")

# ─── TAB 4: 3D Scatter ───────────────────────────────────────────────────────────
with tab_3d:
    st.markdown("### 3D Interactive Scatter")
    points = []
    for name in selected:
        run_dir = runs[run_names.index(name)]
        csv = run_dir / "results.csv"
        if not csv.exists():
            continue
        last = pd.read_csv(csv).iloc[-1]
        x = get_metric(last, x_metric)
        y = get_metric(last, y_metric)
        z = get_metric(last, metric)
        if pd.notna(x) and pd.notna(y) and pd.notna(z):
            points.append({"run": name, "x": x, "y": y, "z": z})

    if not points:
        st.info("Not enough data for 3D plot")
    else:
        df3 = pd.DataFrame(points)
        fig = px.scatter_3d(
            df3, x="x", y="y", z="z", color="run",
            labels={
                "x": x_metric.split("/")[-1],
                "y": y_metric.split("/")[-1],
                "z": metric.split("/")[-1],
            },
            title="Metric vs Metric vs Metric"
        )
        fig.update_traces(marker=dict(size=6))
        st.plotly_chart(fig, use_container_width=True)

