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
import plotly.express as px

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="YOLOxBench Dashboard", layout="wide", initial_sidebar_state="expanded")


# ─── Helpers ───────────────────────────────────────────────────────────────────
def list_run_dirs(root: Path) -> list[Path]:
    detect = root / "detect"
    if not detect.exists():
        return []
    return sorted(p.parent for p in detect.glob("*/results.csv"))


def get_metric(row: pd.Series, key: str):
    if key in row:
        return row[key]
    for c in row.index:
        if c.endswith(key) or c.startswith(key):
            return row[c]
    return None


# ─── Collect runs ───────────────────────────────────────────────────────────────
logdir = Path(st.experimental_get_query_params().get("logdir", ["runs"])[0]).expanduser()

custom = st.sidebar.text_area(
    "Custom run dirs (one per line, abs or relative to logdir/detect)",
    height=120
).strip()

if custom:
    run_dirs = []
    for line in custom.splitlines():
        p = Path(line.strip())
        if not p.is_absolute():
            p = logdir / "detect" / p
        if p.is_dir() and (p / "results.csv").exists():
            run_dirs.append(p)
    run_dirs = sorted(run_dirs)
else:
    run_dirs = list_run_dirs(logdir)

if not run_dirs:
    st.sidebar.error("No runs found under `runs/detect/` or in custom list.")
    st.stop()

run_names = [d.name for d in run_dirs]


# ─── Sidebar controls ───────────────────────────────────────────────────────────
st.sidebar.title("YOLOxBench 🚀")
selected = st.sidebar.multiselect("Pick runs to compare", run_names, default=[run_names[0]])
metric = st.sidebar.selectbox("Scalar metric", ["metrics/mAP50", "metrics/mAP50-95", "metrics/precision", "metrics/recall"])
x_metric = st.sidebar.selectbox("X‑axis (3D)", ["metrics/mAP50", "metrics/mAP50-95", "metrics/precision", "metrics/recall"], index=1)
y_metric = st.sidebar.selectbox("Y‑axis (3D)", ["metrics/precision", "metrics/recall", "metrics/mAP50-95", "metrics/mAP50"], index=2)

# ─── Tabs ───────────────────────────────────────────────────────────────────────
tab_overview, tab_train, tab_table, tab_3d = st.tabs(["📊 Overview", "📈 Training", "📋 Table", "🕹️ 3D Comparison"])


# ─── TAB 1: Overview ────────────────────────────────────────────────────────────
with tab_overview:
    cols = st.columns(len(selected) or 1)
    for col, name in zip(cols, selected):
        d = run_dirs[run_names.index(name)]
        csv = d / "results.csv"
        col.subheader(name)
        if not csv.exists():
            col.error("No results.csv")
            continue

        df = pd.read_csv(csv)
        last = df.iloc[-1]
        val = get_metric(last, metric)
        if pd.notna(val):
            col.metric(metric.split("/")[-1], f"{val:.3f}")
        else:
            col.info("–")

        # static plots
        for fn, cap in [
            ("PR_curve.png", "PR"), ("F1_curve.png", "F1"),
            ("confusion_matrix.png", "CM"), ("confusion_matrix_normalized.png", "CM norm")
        ]:
            p = d / fn
            if p.exists():
                col.image(mpimg.imread(p), caption=cap, use_container_width=True)


# ─── TAB 2: Training curves ─────────────────────────────────────────────────────
with tab_train:
    for name in selected:
        d = run_dirs[run_names.index(name)]
        csv = d / "results.csv"
        st.markdown(f"### `{name}`")
        if not csv.exists():
            st.warning("No results.csv")
            continue
        df = pd.read_csv(csv)

        if "epoch" not in df:
            st.info("No per‑epoch data in results.csv")
            continue

        losses = df[["epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss"]].set_index("epoch")
        st.line_chart(losses.rename(columns={
            "train/box_loss": "Box", "train/cls_loss": "Cls", "train/dfl_loss": "DFL"
        }))

        online = [c for c in df.columns if c.startswith("metrics/")]
        if online:
            online_df = df[["epoch"] + online].set_index("epoch").rename(columns=lambda c: c.split("/",1)[1].replace("(B)",""))
            st.line_chart(online_df)


# ─── TAB 3: Table + CSV/MD download ───────────────────────────────────────────────
with tab_table:
    rows = []
    for name in selected:
        d = run_dirs[run_names.index(name)]
        csv = d / "results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        last = df.iloc[-1]
        val = get_metric(last, metric)
        rows.append({"run": name, metric.split("/")[-1]: val})
    table = pd.DataFrame(rows).set_index("run")

    st.dataframe(table, use_container_width=True)

    if not table.empty:
        csv_bytes = table.reset_index().to_csv(index=False).encode()
        st.download_button("Download CSV", csv_bytes, "compare.csv", "text/csv")
        md_bytes = table.to_markdown().encode()
        st.download_button("Download MD", md_bytes, "compare.md", "text/markdown")


# ─── TAB 4: 3D interactive scatter ───────────────────────────────────────────────
with tab_3d:
    # build a little DataFrame of final values
    data_3d = []
    for name in selected:
        d = run_dirs[run_names.index(name)]
        csv = d / "results.csv"
        if not csv.exists():
            continue
        last = pd.read_csv(csv).iloc[-1]
        data_3d.append({
            "run": name,
            "x": get_metric(last, x_metric),
            "y": get_metric(last, y_metric),
            "z": get_metric(last, metric),
        })
    df3 = pd.DataFrame(data_3d).dropna()
    if df3.empty:
        st.info("No numeric metrics to plot in 3D.")
    else:
        fig = px.scatter_3d(
            df3, x="x", y="y", z="z", color="run",
            labels={"x": x_metric.split("/")[-1], "y": y_metric.split("/")[-1], "z": metric.split("/")[-1]},
            title="3D metric comparison"
        )
        fig.update_traces(marker_size=6)
        st.plotly_chart(fig, use_container_width=True)
