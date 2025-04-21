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

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="YOLOxBench Dashboard", layout="wide")


# ─── Helpers ───────────────────────────────────────────────────────────────────
def list_run_dirs(root: Path) -> list[Path]:
    """Return every subdirectory under `root/detect/` that contains a results.csv."""
    detect_dir = root / "detect"
    if not detect_dir.exists():
        return []
    return sorted(p.parent for p in detect_dir.glob("*/results.csv"))


def get_metric(row: pd.Series, key: str):
    """
    Lookup `row[key]`, or first column that endswith/startswith key
    (handles Ultralytics suffixes like '(B)' or '_B').
    """
    if key in row:
        return row[key]
    for colname in row.index:
        if colname.endswith(key) or colname.startswith(key):
            return row[colname]
    return None


# ─── UI: collect runs ──────────────────────────────────────────────────────────
logdir = Path(st.query_params.get("logdir", ["runs"])[0]).expanduser()

custom = st.sidebar.text_area(
    "Custom run dirs (one per line, absolute or relative to logdir/detect)",
    value="",
    height=120,
).strip()

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


# ─── Sidebar controls ───────────────────────────────────────────────────────────
st.sidebar.title("YOLOxBench runs")
selected = st.sidebar.multiselect(
    "Select runs to compare",
    options=run_names,
    default=[run_names[0]],
)

metric = st.sidebar.selectbox(
    "Scalar metric to display",
    ["metrics/mAP50", "metrics/mAP50-95", "metrics/precision", "metrics/recall"],
)


# ─── Main display: per‑run columns ───────────────────────────────────────────────
cols = st.columns(len(selected) or 1)
rows = []

for col, run_name in zip(cols, selected):
    run_dir = next(d for d in run_dirs if d.name == run_name)
    csv_path = run_dir / "results.csv"

    col.header(run_name)
    if not csv_path.exists():
        col.warning("No results.csv")
        continue

    # Load the full CSV
    df = pd.read_csv(csv_path)
    last = df.iloc[-1]
    value = get_metric(last, metric)
    rows.append({"run": run_name, metric: value})

    # Show the chosen scalar metric
    if pd.notna(value):
        col.metric(metric.split("/")[-1], f"{value:.3f}")
    else:
        col.info(f"{metric} not found")

    # Show training curves if available
    if "epoch" in df.columns and "train/box_loss" in df.columns:
        with col.expander("Training curves", expanded=False):
            # Losses
            loss_cols = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
            losses = df[loss_cols].rename(columns={
                "train/box_loss": "Box Loss",
                "train/cls_loss": "Class Loss",
                "train/dfl_loss": "DFL Loss",
            })
            st.line_chart(losses.set_index(df["epoch"]))

            # On‑line metrics
            metric_cols = [
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
            ]
            # only keep those that exist
            metric_cols = [c for c in metric_cols if c in df.columns]
            if metric_cols:
                metrics_trend = df[metric_cols].rename(columns=lambda c: c.split("/")[1].replace("(B)", ""))
                st.line_chart(metrics_trend.set_index(df["epoch"]))

    # Show all four static plots
    for fname, caption in [
        ("PR_curve.png", "PR Curve"),
        ("F1_curve.png", "F1 Curve"),
        ("confusion_matrix.png", "Confusion Matrix"),
        ("confusion_matrix_normalized.png", "Confusion Matrix (Normalized)"),
    ]:
        img_path = run_dir / fname
        if img_path.exists():
            img = mpimg.imread(img_path)
            col.image(img, caption=caption, use_container_width=True)


# ─── Tabular comparison + CSV download ──────────────────────────────────────────
st.markdown("---")
st.subheader("Tabular comparison")

if not rows:
    st.info("No runs produced numeric metrics for the current selection.")
    st.stop()

table = pd.DataFrame(rows).set_index("run")

if metric not in table.columns or table[metric].isna().all():
    st.info("No numeric metrics available for the current selection.")
else:
    st.dataframe(table)

    # Download CSV
    csv_bytes = table.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download comparison CSV",
        data=csv_bytes,
        file_name="yoloxbench_comparison.csv",
        mime="text/csv",
    )
    #download markdown
    md_bytes = table.to_markdown(index=True).encode("utf-8")
    st.download_button(
        "📥 Download comparison Markdown",
        data=md_bytes,
        file_name="yoloxbench_comparison.md",
        mime="text/markdown",
    )
    
