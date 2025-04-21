# yoloxbench/cli.py
"""
YOLOxBench command‑line interface
---------------------------------
train   – fine‑tune or resume a model
val     – evaluate one model on one dataset
test    – compare N models across M datasets (collect metrics)
cmp     – plot existing run folders
ui      – launch Streamlit dashboard
compare – advanced CSV comparison across all runs
report  – generate Markdown report from comparison CSV
video   – run inference on video or launch PyQt viewer
"""

from pathlib import Path
from typing import List, Optional

import typer
from rich import print, box
from rich.console import Console
from rich.table import Table

from .cfg import YoxConfig
from .engine import trainer, validator, tester, comparator
from .engine.video import annotate_and_save
from .gui import run_gui
from .analysis.advanced_compare import compare as adv_compare
from .reporting.markdown_report import make_markdown

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    help="YOLOxBench – unified train / validate / compare toolkit for YOLO",
)

console = Console()


# ---------- root banner ----------
@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        banner = (
            "[bold cyan]\nYOLOxBench[/] – what would you like to do?\n"
            "  • [bold]train[/]   Fine‑tune or resume a model\n"
            "  • [bold]val[/]     Validate one model\n"
            "  • [bold]test[/]    Compare multiple models × datasets\n"
            "  • [bold]cmp[/]     Plot existing run directories\n"
            "  • [bold]ui[/]      Launch Streamlit dashboard\n"
            "  • [bold]compare[/] Advanced CSV comparison\n"
            "  • [bold]report[/] Generate Markdown report\n"
            "  • [bold]video[/]   Run inference on video\n"
        )
        print(banner)
        print("Try: [italic]yox train --help[/] or [italic]yox test --help[/]\n")


# ---------- TRAIN ----------
@app.command()
def train(
    cfg: Optional[Path] = typer.Option(None, help="YAML config file (overrides CLI flags)"),
    model: str = typer.Option(..., help=".pt or model alias"),
    data: str = typer.Option(..., help="dataset YAML"),
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    workers: int = 4,
    device: str = "0",
    name: str = "run",
):
    """Fine‑tune a model."""
    cfg_obj = YoxConfig.load(
        cfg,
        mode="train",
        model=model,
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        extra={"workers": workers, "name": name},
    )
    _print_cfg(cfg_obj)
    run_dir = trainer.train(cfg_obj)
    print(f"[bold green]✓[/] Training complete – results in {run_dir}")


# ---------- VALIDATE ----------
@app.command()
def val(
    cfg: Optional[Path] = typer.Option(None),
    model: str = typer.Option(...),
    data: str = typer.Option(...),
    iou: float = 0.5,
    conf: float = 0.25,
    imgsz: int = 640,
    batch: int = 16,
):
    """Validate one model on one dataset."""
    cfg_obj = YoxConfig.load(
        cfg,
        mode="val",
        model=model,
        data=data,
        iou=iou,
        conf=conf,
        imgsz=imgsz,
        batch=batch,
    )
    metrics = validator.validate(cfg_obj)
    _print_metrics(metrics)


# ---------- TEST ----------
@app.command()
def test(
    models: List[Path] = typer.Argument(
        ..., exists=True, readable=True, dir_okay=False, help="One or more .pt checkpoints"
    ),
    datasets: List[Path] = typer.Option(..., help="One or more dataset YAMLs"),
    iou: float = 0.5,
    conf: float = 0.25,
    imgsz: int = 640,
    batch: int = 16,
    cfg: Optional[Path] = None,
):
    """Run every model against every dataset and write a CSV of metrics."""
    cfg_obj = YoxConfig.load(
        cfg,
        mode="test",
        models=[str(p) for p in models],
        datasets=[str(p) for p in datasets],
        iou=iou,
        conf=conf,
        imgsz=imgsz,
        batch=batch,
    )
    csv_path = tester.test(cfg_obj)
    print(f"[bold green]✓[/] Results saved to {csv_path}")


# ---------- CMP ----------
@app.command()
def cmp(
    run_dirs: List[Path] = typer.Argument(..., help="Run directories to compare"),
    out: Path = typer.Option(Path("cmp_out"), help="Where to save comparison plots"),
):
    """Create comparison bar‑plots from Ultralytics run folders."""
    comparator.compare(run_dirs, out)
    print(f"[bold green]✓[/] Comparison plots saved to {out}")


# ---------- UI ----------
@app.command()
def ui(logdir: Path = Path("runs")):
    """Launch Streamlit dashboard in a separate process."""
    import subprocess, sys

    dash = Path(__file__).parent / "plotting" / "dashboard.py"
    if not dash.exists():
        typer.echo(f"Dashboard script not found at {dash}", err=True)
        raise typer.Exit(1)

    cmd = [sys.executable, "-m", "streamlit", "run", str(dash), "--", f"--logdir={logdir}"]
    subprocess.run(cmd)


# ---------- COMPARE (advanced) ----------
@app.command()
def compare(logdir: Path = Path("runs")):
    """Advanced comparison across *all* runs under logdir/detect/."""
    csv = adv_compare(logdir)
    print(f"[bold green]✓[/] Advanced comparison written to {csv}")


# ---------- REPORT ----------
@app.command()
def report(
    csv: Path = typer.Argument(..., help="CSV produced by `yox test`"),
    runs: List[str] = typer.Argument(..., help="Run names in same order as CSV rows"),
):
    """Generate a Markdown report from a YOLOxBench CSV."""
    rep = make_markdown(csv, runs)
    print(f"[bold green]✓[/] Report written to {rep}")


# ---------- VIDEO ----------
@app.command()
def video(
    model: Path = typer.Option(..., exists=True, readable=True, help="Path to .pt checkpoint"),
    source: Path = typer.Option(..., exists=True, readable=True, help="Video file to run inference on"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Where to save annotated video"),
    conf: float = typer.Option(0.25, help="Confidence threshold"),
    iou: float = typer.Option(0.5, help="IOU threshold"),
    gui: bool = typer.Option(False, help="Launch interactive PyQt GUI instead of saving"),
):
    """
    Run YOLO inference on a video file (with a Rich progress bar),
    or open an interactive PyQt viewer if --gui is passed.
    """
    model_str, source_str = str(model), str(source)

    if gui:
        run_gui(model_str)
    else:
        out_path = output or (source.parent / f"{source.stem}_annotated.mp4")
        video_out = annotate_and_save(model_str, source_str, str(out_path), conf, iou)
        print(f"[bold green]✓[/] Annotated video saved to {video_out}")


# ---------- Helpers ----------
def _print_cfg(cfg: YoxConfig):
    table = Table(title="Resolved Training Config", box=box.SIMPLE)
    for k, v in cfg.__dict__.items():
        table.add_row(str(k), str(v))
    console.print(table)


def _print_metrics(metrics: dict):
    table = Table(title="Validation metrics", box=box.SIMPLE)
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)


if __name__ == "__main__":
    app()
