# yoloxbench/cli.py
"""
YOLOxBench command‑line interface
---------------------------------
train   – fine‑tune or resume a model
val     – evaluate one model on one dataset
test    – compare N models across M datasets (collect metrics)
cmp     – plot existing run folders
"""

from pathlib import Path
from typing import List, Optional

import typer
from rich import print, box
from rich.console import Console
from rich.table import Table

from .cfg import YoxConfig
from .engine import trainer, validator, tester, comparator

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
        banner = "[bold cyan]\nYOLOxBench[/] – what would you like to do?\n" \
                 "  • [bold]train[/]   Fine‑tune or resume a model\n" \
                 "  • [bold]val[/]     Validate one model\n" \
                 "  • [bold]test[/]    Compare multiple models × datasets\n" \
                 "  • [bold]cmp[/]     Plot existing run directories\n"
        print(banner)
        print("Try: [italic]yox train --help[/]  or  [italic]yox test --help[/]\n")


# ---------- TRAIN ----------
@app.command()
def train(
    cfg: Optional[Path] = typer.Option(
        None, help="YAML config file (overrides CLI flags)"),
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


# ---------- VALIDATE single model ----------
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


# ---------- TEST multi‑model × multi‑dataset ----------
@app.command()
def test(
    models: List[Path] = typer.Argument(..., exists=True, readable=True, dir_okay=False,
                                        help="One or more .pt checkpoints"),
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


# ---------- COMPARE existing run folders ----------
@app.command()
def cmp(
    run_dirs: List[Path] = typer.Argument(..., help="run directories to compare"),
    out: Path = Path("cmp_out"),
):
    """Create comparison bar‑plots from Ultralytics run folders."""
    comparator.compare(run_dirs, out)
    print(f"[bold green]✓[/] Comparison plots saved to {out}")


# ---------- helpers ----------
def _print_cfg(cfg):
    table = Table(title="Resolved Training Config", box=box.SIMPLE)
    for k, v in cfg.__dict__.items():
        table.add_row(str(k), str(v))
    console.print(table)


def _print_metrics(metrics: dict):
    table = Table(title="Validation metrics", box=box.SIMPLE)
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)


# ---------- UI ----------
# ---------- UI ----------
@app.command()
def ui(logdir: Path = Path("runs")):
    """Launch Streamlit dashboard in a separate process."""
    import subprocess, sys
    from pathlib import Path

    dash = Path(__file__).parent / "plotting" / "dashboard.py"
    if not dash.exists():
        typer.echo(f"Dashboard script not found at {dash}", err=True)
        raise typer.Exit(1)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dash), "--", f"--logdir={logdir}"
    ]
    subprocess.run(cmd)
from yoloxbench.analysis.advanced_compare import compare as adv_compare

@app.command()
def compare(logdir: Path = Path("runs")):
    """Advanced comparison across *all* runs under logdir/detect/."""
    csv = adv_compare(logdir)
    print(f"[bold green]✓[/] Comparison written to {csv}")



from typing import List
from pathlib import Path
import typer
from yoloxbench.reporting.markdown_report import make_markdown

@app.command()
def report(
    csv: Path = typer.Argument(..., help="CSV produced by `yox test`"),
    runs: List[str] = typer.Argument(
        ..., help="Run names in the same order as the CSV rows"
    ),
):
    """
    Generate a Markdown report from a YOLOxBench CSV.
    """
    rep = make_markdown(csv, runs)
    print(f"[bold green]✓[/] Report written to {rep}")

import subprocess, sys
from pathlib import Path
# … existing imports …

from .engine.video import annotate_and_save
from .gui          import run_gui

@app.command()
def video(
    model: Path = typer.Option(..., exists=True, readable=True, help="path to .pt"),
    source: Path = typer.Option(..., exists=True, readable=True, help="video file"),
    output: Path = typer.Option(
        None,
        "--output", "-o",
        help="where to save annotated video (if not using --gui)"
    ),
    conf: float = 0.25,
    iou:  float = 0.5,
    gui:  bool  = typer.Option(False, help="launch interactive PyQt GUI"),
):
    """
    Run YOLO inference on a video file.  By default saves an annotated copy;
    if --gui is passed, opens the interactive Qt viewer instead.
    """
    model_str  = str(model)
    source_str = str(source)

    if gui:
        run_gui(model_str)
    else:
        out_path = str(output or (source.parent / f"{source.stem}_annotated.mp4"))
        video_out = annotate_and_save(model_str, source_str, out_path, conf, iou)
        print(f"[bold green]✓[/] Annotated video saved to {video_out}")

# ---------- main ----------
if __name__ == "__main__":
    app()
