import typer
from pathlib import Path
from rich import print
from .cfg import YoxConfig
from .engine import trainer, validator, comparator

app = typer.Typer(help="YOLOxBench — train, validate, compare.")

@app.command()
def train(cfg: Path = typer.Option(None, help="YAML config file."),
          model: str = None, data: str = None):
    cfg_obj = YoxConfig.load(cfg, mode="train", model=model, data=data)
    run_dir = trainer.train(cfg_obj)
    print(f"[bold green]✓[/] Training done — results in {run_dir}")

@app.command()
def val(cfg: Path = typer.Option(None), model: str = None, data: str = None):
    cfg_obj = YoxConfig.load(cfg, mode="val", model=model, data=data)
    metrics = validator.validate(cfg_obj)
    print(metrics)

@app.command()
def cmp(run_dirs: list[Path], out: Path = Path("cmp_out")):
    comparator.compare(run_dirs, out)
    print(f"Comparison saved to {out}")

if __name__ == "__main__":
    app()