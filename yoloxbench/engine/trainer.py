"""Training wrapper that forwards all config fields to Ultralytics YOLO."""
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..adapters import load_model
from ..exceptions import TrainError, _fmt, _smart_hint
from ..utils import ensure_dir
from ..cfg import YoxConfig


def train(cfg: YoxConfig) -> Path:
    model = load_model(cfg.model)
    spinner = Progress(SpinnerColumn(), TextColumn("[green]training…"), transient=True)
    task = spinner.add_task("run", total=None)
    try:
        with spinner:
            model.train(
                data=str(cfg.data),
                epochs=cfg.epochs or 100,
                imgsz=cfg.imgsz,
                batch=cfg.batch or 16,
                device=cfg.device,
                amp=cfg.fp16,
                seed=cfg.seed,
                **cfg.extra,
            )
    except Exception as e:
        raise TrainError(_fmt(str(e), _smart_hint(str(e)))) from e
    run_dir = Path(model.save_dir)
    ensure_dir(run_dir)
    return run_dir