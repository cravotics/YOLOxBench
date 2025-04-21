"""Training wrapper that forwards all config fields to Ultralytics YOLO."""
from pathlib import Path
from typing import Any
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..adapters import load_model
from ..exceptions import TrainError, _fmt, _smart_hint
from ..utils import ensure_dir
from ..cfg import YoxConfig


def train(cfg: YoxConfig) -> Path:
    """Run model.train() and return run_dir."""
    model = load_model(cfg.model)

    # Progress spinner while Ultralytics does its own tqdm
    spinner = Progress(
        SpinnerColumn(), TextColumn("[bold green]training…[/] {task.description}"), transient=True
    )
    task_id = spinner.add_task("running", total=None)

    try:
        with spinner:
            results: Any = model.train(
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