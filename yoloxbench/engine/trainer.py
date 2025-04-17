from pathlib import Path
from ..cfg import YoxConfig
from ..adapters import load_model
from ..exceptions import TrainError
from ..utils import ensure_dir


def train(cfg: YoxConfig):
    model = load_model(cfg.model)
    try:
        results = model.train(
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
        raise TrainError(str(e)) from e
    run_dir = Path(model.save_dir)
    ensure_dir(run_dir)
    return run_dir