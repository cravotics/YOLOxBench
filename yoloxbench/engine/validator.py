from pathlib import Path
from ..cfg import YoxConfig
from ..adapters import load_model
from ..exceptions import ValError


def validate(cfg: YoxConfig):
    model = load_model(cfg.model)
    try:
        metrics = model.val(data=str(cfg.data), imgsz=cfg.imgsz, batch=cfg.batch or 16)
    except Exception as e:
        raise ValError(str(e)) from e
    return metrics  # dict from Ultralytics