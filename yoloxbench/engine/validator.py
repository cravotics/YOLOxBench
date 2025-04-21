"""Single‑model validation helper."""
from ..cfg import YoxConfig
from ..adapters import load_model
from ..exceptions import ValError, _fmt, _smart_hint


def validate(cfg: YoxConfig):
    model = load_model(cfg.model)
    try:
        metrics = model.val(
            data=str(cfg.data),
            imgsz=cfg.imgsz,
            batch=cfg.batch or 16,
            iou=cfg.iou,
            conf=cfg.conf,
            verbose=False,
        )
    except Exception as e:
        raise ValError(_fmt(str(e), _smart_hint(str(e)))) from e
    return metrics