"""Benchmark many models across many datasets, producing a CSV."""
from pathlib import Path
import pandas as pd

from ..cfg import YoxConfig
from ..adapters import load_model
from ..exceptions import ValError, _fmt, _smart_hint
from ..utils import ensure_dir


def test(cfg: YoxConfig) -> Path:
    rows = []
    for model_path in cfg.models:
        model = load_model(model_path)
        for ds in cfg.datasets:
            try:
                m = model.val(
                    data=ds,
                    imgsz=cfg.imgsz,
                    batch=cfg.batch or 16,
                    iou=cfg.iou,
                    conf=cfg.conf,
                    verbose=False,
                )
            except Exception as e:
                raise ValError(_fmt(str(e), _smart_hint(str(e)))) from e
            row = {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}
            row.update(model=Path(model_path).name, dataset=Path(ds).stem,
                       iou=cfg.iou, conf=cfg.conf)
            rows.append(row)
    df = pd.DataFrame(rows)
    out_dir = Path("cmp_runs"); ensure_dir(out_dir)
    out_file = out_dir / f"test_{len(cfg.models)}x{len(cfg.datasets)}.csv"
    df.to_csv(out_file, index=False)
    return out_file