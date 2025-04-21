"""
Benchmark many models across many datasets, producing a CSV.
"""
from pathlib import Path
import pandas as pd

from ..cfg import YoxConfig
from ..adapters import load_model
from ..exceptions import ValError, _fmt, _smart_hint
from ..utils import ensure_dir


def test(cfg: YoxConfig) -> Path:
    """
    Run each model against each dataset, collect detection metrics,
    and write them to a CSV for later comparison.
    """
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

            # Build a dict of numeric metrics
            metrics: dict[str, float] = {}

            # Top‑level numeric attributes on the DetMetrics object
            for attr in dir(m):
                if attr.startswith("_"):
                    continue
                val = getattr(m, attr)
                if isinstance(val, (int, float)):
                    metrics[attr] = float(val)

            # Include core box metrics if present
            box = getattr(m, 'box', None)
            if box is not None:
                for key in ('map50', 'map50_95', 'precision', 'recall'):
                    if hasattr(box, key):
                        metrics[key] = float(getattr(box, key))

            # Add identifying columns
            metrics['model']   = Path(model_path).name
            metrics['dataset'] = Path(ds).stem
            metrics['iou']     = cfg.iou
            metrics['conf']    = cfg.conf

            rows.append(metrics)

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    out_dir = Path("cmp_runs")
    ensure_dir(out_dir)
    out_file = out_dir / f"test_{len(cfg.models)}x{len(cfg.datasets)}.csv"
    df.to_csv(out_file, index=False)
    return out_file
