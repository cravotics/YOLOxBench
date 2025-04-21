from importlib import import_module
from pathlib import Path
from typing import Any

_ADAPTERS = {
    "ultralytics": ".ultralytics",
    "v11": ".stub_future",
}

def load_model(spec: str | Path, *args: Any, **kwargs: Any):
    """Return a model object using the right adapter."""
    spec_str = str(spec)
    # if it’s a YOLO alias OR any .pt checkpoint ⇒ use Ultralytics
    if spec_str.lower().startswith("yolo") or spec_str.endswith(".pt"):
        mod = import_module(__name__ + _ADAPTERS["ultralytics"])
        return mod.load(spec, *args, **kwargs)

    # fallback (future adapters)
    mod = import_module(__name__ + _ADAPTERS["v11"])
    return mod.load(spec, *args, **kwargs)
