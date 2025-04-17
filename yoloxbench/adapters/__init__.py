from importlib import import_module
from pathlib import Path
from typing import Any

_ADAPTERS = {
    "ultralytics": ".ultralytics",
    "v11": ".stub_future",
}

def load_model(spec: str | Path, *args: Any, **kwargs: Any):
    """Dispatch to the correct adapter based on file/alias."""
    if isinstance(spec, str) and spec.lower().startswith("yolo"):
        mod = import_module(__name__ + _ADAPTERS["ultralytics"])
        return mod.load(spec, *args, **kwargs)
    # fallback
    mod = import_module(__name__ + _ADAPTERS["v11"])
    return mod.load(spec, *args, **kwargs)