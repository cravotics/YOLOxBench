class YoxError(Exception):
    """Base for all custom exceptions raised by YOLOxBench."""

# ---------------------------------------------------------------------------
# Friendly error‑message helpers
# ---------------------------------------------------------------------------

_KNOWN_HINTS = {
    "CUDA out of memory": "Lower the --batch size or use device=cpu to debug.",
    "No such file or directory": "Check your --data YAML and --model path.",
    "could not find class names": "Your dataset YAML is missing a 'names:' list.",
}

def _smart_hint(msg: str) -> str | None:
    for key, hint in _KNOWN_HINTS.items():
        if key in msg:
            return hint
    return None

def _fmt(msg: str, hint: str | None = None) -> str:
    """Return *msg* plus an optional hint formatted on a new line."""
    return f"{msg}\n\nHint: {hint}" if hint else msg   # <- fixed line

class ConfigError(YoxError):
    """Bad YAML or unsupported CLI arg."""

class DataError(YoxError):
    """Dataset missing or malformed."""

class TrainError(YoxError):
    """Failure inside model.train()."""

class ValError(YoxError):
    """Failure inside model.val()."""
