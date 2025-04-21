from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml

from .exceptions import ConfigError, _fmt   # import _fmt !

@dataclass(frozen=True, slots=True)
class YoxConfig:
    # ---------- core ----------
    mode: str                    # 'train' | 'val' | 'test'
    model: str | Path | None = None
    data: str | Path | None = None

    # ---------- training/val params ----------
    epochs: int | None = None
    imgsz: int = 640
    batch: int | None = None
    device: str = "cuda"
    fp16: bool = True
    seed: int = 42
    extra: dict = field(default_factory=dict)

    # ---------- thresholds ----------
    iou: float = 0.5
    conf: float = 0.25

    # ---------- test‑mode lists ----------
    models: list[str] | None = None
    datasets: list[str] | None = None

    # ---------- reporting ----------
    metrics: list[str] = field(default_factory=lambda: ["mAP50", "mAP50-95"])

    # ------------------------------------------------------------------
    @staticmethod
    def load(path: str | Path | None = None, **overrides) -> "YoxConfig":
        cfg_dict: dict = {}
        if path:
            try:
                cfg_dict = yaml.safe_load(Path(path).read_text()) or {}
            except FileNotFoundError as e:
                raise ConfigError(str(e)) from e

        cfg_dict.update({k: v for k, v in overrides.items() if v is not None})

        # accept comma‑separated strings for CLI convenience
        for key in ("models", "datasets"):
            if key in cfg_dict and isinstance(cfg_dict[key], str):
                cfg_dict[key] = [p.strip() for p in cfg_dict[key].split(",") if p.strip()]

        try:
            return YoxConfig(**cfg_dict)
        except TypeError as e:
            raise ConfigError(_fmt("Invalid config keys", str(e))) from e
