from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from .exceptions import ConfigError, _fmt

@dataclass(frozen=True, slots=True)
class YoxConfig:
    mode: str                            # 'train' | 'val'
    model: str | Path                    # path or alias
    data: str | Path
    epochs: int | None = None
    imgsz: int = 640
    batch: int | None = None
    device: str = "cuda"
    fp16: bool = True
    seed: int = 42
    extra: dict = field(default_factory=dict)
    iou: float = 0.5
    conf: float = 0.25
    models: list[str] | None = None      # only for test mode
    datasets: list[str] | None = None    # only for test mode
    metrics: list[str] = field(default_factory=lambda: ["mAP50", "mAP50-95"])

    @staticmethod
    def load(path: str | Path | None = None, **overrides) -> "YoxConfig":
        cfg_dict: dict = {}
        if path:
            try:
                cfg_dict = yaml.safe_load(Path(path).read_text()) or {}
            except FileNotFoundError as e:
                raise ConfigError(str(e)) from e
        cfg_dict.update({k: v for k, v in overrides.items() if v is not None})
        try:
            return YoxConfig(**cfg_dict)
        except TypeError as e:
            raise ConfigError(_fmt("Invalid config keys", str(e))) from e