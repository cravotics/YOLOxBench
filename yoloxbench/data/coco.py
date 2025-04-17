from pathlib import Path
from ultralytics.cfg import get_cfg

class CocoDataset:
    """Thin wrapper until we need more."""
    def __init__(self, cfg_yaml: str | Path):
        self.yaml = str(cfg_yaml)
        self.info = get_cfg(self.yaml)