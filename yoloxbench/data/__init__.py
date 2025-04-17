from pathlib import Path
from .coco import CocoDataset

_DS_LOADERS = {
    "coco": CocoDataset,
}

def get_dataset(path_or_yaml: str | Path):
    # naive: assume coco for now
    return CocoDataset(path_or_yaml)