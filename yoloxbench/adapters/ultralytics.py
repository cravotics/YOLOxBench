"""
Adapter for Ultralytics YOLO v8 (v8.3+)

Turns any model alias (e.g. 'yolov8m.pt') **or** arbitrary .pt checkpoint
into an object that exposes .train() and .val().
"""
from pathlib import Path
from ultralytics import YOLO


def load(spec: str | Path, *_, **__) -> YOLO:
    """
    Return a YOLO model ready for train/val/predict.

    Parameters
    ----------
    spec : str | Path
        Model alias ('yolov8n.pt', 'yolov8x.yaml') **or** path to a .pt
        checkpoint produced by Ultralytics training (e.g. runs/detect/.../best.pt).
    """
    return YOLO(str(spec))
