# yoloxbench/engine/video.py
import cv2
import torch
from ultralytics import YOLO
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
)


def annotate_video(
    model_path: str,
    source_path: str,
    output_path: str,
    device: str = "0",
):
    """
    Load `model_path`, open `source_path` video, run frame-by-frame
    inference, draw boxes, and write to `output_path`, while showing
    a Rich progress bar.
    """
    # 1) load model
    model = YOLO(model_path)
    model.to(device)

    # 2) open input & output
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video '{source_path}'")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 3) set up Rich progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    task = progress.add_task("Annotating frames…", total=total)

    # 4) process
    with progress:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # run inference
            results = model(frame, device=device, verbose=False)[0]

            # draw boxes & labels
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls  = int(box.cls[0].item())
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 200, 100), 2)
                cv2.putText(
                    frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (16, 200, 100), 1
                )

            writer.write(frame)
            progress.update(task, advance=1)

    # 5) cleanup
    cap.release()
    writer.release()
