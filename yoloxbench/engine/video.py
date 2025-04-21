# yoloxbench/engine/video.py
from pathlib import Path
import cv2
from ultralytics import YOLO
from rich.progress import track

def annotate_and_save(
    model_path: str,
    source_path: str,
    output_path: str,
    conf: float = 0.25,
    iou: float = 0.5
) -> str:
    """
    Run YOLO inference on a video and write out an annotated copy,
    showing a progress bar in your terminal.
    Returns the path to the saved video.
    """
    # Load the model
    model = YOLO(model_path)

    # Open input video
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {source_path!r}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Process frame by frame
    for _ in track(range(total), description="🔎 running inference"):
        ret, frame = cap.read()
        if not ret:
            break

        # model(frame) returns a Results list; take first
        res = model(frame, conf=conf, iou=iou)[0]

        # Draw boxes & labels
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            score = box.conf[0].cpu().item()
            cls   = int(box.cls[0].cpu().item())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                frame,
                f"{cls}:{score:.2f}",
                (x1, y1-6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,0,255),
                1,
            )

        out.write(frame)

    cap.release()
    out.release()
    return str(output_path)
