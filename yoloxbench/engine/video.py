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
    Boxes are drawn in a green gradient: pale at the conf threshold,
    darker as confidence approaches 1.0.
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

    # Predefine gradient endpoints
    low_color  = (200, 255, 200)  # pale green (B, G, R)
    high_color = (0,   128,   0)  # dark green

    # Process frame by frame with a progress bar
    for _ in track(range(total), description="🔎 running inference"):
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        res = model(frame, conf=conf, iou=iou)[0]

        # Draw boxes & labels with gradient color
        for box in res.boxes:
            # extract box coords
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            score = float(box.conf[0].cpu().item())
            cls   = int(box.cls[0].cpu().item())

            # map score ∈ [conf, 1.0] to t ∈ [0,1]
            t = (score - conf) / (1.0 - conf)
            t = max(0.0, min(1.0, t))

            # linear interpolate each channel
            b = int(low_color[0] * (1 - t) + high_color[0] * t)
            g = int(low_color[1] * (1 - t) + high_color[1] * t)
            r = int(low_color[2] * (1 - t) + high_color[2] * t)
            color = (b, g, r)

            # draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{cls}:{score:.2f}",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        # write annotated frame out
        out.write(frame)

    cap.release()
    out.release()
    return str(output_path)
