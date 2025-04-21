import cv2
from pathlib import Path
from ultralytics import YOLO

def annotate_and_save(
    model_path: str,
    source_path: str,
    output_path: str,
    conf: float = 0.25,
    iou: float = 0.5,
):
    """
    Run YOLO inference on a video and write an annotated copy.
    Returns Path to the saved file.
    """
    model = YOLO(model_path)
    cap   = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {source_path}")

    # Prepare output writer at same size/fps:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, iou=iou, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_score    = box.conf[0].item()
            cls           = int(box.cls[0].item())
            label         = f"{cls}:{conf_score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,0,0),
                2
            )

        out.write(frame)

    cap.release()
    out.release()
    return Path(output_path)
