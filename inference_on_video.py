import cv2
import time
from ultralytics import YOLO
import os

def run_inference_on_video(model_path, input_video, output_video, conf_threshold=0.3):
    model = YOLO(model_path)
    model.to('cuda')  # Force GPU usage

    cap = cv2.VideoCapture(input_video)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        for box in results.boxes:
            conf = box.conf[0].item()
            if conf < conf_threshold:
                continue  # Skip low confidence

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"person ({conf:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(y1 - 15, 0)), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, (0, 0, 255), 3, cv2.LINE_AA)

        # Optional FPS overlay
        fps_display = f"FPS: {1 / (time.time() - start):.2f}"
        cv2.putText(frame, fps_display, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Saved: {output_video}")


def generate_comparison_video(video1, video2, output_video):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    width1  = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap1.get(cv2.CAP_PROP_FPS)

    width2  = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    assert height1 == height2, "Both videos must have the same height"
    
    out_width = width1 + width2
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_width, height1))

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        combined = cv2.hconcat([frame1, frame2])
        out.write(combined)

    cap1.release()
    cap2.release()
    out.release()
    print(f"✅ Comparison video saved: {output_video}")


# --- 🔁 Inference ---
run_inference_on_video(
    model_path=r"runs/detect/visdrone_yolo8m_person4/weights/best.pt",  # Fine-tuned model
    input_video="uav1_rectified_fisheye.mp4",
    output_video="uav1_yolo8m_epoch100.mp4"
)

run_inference_on_video(
    model_path="yolov8m.pt",  # Pretrained model on COCO
    input_video="uav1_rectified_fisheye.mp4",
    output_video="uav1_yolo8m_pretrained.mp4"
)

# --- 🎥 Side-by-side comparison ---
generate_comparison_video(
    video1="uav1_yolo8m_pretrained.mp4",
    video2="uav1_yolo8m_epoch100.mp4",
    output_video="uav1_comparison.mp4"
)
