import cv2
import numpy as np
import os
from tqdm import tqdm

# === EDIT THESE PATHS ===
input_video_path = "uav1.mp4"
output_video_path = "uav1_rectified_fisheye.mp4"

# === CAMERA INTRINSICS (Chimera-55deg) ===
K = np.array([[1978.56053, 0.0, 1091.80278],
              [0.0, 1986.37639, 549.97315],
              [0.0, 0.0, 1.0]])

D = np.array([-0.054078, 0.200858, -0.011292, -0.000894])  # Only first 4 used in fisheye

# === CHECK VIDEO EXISTS ===
if not os.path.exists(input_video_path):
    print(f"[❌] Input video not found: {input_video_path}")
    exit()

print(f"[🔍] Found input video: {input_video_path}")
cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"[ℹ️] Resolution: {width}x{height} | FPS: {fps} | Total frames: {frame_count}")

# === SETUP VIDEO WRITER ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# === FISHEYE UNDISTORT MAP ===
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (width, height), np.eye(3), balance=0)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (width, height), cv2.CV_16SC2)

print(f"[🚀] Starting fisheye rectification... Press 'q' to quit early.")

# === PROCESS FRAMES ===
for _ in tqdm(range(frame_count), desc="Rectifying"):
    ret, frame = cap.read()
    if not ret:
        break

    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    # # === Optional: Crop center 80% area for better visual alignment
    # crop_x = int(width * 0.1)
    # crop_y = int(height * 0.1)
    # cropped = undistorted[crop_y:height - crop_y, crop_x:width - crop_x]

    # Pad back to original size if cropped
    # Save to output
    out.write(undistorted)

    # === Preview side-by-side
    side_by_side = np.hstack((frame, undistorted))
    preview_w = 1280
    scale = preview_w / side_by_side.shape[1]
    preview_resized = cv2.resize(side_by_side, (preview_w, int(side_by_side.shape[0] * scale)))

    cv2.imshow("Original | Rectified", preview_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("❗ Quit requested.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"[✅] Rectified fisheye video saved to: {output_video_path}")
