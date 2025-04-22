import sys
import cv2
import torch
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from ultralytics import YOLO

class VideoVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Video Visualizer")
        self.model = YOLO("runs/detect/yolov8m_finetuned_visdrone_person5/weights/best.pt")  # pretrained weights

        self.video_path = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # GUI
        self.label = QLabel(self)
        self.button = QPushButton("Load Video")
        self.button.clicked.connect(self.load_video)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if file_name:
            self.video_path = file_name
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print("Error opening video file")
                return

            # Automatically match window size to video
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.label.setFixedSize(width, height)
            self.setFixedSize(width + 50, height + 80)

            self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            return

        # ===== Optional Rectification (use only if camera calibration available) =====
        # frame = self.rectify_frame(frame)

        results = self.model(frame, verbose=False)[0]  # Runs on GPU by default if available
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"ID: {cls} ({conf:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(q_img))

    def rectify_frame(self, frame):
        # TODO: Replace with your camera matrix & distortion coefficients
        # If you don't have them, skip this function
        K = ...  # camera matrix
        dist = ...  # distortion coefficients
        h, w = frame.shape[:2]
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        return cv2.undistort(frame, K, dist, None, new_camera_matrix)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = VideoVisualizer()
    viewer.show()
    sys.exit(app.exec_())
