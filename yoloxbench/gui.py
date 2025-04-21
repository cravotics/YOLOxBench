import sys
import cv2
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout,
    QWidget, QFileDialog
)
from PyQt5.QtGui  import QImage, QPixmap
from PyQt5.QtCore import QTimer
from ultralytics import YOLO

class VideoVisualizer(QWidget):
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.5):
        super().__init__()
        self.setWindowTitle("YOLOxBench Video Viewer")

        # load model with your thresholds
        self.model = YOLO(model_path)
        self.conf, self.iou = conf, iou

        self.cap   = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)

        self.label  = QLabel(alignment=0x84)  # center
        load_btn    = QPushButton("Load Video…")
        load_btn.clicked.connect(self._load_video)

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(load_btn)

    def _load_video(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open video")
        if not fn:
            return
        self.cap = cv2.VideoCapture(fn)
        if not self.cap.isOpened():
            self.label.setText("❌ Failed to open video")
            return

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.label.setFixedSize(w, h)
        self.setFixedSize(w+20, h+60)
        self.timer.start(30)

    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            return

        # run inference with conf and iou
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
        for box in results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            conf_score = box.conf[0].item()
            cls_id     = int(box.cls[0].item())
            lbl        = f"{cls_id}:{conf_score:.2f}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(
                frame, lbl, (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_img))

def run_gui(model_path: str, conf: float = 0.25, iou: float = 0.5):
    # create the Qt app once
    app = QApplication(sys.argv)
    viewer = VideoVisualizer(model_path, conf, iou)
    viewer.show()
    sys.exit(app.exec_())

