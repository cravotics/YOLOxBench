# yoloxbench/gui.py
import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui     import QImage, QPixmap
from PyQt5.QtCore    import QTimer
from ultralytics     import YOLO

class VideoVisualizer(QWidget):
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.5):
        super().__init__()
        self.setWindowTitle("YOLOxBench Video Viewer")
        self.model = YOLO(model_path)
        self.conf  = conf
        self.iou   = iou
        self.cap   = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)

        self.label = QLabel(alignment=0x84)  # center
        load_btn   = QPushButton("Load Video…")
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
        self.setFixedSize(w + 20, h + 60)
        self.timer.start(30)

    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            return

        # ------- run YOLO inference -------
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)[0]

        # create an overlay for semi‑transparent fills
        overlay = frame.copy()

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score         = float(box.conf[0])
            cls_id        = int(box.cls[0])

            # compute interpolation t in [0,1]
            t = (score - self.conf) / (1.0 - self.conf)
            t = max(0.0, min(1.0, t))

            # border color will vary from pale→dark green
            low_color  = (200, 255, 200)
            high_color = (0,   128,   0)
            b = int(low_color[0] * (1 - t) + high_color[0] * t)
            g = int(low_color[1] * (1 - t) + high_color[1] * t)
            r = int(low_color[2] * (1 - t) + high_color[2] * t)
            color = (b, g, r)

            # alpha for fill: low confidence → very transparent; high → more opaque
            alpha_min, alpha_max = 0.1, 0.6
            alpha = alpha_min + (alpha_max - alpha_min) * t

            # draw filled rectangle on the overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)

            # label text
            lbl = f"{cls_id}:{score:.2f}"

            # combine only the overlay region with original frame
            cv2.addWeighted(overlay[y1:y2, x1:x2], alpha,
                            frame[y1:y2, x1:x2], 1 - alpha, 0,
                            dst=frame[y1:y2, x1:x2])

            # draw the solid border
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
            cv2.putText(frame, lbl, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # convert to QImage and display
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img  = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_img))

def run_gui(model_path: str, conf: float = 0.25, iou: float = 0.5):
    """
    Launch the Qt window.  Pass this into your CLI when --gui is set.
    """
    app = QApplication(sys.argv)
    viewer = VideoVisualizer(model_path, conf, iou)
    viewer.show()
    sys.exit(app.exec_())
