import sys, os, cv2, argparse
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QListWidget,
    QHBoxLayout, QFileDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class LabelViewer(QWidget):
    def __init__(self, image_dir: Path, label_dir: Path):
        super().__init__()
        self.setWindowTitle("YOLOxBench Image + Label Viewer")
        self.image_dir = image_dir
        self.label_dir = label_dir

        # file‐list on the left
        self.list_widget = QListWidget()
        self.list_widget.currentTextChanged.connect(self.load_and_display)

        # image display on the right
        self.image_label = QLabel(alignment=Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)

        layout = QHBoxLayout(self)
        layout.addWidget(self.list_widget, 1)
        layout.addWidget(self.image_label, 4)

        self.populate_list()

    def populate_list(self):
        for imgfile in sorted(self.image_dir.iterdir()):
            if imgfile.suffix.lower() in {".jpg", ".png", ".jpeg"}:
                self.list_widget.addItem(imgfile.name)

    def load_and_display(self, filename: str):
        img_path = self.image_dir / filename
        lbl_path = self.label_dir / (Path(filename).stem + ".txt")
        frame = cv2.imread(str(img_path))
        h, w, _ = frame.shape

        # draw YOLO‐format boxes
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    cls, x_c, y_c, bw, bh = map(float, line.split())
                    x1 = int((x_c - bw/2) * w)
                    y1 = int((y_c - bh/2) * h)
                    x2 = int((x_c + bw/2) * w)
                    y2 = int((y_c + bh/2) * h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(
                        frame, f"ID:{int(cls)}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1
                    )

        # convert to QImage and display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img)
        self.image_label.setPixmap(
            pix.scaled(self.image_label.size(),
                       Qt.KeepAspectRatio,
                       Qt.SmoothTransformation)
        )

def run_label_viewer(image_dir: str, label_dir: str):
    """Entry point for CLI."""
    img_dir = Path(image_dir)
    lbl_dir = Path(label_dir)
    if not img_dir.is_dir():
        print(f"❌ Images folder not found: {img_dir}")
        sys.exit(1)
    if not lbl_dir.is_dir():
        print(f"❌ Labels folder not found: {lbl_dir}")
        sys.exit(1)

    app = QApplication(sys.argv)
    viewer = LabelViewer(img_dir, lbl_dir)
    viewer.resize(1200, 800)
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-i","--images", required=True,
                   help="Folder containing .jpg/.png images")
    p.add_argument("-l","--labels", required=True,
                   help="Folder containing YOLO .txt labels")
    args = p.parse_args()
    run_label_viewer(args.images, args.labels)
