import os
import cv2
import matplotlib.pyplot as plt

image_dir = 'VisDrone2019-DET-train/images'
label_dir = 'VisDrone2019-DET-train/labels'

# List of images with matching labels
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') and os.path.exists(os.path.join(label_dir, f.replace('.jpg', '.txt')))]

# Limit to first 20
image_files = image_files[:20]

for img_file in image_files:
    label_file = img_file.replace('.jpg', '.txt')
    img_path = os.path.join(image_dir, img_file)
    label_path = os.path.join(label_dir, label_file)

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Skipping unreadable image: {img_path}")
        continue

    h, w = img.shape[:2]

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, bw, bh = map(float, parts)
            # Denormalize
            cx, cy, bw, bh = x * w, y * h, bw * w, bh * h
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"{img_file} — Bounding Boxes")
    plt.axis('off')
    plt.show()
