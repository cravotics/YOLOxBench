import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

def visdrone2yolo_person_only(dir_path):
    ann_path = Path(dir_path) / 'annotations'
    img_path = Path(dir_path) / 'images'
    label_path = Path(dir_path) / 'labels'
    label_path.mkdir(parents=True, exist_ok=True)

    for f in tqdm(list(ann_path.glob('*.txt')), desc=f'Converting {dir_path}'):
        image_file = img_path / f.with_suffix('.jpg').name
        if not image_file.exists():
            continue
        img_size = Image.open(image_file).size
        lines = []

        with open(f, 'r') as file:
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # Ignore ignored regions
                    continue
                if row[5] in ['1', '2']:  # pedestrian or people
                    cls = 0
                    box = convert_box(img_size, tuple(map(int, row[:4])))
                    lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")

        if lines:
            out_file = label_path / f.name
            with open(out_file, 'w') as out:
                out.writelines(lines)

if __name__ == '__main__':
    base_dir = Path('C:/CDCL')  # <- change this to your dataset root

    for subset in ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']:
        visdrone2yolo_person_only(base_dir / subset)
