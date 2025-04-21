# YOLOxBench Documentation

Welcome to **YOLOxBench**, a unified toolkit for training, validating, comparing, and visualizing YOLO-based object detection models. This guide walks new users through installation, commands, architecture, and workflows.

---

## Table of Contents

1. [Installation](#installation)  
2. [Command-Line Interface](#command-line-interface)  
   - [Overview of commands](#overview-of-commands)  
   - [`train`](#train)  
   - [`val`](#val)  
   - [`test`](#test)  
   - [`cmp`](#cmp)  
   - [`ui`](#ui)  
   - [`compare`](#compare)  
   - [`report`](#report)  
   - [`video`](#video)  
3. [Project Structure](#project-structure)  
4. [Configuration (`YoxConfig`)](#configuration-yoxconfig)  
5. [Engine Modules](#engine-modules)  
6. [Reporting & Dashboard](#reporting--dashboard)  
7. [Extending YOLOxBench](#extending-yoloxbench)  
8. [Example Workflows](#example-workflows)  
9. [Troubleshooting & FAQs](#troubleshooting--faqs)

---

## Installation

YOLOxBench uses Poetry for dependency management. To install:

```bash
git clone https://github.com/yourusername/yoloxbench.git
cd yoloxbench
poetry install
poetry shell
```

Optionally, add PyQt5 and OpenCV if you plan to use the interactive video GUI:

```bash
poetry add opencv-python PyQt5
```

Once installed, the CLI entrypoint `yox` is available:

```bash
yox --help
```

---

## Command-Line Interface

All commands follow:

```bash
yox [COMMAND] [OPTIONS]
```

### Overview of commands

| Command   | Description                                                             |
|-----------|-------------------------------------------------------------------------|
| `train`   | Fine‑tune or resume a YOLO model                                        |
| `val`     | Validate one model on one dataset                                       |
| `test`    | Compare multiple models across multiple datasets (outputs CSV)          |
| `cmp`     | Plot bar‑charts from existing run directories                           |
| `ui`      | Launch an interactive Streamlit dashboard                               |
| `compare` | Advanced, recursive comparison over all runs under `runs/detect/`       |
| `report`  | Generate a Markdown report from a comparison CSV                        |
| `video`   | Run inference on video files, with optional PyQt5 viewer                |

### `train`

```bash
yox train \
  --model <model.pt|alias> \
  --data <dataset.yaml> \
  [--epochs N] [--imgsz SIZE] [--batch B] [--workers W] [--device DEVICE] [--name RUN_NAME]
```

### `val`

```bash
yox val \
  --model <path/to/weights.pt> \
  --data <dataset.yaml> \
  [--iou F] [--conf F] [--imgsz SIZE] [--batch B]
```

### `test`

```bash
yox test \
  path/to/model1.pt path/to/model2.pt ... \
  --datasets ds1.yaml ds2.yaml ... \
  [--iou F] [--conf F] [--imgsz SIZE] [--batch B]
```

### `cmp`

```bash
yox cmp runs/detect/runA runs/detect/runB ... [--out out_dir]
```

### `ui`

```bash
yox ui --logdir runs
```

### `compare`

```bash
yox compare --logdir runs
```

### `report`

```bash
yox report <comparison.csv> <runA> <runB> ...
```

### `video`

```bash
yox video \
  --model runs/detect/.../weights/best.pt \
  --source input.mp4 \
  [--output out.mp4] [--conf F] [--iou F] [--gui]
```

---

## Project Structure

```
yoloxbench/
├── cfg.py
├── cli.py
├── engine/
│   ├── trainer.py
│   ├── validator.py
│   ├── tester.py
│   ├── comparator.py
│   ├── video.py
│   └── ...
├── gui.py
├── analysis/
│   └── advanced_compare.py
├── plotting/
│   └── dashboard.py
├── reporting/
│   └── markdown_report.py
├── utils.py
├── exceptions.py
└── ...
```

---

## Configuration (`YoxConfig`)

```python
@dataclass(frozen=True, slots=True)
class YoxConfig:
    mode: str                  # 'train' | 'val' | 'test'
    model: str | Path          # path or alias
    data: str | Path           # dataset YAML
    epochs: int | None = None
    imgsz: int = 640
    batch: int | None = None
    device: str = 'cuda'
    ...
```

---

## Reporting & Dashboard

### Streamlit dashboard

- Auto‑discovers `runs/detect/*/results.csv`.
- Side‑panel to select specific runs, metrics.
- Displays per-run scalars, training curves, PR/F1/confusion images.
- Export comparison as CSV or Markdown.

### Markdown report

Uses `make_markdown()` to create a self‑contained `.md` folder:

```
cmp_runs/compare_2x1.csv_report/
├── bar_metrics.png
├── pr_curves/
├── confusion_matrices/
└── report.md
```

---

## Extending YOLOxBench

1. Add a new metric: edit `tester.py`.
2. Add new video format: tweak `video.py`.
3. Improve dashboard: modify `dashboard.py`.
4. Add CLI command: modify `cli.py` and create new module in `engine/`.

---

## Example Workflows

### Training + Validation

```bash
yox train --model yolov8m.pt --data VisDrone.yaml --epochs 50 --name visdrone_yolov8m
yox val --model runs/detect/visdrone_yolov8m/weights/best.pt --data VisDrone.yaml
```

### Batch Comparison

```bash
yox test runs/detect/visdrone_yolov8m/weights/best.pt runs/detect/visdrone_yolov8n/weights/best.pt \
  --datasets VisDrone.yaml
```

### Streamlit Dashboard

```bash
yox ui --logdir runs
```

### Generate Markdown Report

```bash
yox report cmp_runs/test_2x1.csv visdrone_yolov8m visdrone_yolov8n
```

### Video Inference CLI

```bash
yox video --model runs/detect/visdrone_yolov8m/weights/best.pt \
           --source uav1.mp4 --output uav1_annotated.mp4
```

### Interactive PyQt Viewer

```bash
yox video --model runs/detect/visdrone_yolov8m/weights/best.pt \
           --source uav1.mp4 --gui
```

---

## Troubleshooting & FAQs

### `AttributeError: annotate_and_save`

- Make sure `engine/video.py` defines it and is correctly imported.

### `Missing 'tabulate' in dashboard`

```bash
poetry add tabulate
```

### `KeyError: 'metrics/mAP50'` in dashboard

- Check if `results.csv` has the column `metrics/mAP50(B)` and select it via the sidebar.

### No CLI entrypoint?

Ensure this exists in `pyproject.toml`:

```toml
[project.scripts]
yox = "yoloxbench.cli:app"
```

---

**Happy benchmarking!**  
For bugs, suggestions, or contributions, open an issue or PR on GitHub.
