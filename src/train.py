"""Train YOLOv8n on the military vehicle dataset.

This script assumes the dataset has already been downloaded to data/ via
src/download_data.py and that you're running on a machine with a GPU
(e.g., Google Colab with T4 enabled).
"""
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent

dataset_folders = [f for f in (PROJECT_ROOT / "data").iterdir() if f.is_dir()]
if not dataset_folders:
    raise FileNotFoundError("No dataset in data/. Run download_data.py first.")
DATA_YAML = dataset_folders[0] / "data.yaml"


def train():
    model = YOLO("yolov8n.pt")
    return model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project=str(PROJECT_ROOT / "models"),
        name="vehicle_detector_v1",
        patience=15,
        save=True,
        plots=True,
    )


if __name__ == "__main__":
    train()