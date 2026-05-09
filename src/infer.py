"""Test the trained model on a single image."""
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "vehicle_detector_v1" / "weights" / "best.pt"


def predict(image_path: str, conf_threshold: float = 0.25):
    model = YOLO(str(MODEL_PATH))
    results = model.predict(image_path, conf=conf_threshold, save=True, project=str(PROJECT_ROOT / "runs"))

    for r in results:
        print(f"\nFound {len(r.boxes)} objects in {Path(image_path).name}:")
        for box in r.boxes:
            cls = model.names[int(box.cls)]
            conf = float(box.conf)
            print(f"  - {cls}: {conf:.2%} confidence")

    return results


if __name__ == "__main__":
    dataset_folders = [f for f in (PROJECT_ROOT / "data").iterdir() if f.is_dir()]
    sample_image = next((dataset_folders[0] / "valid" / "images").glob("*.jpg"))
    print(f"Testing on: {sample_image.name}")
    predict(str(sample_image))