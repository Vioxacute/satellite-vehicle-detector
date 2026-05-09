"""Visualise a few labelled training images to confirm data looks right."""
import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).parent.parent

# Find the dataset folder Roboflow created
dataset_folders = [f for f in (PROJECT_ROOT / "data").iterdir() if f.is_dir()]
if not dataset_folders:
    raise FileNotFoundError("No dataset folder found in data/. Run download_data.py first.")
DATASET_DIR = dataset_folders[0]
print(f"Inspecting dataset: {DATASET_DIR.name}")

# Load class names from data.yaml
with open(DATASET_DIR / "data.yaml") as f:
    config = yaml.safe_load(f)
class_names = config["names"]
print(f"Classes ({len(class_names)}): {class_names}")

# Pick 4 random training images
images_dir = DATASET_DIR / "train" / "images"
labels_dir = DATASET_DIR / "train" / "labels"
all_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
print(f"Total training images: {len(all_images)}")

sample_images = random.sample(all_images, min(4, len(all_images)))

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for ax, img_path in zip(axes.flat, sample_images):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    label_path = labels_dir / (img_path.stem + ".txt")
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id, xc, yc, bw, bh = map(float, parts[:5])
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, class_names[int(cls_id)], (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    ax.imshow(img)
    ax.axis("off")
    ax.set_title(img_path.name, fontsize=10)

samples_dir = PROJECT_ROOT / "samples"
samples_dir.mkdir(exist_ok=True)
output_path = samples_dir / "data_inspection.png"
plt.tight_layout()
plt.savefig(output_path, dpi=100)
plt.show()
print(f"\nSaved to {output_path}")