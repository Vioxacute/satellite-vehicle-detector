"""Count instances per class to spot imbalance."""
from pathlib import Path
from collections import Counter
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
dataset_folders = [f for f in (PROJECT_ROOT / "data").iterdir() if f.is_dir()]
DATASET_DIR = dataset_folders[0]

with open(DATASET_DIR / "data.yaml") as f:
    config = yaml.safe_load(f)
class_names = config["names"]

counts = Counter()
for split in ["train", "valid", "test"]:
    labels_dir = DATASET_DIR / split / "labels"
    if not labels_dir.exists():
        continue
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                parts = line.split()
                if parts:
                    cls_id = int(parts[0])
                    counts[class_names[cls_id]] += 1

print(f"{'Class':<25} {'Count':>8}")
print("-" * 35)
for cls, count in counts.most_common():
    print(f"{cls:<25} {count:>8}")
print("-" * 35)
print(f"{'TOTAL':<25} {sum(counts.values()):>8}")