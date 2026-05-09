import os
from pathlib import Path
from roboflow import Roboflow

# Get project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Change working directory so Roboflow downloads into data/
os.chdir(DATA_DIR)

rf = Roboflow(api_key="YOUR_API_KEY_HERE")
project = rf.workspace("militaryvehiclerecognition").project("military-vehicle-recognition")
version = project.version(7)
dataset = version.download("yolov8")

print(f"\nDataset downloaded to: {dataset.location}")
print("Look inside that folder — you should see train/, valid/, test/, and data.yaml")