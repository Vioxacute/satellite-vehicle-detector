"""Streamlit demo for the aerial military vehicle detector."""
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "vehicle_detector_v1" / "weights" / "best.pt"

st.set_page_config(page_title="Aerial Military Vehicle Detector", layout="wide")
st.title("Aerial Military Vehicle Detector")
st.caption(
    "YOLOv8 fine-tuned on a public OSINT dataset of drone and satellite imagery. "
    "Detects 5 classes: tank, armoured personnel carrier, air-fighter, bomber, soldier."
)


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


model = load_model()

# Sidebar
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider(
    "Confidence threshold", 0.0, 1.0, 0.25, 0.05,
    help="Lower = more detections (some false positives). Higher = fewer detections (more confident)."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Classes")
for name in model.names.values():
    st.sidebar.markdown(f"- {name}")

# Main UI
uploaded = st.file_uploader("Upload an aerial / overhead image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image above to run detection. Try drone footage or satellite imagery of an airfield, base, or vehicle column.")
else:
    image = Image.open(uploaded).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detections")
        with st.spinner("Running inference..."):
            results = model.predict(img_array, conf=conf_threshold, verbose=False)
        annotated = results[0].plot()
        st.image(annotated[..., ::-1], use_container_width=True)

    # Summary
    st.subheader("Detection summary")
    boxes = results[0].boxes
    if len(boxes) == 0:
        st.write("No objects detected above the confidence threshold. Try lowering the threshold in the sidebar.")
    else:
        st.write(f"**{len(boxes)} object(s) detected**")
        class_counts = {}
        for box in boxes:
            cls = model.names[int(box.cls)]
            class_counts[cls] = class_counts.get(cls, 0) + 1
        for cls, count in sorted(class_counts.items()):
            st.write(f"- {cls}: {count}")

st.markdown("---")
st.caption(
    "**Limitations**: trained on a small (~5,000 instance) public dataset of mixed drone and satellite imagery from "
    "the Russo-Ukrainian war. Performance degrades on imagery from different sensors, weather conditions, geographies, "
    "or angles than the training distribution. Bomber detection is weakest due to limited training examples (n≈285)."
)