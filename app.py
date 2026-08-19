import streamlit as st
from PIL import Image
import torch
import pandas as pd
from torchvision import transforms

from model import DualModeClassifier
from inference_engine import predict_standard, predict_bnn_mcdropout
from labels import CLASSES, NUM_CLASSES
from config import IMG_SIZE

st.set_page_config(page_title="Safeguard AI Demo", layout="wide")
st.title("🛡️ PathoVision AI: Standard vs Bayesian Medical Inference")

MODEL_PATH = "fine_tuned_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@st.cache_resource
def load_model():
    """Loaded once per server session, not on every rerun/upload."""
    model = DualModeClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"Couldn't find model weights at '{MODEL_PATH}'. "
        f"Run train.py first to produce this file."
    )
    st.stop()

st.sidebar.header("Bayesian Inference Settings")
num_samples = st.sidebar.slider("MC Dropout passes", min_value=5, max_value=100, value=30, step=5)
use_tta = st.sidebar.checkbox("Use Test-Time Augmentation", value=True)
st.sidebar.caption(f"Running on: **{DEVICE}**")

uploaded_file = st.file_uploader("Upload Skin Lesion Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2, col3 = st.columns([1, 1, 1])

    image = Image.open(uploaded_file).convert("RGB")
    col1.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = EVAL_TRANSFORM(image).unsqueeze(0)

    with st.spinner(f"Running standard inference and {num_samples}-pass MC Dropout..."):
        res_std = predict_standard(model, img_tensor, DEVICE)
        res_mcd = predict_bnn_mcdropout(
            model, img_tensor, num_samples=num_samples, use_tta=use_tta, device=DEVICE
        )

    # Column 2: Standard Model Output
    with col2:
        st.subheader("Standard Model")
        st.write(f"**Predicted Class:** {res_std['class_name']}")
        st.write(f"**Confidence:** {res_std['confidence'] * 100:.1f}%")
        st.warning("⚠️ Cannot detect out-of-distribution errors or artifacts.")
        st.bar_chart(pd.Series(res_std["probs"].numpy(), index=CLASSES))

    # Column 3: BNN MC Dropout Output
    with col3:
        st.subheader("Bayesian Model (MCD)")
        st.write(f"**Predicted Class:** {res_mcd['class_name']}")
        st.write(f"**Confidence:** {res_mcd['confidence'] * 100:.1f}%")
        st.write(f"**Uncertainty ({res_mcd['uncertainty_metric']}):** {res_mcd['uncertainty']:.5f}")
        st.bar_chart(pd.Series(res_mcd["mean_probs"].numpy(), index=CLASSES))

        triage = res_mcd["triage"]
        if triage["status"] == "RED LIGHT":
            st.error(f"🚨 {triage['status']}: {triage['action']}")
        elif triage["status"] == "YELLOW LIGHT":
            st.warning(f"⚠️ {triage['status']}: {triage['action']}")
        else:
            st.success(f"✅ {triage['status']}: {triage['action']}")