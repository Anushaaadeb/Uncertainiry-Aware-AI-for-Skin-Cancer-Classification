import html

import altair as alt
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from config import IMG_SIZE
from inference_engine import predict_bnn_mcdropout, predict_standard
from labels import CLASSES, DISEASE_DETAILS, NUM_CLASSES
from model import DualModeClassifier

st.set_page_config(page_title="PathoVision AI", layout="wide")

CATEGORY_COLORS = {
    "Malignant": ("#b42318", "#fef3f2"),
    "Precancerous": ("#b54708", "#fffaeb"),
    "Benign": ("#027a48", "#ecfdf3"),
    "Usually benign": ("#026aa2", "#f0f9ff"),
}

MODEL_PATH = "fine_tuned_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .pv-hero { margin-bottom: 0.4rem; }
        .pv-hero h1 { font-size: 1.85rem; margin-bottom: 0.25rem; }
        .pv-subtle { color: #475467; font-size: 0.95rem; margin-bottom: 1rem; }
        .pv-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin: 0.4rem 0 1.1rem 0;
        }
        .pv-card {
            position: relative;
            border: 1px solid #e4e7ec;
            border-radius: 12px;
            padding: 10px 12px 12px 12px;
            background: #ffffff;
            cursor: help;
            min-height: 74px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .pv-card:hover {
            border-color: #98a2b3;
            box-shadow: 0 8px 18px rgba(16, 24, 40, 0.10);
            z-index: 20;
        }
        .pv-name { font-weight: 650; font-size: 0.92rem; line-height: 1.25; color: #101828; }
        .pv-code { font-size: 0.75rem; color: #667085; margin-top: 2px; }
        .pv-badge {
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 650;
            letter-spacing: 0.02em;
            padding: 2px 8px;
            border-radius: 999px;
            margin-bottom: 6px;
        }
        .pv-tip {
            display: none;
            position: absolute;
            left: 0;
            top: calc(100% + 8px);
            width: min(340px, 70vw);
            background: #101828;
            color: #f9fafb;
            padding: 12px 13px;
            border-radius: 10px;
            z-index: 50;
            box-shadow: 0 12px 30px rgba(16, 24, 40, 0.28);
            font-size: 0.82rem;
            line-height: 1.4;
        }
        .pv-card:hover .pv-tip { display: block; }
        .pv-tip strong { color: #eaecf0; }
        .pv-tip p { margin: 0 0 0.45rem 0; }
        .pv-tip p:last-child { margin-bottom: 0; }
        .pv-pred {
            border: 1px solid #d0d5dd;
            border-radius: 12px;
            padding: 0.85rem 1rem;
            background: #f9fafb;
            margin-bottom: 0.6rem;
        }
        .pv-pred .label { font-size: 0.78rem; color: #667085; text-transform: uppercase; letter-spacing: 0.04em; }
        .pv-pred .value { font-size: 1.15rem; font-weight: 700; color: #101828; }
        .pv-pred .hint { font-size: 0.8rem; color: #475467; margin-top: 0.35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def disease_cards_html() -> str:
    cards = ['<div class="pv-grid">']
    for name in CLASSES:
        info = DISEASE_DETAILS[name]
        fg, bg = CATEGORY_COLORS[info["category"]]
        cards.append(
            f"""
            <div class="pv-card">
              <span class="pv-badge" style="color:{fg};background:{bg};">{_esc(info["category"])}</span>
              <div class="pv-name">{_esc(name)}</div>
              <div class="pv-code">{_esc(info["code"])}</div>
              <div class="pv-tip">
                <p><strong>{_esc(name)}</strong> ({_esc(info["code"])})</p>
                <p>{_esc(info["summary"])}</p>
                <p><strong>Also called:</strong> {_esc(info["also_called"])}</p>
                <p><strong>Typical look:</strong> {_esc(info["looks_like"])}</p>
                <p><strong>Why it matters:</strong> {_esc(info["why_it_matters"])}</p>
              </div>
            </div>
            """
        )
    cards.append("</div>")
    return "".join(cards)


def tooltip_text(name: str) -> str:
    info = DISEASE_DETAILS[name]
    return (
        f"{info['category']} ({info['code']}). {info['summary']} "
        f"Typical look: {info['looks_like']} Why it matters: {info['why_it_matters']}"
    )


def probability_chart(probs, predicted_name: str):
    """Horizontal bars so full HAM10000 class names stay readable on the axis."""
    df = pd.DataFrame({
        "Disease": CLASSES,
        "Probability": [float(p) for p in probs],
        "Details": [tooltip_text(name) for name in CLASSES],
        "Predicted": [name == predicted_name for name in CLASSES],
    })
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y(
                "Disease:N",
                sort="-x",
                title=None,
                axis=alt.Axis(
                    labelLimit=1000,
                    labelFontSize=13,
                    labelPadding=8,
                    tickSize=0,
                ),
            ),
            x=alt.X(
                "Probability:Q",
                title="Probability",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", grid=True),
            ),
            color=alt.condition(
                alt.datum.Predicted,
                alt.value("#b42318"),
                alt.value("#1570ef"),
            ),
            tooltip=[
                alt.Tooltip("Disease:N", title="Lesion"),
                alt.Tooltip("Probability:Q", title="Probability", format=".1%"),
                alt.Tooltip("Details:N", title="About this class"),
            ],
        )
        .properties(height=280)
        .configure_axis(labelFont="sans-serif", titleFont="sans-serif")
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)


def prediction_panel(title: str, class_name: str, confidence: float, extra_html: str = "") -> None:
    info = DISEASE_DETAILS[class_name]
    st.subheader(title)
    st.markdown(
        f"""
        <div class="pv-pred pv-card" style="min-height:auto;">
          <div class="label">Predicted class — hover here or a chart bar for details</div>
          <div class="value">{_esc(class_name)} <span style="font-weight:500;color:#667085;font-size:0.9rem;">({_esc(info["code"])} · {_esc(info["category"])})</span></div>
          <div class="hint">{_esc(info["summary"])}</div>
          {extra_html}
          <div class="pv-tip">
            <p><strong>{_esc(class_name)}</strong> ({_esc(info["code"])})</p>
            <p>{_esc(info["summary"])}</p>
            <p><strong>Also called:</strong> {_esc(info["also_called"])}</p>
            <p><strong>Typical look:</strong> {_esc(info["looks_like"])}</p>
            <p><strong>Why it matters:</strong> {_esc(info["why_it_matters"])}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.metric("Confidence", f"{confidence * 100:.1f}%")


@st.cache_resource
def load_model():
    """Loaded once per server session, not on every rerun/upload."""
    model = DualModeClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


inject_styles()
st.markdown(
    '<div class="pv-hero"><h1>PathoVision AI: Standard vs Bayesian Medical Inference</h1></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="pv-subtle">Upload a lesion photo to compare a point-estimate classifier with Monte Carlo Dropout. '
    "Hover a class card (or a bar in the charts) for what that HAM10000 label means.</p>",
    unsafe_allow_html=True,
)
st.markdown("**HAM10000 classes** — hover a card")
st.markdown(disease_cards_html(), unsafe_allow_html=True)

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
st.sidebar.markdown("---")
st.sidebar.markdown("**Class atlas**")
st.sidebar.caption("Hover a card in the main view, or pick a class here.")
atlas_pick = st.sidebar.selectbox("Read about a lesion class", CLASSES)
picked = DISEASE_DETAILS[atlas_pick]
st.sidebar.markdown(f"**{atlas_pick}** (`{picked['code']}`)")
st.sidebar.caption(picked["category"])
st.sidebar.write(picked["summary"])
st.sidebar.write(f"**Typical look:** {picked['looks_like']}")
st.sidebar.write(f"**Why it matters:** {picked['why_it_matters']}")

uploaded_file = st.file_uploader("Upload Skin Lesion Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_col, std_col, bnn_col = st.columns([0.9, 1.15, 1.15], gap="large")

    with img_col:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = EVAL_TRANSFORM(image).unsqueeze(0)

    with st.spinner(f"Running standard inference and {num_samples}-pass MC Dropout..."):
        res_std = predict_standard(model, img_tensor, DEVICE)
        res_mcd = predict_bnn_mcdropout(
            model, img_tensor, num_samples=num_samples, use_tta=use_tta, device=DEVICE
        )

    with std_col:
        prediction_panel("Standard Model", res_std["class_name"], res_std["confidence"])
        st.warning("Cannot detect out-of-distribution errors or artifacts.")
        probability_chart(res_std["probs"].numpy(), res_std["class_name"])

    with bnn_col:
        prediction_panel(
            "Bayesian Model (MCD)",
            res_mcd["class_name"],
            res_mcd["confidence"],
            extra_html=(
                f'<div class="hint"><strong>Uncertainty ({_esc(res_mcd["uncertainty_metric"])}):</strong> '
                f'{res_mcd["uncertainty"]:.5f}</div>'
            ),
        )
        probability_chart(res_mcd["mean_probs"].numpy(), res_mcd["class_name"])

        triage = res_mcd["triage"]
        if triage["status"] == "RED LIGHT":
            st.error(f"{triage['status']}: {triage['action']}")
        elif triage["status"] == "YELLOW LIGHT":
            st.warning(f"{triage['status']}: {triage['action']}")
        else:
            st.success(f"{triage['status']}: {triage['action']}")
