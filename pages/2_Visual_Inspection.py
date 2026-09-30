import streamlit as st
from pathlib import Path
from PIL import Image

from modules.visual_inspection import (
    create_reference_features,
    train_anomaly_model,
    predict_anomaly,
    image_quality_metrics,
    generate_anomaly_map,
)


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Visual Inspection",
    page_icon="🔍",
    layout="wide",
)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
NORMAL_IMAGE_DIR = PROJECT_ROOT / "data" / "normal_images"


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .result-pass {
        padding: 22px;
        border-radius: 12px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.4);
        text-align: center;
    }

    .result-review {
        padding: 22px;
        border-radius: 12px;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.4);
        text-align: center;
    }

    .result-heading {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .result-anomaly-score {
        font-size: 1.15rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🔍 AI Visual Inspection</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-powered anomaly detection using the official MVTec Bottle dataset"
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Inspection Settings")

threshold = st.sidebar.slider(
    "Anomaly Review Threshold",
    min_value=0,
    max_value=100,
    value=50,
    step=1,
)

st.sidebar.markdown("---")

st.sidebar.write("### Reference Dataset")

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

reference_paths = []

if NORMAL_IMAGE_DIR.exists():
    reference_paths = sorted(
        [
            p
            for p in NORMAL_IMAGE_DIR.iterdir()
            if p.is_file()
            and p.suffix.lower() in image_extensions
        ]
    )


st.sidebar.metric(
    "Normal Reference Images",
    len(reference_paths),
)

st.sidebar.caption(
    "These images are used as defect-free reference samples."
)


# ---------------------------------------------------------
# CHECK REFERENCE DATASET
# ---------------------------------------------------------
if len(reference_paths) < 5:

    st.warning(
        "At least 5 normal reference images are required. "
        "Please place the MVTec Bottle training images inside "
        "`data/normal_images/`."
    )

    st.stop()


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------
@st.cache_resource
def build_model(image_paths):

    reference_features = create_reference_features(
        image_paths
    )

    model, scaler = train_anomaly_model(
        reference_features
    )

    return model, scaler


with st.spinner(
    "Training AI anomaly detection model..."
):

    model, scaler = build_model(
        tuple(
            str(p)
            for p in reference_paths
        )
    )


st.success(
    f"AI model ready using "
    f"{len(reference_paths)} normal reference images."
)


# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------
st.markdown("## 📤 Upload Product Image")

uploaded_file = st.file_uploader(
    "Upload a bottle image for inspection",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)


# ---------------------------------------------------------
# NO IMAGE UPLOADED
# ---------------------------------------------------------
if uploaded_file is None:

    st.info(
        "Upload a product image above to start "
        "the AI visual inspection."
    )

    st.markdown("---")

    st.markdown(
        "### Inspection Pipeline"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.markdown(
        "**1️⃣ Image**\n\n"
        "Product image"
    )

    col2.markdown(
        "**2️⃣ Preprocess**\n\n"
        "Resize & normalize"
    )

    col3.markdown(
        "**3️⃣ Features**\n\n"
        "Visual features"
    )

    col4.markdown(
        "**4️⃣ AI Model**\n\n"
        "Anomaly detection"
    )

    col5.markdown(
        "**5️⃣ Result**\n\n"
        "PASS / REVIEW"
    )

    st.stop()


# ---------------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------------
image = Image.open(
    uploaded_file
).convert("RGB")


# ---------------------------------------------------------
# IMAGE DISPLAY
# ---------------------------------------------------------
st.markdown(
    "## 🖼️ Inspection Image"
)

col1, col2 = st.columns(2)


with col1:

    st.markdown(
        "### Original Image"
    )

    st.image(
        image,
        use_container_width=True,
    )


# ---------------------------------------------------------
# IMAGE QUALITY METRICS
# ---------------------------------------------------------
metrics = image_quality_metrics(
    image
)


with col2:

    st.markdown(
        "### Image Quality"
    )

    m1, m2 = st.columns(2)

    with m1:

        st.metric(
            "Brightness",
            f"{metrics['Brightness']:.2f}",
        )

        st.metric(
            "Contrast",
            f"{metrics['Contrast']:.2f}",
        )

    with m2:

        st.metric(
            "Sharpness",
            f"{metrics['Sharpness']:.2f}",
        )

        st.metric(
            "Edge Density",
            f"{metrics['Edge Density']:.2f}%",
        )


# ---------------------------------------------------------
# AI PREDICTION
# ---------------------------------------------------------
st.markdown("---")

st.markdown(
    "## 🤖 AI Inspection Result"
)


with st.spinner(
    "Analyzing product image..."
):

    result = predict_anomaly(
        image,
        model,
        scaler,
    )


# ---------------------------------------------------------
# GET MODEL RESULT
# ---------------------------------------------------------
anomaly_score = float(
    result["anomaly_score"]
)

prediction = result["prediction"]

model_decision = result["decision"]


# ---------------------------------------------------------
# APPLY USER THRESHOLD
# ---------------------------------------------------------
if anomaly_score >= threshold:

    final_result = "REVIEW"

else:

    final_result = "PASS"


# ---------------------------------------------------------
# RESULT DISPLAY
# ---------------------------------------------------------

if final_result == "PASS":

    st.markdown(
        f'<div class="result-pass">'
        f'<div class="result-heading">✅ PASS</div>'
        f'<div class="result-anomaly-score">'
        f'Anomaly Score: <strong>{anomaly_score:.1f}/100</strong>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        f'<div class="result-review">'
        f'<div class="result-heading">⚠️ REVIEW REQUIRED</div>'
        f'<div class="result-anomaly-score">'
        f'Anomaly Score: <strong>{anomaly_score:.1f}/100</strong>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# ANOMALY SCORE
# ---------------------------------------------------------
st.markdown(
    "### Anomaly Score"
)

st.progress(
    min(
        max(
            anomaly_score / 100,
            0.0
        ),
        1.0
    )
)

st.caption(
    f"Configured review threshold: "
    f"{threshold}/100"
)


# ---------------------------------------------------------
# VISUAL ANOMALY REPRESENTATION
# ---------------------------------------------------------
st.markdown("---")

st.markdown(
    "## 🔥 Visual Anomaly Representation"
)


try:

    anomaly_visual = generate_anomaly_map(
        image
    )

    st.image(
        anomaly_visual,
        caption=(
            "AI-generated anomaly visualization"
        ),
        use_container_width=True,
    )

except Exception as e:

    st.warning(
        "Anomaly visualization could not "
        f"be generated: {e}"
    )


# ---------------------------------------------------------
# INSPECTION SUMMARY
# ---------------------------------------------------------
st.markdown("---")

st.markdown(
    "## 📋 Inspection Summary"
)


summary_col1, summary_col2, summary_col3 = (
    st.columns(3)
)


with summary_col1:

    st.metric(
        "Inspection Result",
        final_result,
    )


with summary_col2:

    st.metric(
        "Anomaly Score",
        f"{anomaly_score:.1f}/100",
    )


with summary_col3:

    st.metric(
        "Reference Images",
        len(reference_paths),
    )


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------
st.markdown(
    "### 🤖 Model Decision"
)

st.write(
    f"Isolation Forest prediction: "
    f"**{model_decision}**"
)


# ---------------------------------------------------------
# DOWNLOAD REPORT
# ---------------------------------------------------------
report = f"""
AI VISUAL INSPECTION REPORT
============================

Product: MVTec Bottle

Inspection Result: {final_result}

Anomaly Score: {anomaly_score:.2f}/100

Review Threshold: {threshold}/100

Reference Images Used: {len(reference_paths)}

Model Decision: {model_decision}

Image Quality
-------------
Brightness: {metrics['Brightness']:.2f}
Contrast: {metrics['Contrast']:.2f}
Sharpness: {metrics['Sharpness']:.2f}
Edge Density: {metrics['Edge Density']:.2f}%

This inspection is an AI anomaly-detection prototype.
"""


st.download_button(
    "📥 Download Inspection Report",
    data=report,
    file_name="visual_inspection_report.txt",
    mime="text/plain",
)