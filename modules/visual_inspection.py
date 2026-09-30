from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image: Image.Image, size=(128, 128)):
    """
    Convert uploaded PIL image into a normalized OpenCV image.
    """

    image = image.convert("RGB")

    rgb = np.array(image)

    # Convert RGB -> BGR for OpenCV
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    resized = cv2.resize(
        bgr,
        size,
        interpolation=cv2.INTER_AREA
    )

    return resized


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image: Image.Image):
    """
    Extract lightweight visual features from the product image.

    Features include:
    - color statistics
    - brightness
    - contrast
    - edge density
    - sharpness
    - texture information
    """

    processed = preprocess_image(image)

    gray = cv2.cvtColor(
        processed,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Color statistics
    # --------------------------------------------------------

    mean_bgr = processed.mean(axis=(0, 1))
    std_bgr = processed.std(axis=(0, 1))

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    brightness = float(gray.mean())

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    contrast = float(gray.std())

    # --------------------------------------------------------
    # Sharpness
    # --------------------------------------------------------

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    sharpness = float(
        laplacian.var()
    )

    # --------------------------------------------------------
    # Edge density
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        threshold1=50,
        threshold2=150
    )

    edge_density = float(
        np.mean(edges > 0)
    )

    # --------------------------------------------------------
    # Texture
    # --------------------------------------------------------

    texture = float(
        np.mean(
            np.abs(
                gray.astype(np.float32)
                - cv2.GaussianBlur(
                    gray,
                    (5, 5),
                    0
                ).astype(np.float32)
            )
        )
    )

    features = np.array([
        mean_bgr[0],
        mean_bgr[1],
        mean_bgr[2],

        std_bgr[0],
        std_bgr[1],
        std_bgr[2],

        brightness,
        contrast,
        sharpness,
        edge_density,
        texture,
    ], dtype=np.float32)

    return features


# ============================================================
# NORMAL REFERENCE MODEL
# ============================================================

def create_reference_features(
    image_paths,
):
    """
    Extract features from a collection of normal/reference images.
    """

    feature_list = []

    for path in image_paths:

        try:
            image = Image.open(path)

            features = extract_features(image)

            feature_list.append(features)

        except Exception:
            continue

    if not feature_list:
        return np.empty((0, 11))

    return np.vstack(feature_list)


# ============================================================
# TRAIN ANOMALY MODEL
# ============================================================

def train_anomaly_model(
    reference_features,
):
    """
    Train Isolation Forest using normal/reference images.
    """

    if len(reference_features) < 5:
        return None, None

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        reference_features
    )

    model = IsolationForest(
        n_estimators=150,
        contamination=0.08,
        random_state=42
    )

    model.fit(
        scaled_features
    )

    return model, scaler


# ============================================================
# ANOMALY PREDICTION
# ============================================================

def predict_anomaly(
    image,
    model,
    scaler,
):
    """
    Predict whether the uploaded image is similar
    to the normal reference images.
    """

    features = extract_features(image)

    scaled = scaler.transform(
        features.reshape(1, -1)
    )

    prediction = model.predict(
        scaled
    )[0]

    raw_score = model.decision_function(
        scaled
    )[0]

    # Convert model score into a simple 0-100
    # anomaly score where higher = more suspicious.
    anomaly_score = float(
        np.clip(
            50 - (raw_score * 100),
            0,
            100
        )
    )

    if prediction == -1:
        decision = "REVIEW"
    else:
        decision = "PASS"

    return {
        "prediction": prediction,
        "anomaly_score": anomaly_score,
        "decision": decision,
        "features": features,
    }


# ============================================================
# IMAGE QUALITY ANALYSIS
# ============================================================

def image_quality_metrics(image):
    """
    Calculate useful image-quality indicators.
    """

    processed = preprocess_image(image)

    gray = cv2.cvtColor(
        processed,
        cv2.COLOR_BGR2GRAY
    )

    brightness = float(
        gray.mean()
    )

    contrast = float(
        gray.std()
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    edge_density = float(
        np.mean(edges > 0) * 100
    )

    return {
        "Brightness": brightness,
        "Contrast": contrast,
        "Sharpness": sharpness,
        "Edge Density": edge_density,
    }


# ============================================================
# SIMPLE VISUAL ANOMALY MAP
# ============================================================

def generate_anomaly_map(image):
    """
    Generate a simple visual representation of
    high-frequency regions in the image.

    This is an inspection aid, not a trained
    pixel-level defect segmentation model.
    """

    processed = preprocess_image(
        image,
        size=(256, 256)
    )

    gray = cv2.cvtColor(
        processed,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (11, 11),
        0
    )

    difference = cv2.absdiff(
        gray,
        blurred
    )

    normalized = cv2.normalize(
        difference,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    heatmap = cv2.applyColorMap(
        normalized.astype(np.uint8),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    original = cv2.cvtColor(
        processed,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.65,
        heatmap,
        0.35,
        0
    )

    return Image.fromarray(overlay)