from PIL import Image
import cv2
import numpy as np


def visual_inspection(uploaded_file):
    """
    Prototype visual anomaly screening.

    It measures image-level properties (edge density, brightness variation,
    colour variation) and converts them into a screening score. It does not
    identify a manufacturing defect category by itself.
    """
    image = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())

    brightness_std = float(gray.std())
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation_std = float(hsv[:, :, 1].std())

    # Normalized heuristic components for a portfolio prototype.
    edge_component = np.clip(edge_density / 0.20, 0, 1)
    brightness_component = np.clip(brightness_std / 90, 0, 1)
    saturation_component = np.clip(saturation_std / 90, 0, 1)

    anomaly_score = float(
        0.45 * edge_component
        + 0.30 * brightness_component
        + 0.25 * saturation_component
    )

    if anomaly_score >= 0.70:
        status = "High visual anomaly — manual inspection recommended"
    elif anomaly_score >= 0.45:
        status = "Moderate visual anomaly — review recommended"
    else:
        status = "Low visual anomaly"

    return {
        "image": image,
        "edge_density": edge_density,
        "brightness_std": brightness_std,
        "saturation_std": saturation_std,
        "anomaly_score": anomaly_score,
        "status": status,
        "edge_image": Image.fromarray(edges),
    }
