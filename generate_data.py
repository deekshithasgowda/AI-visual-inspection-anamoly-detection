import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

N = 2500
products = ["Mattress", "Sofa", "Bed", "Chair", "Pillow"]
locations = ["Bengaluru", "Hyderabad", "Chennai", "Pune", "Mumbai"]
suppliers = ["SUP-01", "SUP-02", "SUP-03", "SUP-04", "SUP-05"]
defect_types = ["Surface", "Dimension", "Stitching", "Material", "Packaging", "None"]

dates = pd.date_range("2026-01-01", "2026-09-29", periods=N)
product = rng.choice(products, N)
location = rng.choice(locations, N)
supplier = rng.choice(suppliers, N)
batch_num = rng.integers(100, 151, N)
batch = np.array([f"B{n}" for n in batch_num])

# A few batches/suppliers are intentionally made more defect-prone so
# the analytics layer has meaningful patterns to discover.
risk = np.zeros(N)
risk += np.where(batch_num >= 140, 0.18, 0)
risk += np.where(supplier == "SUP-03", 0.12, 0)
risk += np.where(location == "Hyderabad", 0.05, 0)

base_defect_prob = 0.12 + risk
is_defect = rng.random(N) < base_defect_prob

defect = np.where(
    is_defect,
    rng.choice(defect_types[:-1], N, p=[0.25, 0.22, 0.20, 0.20, 0.13]),
    "None",
)

severity = np.where(
    defect == "None",
    "None",
    rng.choice(["Low", "Medium", "High"], N, p=[0.45, 0.40, 0.15]),
)

expected_length = rng.normal(200, 3, N)
expected_width = rng.normal(150, 2, N)
expected_height = rng.normal(20, 1, N)

length = expected_length + rng.normal(0, 0.55, N)
width = expected_width + rng.normal(0, 0.45, N)
height = expected_height + rng.normal(0, 0.30, N)

# Dimension defects are deliberately given larger deviations.
dim_mask = defect == "Dimension"
length[dim_mask] += rng.normal(0, 2.2, dim_mask.sum())
width[dim_mask] += rng.normal(0, 1.8, dim_mask.sum())
height[dim_mask] += rng.normal(0, 1.0, dim_mask.sum())

surface_score = np.clip(rng.normal(0.18, 0.07, N), 0, 1)
surface_score[defect == "Surface"] += rng.uniform(0.25, 0.55, (defect == "Surface").sum())
surface_score = np.clip(surface_score, 0, 1)

inspection_status = np.where(is_defect, "FAIL", "PASS")

df = pd.DataFrame({
    "inspection_id": [f"INS-{i:05d}" for i in range(1, N + 1)],
    "inspection_date": dates.strftime("%Y-%m-%d"),
    "product_category": product,
    "batch_id": batch,
    "supplier_id": supplier,
    "location": location,
    "length_cm": np.round(length, 2),
    "width_cm": np.round(width, 2),
    "height_cm": np.round(height, 2),
    "expected_length_cm": np.round(expected_length, 2),
    "expected_width_cm": np.round(expected_width, 2),
    "expected_height_cm": np.round(expected_height, 2),
    "surface_anomaly_score": np.round(surface_score, 3),
    "defect_type": defect,
    "severity": severity,
    "inspection_status": inspection_status,
})

out = Path(__file__).parent / "data" / "quality_inspections.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Generated {len(df)} inspections at {out}")
