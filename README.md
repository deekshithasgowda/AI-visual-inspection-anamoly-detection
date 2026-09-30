# AI-Powered Product Quality Inspection & Analytics System

An industry-inspired portfolio project for product quality monitoring, inspection analytics,
anomaly detection, and visual inspection prototyping.

## Features
- Quality-control dashboard
- Batch/product/supplier/location analysis
- Defect Pareto analysis
- Anomaly detection using Isolation Forest
- Product inspection form with tolerance checks
- Image-based visual anomaly prototype using OpenCV
- CSV export of filtered inspection data
- Synthetic dataset generator

> The included data is synthetic and is not from Wakefit or any other company.
> The computer-vision module is a prototype for visual anomaly screening, not a certified
> manufacturing inspection system.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL shown by Streamlit, usually http://localhost:8501.

## Project structure

```text
AI_Product_Quality_Inspection/
├── app.py
├── generate_data.py
├── requirements.txt
├── README.md
├── data/
│   └── quality_inspections.csv
└── modules/
    ├── __init__.py
    ├── analytics.py
    └── vision.py
```
