import pandas as pd
from sklearn.ensemble import IsolationForest


def add_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["inspection_date"] = pd.to_datetime(out["inspection_date"])

    out["length_deviation_pct"] = (
        (out["length_cm"] - out["expected_length_cm"]).abs()
        / out["expected_length_cm"] * 100
    )

    out["width_deviation_pct"] = (
        (out["width_cm"] - out["expected_width_cm"]).abs()
        / out["expected_width_cm"] * 100
    )

    out["height_deviation_pct"] = (
        (out["height_cm"] - out["expected_height_cm"]).abs()
        / out["expected_height_cm"] * 100
    )

    out["max_dimension_deviation_pct"] = out[
        [
            "length_deviation_pct",
            "width_deviation_pct",
            "height_deviation_pct",
        ]
    ].max(axis=1)

    out["rule_based_anomaly"] = (
        (out["surface_anomaly_score"] > 0.55)
        | (out["max_dimension_deviation_pct"] > 2.0)
    )

    return out


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if len(out) < 10:
        out["ml_anomaly"] = "Normal"
        return out

    features = [
        "length_deviation_pct",
        "width_deviation_pct",
        "height_deviation_pct",
        "surface_anomaly_score",
    ]

    model = IsolationForest(
        n_estimators=150,
        contamination=0.08,
        random_state=42,
    )

    out["ml_prediction"] = model.fit_predict(out[features])
    out["ml_anomaly"] = out["ml_prediction"].map(
        {1: "Normal", -1: "Anomaly"}
    )

    return out


def get_kpis(df: pd.DataFrame) -> dict:
    total = len(df)

    if total == 0:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0,
            "defect_rate": 0,
            "ml_anomalies": 0,
            "high_severity": 0,
        }

    passed = int((df["inspection_status"] == "PASS").sum())
    failed = int((df["inspection_status"] == "FAIL").sum())
    high_severity = int((df["severity"] == "High").sum())

    ml_df = detect_anomalies(df)
    ml_anomalies = int((ml_df["ml_anomaly"] == "Anomaly").sum())

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total * 100,
        "defect_rate": failed / total * 100,
        "ml_anomalies": ml_anomalies,
        "high_severity": high_severity,
    }


def get_weekly_trend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["week", "defect_rate"])

    work = df.copy()
    work["inspection_date"] = pd.to_datetime(work["inspection_date"])
    work["week"] = work["inspection_date"].dt.to_period("W").apply(
        lambda x: x.start_time
    )

    result = (
        work.groupby("week")
        .agg(
            inspected=("inspection_id", "count"),
            failed=("inspection_status", lambda x: (x == "FAIL").sum()),
        )
        .reset_index()
    )

    result["defect_rate"] = (
        result["failed"] / result["inspected"] * 100
    )

    return result


def get_severity_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("severity")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def get_supplier_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "supplier_id",
                "inspected",
                "failed",
                "defect_rate",
            ]
        )

    result = (
        df.groupby("supplier_id")
        .agg(
            inspected=("inspection_id", "count"),
            failed=("inspection_status", lambda x: (x == "FAIL").sum()),
        )
        .reset_index()
    )

    result["defect_rate"] = result["failed"] / result["inspected"] * 100

    return result.sort_values("defect_rate", ascending=False)


def get_batch_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "batch_id",
                "inspected",
                "failed",
                "defect_rate",
            ]
        )

    result = (
        df.groupby("batch_id")
        .agg(
            inspected=("inspection_id", "count"),
            failed=("inspection_status", lambda x: (x == "FAIL").sum()),
        )
        .reset_index()
    )

    result["defect_rate"] = result["failed"] / result["inspected"] * 100

    return result.sort_values("defect_rate", ascending=False)


def get_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    result = (
        df.groupby("product_category")
        .agg(
            inspected=("inspection_id", "count"),
            failed=("inspection_status", lambda x: (x == "FAIL").sum()),
        )
        .reset_index()
    )

    result["defect_rate"] = result["failed"] / result["inspected"] * 100

    return result.sort_values("defect_rate", ascending=False)


def get_location_summary(df: pd.DataFrame) -> pd.DataFrame:
    result = (
        df.groupby("location")
        .agg(
            inspected=("inspection_id", "count"),
            failed=("inspection_status", lambda x: (x == "FAIL").sum()),
        )
        .reset_index()
    )

    result["defect_rate"] = result["failed"] / result["inspected"] * 100

    return result.sort_values("defect_rate", ascending=False)


def get_alerts(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []

    alerts = []
    overall_rate = (df["inspection_status"] == "FAIL").mean() * 100

    # High severity alert
    high_count = int((df["severity"] == "High").sum())
    if high_count > 0:
        alerts.append(
            f"⚠️ {high_count} high-severity inspection(s) require review."
        )

    # Supplier alert
    supplier = get_supplier_summary(df)
    if not supplier.empty:
        supplier_threshold = overall_rate + 5
        risky = supplier[supplier["defect_rate"] > supplier_threshold]

        for _, row in risky.head(3).iterrows():
            alerts.append(
                f"⚠️ {row['supplier_id']} has a defect rate of "
                f"{row['defect_rate']:.1f}%, above the configured "
                f"investigation threshold of {supplier_threshold:.1f}%."
            )

    # Batch alert
    batch = get_batch_summary(df)
    if not batch.empty:
        batch_threshold = overall_rate + 8
        risky_batches = batch[
            batch["defect_rate"] > batch_threshold
        ]

        for _, row in risky_batches.head(3).iterrows():
            alerts.append(
                f"⚠️ {row['batch_id']} has a defect rate of "
                f"{row['defect_rate']:.1f}%, above the configured "
                f"investigation threshold of {batch_threshold:.1f}%."
            )

    # Dimension alert
    dimension_alerts = int(
        (df["max_dimension_deviation_pct"] > 2.0).sum()
    )
    if dimension_alerts:
        alerts.append(
            f"⚠️ {dimension_alerts} inspection(s) exceed the "
            f"configured 2% maximum dimension deviation."
        )

    return alerts
