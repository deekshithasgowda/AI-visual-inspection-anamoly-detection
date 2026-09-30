from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.analytics import (
    add_quality_features,
    get_alerts,
    get_kpis,
    get_weekly_trend,
    get_severity_summary,
    get_supplier_summary,
    get_batch_summary,
    get_product_summary,
    get_location_summary,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Quality Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DATA PATH
# ============================================================

DATA_PATH = Path(__file__).parent / "data" / "quality_inspections.csv"


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }

    .dashboard-title {
        font-size: 2.1rem;
        font-weight: 750;
        margin-bottom: 0.1rem;
    }

    .dashboard-subtitle {
        color: #8d93a1;
        margin-bottom: 1.2rem;
    }

    .kpi-card {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 14px;
        background: rgba(128,128,128,.06);
    }

    .alert-box {
        padding: .75rem 1rem;
        border-radius: 10px;
        margin-bottom: .55rem;
        border: 1px solid rgba(255,165,0,.25);
        background: rgba(255,165,0,.08);
    }

    .small-muted {
        color: #8d93a1;
        font-size: .88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = add_quality_features(df)
    return df


df = load_data()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">🔍 AI-Powered Product Quality Inspection & Analytics</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Stage 1 — Quality Analytics Dashboard | Synthetic portfolio dataset'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

with st.sidebar:

    st.header("🎛️ Quality Filters")

    # --------------------------------------------------------
    # Product Category
    # --------------------------------------------------------

    product_options = sorted(
        df["product_category"].dropna().unique()
    )

    selected_products = st.multiselect(
        "Product Category",
        product_options,
        default=product_options,
        key="product_filter_stage1_v2",
    )

    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    location_options = sorted(
        df["location"].dropna().unique()
    )

    selected_locations = st.multiselect(
        "Location",
        location_options,
        default=location_options,
        key="location_filter_stage1_v2",
    )

    # --------------------------------------------------------
    # Supplier
    # --------------------------------------------------------

    supplier_options = sorted(
        df["supplier_id"].dropna().unique()
    )

    selected_suppliers = st.multiselect(
        "Supplier",
        supplier_options,
        default=supplier_options,
        key="supplier_filter_stage1_v2",
    )

    # --------------------------------------------------------
    # Defect Type
    # --------------------------------------------------------

    defect_options = ["All"] + sorted(
        df["defect_type"].dropna().unique()
    )

    selected_defect = st.selectbox(
        "Defect Type",
        defect_options,
        index=0,
        key="defect_filter_stage1_v3",
    )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------
    severity_options = ["All"] + sorted(
        df["severity"].dropna().unique()
    )
    selected_severity = st.selectbox(
        "Severity",
        severity_options,
        index=0,
        key="severity_filter_stage1_v3",
    )


    # --------------------------------------------------------
    # Inspection Status
    # --------------------------------------------------------

    status_options = ["All", "PASS", "FAIL"]

    selected_status = st.selectbox(
        "Inspection Status",
        status_options,
        index=0,
        key="inspection_status_filter_stage1_v2",
    )

    # --------------------------------------------------------
    # Date Range
    # --------------------------------------------------------

    min_date = pd.to_datetime(
        df["inspection_date"]
    ).min().date()

    max_date = pd.to_datetime(
        df["inspection_date"]
    ).max().date()

    selected_dates = st.date_input(
        "Inspection Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="date_filter_stage1_v2",
    )


# ============================================================
# DATE HANDLING
# ============================================================

if (
    isinstance(selected_dates, (tuple, list))
    and len(selected_dates) == 2
):
    start_date, end_date = selected_dates
else:
    start_date = selected_dates
    end_date = selected_dates


# ============================================================
# APPLY FILTERS
# ============================================================

filtered = df.copy()


# ------------------------------------------------------------
# Product Filter
# ------------------------------------------------------------

filtered = filtered[
    filtered["product_category"].isin(selected_products)
]


# ------------------------------------------------------------
# Location Filter
# ------------------------------------------------------------

filtered = filtered[
    filtered["location"].isin(selected_locations)
]


# ------------------------------------------------------------
# Supplier Filter
# ------------------------------------------------------------

filtered = filtered[
    filtered["supplier_id"].isin(selected_suppliers)
]


# ------------------------------------------------------------
# Defect Type Filter
# ------------------------------------------------------------

# "All" means do not filter by defect type.
# This is important because PASS records have defect_type = "None".

if selected_defect != "All":
    filtered = filtered[
        filtered["defect_type"] == selected_defect
    ]


# ------------------------------------------------------------
# Severity Filter
# ------------------------------------------------------------

# ------------------------------------------------------------
# Severity Filter
# ------------------------------------------------------------

if selected_severity != "All":
    filtered = filtered[
        filtered["severity"] == selected_severity
    ]


# ------------------------------------------------------------
# Date Filter
# ------------------------------------------------------------

filtered = filtered[
    (filtered["inspection_date"].dt.date >= start_date)
    & (filtered["inspection_date"].dt.date <= end_date)
]


# ------------------------------------------------------------
# Inspection Status Filter
# ------------------------------------------------------------

if selected_status != "All":
    filtered = filtered[
        filtered["inspection_status"] == selected_status
    ]


# Make a clean copy
filtered = filtered.copy()


# ============================================================
# EMPTY DATA PROTECTION
# ============================================================

if filtered.empty:

    st.warning(
        "No inspection records match the selected filters. "
        "Please change the filters and try again."
    )

    st.stop()


# ============================================================
# KPIs
# ============================================================

kpis = get_kpis(filtered)

st.markdown("---")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)


kpi1.metric(
    "Total Inspected",
    f'{kpis["total"]:,}',
)


kpi2.metric(
    "Pass Rate",
    f'{kpis["pass_rate"]:.2f}%',
)


kpi3.metric(
    "Defect Rate",
    f'{kpis["defect_rate"]:.2f}%',
)


kpi4.metric(
    "ML Anomalies",
    f'{kpis["ml_anomalies"]:,}',
)


kpi5.metric(
    "High Severity",
    f'{kpis["high_severity"]:,}',
)


st.caption(
    f"Showing {len(filtered):,} inspection records "
    f"from {start_date} to {end_date}."
)


# ============================================================
# QUALITY ALERTS
# ============================================================

alerts = get_alerts(filtered)

st.subheader("🚨 Quality Alerts")


if alerts:

    for alert in alerts:

        st.markdown(
            f'<div class="alert-box">{alert}</div>',
            unsafe_allow_html=True,
        )

else:

    st.success(
        "No configured quality alerts were triggered "
        "for the current filters."
    )


st.divider()


# ============================================================
# MAIN ANALYTICS TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Overview",
        "🏭 Supplier & Batch",
        "📦 Product & Location",
        "📋 Inspection Records",
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    left, right = st.columns(2)


    # --------------------------------------------------------
    # Defect Distribution
    # --------------------------------------------------------

    with left:

        defect_df = (
            filtered[
                filtered["defect_type"] != "None"
            ]
            .groupby("defect_type")
            .size()
            .reset_index(name="count")
            .sort_values(
                "count",
                ascending=False,
            )
        )

        st.subheader("Defect Distribution")

        if not defect_df.empty:

            fig = px.bar(
                defect_df,
                x="defect_type",
                y="count",
                text="count",
                labels={
                    "defect_type": "Defect Type",
                    "count": "Inspections",
                },
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_layout(
                height=380,
                margin=dict(
                    l=10,
                    r=10,
                    t=30,
                    b=10,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.info(
                "No defects match the current filters."
            )


    # --------------------------------------------------------
    # Defect Severity
    # --------------------------------------------------------

    with right:

        severity_df = get_severity_summary(
            filtered
        )

        st.subheader("Defect Severity")

        severity_df = severity_df[
            severity_df["severity"] != "None"
        ]

        if not severity_df.empty:

            fig = px.bar(
                severity_df,
                x="severity",
                y="count",
                text="count",
                category_orders={
                    "severity": [
                        "High",
                        "Medium",
                        "Low",
                    ]
                },
                labels={
                    "severity": "Severity",
                    "count": "Defects",
                },
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_layout(
                height=380,
                margin=dict(
                    l=10,
                    r=10,
                    t=30,
                    b=10,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.info(
                "No severity data matches "
                "the current filters."
            )


    # --------------------------------------------------------
    # Weekly Quality Trend
    # --------------------------------------------------------

    st.subheader("📈 Weekly Quality Trend")

    weekly = get_weekly_trend(filtered)

    if not weekly.empty:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=weekly["week"],
                y=weekly["defect_rate"],
                mode="lines+markers",
                name="Defect Rate",
                line=dict(width=3),
                hovertemplate=(
                    "%{x}<br>"
                    "Defect Rate: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            xaxis_title="Week",
            yaxis_title="Defect Rate (%)",
            height=420,
            hovermode="x unified",
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # Pareto Analysis
    # --------------------------------------------------------

    st.subheader("📌 Defect Pareto Analysis")

    pareto = (
        filtered[
            filtered["defect_type"] != "None"
        ]
        .groupby("defect_type")
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
    )

    if not pareto.empty:

        pareto["cumulative_pct"] = (
            pareto["count"].cumsum()
            / pareto["count"].sum()
            * 100
        )

        fig = go.Figure()


        fig.add_trace(
            go.Bar(
                x=pareto["defect_type"],
                y=pareto["count"],
                name="Defect Count",
            )
        )


        fig.add_trace(
            go.Scatter(
                x=pareto["defect_type"],
                y=pareto["cumulative_pct"],
                name="Cumulative %",
                yaxis="y2",
                mode="lines+markers",
            )
        )


        fig.update_layout(
            yaxis=dict(
                title="Defect Count"
            ),
            yaxis2=dict(
                title="Cumulative %",
                overlaying="y",
                side="right",
                range=[0, 105],
            ),
            height=420,
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# TAB 2 — SUPPLIER & BATCH
# ============================================================

with tab2:

    st.subheader("🏭 Supplier Quality Analysis")

    supplier_df = get_supplier_summary(
        filtered
    )

    if not supplier_df.empty:

        fig = px.bar(
            supplier_df.sort_values(
                "defect_rate",
                ascending=False,
            ),
            x="supplier_id",
            y="defect_rate",
            text="defect_rate",
            labels={
                "supplier_id": "Supplier",
                "defect_rate": "Defect Rate (%)",
            },
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )

        fig.update_layout(
            height=400,
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.dataframe(
            supplier_df,
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------------
    # Batch Quality
    # --------------------------------------------------------

    st.subheader("🏷️ Batch Quality Analysis")

    batch_df = get_batch_summary(
        filtered
    )

    if not batch_df.empty:

        top_batches = batch_df.head(15)

        fig = px.bar(
            top_batches,
            x="batch_id",
            y="defect_rate",
            text="defect_rate",
            labels={
                "batch_id": "Batch",
                "defect_rate": "Defect Rate (%)",
            },
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )

        fig.update_layout(
            height=430,
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.dataframe(
            batch_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TAB 3 — PRODUCT & LOCATION
# ============================================================

with tab3:

    left, right = st.columns(2)


    # --------------------------------------------------------
    # Product Quality
    # --------------------------------------------------------

    with left:

        st.subheader("📦 Product Quality")

        product_df = get_product_summary(
            filtered
        )

        if not product_df.empty:

            fig = px.bar(
                product_df.sort_values(
                    "defect_rate",
                    ascending=False,
                ),
                x="product_category",
                y="defect_rate",
                text="defect_rate",
                labels={
                    "product_category": "Product",
                    "defect_rate": "Defect Rate (%)",
                },
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
            )

            fig.update_layout(
                height=400,
                margin=dict(
                    l=10,
                    r=10,
                    t=30,
                    b=10,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


    # --------------------------------------------------------
    # Location Quality
    # --------------------------------------------------------

    with right:

        st.subheader("📍 Location Quality")

        location_df = get_location_summary(
            filtered
        )

        if not location_df.empty:

            fig = px.bar(
                location_df.sort_values(
                    "defect_rate",
                    ascending=False,
                ),
                x="location",
                y="defect_rate",
                text="defect_rate",
                labels={
                    "location": "Location",
                    "defect_rate": "Defect Rate (%)",
                },
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
            )

            fig.update_layout(
                height=400,
                margin=dict(
                    l=10,
                    r=10,
                    t=30,
                    b=10,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


    # --------------------------------------------------------
    # High Severity Review
    # --------------------------------------------------------

    st.subheader(
        "🔎 High-Severity Inspection Review"
    )

    high = (
        filtered[
            filtered["severity"] == "High"
        ]
        .sort_values(
            "inspection_date",
            ascending=False,
        )
    )

    if high.empty:

        st.success(
            "No high-severity records match "
            "the current filters."
        )

    else:

        st.dataframe(
            high[
                [
                    "inspection_id",
                    "inspection_date",
                    "product_category",
                    "batch_id",
                    "supplier_id",
                    "location",
                    "defect_type",
                    "severity",
                    "inspection_status",
                    "max_dimension_deviation_pct",
                    "surface_anomaly_score",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TAB 4 — INSPECTION RECORDS
# ============================================================

with tab4:

    st.subheader("📋 Inspection Records")

    display_df = (
        filtered
        .sort_values(
            "inspection_date",
            ascending=False,
        )
        .copy()
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    csv = display_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered Inspection Data",
        data=csv,
        file_name="quality_inspections_filtered.csv",
        mime="text/csv",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="small-muted">'
    'Portfolio project • Synthetic data • Stage 1 Quality Analytics'
    '</div>',
    unsafe_allow_html=True,
)