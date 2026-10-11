from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = ROOT / "data" / "predictions" / "xgboost_predictions.csv"
EVALUATION_PATH = ROOT / "reports" / "evaluation_report.txt"
BASELINE_PATH = ROOT / "reports" / "baseline_comparison.txt"

st.set_page_config(
    page_title="Gulf Energy AI | Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp { background: #f5f8fc; color: #172b4d; }
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e4ebf5;
}
[data-testid="stSidebar"] * { color: #243b5a; }
.hero {
    background: linear-gradient(120deg, #123c75 0%, #1768ac 65%, #0b9c83 100%);
    color: white;
    padding: 30px 34px;
    border-radius: 20px;
    margin: 0 0 24px 0;
    box-shadow: 0 12px 30px rgba(24, 76, 130, .12);
}
.hero h1 { color: white; font-size: 32px; margin: 0 0 8px 0; }
.hero p { color: #e4f2ff; font-size: 14px; margin: 0; }
.eyebrow {
    color: #d0f9ec;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 11px;
    font-weight: 700;
    margin-bottom: 12px;
}
.metric-card {
    background: white;
    padding: 20px;
    border: 1px solid #e4ebf5;
    border-radius: 16px;
    min-height: 116px;
    box-shadow: 0 4px 16px rgba(26, 56, 95, .035);
}
.metric-label { color: #64748b; font-size: 12px; font-weight: 600; }
.metric-value { color: #163c70; font-size: 27px; font-weight: 800; margin: 9px 0 4px 0; }
.metric-note { color: #0a9478; font-size: 12px; }
.section-title { font-size: 21px; font-weight: 750; color: #173b69; margin: 14px 0 4px; }
.section-note { color: #718096; font-size: 13px; margin-bottom: 14px; }
.notice {
    background: #eaf4ff;
    border-left: 4px solid #2878cc;
    padding: 13px 16px;
    border-radius: 8px;
    color: #244b77;
    font-size: 13px;
    margin: 16px 0;
}
.footer {
    text-align: center;
    color: #8391a7;
    border-top: 1px solid #e4ebf5;
    padding-top: 20px;
    margin-top: 30px;
    font-size: 12px;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e4ebf5;
    padding: 16px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_predictions(path_string, modified_time):
    df = pd.read_csv(path_string)
    required = {"Target_Datetime", "Actual_MW", "Predicted_MW"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")

    df["Target_Datetime"] = pd.to_datetime(df["Target_Datetime"], errors="coerce")
    for col in ["Actual_MW", "Predicted_MW"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Target_Datetime", "Actual_MW", "Predicted_MW"])
    df = df.sort_values("Target_Datetime").reset_index(drop=True)
    df["Error_MW"] = df["Predicted_MW"] - df["Actual_MW"]
    df["Absolute_Error_MW"] = df["Error_MW"].abs()
    df["Hour"] = df["Target_Datetime"].dt.hour
    df["Month"] = df["Target_Datetime"].dt.strftime("%b")
    df["Day"] = df["Target_Datetime"].dt.date.astype(str)
    return df


def read_report(path):
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    return "Report not found."


def metric_card(label, value, note):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


if not PREDICTIONS_PATH.exists():
    st.error(f"Prediction file not found: {PREDICTIONS_PATH}")
    st.info("Generate predictions using the existing project pipeline, then refresh this page.")
    st.stop()

try:
    predictions = load_predictions(
        str(PREDICTIONS_PATH),
        PREDICTIONS_PATH.stat().st_mtime,
    )
except Exception as exc:
    st.error(f"Could not load prediction data: {exc}")
    st.stop()

if predictions.empty:
    st.error("The prediction file contains no valid rows.")
    st.stop()

mae = predictions["Absolute_Error_MW"].mean()
rmse = (predictions["Error_MW"].pow(2).mean()) ** 0.5
actual_mean = predictions["Actual_MW"].mean()
ss_total = ((predictions["Actual_MW"] - actual_mean) ** 2).sum()
ss_residual = predictions["Error_MW"].pow(2).sum()
r2 = 1 - ss_residual / ss_total if ss_total else float("nan")

st.sidebar.markdown("## ⚡ Gulf Energy AI")
st.sidebar.caption("ENERGY INTELLIGENCE PLATFORM")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Forecast Explorer", "Model Performance", "Data Explorer", "About Project"],
)
st.sidebar.divider()
st.sidebar.markdown("**Model status**")
st.sidebar.success("Prediction artifact loaded")
st.sidebar.caption(f"{len(predictions):,} valid prediction rows")
st.sidebar.caption("Historical offline evaluation — not a live feed.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">Energy intelligence · ML forecasting</div>
  <h1>Energy Forecasting Dashboard</h1>
  <p>Explore electricity demand, forecast accuracy and model performance in one place.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="notice"><b>Dataset transparency:</b> This project currently uses '
    'the historical US AEP electricity-demand dataset. Values shown are offline '
    'evaluation results, not live Gulf-region energy measurements.</div>',
    unsafe_allow_html=True,
)

st.markdown(f'<div class="section-title">{page}</div>', unsafe_allow_html=True)

if page == "Overview":
    st.markdown(
        '<div class="section-note">A high-level view of the existing model evaluation.</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("PREDICTIONS", f"{len(predictions):,}", "Evaluated records")
    with c2:
        metric_card("MEAN ABSOLUTE ERROR", f"{mae:,.1f} MW", "Lower is better")
    with c3:
        metric_card("ROOT MEAN SQUARED ERROR", f"{rmse:,.1f} MW", "Penalizes large errors")
    with c4:
        metric_card("R-SQUARED", f"{r2:.4f}", "Explained variance score")

    st.write("")
    left, right = st.columns([1.65, 1])
    with left:
        st.markdown("#### Actual demand vs model prediction")
        sample = predictions
        if len(sample) > 2500:
            sample = sample.iloc[::max(1, len(sample) // 2500)]
        chart_data = sample.melt(
            id_vars="Target_Datetime",
            value_vars=["Actual_MW", "Predicted_MW"],
            var_name="Series",
            value_name="Demand_MW",
        )
        fig = px.line(
            chart_data,
            x="Target_Datetime",
            y="Demand_MW",
            color="Series",
            color_discrete_map={
                "Actual_MW": "#2563eb",
                "Predicted_MW": "#10a37f",
            },
            labels={
                "Target_Datetime": "Target time",
                "Demand_MW": "Electricity demand (MW)",
                "Series": "Series",
            },
        )
        fig.update_layout(
            template="plotly_white",
            height=390,
            margin=dict(l=10, r=10, t=20, b=10),
            legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Prediction error distribution")
        fig = px.histogram(
            predictions,
            x="Error_MW",
            nbins=45,
            color_discrete_sequence=["#2878cc"],
            labels={"Error_MW": "Predicted − actual (MW)"},
        )
        fig.update_layout(
            template="plotly_white",
            height=390,
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Latest available evaluation records")
    st.dataframe(
        predictions[["Target_Datetime", "Actual_MW", "Predicted_MW", "Absolute_Error_MW"]]
        .tail(10)
        .sort_values("Target_Datetime", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

elif page == "Forecast Explorer":
    st.markdown(
        '<div class="section-note">Explore predictions from the saved historical evaluation file.</div>',
        unsafe_allow_html=True,
    )
    min_date = predictions["Target_Datetime"].min().date()
    max_date = predictions["Target_Datetime"].max().date()
    date_range = st.date_input(
        "Select target date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start, end = date_range
        view = predictions[
            predictions["Target_Datetime"].dt.date.between(start, end)
        ]
    else:
        view = predictions[predictions["Target_Datetime"].dt.date == date_range]

    if view.empty:
        st.warning("No predictions found for this date range.")
    else:
        a, b, c = st.columns(3)
        a.metric("Records in range", f"{len(view):,}")
        b.metric("MAE in range", f"{view['Absolute_Error_MW'].mean():,.2f} MW")
        c.metric("Maximum absolute error", f"{view['Absolute_Error_MW'].max():,.2f} MW")

        chart_data = view.melt(
            id_vars="Target_Datetime",
            value_vars=["Actual_MW", "Predicted_MW"],
            var_name="Series",
            value_name="Demand_MW",
        )
        fig = px.line(
            chart_data,
            x="Target_Datetime",
            y="Demand_MW",
            color="Series",
            color_discrete_map={"Actual_MW": "#2563eb", "Predicted_MW": "#10a37f"},
            labels={"Target_Datetime": "Target time", "Demand_MW": "Demand (MW)", "Series": ""},
        )
        fig.update_layout(template="plotly_white", height=460)
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download selected predictions as CSV",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="gulf_energy_ai_selected_predictions.csv",
            mime="text/csv",
        )

elif page == "Model Performance":
    st.markdown(
        '<div class="section-note">Metrics recalculated from the saved prediction artifact.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.markdown("#### Evaluation report")
        st.code(read_report(EVALUATION_PATH), language="text")
    with right:
        st.markdown("#### Baseline comparison")
        st.code(read_report(BASELINE_PATH), language="text")

    error_summary = pd.DataFrame({
        "Metric": ["MAE (MW)", "RMSE (MW)", "Maximum absolute error (MW)", "Mean signed error (MW)", "R-squared"],
        "Value": [
            mae,
            rmse,
            predictions["Absolute_Error_MW"].max(),
            predictions["Error_MW"].mean(),
            r2,
        ],
    })
    st.dataframe(error_summary, use_container_width=True, hide_index=True)

    st.markdown("#### Largest prediction errors")
    st.dataframe(
        predictions.nlargest(10, "Absolute_Error_MW")[
            ["Target_Datetime", "Actual_MW", "Predicted_MW", "Error_MW", "Absolute_Error_MW"]
        ],
        use_container_width=True,
        hide_index=True,
    )

elif page == "Data Explorer":
    st.markdown(
        '<div class="section-note">Explore historical actual demand and prediction errors.</div>',
        unsafe_allow_html=True,
    )
    low, high = st.columns(2)
    low_value = float(predictions["Actual_MW"].min())
    high_value = float(predictions["Actual_MW"].max())
    demand_range = st.slider(
        "Filter actual demand (MW)",
        min_value=low_value,
        max_value=high_value,
        value=(low_value, high_value),
    )
    view = predictions[predictions["Actual_MW"].between(*demand_range)]

    a, b, c = st.columns(3)
    a.metric("Filtered records", f"{len(view):,}")
    b.metric("Average actual demand", f"{view['Actual_MW'].mean():,.1f} MW" if not view.empty else "—")
    c.metric("Average absolute error", f"{view['Absolute_Error_MW'].mean():,.1f} MW" if not view.empty else "—")

    if not view.empty:
        fig = px.scatter(
            view,
            x="Actual_MW",
            y="Predicted_MW",
            color="Absolute_Error_MW",
            color_continuous_scale=["#a7f3d0", "#0d9488", "#1d4ed8"],
            hover_data=["Target_Datetime", "Error_MW"],
            labels={
                "Actual_MW": "Actual demand (MW)",
                "Predicted_MW": "Predicted demand (MW)",
                "Absolute_Error_MW": "Absolute error (MW)",
            },
        )
        fig.update_layout(template="plotly_white", height=470)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Filtered records")
        st.dataframe(
            view[["Target_Datetime", "Actual_MW", "Predicted_MW", "Error_MW", "Absolute_Error_MW"]],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download filtered data",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="gulf_energy_ai_filtered_data.csv",
            mime="text/csv",
        )
    else:
        st.info("No rows match the selected demand range.")

elif page == "About Project":
    st.markdown("""
### About Gulf Energy AI

A machine-learning energy-demand forecasting project with a focus on
reproducible data preparation, evaluation and future deployment.

**Current model**
- Algorithm: XGBoost Regressor
- Forecast horizon: next hour in the existing prediction pipeline
- Output unit: megawatts (MW)

**Current technology stack**
- Python, Pandas, XGBoost
- Streamlit and Plotly for the dashboard
- DVC and MLflow are intended for data and experiment management
- FastAPI, Docker and monitoring are planned integration stages

**Dataset transparency**
The current artifact is based on historical AEP electricity-demand data
from the United States. This dashboard does not show live energy data and
does not represent measured demand in the Gulf region.

**Production status**
This is a project dashboard for historical model evaluation. Time-series
cross-validation, data-gap investigation, deployment checks and monitoring
must be completed before describing the system as production-ready.
""")

st.markdown(
    '<div class="footer">Gulf Energy AI · Forecasting & Model Analytics · '
    'Historical offline evaluation · Built with Streamlit and Plotly</div>',
    unsafe_allow_html=True,
)
