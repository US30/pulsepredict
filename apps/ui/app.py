import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import httpx
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PulsePredict",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE_URL = "http://localhost:8000"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

AVAILABLE_MODELS = ["patchtst", "chronos", "nhits", "nbeats", "statsforecast-ets"]
M5_ITEM_IDS = [
    "FOODS_1_001_CA_1",
    "FOODS_1_002_CA_1",
    "HOBBIES_1_001_CA_1",
    "HOUSEHOLD_1_001_CA_1",
]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _plotly_forecast_chart(
    history: list[float],
    forecast: list[float],
    q10: list[float],
    q90: list[float],
    title: str = "Forecast",
) -> go.Figure:
    """Build a Plotly figure with history line, forecast line, and PI ribbon."""
    h_idx = list(range(len(history)))
    f_idx = list(range(len(history), len(history) + len(forecast)))

    fig = go.Figure()

    # Historical
    fig.add_trace(
        go.Scatter(
            x=h_idx, y=history, mode="lines", name="History",
            line=dict(color="#4C9BE8", width=2),
        )
    )
    # 90% PI ribbon
    if q10 and q90:
        fig.add_trace(
            go.Scatter(
                x=f_idx + f_idx[::-1],
                y=q90 + q10[::-1],
                fill="toself",
                fillcolor="rgba(255, 165, 0, 0.20)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="90% PI",
            )
        )
    # Forecast mean
    fig.add_trace(
        go.Scatter(
            x=f_idx, y=forecast, mode="lines", name="Forecast",
            line=dict(color="#FF7F0E", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time step",
        yaxis_title="Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _load_json(path: Path) -> dict | list | None:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Page implementations
# ---------------------------------------------------------------------------

def page_forecast_explorer() -> None:
    st.title("Forecast Explorer")
    st.markdown("Upload your own series or select an M5 item to generate a probabilistic forecast.")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        data_source = st.radio("Data source", ["Upload CSV", "M5 Item"])

        history: list[float] = []

        if data_source == "Upload CSV":
            uploaded = st.file_uploader("CSV with a single 'value' column", type=["csv"])
            if uploaded:
                df = pd.read_csv(uploaded)
                if "value" in df.columns:
                    history = df["value"].dropna().tolist()
                    st.success(f"Loaded {len(history)} observations.")
                else:
                    st.error("CSV must contain a 'value' column.")
        else:
            item_id = st.selectbox("M5 Item", M5_ITEM_IDS)
            # Synthetic placeholder — replace with actual M5 loader
            rng = np.random.default_rng(abs(hash(item_id)) % (2**32))
            history = (rng.poisson(lam=5, size=56) * (1 + 0.3 * rng.random(56))).tolist()
            st.info(f"Using synthetic data for {item_id} ({len(history)} obs).")

        model_choice = st.selectbox("Model", AVAILABLE_MODELS)
        horizon = st.slider("Horizon (days)", min_value=7, max_value=90, value=28, step=7)
        return_quantiles = st.checkbox("Return 90% Prediction Interval", value=True)
        unique_id = st.text_input("Series ID", value="series_001")

        run = st.button("Run Forecast", type="primary", disabled=len(history) == 0)

    with col_right:
        if run and history:
            payload = {
                "unique_id": unique_id,
                "history": history,
                "horizon": horizon,
                "model": model_choice,
                "return_quantiles": return_quantiles,
            }
            with st.spinner("Calling forecast API..."):
                try:
                    resp = httpx.post(
                        f"{API_BASE_URL}/forecast",
                        json=payload,
                        timeout=30.0,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    st.success(f"Latency: {result['latency_ms']:.1f} ms  |  Model: {result['model']}")
                    fig = _plotly_forecast_chart(
                        history=history,
                        forecast=result["forecasts"],
                        q10=result.get("q10", []),
                        q90=result.get("q90", []),
                        title=f"{unique_id} — {model_choice} (H={horizon})",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("Raw response"):
                        st.json(result)
                except httpx.ConnectError:
                    st.error(f"Could not reach API at {API_BASE_URL}. Is the server running?")
                except Exception as exc:
                    st.error(f"Request failed: {exc}")
        elif not run:
            st.info("Configure the inputs on the left and click **Run Forecast**.")


def page_backtest_results() -> None:
    st.title("Backtest Results")

    metrics_path = REPORTS_DIR / "backtest" / "metrics.json"
    data = _load_json(metrics_path)

    if data is None:
        st.warning(f"No backtest metrics found at `{metrics_path}`. Run `make backtest` first.")
        return

    # Expect structure: {model: {MAE: float, MASE: float, Coverage: float}, ...}
    df = pd.DataFrame(data).T.reset_index().rename(columns={"index": "Model"})
    numeric_cols = [c for c in df.columns if c != "Model"]

    st.subheader("Metrics table")
    st.dataframe(df.style.highlight_min(subset=["MAE", "MASE"], color="#d4f1d4"), use_container_width=True)

    st.subheader("MAE / MASE / Coverage by model")
    for metric in ["MAE", "MASE", "Coverage"]:
        if metric not in df.columns:
            continue
        fig = go.Figure(
            go.Bar(x=df["Model"], y=df[metric], name=metric, marker_color="#4C9BE8")
        )
        fig.update_layout(title=metric, xaxis_title="Model", yaxis_title=metric)
        st.plotly_chart(fig, use_container_width=True)


def page_conformal_coverage() -> None:
    st.title("Conformal Coverage")

    cov_path = REPORTS_DIR / "conformal_coverage.json"
    data = _load_json(cov_path)

    if data is None:
        st.warning(f"No conformal coverage report found at `{cov_path}`. Run `make conformal` first.")
        return

    # Expected keys: alpha_levels, achieved_coverage, pi_width_over_time, timestamps
    alpha_levels: list[float] = data.get("alpha_levels", [])
    achieved: list[float] = data.get("achieved_coverage", [])
    pi_width: list[float] = data.get("pi_width_over_time", [])
    timestamps: list[str] = data.get("timestamps", list(range(len(pi_width))))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Coverage vs Target Alpha")
        if alpha_levels and achieved:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=alpha_levels, y=[1 - a for a in alpha_levels],
                                     mode="lines", name="Ideal (1-alpha)",
                                     line=dict(color="gray", dash="dot")))
            fig.add_trace(go.Scatter(x=alpha_levels, y=achieved, mode="lines+markers",
                                     name="Achieved coverage", line=dict(color="#4C9BE8")))
            fig.update_layout(xaxis_title="Alpha", yaxis_title="Coverage",
                              yaxis=dict(range=[0, 1.05]))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("PI Width Over Time")
        if pi_width:
            fig = go.Figure(go.Scatter(x=list(range(len(pi_width))), y=pi_width,
                                       mode="lines", name="PI width",
                                       line=dict(color="#FF7F0E")))
            fig.update_layout(xaxis_title="Time step", yaxis_title="PI width")
            st.plotly_chart(fig, use_container_width=True)


def page_hierarchy_view() -> None:
    st.title("Hierarchy View")

    reconcile_path = REPORTS_DIR / "reconciliation" / "summary.json"
    data = _load_json(reconcile_path)

    if data is None:
        st.warning(f"No reconciliation summary found at `{reconcile_path}`. Run `make reconcile` first.")
        return

    # Expected: {levels: [{name, mae, mase}, ...]}
    levels = data.get("levels", [])
    if not levels:
        st.error("Reconciliation summary is empty or malformed.")
        return

    df = pd.DataFrame(levels)
    st.subheader("Error by Hierarchy Level")
    st.dataframe(df, use_container_width=True)

    fig = go.Figure()
    for metric in ["mae", "mase"]:
        if metric in df.columns:
            fig.add_trace(go.Bar(x=df["name"], y=df[metric], name=metric.upper()))
    fig.update_layout(barmode="group", xaxis_title="Hierarchy Level",
                      yaxis_title="Error metric", title="Reconciliation Error by Level")
    st.plotly_chart(fig, use_container_width=True)


def page_intervention_analysis() -> None:
    st.title("Intervention Analysis")
    st.markdown("Causal impact case study — synthetic intervention injected into a series.")

    intervention_dir = REPORTS_DIR / "intervention"
    summary_path = intervention_dir / "summary.json"
    data = _load_json(summary_path)

    if data is None:
        st.warning(f"No intervention report found at `{summary_path}`. Run `make intervention` first.")
        return

    pre: list[float] = data.get("pre_period", [])
    post_actual: list[float] = data.get("post_actual", [])
    post_counterfactual: list[float] = data.get("post_counterfactual", [])
    post_ci_lower: list[float] = data.get("post_ci_lower", [])
    post_ci_upper: list[float] = data.get("post_ci_upper", [])

    fig = go.Figure()
    pre_idx = list(range(len(pre)))
    post_idx = list(range(len(pre), len(pre) + len(post_actual)))

    fig.add_trace(go.Scatter(x=pre_idx, y=pre, mode="lines", name="Pre-intervention",
                             line=dict(color="#4C9BE8")))
    fig.add_trace(go.Scatter(x=post_idx, y=post_actual, mode="lines", name="Post-intervention (actual)",
                             line=dict(color="#FF7F0E")))
    fig.add_trace(go.Scatter(x=post_idx, y=post_counterfactual, mode="lines",
                             name="Counterfactual", line=dict(color="gray", dash="dash")))
    if post_ci_lower and post_ci_upper:
        fig.add_trace(go.Scatter(
            x=post_idx + post_idx[::-1],
            y=post_ci_upper + post_ci_lower[::-1],
            fill="toself", fillcolor="rgba(128,128,128,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip", name="95% CI",
        ))

    fig.add_vline(x=len(pre) - 0.5, line_dash="dot", line_color="red",
                  annotation_text="Intervention", annotation_position="top left")
    fig.update_layout(title="Causal Impact", xaxis_title="Time step",
                      yaxis_title="Value", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    summary_stats = data.get("summary_stats", {})
    if summary_stats:
        st.subheader("Effect Summary")
        stat_df = pd.DataFrame([summary_stats])
        st.dataframe(stat_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES = {
    "Forecast Explorer": page_forecast_explorer,
    "Backtest Results": page_backtest_results,
    "Conformal Coverage": page_conformal_coverage,
    "Hierarchy View": page_hierarchy_view,
    "Intervention Analysis": page_intervention_analysis,
}

PAGE_ICONS = {
    "Forecast Explorer": "📊",
    "Backtest Results": "📈",
    "Conformal Coverage": "🎯",
    "Hierarchy View": "🏗️",
    "Intervention Analysis": "⚡",
}

with st.sidebar:
    st.image("https://img.shields.io/badge/PulsePredict-0.1.0-blue", width=180)
    st.markdown("---")
    selection = st.selectbox(
        "Navigate",
        list(PAGES.keys()),
        format_func=lambda p: f"{PAGE_ICONS[p]}  {p}",
    )
    st.markdown("---")
    st.markdown(f"**API:** `{API_BASE_URL}`")

PAGES[selection]()
