import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(
    page_title="PulsePredict",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

MODELS = ["deepar", "patchtst", "tft", "nbeatsx", "chronos"]


def _load_json(path: Path) -> dict | list | None:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return None


# =========================================================================
# Page 1: Model Comparison Dashboard
# =========================================================================

def page_model_comparison():
    st.title("Model Comparison")

    # Load test predictions for all models
    test_results = {}
    for model in MODELS:
        test_path = REPORTS_DIR / f"test_predictions_{model}.csv"
        if test_path.exists():
            df = pd.read_csv(test_path)
            df["ds"] = pd.to_datetime(df["ds"])
            y_true = df["y"].values
            y_hat = df["y_hat"].values
            valid = ~(np.isnan(y_true) | np.isnan(y_hat))
            mae = float(np.mean(np.abs(y_true[valid] - y_hat[valid])))
            rmse = float(np.sqrt(np.mean((y_true[valid] - y_hat[valid]) ** 2)))
            test_results[model] = {"mae": mae, "rmse": rmse, "n": int(valid.sum())}

    # Chronos metrics from separate report
    chronos_path = REPORTS_DIR / "chronos_metrics.json"
    chronos_data = _load_json(chronos_path)
    if chronos_data and "chronos" not in test_results:
        test_results["chronos"] = {
            "mae": chronos_data.get("test_mae", chronos_data.get("val_mae", 0)),
            "rmse": 0,
            "n": chronos_data.get("n_observations", 0),
        }

    # Conformal summary
    conformal_path = REPORTS_DIR / "conformal_summary.csv"
    conformal_df = None
    if conformal_path.exists():
        conformal_df = pd.read_csv(conformal_path)

    if test_results:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Test Set MAE")
            models = list(test_results.keys())
            maes = [test_results[m]["mae"] for m in models]
            colors = ["#4CAF50" if m == models[np.argmin(maes)] else "#4C9BE8" for m in models]
            fig = go.Figure(go.Bar(x=[m.upper() for m in models], y=maes, marker_color=colors))
            fig.update_layout(yaxis_title="MAE", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Model Summary")
            summary_rows = []
            for m in models:
                row = {"Model": m.upper(), "MAE": round(test_results[m]["mae"], 4)}
                if test_results[m]["rmse"]:
                    row["RMSE"] = round(test_results[m]["rmse"], 4)
                summary_rows.append(row)
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Conformal coverage comparison
    if conformal_df is not None:
        st.subheader("Conformal Prediction Coverage (ACI, gamma=0.005)")
        aci_df = conformal_df[conformal_df["method"] == "aci_g0.005"]
        if len(aci_df) > 0:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=[m.upper() for m in aci_df["model"]],
                y=aci_df["coverage"],
                name="Coverage",
                marker_color="#4CAF50",
            ))
            fig2.add_hline(y=0.90, line_dash="dash", line_color="red",
                          annotation_text="90% target")
            fig2.update_layout(yaxis_title="Coverage", yaxis=dict(range=[0.85, 0.95]), height=400)
            st.plotly_chart(fig2, use_container_width=True)


# =========================================================================
# Page 2: Forecast Explorer
# =========================================================================

def page_forecast_explorer():
    st.title("Forecast Explorer")

    col_left, col_right = st.columns([1, 3])

    with col_left:
        model = st.selectbox("Model", [m for m in MODELS if (ARTIFACTS_DIR / m / "val_predictions.csv").exists()])
        test_path = REPORTS_DIR / f"test_predictions_{model}.csv"

        if test_path.exists():
            df = pd.read_csv(test_path)
            df["ds"] = pd.to_datetime(df["ds"])
            series_ids = sorted(df["unique_id"].unique())
            selected = st.selectbox("Series", series_ids[:50])
        else:
            st.warning(f"No test predictions for {model}")
            return

    with col_right:
        sub = df[df["unique_id"] == selected].sort_values("ds")
        if len(sub) == 0:
            st.warning("No data for selected series")
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sub["ds"], y=sub["y"], mode="lines+markers",
            name="Actual", line=dict(color="#4C9BE8", width=2),
            marker=dict(size=4),
        ))
        fig.add_trace(go.Scatter(
            x=sub["ds"], y=sub["y_hat"], mode="lines",
            name="Forecast", line=dict(color="#FF7F0E", width=2, dash="dash"),
        ))

        if "conformal_lo_90" in sub.columns and "conformal_hi_90" in sub.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([sub["ds"], sub["ds"][::-1]]),
                y=pd.concat([sub["conformal_hi_90"], sub["conformal_lo_90"][::-1]]),
                fill="toself", fillcolor="rgba(255,165,0,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip", name="90% Conformal PI",
            ))

        fig.update_layout(
            title=f"{selected} - {model.upper()} Test Forecast",
            xaxis_title="Date", yaxis_title="Sales",
            hovermode="x unified", height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics for this series
        mae = float(np.mean(np.abs(sub["y"] - sub["y_hat"])))
        if "conformal_lo_90" in sub.columns:
            cov = float(np.mean((sub["y"] >= sub["conformal_lo_90"]) & (sub["y"] <= sub["conformal_hi_90"])))
            width = float(np.mean(sub["conformal_hi_90"] - sub["conformal_lo_90"]))
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{mae:.3f}")
            c2.metric("90% Coverage", f"{cov:.1%}")
            c3.metric("Avg PI Width", f"{width:.2f}")
        else:
            st.metric("MAE", f"{mae:.3f}")


# =========================================================================
# Page 3: Conformal Coverage
# =========================================================================

def page_conformal_coverage():
    st.title("Conformal Prediction Analysis")

    # ACI adaptation plots
    aci_plot = REPORTS_DIR / "aci_adaptation_plots.png"
    if aci_plot.exists():
        st.subheader("ACI Alpha_t Adaptation Over Time")
        st.image(str(aci_plot), use_container_width=True)

    # Per-series coverage
    series_plot = REPORTS_DIR / "per_series_coverage.png"
    if series_plot.exists():
        st.subheader("Per-Series Coverage Distribution")
        st.image(str(series_plot), use_container_width=True)

    # Full conformal report
    report = _load_json(REPORTS_DIR / "conformal_report.json")
    if report:
        st.subheader("Detailed Results")
        for model_name, model_data in report.items():
            with st.expander(f"{model_name.upper()}", expanded=False):
                rows = []
                for key, val in model_data.items():
                    if isinstance(val, dict) and "empirical_coverage" in val:
                        rows.append({
                            "Method": key,
                            "Coverage": val["empirical_coverage"],
                            "Width": val["mean_interval_width"],
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ACI traces
    st.subheader("ACI Online Trace")
    trace_model = st.selectbox("Model", MODELS, key="aci_trace_model")
    trace_path = REPORTS_DIR / f"aci_trace_{trace_model}.csv"
    if trace_path.exists():
        trace = pd.read_csv(trace_path)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trace["t"], y=trace["alpha_t"], mode="lines",
                                 name="alpha_t", line=dict(color="#4CAF50")))
        fig.add_hline(y=0.10, line_dash="dash", line_color="red", annotation_text="target")
        fig.update_layout(xaxis_title="Step", yaxis_title="alpha_t", height=350)
        st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# Page 4: Hierarchy View
# =========================================================================

def page_hierarchy_view():
    st.title("Hierarchical Reconciliation")

    report = _load_json(REPORTS_DIR / "reconciled" / "reconciliation_report.json")
    if report is None:
        st.warning("No reconciliation report found. Run `scripts/run_reconciliation.py` first.")
        return

    hierarchy = report.get("hierarchy", {})
    st.subheader("M5 Hierarchy Structure")
    levels = hierarchy.get("levels", {})
    if levels:
        fig = go.Figure(go.Bar(
            x=list(levels.keys()),
            y=list(levels.values()),
            marker_color=["#FF5722", "#FF9800", "#FFC107", "#4CAF50", "#2196F3"],
        ))
        fig.update_layout(yaxis_title="Number of Series", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Base vs reconciled comparison
    base = report.get("base_forecasts", {})
    recon = report.get("reconciliation", {})

    if base and recon:
        st.subheader("Base vs Reconciled MAE by Level")

        rows = []
        for level in levels.keys():
            row = {"Level": level}
            for col_name, col_data in base.items():
                lm = col_data.get("per_level_mae", {}).get(level)
                if lm is not None:
                    row[f"Base_{col_name}"] = lm
            for method, method_data in recon.items():
                lm = method_data.get("per_level_mae", {}).get(level)
                if lm is not None:
                    row[method] = lm
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Chart
            fig = go.Figure()
            for col in df.columns:
                if col != "Level":
                    fig.add_trace(go.Bar(x=df["Level"], y=df[col], name=col))
            fig.update_layout(barmode="group", yaxis_title="MAE", height=450,
                             yaxis_type="log")
            st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# Page 5: Intervention Analysis (CausalImpact)
# =========================================================================

def page_intervention_analysis():
    st.title("Bayesian CausalImpact")

    intervention_dir = REPORTS_DIR / "intervention"
    if not intervention_dir.exists():
        st.warning("No intervention reports found. Run `python -m ml.intervention.run_case` first.")
        return

    report_files = list(intervention_dir.glob("*_report.json"))
    if not report_files:
        st.warning("No intervention report JSON files found.")
        return

    selected_report = st.selectbox(
        "Case Study",
        report_files,
        format_func=lambda p: p.stem.replace("_report", ""),
    )

    data = _load_json(selected_report)
    if data is None:
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Relative Effect", f"{data.get('relative_effect_pct', 0):+.1f}%")
    col2.metric("Cumulative Effect", f"{data.get('cumulative_effect', 0):+.1f}")
    col3.metric("P(Positive)", f"{data.get('posterior_prob_positive', 0):.3f}")
    ci = data.get("credible_interval_95", [0, 0])
    col4.metric("95% CI", f"[{ci[0]:.1f}, {ci[1]:.1f}]")

    # Show plot if exists
    plot_name = selected_report.stem.replace("_report", "_impact.png")
    plot_path = intervention_dir / plot_name
    if plot_path.exists():
        st.image(str(plot_path), use_container_width=True)

    with st.expander("Raw Results"):
        st.json(data)


# =========================================================================
# Page 6: Drift Monitor
# =========================================================================

def page_drift_monitor():
    st.title("Model Drift Monitor")

    drift_path = REPORTS_DIR / "drift" / "drift_report.json"
    if not drift_path.exists():
        st.warning("No drift report found. Run `python scripts/run_drift_eval.py` first.")
        return

    data = _load_json(drift_path)
    if not data:
        return

    df = pd.DataFrame(data)

    # Summary metrics
    st.subheader("Current Status")
    models = df["model"].unique()
    cols = st.columns(len(models))
    for i, model in enumerate(models):
        model_df = df[df["model"] == model]
        latest = model_df.iloc[-1]
        with cols[i]:
            st.metric(f"{model.upper()} PSI", f"{latest['psi']:.4f}",
                      delta="DRIFT" if latest["drift_detected"] else "OK",
                      delta_color="inverse" if latest["drift_detected"] else "normal")
            st.metric("Coverage", f"{latest['coverage_90']:.1%}")
            st.metric("MAE", f"{latest['mae']:.4f}")

    # PSI over windows
    st.subheader("PSI Trend by Window")
    fig_psi = go.Figure()
    for model in models:
        model_df = df[df["model"] == model]
        fig_psi.add_trace(go.Scatter(
            x=model_df["window"], y=model_df["psi"],
            mode="lines+markers", name=model.upper(),
        ))
    fig_psi.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Moderate")
    fig_psi.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Significant")
    fig_psi.update_layout(yaxis_title="PSI", height=400)
    st.plotly_chart(fig_psi, use_container_width=True)

    # Coverage over windows
    st.subheader("90% Coverage Drift")
    fig_cov = go.Figure()
    for model in models:
        model_df = df[df["model"] == model]
        fig_cov.add_trace(go.Scatter(
            x=model_df["window"], y=model_df["coverage_90"],
            mode="lines+markers", name=model.upper(),
        ))
    fig_cov.add_hline(y=0.90, line_dash="dash", line_color="red", annotation_text="90% target")
    fig_cov.update_layout(yaxis_title="Coverage", yaxis=dict(range=[0.85, 0.95]), height=400)
    st.plotly_chart(fig_cov, use_container_width=True)

    # MAE over windows
    st.subheader("MAE Trend")
    fig_mae = go.Figure()
    for model in models:
        model_df = df[df["model"] == model]
        fig_mae.add_trace(go.Scatter(
            x=model_df["window"], y=model_df["mae"],
            mode="lines+markers", name=model.upper(),
        ))
    fig_mae.update_layout(yaxis_title="MAE", height=400)
    st.plotly_chart(fig_mae, use_container_width=True)

    # Full table
    with st.expander("Full Drift Report"):
        st.dataframe(df, use_container_width=True, hide_index=True)


# =========================================================================
# Navigation
# =========================================================================

PAGES = {
    "Model Comparison": page_model_comparison,
    "Forecast Explorer": page_forecast_explorer,
    "Conformal Coverage": page_conformal_coverage,
    "Hierarchy View": page_hierarchy_view,
    "Intervention Analysis": page_intervention_analysis,
    "Drift Monitor": page_drift_monitor,
}

PAGE_ICONS = {
    "Model Comparison": "📊",
    "Forecast Explorer": "🔍",
    "Conformal Coverage": "🎯",
    "Hierarchy View": "🏗️",
    "Intervention Analysis": "⚡",
    "Drift Monitor": "📈",
}

with st.sidebar:
    st.title("PulsePredict")
    st.caption("Probabilistic Time-Series Forecasting")
    st.markdown("---")
    selection = st.selectbox(
        "Navigate",
        list(PAGES.keys()),
        format_func=lambda p: f"{PAGE_ICONS[p]}  {p}",
    )
    st.markdown("---")

    # Show available data
    st.caption("Available Data")
    for model in MODELS:
        has_val = (ARTIFACTS_DIR / model / "val_predictions.csv").exists()
        has_test = (REPORTS_DIR / f"test_predictions_{model}.csv").exists()
        status = "✅" if has_test else ("📦" if has_val else "❌")
        st.text(f"{status} {model.upper()}")

PAGES[selection]()
