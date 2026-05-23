"""
Week 5: Conformal prediction pipeline.

For each trained model, loads val_predictions + M5 actuals, runs:
  1. SplitConformal (symmetric + asymmetric) at 80% and 90% targets
  2. Adaptive Conformal Inference (ACI, Gibbs & Candes 2021)

Saves per-model JSON reports and a combined summary to reports/.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.conformal.split_conformal import SplitConformalPredictor, SplitConformalConfig
from ml.conformal.adaptive_conformal import AdaptiveConformalPredictor, AdaptiveConformalConfig

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5001")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")

MODELS = {
    "deepar": {"median_col": "DeepAR-median"},
    "patchtst": {"median_col": "PatchTST-median"},
    "tft": {"median_col": "TFT-median"},
    "nbeatsx": {"median_col": "NBEATSx"},
}


def load_m5_actuals() -> pd.DataFrame:
    """Load M5 data and return val+test actuals (unique_id, ds, y)."""
    from ml.data.dataset import M5Dataset, DatasetConfig

    cfg = DatasetConfig(
        data_dir="data/raw/m5",
        max_series=1000,
        train_cutoff="2016-03-27",
        val_cutoff="2016-04-24",
        test_cutoff="2016-05-22",
    )
    ds = M5Dataset(cfg)
    raw = ds.load_raw(Path(cfg.data_dir))
    _, val_df, test_df = ds.get_splits(raw)
    actuals = pd.concat([val_df, test_df], ignore_index=True)
    return actuals


def merge_predictions_with_actuals(
    preds_df: pd.DataFrame, actuals: pd.DataFrame
) -> pd.DataFrame:
    """Merge model predictions with actual y values on (unique_id, ds)."""
    preds_df = preds_df.copy()
    actuals = actuals.copy()
    preds_df["ds"] = pd.to_datetime(preds_df["ds"])
    actuals["ds"] = pd.to_datetime(actuals["ds"])
    merged = preds_df.merge(actuals[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner")
    return merged


def run_split_conformal(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    alpha: float,
    symmetric: bool = True,
) -> dict:
    """Calibrate on first half, evaluate on second half."""
    n = len(y_true)
    split = n // 2
    cal_yt, test_yt = y_true[:split], y_true[split:]
    cal_yh, test_yh = y_hat[:split], y_hat[split:]

    cfg = SplitConformalConfig(alpha=alpha, symmetric=symmetric)
    scp = SplitConformalPredictor(cfg)
    scp.calibrate(cal_yt, cal_yh)
    lo, hi = scp.predict(test_yh)

    mask = ~np.isnan(test_yt)
    covered = (test_yt[mask] >= lo[mask]) & (test_yt[mask] <= hi[mask])
    coverage = float(np.mean(covered))
    width = float(np.mean(hi - lo))

    return {
        "method": "split_conformal",
        "symmetric": symmetric,
        "target_coverage": 1.0 - alpha,
        "empirical_coverage": round(coverage, 4),
        "mean_interval_width": round(width, 4),
        "q_hat": round(scp.q_hat, 4),
        "n_cal": split,
        "n_test": n - split,
    }


def run_aci(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    alpha: float,
    gamma: float = 0.005,
    window_size: int = 200,
) -> dict:
    """Run Adaptive Conformal Inference with rolling calibration window."""
    n = len(y_true)
    split = n // 2
    cal_yt, test_yt = y_true[:split], y_true[split:]
    cal_yh, test_yh = y_hat[:split], y_hat[split:]

    cal_scores = np.abs(cal_yt - cal_yh)
    test_n = len(test_yt)

    cfg = AdaptiveConformalConfig(alpha=alpha, gamma=gamma)
    aci = AdaptiveConformalPredictor(cfg)

    ws = min(window_size, len(cal_scores))
    running_scores = list(cal_scores[-ws:])

    records = []
    aci.alpha_t = alpha

    for t in range(test_n):
        scores_window = np.array(running_scores[-ws:])
        q_hat = aci.calibrate(scores_window)

        lo = test_yh[t] - q_hat
        hi = test_yh[t] + q_hat
        covered = bool(lo <= test_yt[t] <= hi)

        records.append({
            "t": t,
            "alpha_t": aci.alpha_t,
            "q_hat": q_hat,
            "lo": lo,
            "hi": hi,
            "y_true": test_yt[t],
            "y_hat": test_yh[t],
            "covered": covered,
        })

        aci.update(covered)
        running_scores.append(float(np.abs(test_yt[t] - test_yh[t])))

    df = pd.DataFrame(records)
    coverage = float(df["covered"].mean())
    width = float((df["hi"] - df["lo"]).mean())

    return {
        "method": "adaptive_conformal",
        "target_coverage": 1.0 - alpha,
        "empirical_coverage": round(coverage, 4),
        "mean_interval_width": round(width, 4),
        "gamma": gamma,
        "window_size": ws,
        "alpha_t_final": round(float(df["alpha_t"].iloc[-1]), 4),
        "alpha_t_min": round(float(df["alpha_t"].min()), 4),
        "alpha_t_max": round(float(df["alpha_t"].max()), 4),
        "n_cal": split,
        "n_test": test_n,
        "online_trace": df,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading M5 actuals...")
    actuals = load_m5_actuals()
    logger.info(f"Loaded {len(actuals)} actual observations")

    summary_rows = []
    full_report = {}

    for model_name, model_info in MODELS.items():
        preds_path = ARTIFACTS_DIR / model_name / "val_predictions.csv"
        if not preds_path.exists():
            logger.warning(f"Skipping {model_name} — no val_predictions.csv")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"  Conformal calibration: {model_name.upper()}")
        logger.info(f"{'='*60}")

        preds_df = pd.read_csv(preds_path)
        merged = merge_predictions_with_actuals(preds_df, actuals)

        if len(merged) == 0:
            logger.warning(f"No matching actuals for {model_name} — skipping")
            continue

        median_col = model_info["median_col"]
        if median_col not in merged.columns:
            logger.warning(f"Column {median_col} not found in {model_name} predictions")
            continue

        y_true = merged["y"].values.astype(float)
        y_hat = merged[median_col].values.astype(float)

        valid = ~(np.isnan(y_true) | np.isnan(y_hat))
        y_true = y_true[valid]
        y_hat = y_hat[valid]

        logger.info(f"  {len(y_true)} valid observations for calibration")

        model_report = {"model": model_name, "n_total": len(y_true)}

        # Raw model metrics (before conformal)
        raw_mae = float(np.mean(np.abs(y_true - y_hat)))
        model_report["raw_mae"] = round(raw_mae, 4)

        # Check if model has native quantile columns for baseline coverage
        lo_cols = [c for c in merged.columns if "lo" in c.lower()]
        hi_cols = [c for c in merged.columns if "hi" in c.lower()]
        if lo_cols and hi_cols:
            native_lo = merged[lo_cols[0]].values[valid].astype(float)
            native_hi = merged[hi_cols[-1]].values[valid].astype(float)
            native_cov = float(np.mean((y_true >= native_lo) & (y_true <= native_hi)))
            native_width = float(np.mean(native_hi - native_lo))
            model_report["native_quantile_coverage"] = round(native_cov, 4)
            model_report["native_quantile_width"] = round(native_width, 4)
            logger.info(f"  Native quantile coverage: {native_cov:.3f} (width: {native_width:.3f})")

        # Split Conformal at 90% and 80%
        for alpha, label in [(0.10, "90"), (0.20, "80")]:
            for sym, sym_label in [(True, "symmetric"), (False, "asymmetric")]:
                key = f"split_{label}pct_{sym_label}"
                result = run_split_conformal(y_true, y_hat, alpha, symmetric=sym)
                model_report[key] = result
                logger.info(
                    f"  SplitConformal {label}% ({sym_label}): "
                    f"coverage={result['empirical_coverage']:.3f}, "
                    f"width={result['mean_interval_width']:.3f}, "
                    f"q_hat={result['q_hat']:.3f}"
                )
                summary_rows.append({
                    "model": model_name,
                    "method": f"split_{sym_label}",
                    "target": f"{label}%",
                    "coverage": result["empirical_coverage"],
                    "width": result["mean_interval_width"],
                })

        # ACI at 90% with different gamma values
        for gamma in [0.001, 0.005, 0.01, 0.05]:
            key = f"aci_90pct_gamma{gamma}"
            result = run_aci(y_true, y_hat, alpha=0.10, gamma=gamma, window_size=500)
            trace_df = result.pop("online_trace")
            model_report[key] = result
            logger.info(
                f"  ACI 90% (gamma={gamma}): "
                f"coverage={result['empirical_coverage']:.3f}, "
                f"width={result['mean_interval_width']:.3f}, "
                f"alpha_t_final={result['alpha_t_final']:.4f}"
            )
            summary_rows.append({
                "model": model_name,
                "method": f"aci_g{gamma}",
                "target": "90%",
                "coverage": result["empirical_coverage"],
                "width": result["mean_interval_width"],
            })

            # Save ACI trace for best gamma
            if gamma == 0.005:
                trace_path = REPORTS_DIR / f"aci_trace_{model_name}.csv"
                trace_df.to_csv(trace_path, index=False)
                logger.info(f"  ACI trace saved → {trace_path}")

        full_report[model_name] = model_report

    # Summary table
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print(f"\n{'='*80}")
        print("CONFORMAL PREDICTION SUMMARY")
        print(f"{'='*80}")
        print(summary_df.to_string(index=False))

        summary_path = REPORTS_DIR / "conformal_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\nSummary saved → {summary_path}")

    # Full report
    report_path = REPORTS_DIR / "conformal_report.json"
    report_path.write_text(json.dumps(full_report, indent=2, default=str))
    logger.info(f"Full report saved → {report_path}")

    # Log to MLflow
    try:
        import mlflow
        mlflow.set_experiment("pulsepredict-conformal")
        with mlflow.start_run(run_name="conformal_calibration"):
            for model_name, model_report in full_report.items():
                for key, val in model_report.items():
                    if isinstance(val, dict) and "empirical_coverage" in val:
                        mlflow.log_metric(
                            f"{model_name}/{key}/coverage",
                            val["empirical_coverage"],
                        )
                        mlflow.log_metric(
                            f"{model_name}/{key}/width",
                            val["mean_interval_width"],
                        )
            mlflow.log_artifacts(str(REPORTS_DIR))
        logger.info("Results logged to MLflow")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")


if __name__ == "__main__":
    main()
