"""
Week 5 full evaluation: CQR + ACI visualization + test-set conformal intervals.

1. CQR (Conformal Quantile Regression) on models with native quantile outputs
2. ACI alpha_t adaptation plots
3. Test-set predictions with conformal intervals from all 4 models
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5001")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

from ml.conformal.split_conformal import SplitConformalPredictor, SplitConformalConfig
from ml.conformal.adaptive_conformal import AdaptiveConformalPredictor, AdaptiveConformalConfig
from ml.conformal.cqr import CQRPredictor, CQRConfig

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")

MODELS_WITH_QUANTILES = {
    "deepar": {
        "median_col": "DeepAR-median",
        "lo_col": "DeepAR-lo-80",
        "hi_col": "DeepAR-hi-80",
    },
    "patchtst": {
        "median_col": "PatchTST-median",
        "lo_col": "PatchTST-lo-80.0",
        "hi_col": "PatchTST-hi-80.0",
    },
    "tft": {
        "median_col": "TFT-median",
        "lo_col": "TFT-lo-80.0",
        "hi_col": "TFT-hi-80.0",
    },
}

ALL_MODELS = {
    "deepar": {"median_col": "DeepAR-median"},
    "patchtst": {"median_col": "PatchTST-median"},
    "tft": {"median_col": "TFT-median"},
    "nbeatsx": {"median_col": "NBEATSx"},
}


def load_m5_splits():
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
    train_df, val_df, test_df = ds.get_splits(raw)
    return train_df, val_df, test_df


# ========================================================================
# TASK 1: Conformal Quantile Regression (CQR)
# ========================================================================

def run_cqr_evaluation(val_df: pd.DataFrame):
    logger.info("\n" + "=" * 70)
    logger.info("  TASK 1: Conformal Quantile Regression (CQR)")
    logger.info("=" * 70)

    cqr_results = {}
    val_df = val_df.copy()
    val_df["ds"] = pd.to_datetime(val_df["ds"])

    for model_name, cols in MODELS_WITH_QUANTILES.items():
        preds_path = ARTIFACTS_DIR / model_name / "val_predictions.csv"
        if not preds_path.exists():
            continue

        preds = pd.read_csv(preds_path)
        preds["ds"] = pd.to_datetime(preds["ds"])
        merged = preds.merge(val_df[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner")

        if len(merged) == 0 or cols["lo_col"] not in merged.columns:
            continue

        y_true = merged["y"].values.astype(float)
        q_lo = merged[cols["lo_col"]].values.astype(float)
        q_hi = merged[cols["hi_col"]].values.astype(float)

        valid = ~(np.isnan(y_true) | np.isnan(q_lo) | np.isnan(q_hi))
        y_true, q_lo, q_hi = y_true[valid], q_lo[valid], q_hi[valid]

        n = len(y_true)
        split = n // 2

        native_cov = float(np.mean((y_true >= q_lo) & (y_true <= q_hi)))
        native_width = float(np.mean(q_hi - q_lo))

        for alpha, label in [(0.10, "90%"), (0.20, "80%")]:
            cqr = CQRPredictor(CQRConfig(alpha=alpha))
            cqr.calibrate(y_true[:split], q_lo[:split], q_hi[:split])
            lo_conf, hi_conf = cqr.predict(q_lo[split:], q_hi[split:])
            cov = cqr.coverage(y_true[split:], lo_conf, hi_conf)
            width = float(np.mean(hi_conf - lo_conf))

            key = f"{model_name}_cqr_{label}"
            cqr_results[key] = {
                "model": model_name,
                "method": "CQR",
                "target": label,
                "native_coverage": round(native_cov, 4),
                "native_width": round(native_width, 4),
                "cqr_coverage": round(cov, 4),
                "cqr_width": round(width, 4),
                "q_hat_correction": round(cqr.q_hat, 4),
            }
            logger.info(
                f"  {model_name} CQR {label}: native={native_cov:.3f} → CQR={cov:.3f} "
                f"(width: {native_width:.3f} → {width:.3f}, correction={cqr.q_hat:.3f})"
            )

    return cqr_results


# ========================================================================
# TASK 2: ACI Visualization (alpha_t adaptation over time)
# ========================================================================

def generate_aci_plots(val_df: pd.DataFrame):
    logger.info("\n" + "=" * 70)
    logger.info("  TASK 2: ACI Alpha_t Adaptation Visualization")
    logger.info("=" * 70)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plots")
        return

    val_df = val_df.copy()
    val_df["ds"] = pd.to_datetime(val_df["ds"])

    fig, axes = plt.subplots(len(ALL_MODELS), 2, figsize=(16, 4 * len(ALL_MODELS)))
    if len(ALL_MODELS) == 1:
        axes = axes[np.newaxis, :]

    for idx, (model_name, model_info) in enumerate(ALL_MODELS.items()):
        preds_path = ARTIFACTS_DIR / model_name / "val_predictions.csv"
        if not preds_path.exists():
            continue

        preds = pd.read_csv(preds_path)
        preds["ds"] = pd.to_datetime(preds["ds"])
        merged = preds.merge(val_df[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner")

        median_col = model_info["median_col"]
        if median_col not in merged.columns or len(merged) == 0:
            continue

        y_true = merged["y"].values.astype(float)
        y_hat = merged[median_col].values.astype(float)
        valid = ~(np.isnan(y_true) | np.isnan(y_hat))
        y_true, y_hat = y_true[valid], y_hat[valid]

        n = len(y_true)
        split = n // 2
        cal_scores = np.abs(y_true[:split] - y_hat[:split])
        test_yt, test_yh = y_true[split:], y_hat[split:]

        for gamma, color, ls in [(0.001, "#2196F3", "-"), (0.005, "#4CAF50", "-"),
                                  (0.01, "#FF9800", "--"), (0.05, "#F44336", ":")]:
            cfg = AdaptiveConformalConfig(alpha=0.10, gamma=gamma)
            aci = AdaptiveConformalPredictor(cfg)

            ws = min(500, len(cal_scores))
            running = list(cal_scores[-ws:])
            alpha_trace = []
            width_trace = []

            for t in range(len(test_yt)):
                alpha_trace.append(aci.alpha_t)
                q = aci.calibrate(np.array(running[-ws:]))
                width_trace.append(2 * q)
                covered = bool(test_yh[t] - q <= test_yt[t] <= test_yh[t] + q)
                aci.update(covered)
                running.append(float(np.abs(test_yt[t] - test_yh[t])))

            axes[idx, 0].plot(alpha_trace, color=color, linestyle=ls,
                             label=f"γ={gamma}", alpha=0.8, linewidth=1)
            axes[idx, 1].plot(width_trace, color=color, linestyle=ls,
                             label=f"γ={gamma}", alpha=0.8, linewidth=1)

        axes[idx, 0].axhline(0.10, color="black", linestyle="--", alpha=0.5, label="target α=0.10")
        axes[idx, 0].set_ylabel("α_t")
        axes[idx, 0].set_title(f"{model_name.upper()} — ACI α_t Adaptation")
        axes[idx, 0].legend(fontsize=8)
        axes[idx, 0].set_xlabel("Test step t")

        axes[idx, 1].set_ylabel("Interval width")
        axes[idx, 1].set_title(f"{model_name.upper()} — ACI Interval Width")
        axes[idx, 1].legend(fontsize=8)
        axes[idx, 1].set_xlabel("Test step t")

    plt.tight_layout()
    plot_path = REPORTS_DIR / "aci_adaptation_plots.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  ACI plots saved → {plot_path}")

    # Per-series coverage plot for best model (TFT with ACI)
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    for ax_idx, (model_name, model_info) in enumerate(
        [(k, v) for k, v in ALL_MODELS.items() if k in ("deepar", "patchtst", "tft")]
    ):
        preds_path = ARTIFACTS_DIR / model_name / "val_predictions.csv"
        if not preds_path.exists():
            continue

        preds = pd.read_csv(preds_path)
        preds["ds"] = pd.to_datetime(preds["ds"])
        merged = preds.merge(val_df[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner")
        median_col = model_info["median_col"]
        if median_col not in merged.columns:
            continue

        # Per-series coverage
        series_covs = []
        for uid in merged["unique_id"].unique()[:100]:
            sub = merged[merged["unique_id"] == uid]
            yt = sub["y"].values.astype(float)
            yh = sub[median_col].values.astype(float)
            if len(yt) < 4:
                continue
            sp = len(yt) // 2
            cal_s = np.abs(yt[:sp] - yh[:sp])
            cfg = SplitConformalConfig(alpha=0.10)
            scp = SplitConformalPredictor(cfg)
            scp.calibrate(yt[:sp], yh[:sp])
            lo, hi = scp.predict(yh[sp:])
            cov = float(np.mean((yt[sp:] >= lo) & (yt[sp:] <= hi)))
            series_covs.append(cov)

        if series_covs:
            axes2[ax_idx].hist(series_covs, bins=20, edgecolor="black", alpha=0.7, color="#4CAF50")
            axes2[ax_idx].axvline(0.90, color="red", linestyle="--", label="90% target")
            axes2[ax_idx].set_title(f"{model_name.upper()} — Per-Series Coverage")
            axes2[ax_idx].set_xlabel("Coverage")
            axes2[ax_idx].set_ylabel("Count")
            axes2[ax_idx].legend()

    plt.tight_layout()
    plot2_path = REPORTS_DIR / "per_series_coverage.png"
    fig2.savefig(plot2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"  Per-series coverage plot saved → {plot2_path}")


# ========================================================================
# TASK 3: Test-set predictions with conformal intervals
# ========================================================================

def run_test_set_evaluation(train_df, val_df, test_df):
    logger.info("\n" + "=" * 70)
    logger.info("  TASK 3: Test-Set Conformal Predictions")
    logger.info("=" * 70)

    val_df = val_df.copy()
    test_df = test_df.copy()
    val_df["ds"] = pd.to_datetime(val_df["ds"])
    test_df["ds"] = pd.to_datetime(test_df["ds"])

    test_results = {}

    for model_name, model_info in ALL_MODELS.items():
        model_dir = ARTIFACTS_DIR / model_name / "model"
        preds_path = ARTIFACTS_DIR / model_name / "val_predictions.csv"

        if not model_dir.exists() or not preds_path.exists():
            logger.warning(f"Skipping {model_name} — missing artifacts")
            continue

        logger.info(f"\n  Loading {model_name} model for test-set prediction...")

        try:
            from neuralforecast import NeuralForecast
            nf = NeuralForecast.load(str(model_dir))

            # Predict on test set: NeuralForecast needs the full history up to
            # the prediction point. Concatenate train + val as input.
            full_history = pd.concat([train_df, val_df], ignore_index=True)
            full_history["ds"] = pd.to_datetime(full_history["ds"])
            full_history = full_history.sort_values(["unique_id", "ds"])

            logger.info(f"  Generating test forecasts for {model_name}...")
            test_preds = nf.predict(full_history)
            test_preds = test_preds.reset_index()
            test_preds["ds"] = pd.to_datetime(test_preds["ds"])

            # Merge with test actuals
            merged_test = test_preds.merge(
                test_df[["unique_id", "ds", "y"]],
                on=["unique_id", "ds"],
                how="inner",
            )

            if len(merged_test) == 0:
                logger.warning(f"  No test observations matched for {model_name}")
                continue

            # Identify median column
            forecast_cols = [c for c in merged_test.columns if c not in ("unique_id", "ds", "y", "index")]
            median_col = next(
                (c for c in forecast_cols if "median" in c.lower() or "0.5" in c),
                forecast_cols[0] if forecast_cols else None,
            )

            if median_col is None:
                continue

            test_y = merged_test["y"].values.astype(float)
            test_yhat = merged_test[median_col].values.astype(float)

            # Calibrate on val predictions
            val_preds = pd.read_csv(preds_path)
            val_preds["ds"] = pd.to_datetime(val_preds["ds"])
            merged_val = val_preds.merge(
                val_df[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner"
            )

            val_median_col = model_info["median_col"]
            if val_median_col not in merged_val.columns:
                continue

            cal_y = merged_val["y"].values.astype(float)
            cal_yhat = merged_val[val_median_col].values.astype(float)
            valid_cal = ~(np.isnan(cal_y) | np.isnan(cal_yhat))
            cal_y, cal_yhat = cal_y[valid_cal], cal_yhat[valid_cal]

            valid_test = ~(np.isnan(test_y) | np.isnan(test_yhat))
            test_y, test_yhat = test_y[valid_test], test_yhat[valid_test]

            # Split conformal: calibrate on val, evaluate on test
            scp = SplitConformalPredictor(SplitConformalConfig(alpha=0.10))
            scp.calibrate(cal_y, cal_yhat)
            lo, hi = scp.predict(test_yhat)
            test_cov = float(np.mean((test_y >= lo) & (test_y <= hi)))
            test_width = float(np.mean(hi - lo))
            test_mae = float(np.mean(np.abs(test_y - test_yhat)))

            # ACI on test set
            cal_scores = np.abs(cal_y - cal_yhat)
            aci = AdaptiveConformalPredictor(AdaptiveConformalConfig(alpha=0.10, gamma=0.005))
            ws = min(500, len(cal_scores))
            running = list(cal_scores[-ws:])
            aci_covered = []
            aci_widths = []

            for t in range(len(test_y)):
                q = aci.calibrate(np.array(running[-ws:]))
                lo_t = test_yhat[t] - q
                hi_t = test_yhat[t] + q
                cov = bool(lo_t <= test_y[t] <= hi_t)
                aci_covered.append(cov)
                aci_widths.append(2 * q)
                aci.update(cov)
                running.append(float(np.abs(test_y[t] - test_yhat[t])))

            aci_cov = float(np.mean(aci_covered))
            aci_width = float(np.mean(aci_widths))

            test_results[model_name] = {
                "n_test": len(test_y),
                "test_mae": round(test_mae, 4),
                "split_conformal_90_coverage": round(test_cov, 4),
                "split_conformal_90_width": round(test_width, 4),
                "aci_90_coverage": round(aci_cov, 4),
                "aci_90_width": round(aci_width, 4),
                "q_hat": round(scp.q_hat, 4),
            }

            logger.info(
                f"  {model_name} TEST: MAE={test_mae:.3f}, "
                f"SplitConf={test_cov:.3f} (w={test_width:.3f}), "
                f"ACI={aci_cov:.3f} (w={aci_width:.3f})"
            )

            # Save test predictions with conformal intervals
            test_preds_out = merged_test[["unique_id", "ds", "y"]].copy()
            test_preds_out["y_hat"] = merged_test[median_col].values
            scp2 = SplitConformalPredictor(SplitConformalConfig(alpha=0.10))
            scp2.calibrate(cal_y, cal_yhat)
            all_lo, all_hi = scp2.predict(merged_test[median_col].values.astype(float))
            test_preds_out["conformal_lo_90"] = all_lo
            test_preds_out["conformal_hi_90"] = all_hi
            out_path = REPORTS_DIR / f"test_predictions_{model_name}.csv"
            test_preds_out.to_csv(out_path, index=False)
            logger.info(f"  Test predictions saved → {out_path}")

        except Exception as e:
            logger.error(f"  {model_name} test prediction failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    return test_results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading M5 data splits...")
    train_df, val_df, test_df = load_m5_splits()
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Task 1: CQR
    cqr_results = run_cqr_evaluation(val_df)

    # Task 2: ACI plots
    generate_aci_plots(val_df)

    # Task 3: Test-set evaluation
    test_results = run_test_set_evaluation(train_df, val_df, test_df)

    # Combined report
    full = {
        "cqr": cqr_results,
        "test_set": test_results,
    }
    report_path = REPORTS_DIR / "conformal_full_eval.json"
    report_path.write_text(json.dumps(full, indent=2, default=str))
    logger.info(f"\nFull evaluation report → {report_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("WEEK 5 FULL EVALUATION SUMMARY")
    print(f"{'='*80}")

    if cqr_results:
        print("\n--- CQR (Conformal Quantile Regression) ---")
        for key, val in cqr_results.items():
            print(f"  {val['model']} {val['target']}: "
                  f"native={val['native_coverage']:.3f} → CQR={val['cqr_coverage']:.3f} "
                  f"(correction={val['q_hat_correction']:.3f})")

    if test_results:
        print("\n--- Test-Set Results (calibrated on val, evaluated on test) ---")
        print(f"  {'Model':<10} {'MAE':>8} {'SplitConf':>12} {'ACI':>12} {'SC-Width':>10} {'ACI-Width':>10}")
        for model, res in test_results.items():
            print(f"  {model:<10} {res['test_mae']:>8.3f} "
                  f"{res['split_conformal_90_coverage']:>12.3f} "
                  f"{res['aci_90_coverage']:>12.3f} "
                  f"{res['split_conformal_90_width']:>10.3f} "
                  f"{res['aci_90_width']:>10.3f}")

    # Log to MLflow
    try:
        import mlflow
        mlflow.set_experiment("pulsepredict-conformal")
        with mlflow.start_run(run_name="week5_full_eval"):
            for model, res in test_results.items():
                for k, v in res.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"test/{model}/{k}", v)
            mlflow.log_artifacts(str(REPORTS_DIR))
        logger.info("Results logged to MLflow")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")


if __name__ == "__main__":
    main()
