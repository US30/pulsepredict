"""
Week 6: Chronos-T5 zero-shot inference on M5 dataset.

Loads pretrained Chronos-T5-small, predicts 28-day horizon on 1000 M5 series.
No training required. Generates val and test predictions, runs conformal
calibration, saves to artifacts/ and reports/.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5001")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

from ml.models.chronos import ChronosForecaster, ChronosConfig
from ml.conformal.split_conformal import SplitConformalPredictor, SplitConformalConfig
from ml.conformal.adaptive_conformal import AdaptiveConformalPredictor, AdaptiveConformalConfig

logger = logging.getLogger(__name__)


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
    return ds.get_splits(raw)


def run_chronos_inference(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    forecaster: ChronosForecaster,
    label: str,
) -> pd.DataFrame:
    """Run Chronos zero-shot on each series using history_df, compare to target_df."""
    history_df = history_df.copy()
    target_df = target_df.copy()
    history_df["ds"] = pd.to_datetime(history_df["ds"])
    target_df["ds"] = pd.to_datetime(target_df["ds"])

    series_ids = history_df["unique_id"].unique()
    logger.info(f"  Running Chronos on {len(series_ids)} series ({label})...")

    records = []
    t0 = time.time()
    for i, uid in enumerate(series_ids):
        hist = history_df[history_df["unique_id"] == uid].sort_values("ds")
        history_vals = hist["y"].values.astype(np.float32)

        samples = forecaster.predict_series(history_vals)
        q10 = np.quantile(samples, 0.1, axis=0)
        q50 = np.quantile(samples, 0.5, axis=0)
        q90 = np.quantile(samples, 0.9, axis=0)

        last_date = pd.Timestamp(hist["ds"].iloc[-1])
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(q50), freq="D")

        for t in range(len(q50)):
            records.append({
                "unique_id": uid,
                "ds": future_dates[t],
                "Chronos-lo-80": float(q10[t]),
                "Chronos-median": float(q50[t]),
                "Chronos-hi-80": float(q90[t]),
            })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(series_ids) - i - 1) / rate
            logger.info(f"    {i+1}/{len(series_ids)} done ({rate:.1f} series/s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - t0
    logger.info(f"  Chronos {label} inference done in {elapsed:.1f}s")

    preds_df = pd.DataFrame(records)
    return preds_df


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Loading M5 data splits...")
    train_df, val_df, test_df = load_m5_splits()
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    logger.info("Loading Chronos-T5-small...")
    cfg = ChronosConfig(
        model_id="amazon/chronos-t5-small",
        device="cuda",
        torch_dtype="float32",
        prediction_length=28,
        num_samples=20,
    )
    forecaster = ChronosForecaster(cfg)
    logger.info(f"Chronos loaded (~{forecaster.vram_gb()} GB VRAM)")

    # Val predictions: history = train, predict 28 days → compare to val
    val_preds = run_chronos_inference(train_df, val_df, forecaster, "val")

    # Save val predictions
    artifacts_dir = Path("artifacts/chronos")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    val_preds.to_csv(artifacts_dir / "val_predictions.csv", index=False)
    logger.info(f"Val predictions saved → {artifacts_dir / 'val_predictions.csv'}")

    # Test predictions: history = train+val, predict 28 days → compare to test
    full_history = pd.concat([train_df, val_df], ignore_index=True)
    test_preds = run_chronos_inference(full_history, test_df, forecaster, "test")
    test_preds.to_csv(artifacts_dir / "test_predictions.csv", index=False)

    # Evaluate val predictions
    val_df_ts = val_df.copy()
    val_df_ts["ds"] = pd.to_datetime(val_df_ts["ds"])
    val_preds["ds"] = pd.to_datetime(val_preds["ds"])
    merged_val = val_preds.merge(val_df_ts[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner")

    if len(merged_val) > 0:
        y_true = merged_val["y"].values.astype(float)
        y_hat = merged_val["Chronos-median"].values.astype(float)
        q_lo = merged_val["Chronos-lo-80"].values.astype(float)
        q_hi = merged_val["Chronos-hi-80"].values.astype(float)

        mae = float(np.mean(np.abs(y_true - y_hat)))
        rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
        native_cov = float(np.mean((y_true >= q_lo) & (y_true <= q_hi)))
        native_width = float(np.mean(q_hi - q_lo))

        logger.info(f"\n{'='*60}")
        logger.info(f"  Chronos-T5-small VAL results:")
        logger.info(f"  MAE={mae:.4f}, RMSE={rmse:.4f}")
        logger.info(f"  Native 80% coverage={native_cov:.4f}, width={native_width:.4f}")

        # Conformal calibration
        valid = ~(np.isnan(y_true) | np.isnan(y_hat))
        y_true_v, y_hat_v = y_true[valid], y_hat[valid]
        n = len(y_true_v)
        split = n // 2

        scp = SplitConformalPredictor(SplitConformalConfig(alpha=0.10))
        scp.calibrate(y_true_v[:split], y_hat_v[:split])
        lo, hi = scp.predict(y_hat_v[split:])
        sc_cov = float(np.mean((y_true_v[split:] >= lo) & (y_true_v[split:] <= hi)))
        sc_width = float(np.mean(hi - lo))

        cal_scores = np.abs(y_true_v[:split] - y_hat_v[:split])
        aci = AdaptiveConformalPredictor(AdaptiveConformalConfig(alpha=0.10, gamma=0.005))
        ws = min(500, len(cal_scores))
        running = list(cal_scores[-ws:])
        aci_covered, aci_widths = [], []
        for t in range(n - split):
            q = aci.calibrate(np.array(running[-ws:]))
            lo_t, hi_t = y_hat_v[split + t] - q, y_hat_v[split + t] + q
            cov = bool(lo_t <= y_true_v[split + t] <= hi_t)
            aci_covered.append(cov)
            aci_widths.append(2 * q)
            aci.update(cov)
            running.append(float(np.abs(y_true_v[split + t] - y_hat_v[split + t])))

        aci_cov = float(np.mean(aci_covered))
        aci_width = float(np.mean(aci_widths))

        logger.info(f"  SplitConformal 90%: coverage={sc_cov:.4f}, width={sc_width:.4f}")
        logger.info(f"  ACI 90%: coverage={aci_cov:.4f}, width={aci_width:.4f}")

        metrics = {
            "model": "chronos-t5-small",
            "val_mae": round(mae, 4),
            "val_rmse": round(rmse, 4),
            "native_80_coverage": round(native_cov, 4),
            "native_80_width": round(native_width, 4),
            "split_conformal_90_coverage": round(sc_cov, 4),
            "split_conformal_90_width": round(sc_width, 4),
            "aci_90_coverage": round(aci_cov, 4),
            "aci_90_width": round(aci_width, 4),
            "n_series": len(merged_val["unique_id"].unique()),
            "n_observations": len(merged_val),
        }

        # Test set evaluation
        test_df_ts = test_df.copy()
        test_df_ts["ds"] = pd.to_datetime(test_df_ts["ds"])
        test_preds["ds"] = pd.to_datetime(test_preds["ds"])
        merged_test = test_preds.merge(test_df_ts[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner")

        if len(merged_test) > 0:
            test_y = merged_test["y"].values.astype(float)
            test_yhat = merged_test["Chronos-median"].values.astype(float)
            test_mae = float(np.mean(np.abs(test_y - test_yhat)))

            scp2 = SplitConformalPredictor(SplitConformalConfig(alpha=0.10))
            scp2.calibrate(y_true_v, y_hat_v)
            tlo, thi = scp2.predict(test_yhat)
            test_sc_cov = float(np.mean((test_y >= tlo) & (test_y <= thi)))

            aci2 = AdaptiveConformalPredictor(AdaptiveConformalConfig(alpha=0.10, gamma=0.005))
            all_cal_scores = np.abs(y_true_v - y_hat_v)
            running2 = list(all_cal_scores[-ws:])
            aci2_cov_list = []
            aci2_widths = []
            for t in range(len(test_y)):
                q2 = aci2.calibrate(np.array(running2[-ws:]))
                lo2, hi2 = test_yhat[t] - q2, test_yhat[t] + q2
                c2 = bool(lo2 <= test_y[t] <= hi2)
                aci2_cov_list.append(c2)
                aci2_widths.append(2 * q2)
                aci2.update(c2)
                running2.append(float(np.abs(test_y[t] - test_yhat[t])))

            test_aci_cov = float(np.mean(aci2_cov_list))
            test_aci_width = float(np.mean(aci2_widths))

            metrics["test_mae"] = round(test_mae, 4)
            metrics["test_split_conformal_90_coverage"] = round(test_sc_cov, 4)
            metrics["test_aci_90_coverage"] = round(test_aci_cov, 4)
            metrics["test_aci_90_width"] = round(test_aci_width, 4)

            logger.info(f"\n  Chronos-T5-small TEST results:")
            logger.info(f"  MAE={test_mae:.4f}, SplitConf={test_sc_cov:.4f}, ACI={test_aci_cov:.4f}")

        # Save metrics
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "chronos_metrics.json").write_text(json.dumps(metrics, indent=2))
        logger.info(f"Metrics saved → reports/chronos_metrics.json")

        # Log to MLflow
        try:
            import mlflow
            mlflow.set_experiment("pulsepredict")
            with mlflow.start_run(run_name="chronos_t5_small_zeroshot"):
                mlflow.log_params({"model": "chronos-t5-small", "num_samples": 20, "zero_shot": True})
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
                mlflow.log_artifacts(str(artifacts_dir))
            logger.info("Logged to MLflow")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

    else:
        logger.warning("No matching observations between predictions and actuals")

    print(f"\n{'='*60}")
    print("CHRONOS-T5-SMALL ZERO-SHOT COMPLETE")
    print(f"{'='*60}")
    if 'metrics' in dir():
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
