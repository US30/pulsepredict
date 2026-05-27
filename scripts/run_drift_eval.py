"""
Week 8: Drift monitoring evaluation.

Splits test predictions into time windows, computes PSI / KS / coverage
drift metrics per model, and exports results for Prometheus + Grafana.
"""

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.drift.monitor import evaluate_window, DriftReport

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
DRIFT_DIR = REPORTS_DIR / "drift"
MODELS = ["deepar", "patchtst", "tft", "nbeatsx"]


def load_test_predictions(model: str) -> pd.DataFrame | None:
    path = REPORTS_DIR / f"test_predictions_{model}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def split_windows(df: pd.DataFrame, window_days: int = 7) -> list[tuple[str, pd.DataFrame]]:
    """Split dataframe into weekly windows."""
    min_date = df["ds"].min()
    max_date = df["ds"].max()
    windows = []
    start = min_date
    while start < max_date:
        end = start + pd.Timedelta(days=window_days)
        mask = (df["ds"] >= start) & (df["ds"] < end)
        window_df = df[mask]
        if len(window_df) > 0:
            label = f"{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}"
            windows.append((label, window_df))
        start = end
    return windows


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)

    all_reports: list[dict] = []

    for model in MODELS:
        logger.info(f"Evaluating drift for {model}...")
        df = load_test_predictions(model)
        if df is None:
            logger.warning(f"No test predictions for {model}, skipping")
            continue

        valid = ~(np.isnan(df["y"].values) | np.isnan(df["y_hat"].values))
        y_all = df["y"].values[valid]
        y_hat_all = df["y_hat"].values[valid]

        # Use first window as reference
        windows = split_windows(df)
        if len(windows) < 2:
            logger.warning(f"Only {len(windows)} window(s) for {model}, need >=2")
            continue

        ref_label, ref_df = windows[0]
        ref_valid = ~(np.isnan(ref_df["y"].values) | np.isnan(ref_df["y_hat"].values))
        y_ref = ref_df["y"].values[ref_valid]
        y_hat_ref = ref_df["y_hat"].values[ref_valid]

        has_conformal = "conformal_lo_90" in df.columns and "conformal_hi_90" in df.columns

        for label, window_df in windows:
            w_valid = ~(np.isnan(window_df["y"].values) | np.isnan(window_df["y_hat"].values))

            lower_90 = None
            upper_90 = None
            if has_conformal:
                lower_90 = window_df["conformal_lo_90"].values
                upper_90 = window_df["conformal_hi_90"].values

            report = evaluate_window(
                y_true=window_df["y"].values,
                y_hat=window_df["y_hat"].values,
                y_ref=y_ref,
                y_hat_ref=y_hat_ref,
                model=model,
                window=label,
                lower_90=lower_90,
                upper_90=upper_90,
            )
            all_reports.append(asdict(report))

            status = "DRIFT" if report.drift_detected else "OK"
            logger.info(
                f"  {label}: PSI={report.psi:.4f} KS={report.ks_statistic:.4f} "
                f"MAE={report.mae:.4f} Cov={report.coverage_90:.2%} [{status}]"
            )

    # Save reports
    reports_path = DRIFT_DIR / "drift_report.json"
    with open(reports_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    logger.info(f"Saved {len(all_reports)} drift reports -> {reports_path}")

    # Save as CSV for Grafana/Prometheus
    if all_reports:
        pd.DataFrame(all_reports).to_csv(DRIFT_DIR / "drift_metrics.csv", index=False)

    # Prometheus metrics file (textfile collector format)
    prom_path = DRIFT_DIR / "drift_metrics.prom"
    with open(prom_path, "w") as f:
        f.write("# HELP pulsepredict_psi Population Stability Index per model per window\n")
        f.write("# TYPE pulsepredict_psi gauge\n")
        for r in all_reports:
            f.write(f'pulsepredict_psi{{model="{r["model"]}",window="{r["window"]}"}} {r["psi"]}\n')

        f.write("# HELP pulsepredict_mae Mean Absolute Error per model per window\n")
        f.write("# TYPE pulsepredict_mae gauge\n")
        for r in all_reports:
            f.write(f'pulsepredict_mae{{model="{r["model"]}",window="{r["window"]}"}} {r["mae"]}\n')

        f.write("# HELP pulsepredict_coverage_90 Conformal 90% coverage per model per window\n")
        f.write("# TYPE pulsepredict_coverage_90 gauge\n")
        for r in all_reports:
            f.write(f'pulsepredict_coverage_90{{model="{r["model"]}",window="{r["window"]}"}} {r["coverage_90"]}\n')

        f.write("# HELP pulsepredict_ks_statistic KS test statistic per model per window\n")
        f.write("# TYPE pulsepredict_ks_statistic gauge\n")
        for r in all_reports:
            f.write(f'pulsepredict_ks_statistic{{model="{r["model"]}",window="{r["window"]}"}} {r["ks_statistic"]}\n')

        f.write("# HELP pulsepredict_drift_detected Whether drift was detected (PSI>0.25 or KS p<0.01)\n")
        f.write("# TYPE pulsepredict_drift_detected gauge\n")
        for r in all_reports:
            val = 1 if r["drift_detected"] else 0
            f.write(f'pulsepredict_drift_detected{{model="{r["model"]}",window="{r["window"]}"}} {val}\n')

    logger.info(f"Prometheus metrics -> {prom_path}")

    # Summary
    print(f"\n{'='*80}")
    print("DRIFT MONITORING SUMMARY")
    print(f"{'='*80}")
    for model in MODELS:
        model_reports = [r for r in all_reports if r["model"] == model]
        if not model_reports:
            continue
        n_drift = sum(1 for r in model_reports if r["drift_detected"])
        avg_psi = np.mean([r["psi"] for r in model_reports])
        avg_mae = np.mean([r["mae"] for r in model_reports])
        avg_cov = np.mean([r["coverage_90"] for r in model_reports])
        print(f"\n{model.upper()}: {len(model_reports)} windows, {n_drift} drift alerts")
        print(f"  Avg PSI={avg_psi:.4f}  Avg MAE={avg_mae:.4f}  Avg Coverage={avg_cov:.2%}")


if __name__ == "__main__":
    main()
