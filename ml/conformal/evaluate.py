"""
Conformal evaluation runner.

Loads calibration predictions from the artifacts directory, runs both
SplitConformal and AdaptiveConformal predictors, and saves a JSON report
containing 80% / 90% PI coverage and mean interval width for each method.
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from ml.conformal.split_conformal import SplitConformalPredictor, SplitConformalConfig
from ml.conformal.adaptive_conformal import (
    AdaptiveConformalPredictor,
    AdaptiveConformalConfig,
)

logger = logging.getLogger(__name__)


def _load_calibration_data(artifacts_dir: Path) -> dict:
    """
    Attempt to load calibration predictions from the artifacts directory.

    Expected file: artifacts/calibration/cal_predictions.csv
    with columns: unique_id, ds, y_true, y_hat, [score].

    Falls back to the first val_predictions.csv found under artifacts/.

    Returns
    -------
    dict with keys:
        y_true  : np.ndarray
        y_hat   : np.ndarray
        scores  : np.ndarray  (|y_true - y_hat|)
    """
    cal_path = artifacts_dir / "calibration" / "cal_predictions.csv"
    if not cal_path.exists():
        # Search for any val_predictions.csv
        candidates = list(artifacts_dir.rglob("val_predictions.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No calibration data found under {artifacts_dir}. "
                "Run training first or place cal_predictions.csv in "
                "artifacts/calibration/."
            )
        cal_path = candidates[0]
        logger.warning(f"Calibration file not found; using {cal_path} instead.")

    df = pd.read_csv(cal_path)

    # Identify columns
    if "y_true" in df.columns and "y_hat" in df.columns:
        y_true = df["y_true"].values.astype(float)
        y_hat = df["y_hat"].values.astype(float)
    elif "y" in df.columns:
        y_true = df["y"].values.astype(float)
        # Use first non-identifier, non-y column as y_hat
        forecast_cols = [
            c for c in df.columns if c not in ("unique_id", "ds", "y")
        ]
        if not forecast_cols:
            raise ValueError(
                "Cannot identify forecast column in calibration CSV."
            )
        y_hat = df[forecast_cols[0]].values.astype(float)
    else:
        raise ValueError(
            f"Calibration CSV at {cal_path} must have columns "
            "(y_true, y_hat) or (y, <forecast_col>)."
        )

    scores = np.abs(y_true - y_hat)
    return {"y_true": y_true, "y_hat": y_hat, "scores": scores}


def _coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    mask = ~np.isnan(y_true)
    return float(np.mean((y_true[mask] >= lo[mask]) & (y_true[mask] <= hi[mask])))


def _mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Evaluate conformal prediction intervals for PulsePredict"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Root artifacts directory containing calibration predictions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/conformal_coverage.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load calibration data
    # ------------------------------------------------------------------
    logger.info(f"Loading calibration data from {artifacts_dir} …")
    cal = _load_calibration_data(artifacts_dir)
    y_true = cal["y_true"]
    y_hat = cal["y_hat"]
    scores = cal["scores"]
    n = len(y_true)
    logger.info(f"Loaded {n} calibration observations.")

    # Split into cal / test halves for honest evaluation
    split = n // 2
    cal_yt, test_yt = y_true[:split], y_true[split:]
    cal_yh, test_yh = y_hat[:split], y_hat[split:]
    cal_scores = scores[:split]
    # Calibrate on first half; report coverage on second half

    report = {}

    # ------------------------------------------------------------------
    # Split Conformal — 90% PI (alpha = 0.10)
    # ------------------------------------------------------------------
    for alpha, label in [(0.10, "90"), (0.20, "80")]:
        cfg = SplitConformalConfig(alpha=alpha, symmetric=True)
        scp = SplitConformalPredictor(cfg)
        scp.calibrate(cal_yt, cal_yh)
        lo, hi = scp.predict(test_yh)
        cov = _coverage(test_yt, lo, hi)
        width = _mean_width(lo, hi)
        logger.info(
            f"SplitConformal alpha={alpha}: coverage={cov:.3f}, "
            f"mean_width={width:.4f}, q_hat={scp.q_hat:.4f}"
        )
        report[f"split_conformal_{label}pct"] = {
            "method": "split_conformal",
            "target_coverage": 1.0 - alpha,
            "empirical_coverage": round(cov, 4),
            "mean_interval_width": round(width, 4),
            "q_hat": round(scp.q_hat, 4),
            "n_cal": scp._n_cal,
        }

    # ------------------------------------------------------------------
    # Adaptive Conformal — 90% PI (alpha = 0.10)
    # ------------------------------------------------------------------
    for alpha, label in [(0.10, "90"), (0.20, "80")]:
        cfg_aci = AdaptiveConformalConfig(alpha=alpha, gamma=0.005)
        aci = AdaptiveConformalPredictor(cfg_aci)

        # Online simulation: use calibration scores as a sliding window of size 20
        window_size = min(20, len(cal_scores))
        test_T = len(test_yt)

        # Build calibration score windows for each test step
        # Use the last `window_size` calibration scores as a fixed seed,
        # then append running test scores as we observe them.
        cal_scores_sequence = np.zeros((test_T, window_size))
        running_window = list(cal_scores[-window_size:])
        for t in range(test_T):
            cal_scores_sequence[t] = np.array(running_window[-window_size:])

        online_df = aci.predict_online(
            y_hat_sequence=test_yh,
            cal_scores_sequence=cal_scores_sequence,
            y_true_sequence=test_yt,
        )

        # Update running window with observed errors
        test_scores = np.abs(test_yt - test_yh)
        for t in range(test_T):
            running_window.append(float(test_scores[t]))

        if "covered_t" in online_df.columns:
            aci_cov = float(online_df["covered_t"].mean())
        else:
            aci_cov = _coverage(
                test_yt,
                online_df["lo_t"].values,
                online_df["hi_t"].values,
            )

        aci_width = _mean_width(online_df["lo_t"].values, online_df["hi_t"].values)
        alpha_t_final = float(online_df["alpha_t"].iloc[-1])

        logger.info(
            f"AdaptiveConformal alpha={alpha}: coverage={aci_cov:.3f}, "
            f"mean_width={aci_width:.4f}, alpha_t_final={alpha_t_final:.4f}"
        )

        report[f"adaptive_conformal_{label}pct"] = {
            "method": "adaptive_conformal",
            "target_coverage": 1.0 - alpha,
            "empirical_coverage": round(aci_cov, 4),
            "mean_interval_width": round(aci_width, 4),
            "alpha_t_final": round(alpha_t_final, 4),
            "gamma": cfg_aci.gamma,
            "n_test": test_T,
        }

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    rows = []
    for key, v in report.items():
        rows.append({
            "method_key": key,
            "method": v["method"],
            "target_coverage": v["target_coverage"],
            "empirical_coverage": v["empirical_coverage"],
            "mean_interval_width": v["mean_interval_width"],
        })

    summary_df = pd.DataFrame(rows)
    print("\nConformal Coverage Summary:")
    print(summary_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Save JSON report
    # ------------------------------------------------------------------
    output_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Conformal coverage report saved to {output_path}")


if __name__ == "__main__":
    main()
