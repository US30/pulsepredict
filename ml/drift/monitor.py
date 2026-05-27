"""
Model drift detection for PulsePredict.

Implements Population Stability Index (PSI), Kolmogorov-Smirnov test,
prediction distribution tracking, and conformal coverage monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DriftReport:
    model: str
    window: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    pred_mean: float
    pred_std: float
    actual_mean: float
    actual_std: float
    mae: float
    coverage_90: float
    avg_pi_width: float
    n_observations: int
    drift_detected: bool


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10, eps: float = 1e-4
) -> float:
    """Compute PSI between reference and current distributions.

    PSI < 0.1: no significant shift
    PSI 0.1-0.25: moderate shift
    PSI > 0.25: significant shift
    """
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference) + eps
    cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current) + eps

    psi = float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))
    return psi


def ks_drift_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test for distribution shift."""
    result = stats.ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


def compute_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Compute empirical coverage of prediction intervals."""
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def evaluate_window(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    y_ref: np.ndarray,
    y_hat_ref: np.ndarray,
    model: str,
    window: str,
    lower_90: np.ndarray | None = None,
    upper_90: np.ndarray | None = None,
) -> DriftReport:
    """Evaluate drift metrics for a single time window."""
    valid = ~(np.isnan(y_true) | np.isnan(y_hat))
    y_t = y_true[valid]
    y_h = y_hat[valid]

    residuals_cur = y_t - y_h
    residuals_ref = y_ref - y_hat_ref

    psi = population_stability_index(residuals_ref, residuals_cur)
    ks_stat, ks_pval = ks_drift_test(residuals_ref, residuals_cur)
    mae = float(np.mean(np.abs(residuals_cur)))

    coverage = 0.0
    width = 0.0
    if lower_90 is not None and upper_90 is not None:
        lo = lower_90[valid]
        hi = upper_90[valid]
        coverage = compute_coverage(y_t, lo, hi)
        width = float(np.mean(hi - lo))

    return DriftReport(
        model=model,
        window=window,
        psi=round(psi, 4),
        ks_statistic=round(ks_stat, 4),
        ks_pvalue=round(ks_pval, 4),
        pred_mean=round(float(np.mean(y_h)), 4),
        pred_std=round(float(np.std(y_h)), 4),
        actual_mean=round(float(np.mean(y_t)), 4),
        actual_std=round(float(np.std(y_t)), 4),
        mae=round(mae, 4),
        coverage_90=round(coverage, 4),
        avg_pi_width=round(width, 4),
        n_observations=int(valid.sum()),
        drift_detected=(psi > 0.25 or ks_pval < 0.01),
    )
