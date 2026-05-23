"""
Conformal Quantile Regression (CQR) for time-series prediction intervals.

Wraps native quantile forecasts (e.g. MQLoss 10th/90th percentiles) with
conformal correction to guarantee finite-sample marginal coverage.

Reference:
    Romano, Patterson & Candes (2019). Conformalized Quantile Regression.
    NeurIPS 2019. https://arxiv.org/abs/1905.03222
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CQRConfig:
    alpha: float = 0.10
    symmetric: bool = True


class CQRPredictor:
    """Conformal Quantile Regression predictor.

    Unlike split conformal (which wraps point forecasts), CQR adjusts
    existing quantile-based intervals so they achieve exact coverage.

    Nonconformity score: max(q_lo - y, y - q_hi)
    If score <= 0, the observation is already inside the native interval.
    """

    def __init__(self, config: CQRConfig):
        self.config = config
        self.q_hat: float | None = None
        self._n_cal: int = 0

    def calibrate(
        self,
        cal_y_true: np.ndarray,
        cal_q_lo: np.ndarray,
        cal_q_hi: np.ndarray,
    ) -> "CQRPredictor":
        """Compute conformal correction from calibration residuals."""
        cal_y_true = np.asarray(cal_y_true, dtype=float)
        cal_q_lo = np.asarray(cal_q_lo, dtype=float)
        cal_q_hi = np.asarray(cal_q_hi, dtype=float)

        n = len(cal_y_true)
        self._n_cal = n

        if self.config.symmetric:
            scores = np.maximum(cal_q_lo - cal_y_true, cal_y_true - cal_q_hi)
        else:
            scores = np.maximum(cal_q_lo - cal_y_true, cal_y_true - cal_q_hi)

        corrected_level = min(1.0, (1.0 - self.config.alpha) * (1.0 + 1.0 / n))
        self.q_hat = float(np.quantile(scores, corrected_level))
        return self

    def predict(
        self,
        q_lo: np.ndarray,
        q_hi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Widen native quantile intervals by conformal correction."""
        if self.q_hat is None:
            raise RuntimeError("Not calibrated. Call .calibrate() first.")

        q_lo = np.asarray(q_lo, dtype=float)
        q_hi = np.asarray(q_hi, dtype=float)

        lo = q_lo - self.q_hat
        hi = q_hi + self.q_hat
        return lo, hi

    def coverage(
        self,
        y_true: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> float:
        y_true = np.asarray(y_true, dtype=float)
        mask = ~np.isnan(y_true)
        return float(np.mean((y_true[mask] >= lo[mask]) & (y_true[mask] <= hi[mask])))

    def __repr__(self) -> str:
        cal_str = (
            f"q_hat={self.q_hat:.4f}, n_cal={self._n_cal}"
            if self.q_hat is not None
            else "not calibrated"
        )
        return f"CQRPredictor(alpha={self.config.alpha}, {cal_str})"
