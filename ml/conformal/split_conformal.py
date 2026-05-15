"""
Split conformal prediction for time-series forecasting.

Implements the standard split (inductive) conformal predictor that uses
a held-out calibration set to construct marginal prediction intervals with
finite-sample coverage guarantees.

Reference:
    Papadopoulos et al. (2002); Vovk, Gammerman & Shafer (2005).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class SplitConformalConfig:
    """
    Configuration for :class:`SplitConformalPredictor`.

    Parameters
    ----------
    alpha : float
        Target miscoverage level. A value of 0.10 targets 90% PI coverage.
    symmetric : bool
        If True (default), use symmetric intervals: q_hat applied equally to
        both sides of the point forecast. If False, compute separate lower and
        upper nonconformity scores for asymmetric intervals.
    """

    alpha: float = 0.10
    symmetric: bool = True


class SplitConformalPredictor:
    """
    Split conformal predictor for regression / time-series point forecasts.

    Usage
    -----
    1. Call :meth:`calibrate` with held-out calibration actuals and predictions.
    2. Call :meth:`predict` to wrap new point forecasts with conformal intervals.
    3. Optionally call :meth:`coverage` to verify empirical coverage on a held-out set.
    """

    def __init__(self, config: SplitConformalConfig):
        self.config = config
        self.q_hat: float | None = None
        self.q_hat_lo: float | None = None
        self.q_hat_hi: float | None = None
        self._n_cal: int = 0

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        cal_y_true: np.ndarray,
        cal_y_hat: np.ndarray,
    ) -> "SplitConformalPredictor":
        """
        Compute and store the conformal quantile from calibration residuals.

        For symmetric intervals, the nonconformity score is |y - y_hat|.
        For asymmetric intervals, separate quantiles are stored for the
        lower (y - y_hat, clipped at 0) and upper (y_hat - y, clipped at 0) sides.

        The corrected quantile level ``(1 - alpha)(1 + 1/n)`` ensures marginal
        coverage at level ``1 - alpha`` even in finite samples.

        Parameters
        ----------
        cal_y_true : np.ndarray, shape (n,)
            Calibration-set actuals.
        cal_y_hat : np.ndarray, shape (n,)
            Calibration-set point forecasts.
        """
        cal_y_true = np.asarray(cal_y_true, dtype=float)
        cal_y_hat = np.asarray(cal_y_hat, dtype=float)

        if cal_y_true.shape != cal_y_hat.shape:
            raise ValueError(
                f"Shape mismatch: cal_y_true {cal_y_true.shape} vs "
                f"cal_y_hat {cal_y_hat.shape}"
            )

        n = len(cal_y_true)
        self._n_cal = n
        alpha = self.config.alpha

        # Corrected quantile level per Theorem 1 in Tibshirani et al. (2019)
        corrected_level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))

        if self.config.symmetric:
            scores = np.abs(cal_y_true - cal_y_hat)
            self.q_hat = float(np.quantile(scores, corrected_level))
        else:
            # Separate scores for under- and over-prediction
            scores_lo = cal_y_hat - cal_y_true  # positive when y_hat > y (over)
            scores_hi = cal_y_true - cal_y_hat  # positive when y_hat < y (under)
            self.q_hat_lo = float(np.quantile(scores_lo, corrected_level))
            self.q_hat_hi = float(np.quantile(scores_hi, corrected_level))
            # Set symmetric q_hat as average for convenience
            self.q_hat = (self.q_hat_lo + self.q_hat_hi) / 2.0

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        y_hat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Wrap point forecasts with conformal prediction intervals.

        Parameters
        ----------
        y_hat : np.ndarray
            Point forecasts.

        Returns
        -------
        (lo, hi) : tuple of np.ndarray
            Lower and upper bounds of the ``(1 - alpha)`` prediction interval.

        Raises
        ------
        RuntimeError
            If :meth:`calibrate` has not been called yet.
        """
        if self.q_hat is None:
            raise RuntimeError(
                "Predictor is not calibrated. Call .calibrate() first."
            )

        y_hat = np.asarray(y_hat, dtype=float)

        if self.config.symmetric:
            lo = y_hat - self.q_hat
            hi = y_hat + self.q_hat
        else:
            lo = y_hat - self.q_hat_lo
            hi = y_hat + self.q_hat_hi

        return lo, hi

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def coverage(
        self,
        y_true: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> float:
        """
        Compute empirical coverage: fraction of actuals within [lo, hi].

        Parameters
        ----------
        y_true : np.ndarray
            Test-set actuals.
        lo, hi : np.ndarray
            Lower and upper prediction-interval bounds (from :meth:`predict`).

        Returns
        -------
        float
            Empirical coverage in [0, 1].
        """
        y_true = np.asarray(y_true, dtype=float)
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        mask = ~np.isnan(y_true)
        in_interval = (y_true[mask] >= lo[mask]) & (y_true[mask] <= hi[mask])
        return float(np.mean(in_interval))

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cal_str = (
            f"q_hat={self.q_hat:.4f}, n_cal={self._n_cal}"
            if self.q_hat is not None
            else "not calibrated"
        )
        return (
            f"SplitConformalPredictor("
            f"alpha={self.config.alpha}, "
            f"symmetric={self.config.symmetric}, "
            f"{cal_str})"
        )
