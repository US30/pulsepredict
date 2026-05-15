"""
Adaptive Conformal Inference (ACI) for online time-series prediction intervals.

Implements the Gibbs & Candes (2021) ACI algorithm that dynamically adjusts
the miscoverage level alpha_t to track the target coverage 1-alpha despite
distribution shift.

Reference:
    Gibbs, I. & Candes, E. (2021). Adaptive Conformal Inference Under Distribution Shift.
    NeurIPS 2021. https://arxiv.org/abs/2106.00170
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class AdaptiveConformalConfig:
    """
    Configuration for :class:`AdaptiveConformalPredictor`.

    Parameters
    ----------
    alpha : float
        Target miscoverage level (e.g. 0.10 for 90% PI). This is the long-run
        target; alpha_t is adapted online.
    gamma : float
        Step size for the ACI update rule. Smaller values give smoother
        adaptation; larger values react faster to coverage violations.
        Gibbs & Candes recommend tuning gamma in [0.001, 0.05].
    """

    alpha: float = 0.10
    gamma: float = 0.005


class AdaptiveConformalPredictor:
    """
    Online adaptive conformal predictor.

    The alpha_t is updated at each step so that the long-run miscoverage
    tracks the target alpha:

        alpha_{t+1} = clip(alpha_t + gamma * (alpha - err_t), eps, 1 - eps)

    where err_t = 1 - int(y_t in [lo_t, hi_t]) (i.e. 1 if NOT covered, 0 if covered).

    Usage
    -----
    1. Call :meth:`predict_online` for batch online simulation.
    2. Or drive step-by-step: :meth:`calibrate` → produce interval → observe y → :meth:`update`.
    """

    def __init__(self, config: AdaptiveConformalConfig):
        self.config = config
        self.alpha_t: float = config.alpha
        self._eps: float = 1e-6  # clamp boundary to avoid degenerate quantiles

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(self, covered: bool) -> None:
        """
        Update alpha_t given whether the last observation was covered.

        ACI update rule:
            alpha_{t+1} = alpha_t + gamma * (alpha - indicator(not covered))

        Parameters
        ----------
        covered : bool
            True if y_t fell within the last prediction interval.
        """
        err_t = 0.0 if covered else 1.0
        self.alpha_t = self.alpha_t + self.config.gamma * (self.config.alpha - err_t)
        # Clamp to a valid probability range
        self.alpha_t = float(np.clip(self.alpha_t, self._eps, 1.0 - self._eps))

    # ------------------------------------------------------------------
    # Calibration (per-step)
    # ------------------------------------------------------------------

    def calibrate(self, cal_scores: np.ndarray) -> float:
        """
        Compute the conformal quantile for the *current* alpha_t.

        Parameters
        ----------
        cal_scores : np.ndarray
            Nonconformity scores from a calibration window
            (e.g. |y - y_hat| for recent steps).

        Returns
        -------
        float
            q_hat at the corrected quantile level for current alpha_t.
        """
        cal_scores = np.asarray(cal_scores, dtype=float)
        n = len(cal_scores)
        # Corrected quantile: (1 - alpha_t)(1 + 1/n) clamped to [0, 1]
        corrected_level = min(1.0, (1.0 - self.alpha_t) * (1.0 + 1.0 / max(n, 1)))
        return float(np.quantile(cal_scores, corrected_level))

    # ------------------------------------------------------------------
    # Batch online simulation
    # ------------------------------------------------------------------

    def predict_online(
        self,
        y_hat_sequence: np.ndarray,
        cal_scores_sequence: np.ndarray,
        y_true_sequence: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Simulate online ACI over a sequence of time steps.

        At each step t:
          1. Compute q_hat_t from cal_scores_sequence[t] using current alpha_t.
          2. Produce interval [y_hat_t - q_hat_t, y_hat_t + q_hat_t].
          3. If y_true_sequence is provided, observe y_t and call :meth:`update`.

        Parameters
        ----------
        y_hat_sequence : np.ndarray, shape (T,)
            Point forecasts for each time step.
        cal_scores_sequence : np.ndarray, shape (T, W)
            Calibration nonconformity scores for each step.
            W is the calibration window size (may vary; use object arrays if needed).
            If 1-D of length T, treated as a single score per step.
        y_true_sequence : np.ndarray, shape (T,), optional
            Actuals. If provided, alpha_t is updated online (full ACI simulation).
            If None, intervals are produced without updating alpha_t.

        Returns
        -------
        pd.DataFrame
            Columns: t, alpha_t, q_hat_t, lo_t, hi_t[, covered_t].
        """
        y_hat_sequence = np.asarray(y_hat_sequence, dtype=float)
        T = len(y_hat_sequence)

        if cal_scores_sequence.ndim == 1:
            # Treat as a single calibration score per step (window size = 1)
            cal_scores_sequence = cal_scores_sequence[:, np.newaxis]

        if y_true_sequence is not None:
            y_true_sequence = np.asarray(y_true_sequence, dtype=float)
            if len(y_true_sequence) != T:
                raise ValueError(
                    f"y_true_sequence length {len(y_true_sequence)} != "
                    f"y_hat_sequence length {T}"
                )

        records = []
        # Reset to initial alpha for a fresh simulation
        self.alpha_t = self.config.alpha

        for t in range(T):
            alpha_t_current = self.alpha_t

            cal_scores_t = np.asarray(cal_scores_sequence[t], dtype=float).ravel()
            q_hat_t = self.calibrate(cal_scores_t)

            y_hat_t = y_hat_sequence[t]
            lo_t = y_hat_t - q_hat_t
            hi_t = y_hat_t + q_hat_t

            record = {
                "t": t,
                "alpha_t": alpha_t_current,
                "q_hat_t": q_hat_t,
                "lo_t": lo_t,
                "hi_t": hi_t,
            }

            if y_true_sequence is not None:
                y_t = y_true_sequence[t]
                covered = bool(lo_t <= y_t <= hi_t)
                record["y_true_t"] = y_t
                record["covered_t"] = covered
                self.update(covered)

            records.append(record)

        df = pd.DataFrame(records)
        return df

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AdaptiveConformalPredictor("
            f"alpha={self.config.alpha}, "
            f"gamma={self.config.gamma}, "
            f"alpha_t={self.alpha_t:.4f})"
        )
