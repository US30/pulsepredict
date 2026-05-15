"""
Forecasting evaluation metrics for PulsePredict.

All scalar functions accept array-like inputs and return Python floats.
`evaluate_forecast` is the primary entry point that computes all metrics at once.
`rolling_origin_backtest` provides expanding-window evaluation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Union

ArrayLike = Union[np.ndarray, pd.Series]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ForecastMetrics:
    mae: float
    mse: float
    rmse: float
    mase: float
    smape: float
    coverage_90: float
    coverage_80: float
    winkler_90: float
    pinball_10: float
    pinball_90: float
    crps: float


# ---------------------------------------------------------------------------
# Scalar metric functions
# ---------------------------------------------------------------------------

def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def mase(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_train: ArrayLike,
    seasonality: int = 1,
) -> float:
    """
    Mean Absolute Scaled Error (MASE).

    The denominator is the in-sample MAE of the seasonal naive forecaster:
        naive_error = mean |y_train[t] - y_train[t - seasonality]|

    Parameters
    ----------
    y_true, y_pred : array-like
        Out-of-sample actuals and forecasts.
    y_train : array-like
        Training series used to compute the naive benchmark.
    seasonality : int
        Seasonality lag for the naive benchmark (1 = random walk, 7 = weekly, 12 = monthly).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if len(y_train) <= seasonality:
        # Fall back to MAE if not enough training data
        return mae(y_true, y_pred)

    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)

    if scale == 0.0:
        return float("inf")

    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).

    Uses the standard definition:
        sMAPE = 100 * mean(2 * |y - y_hat| / (|y| + |y_hat|))

    Handles zero denominators by excluding those observations.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    values = 2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return float(100.0 * np.mean(values))


def coverage(
    y_true: ArrayLike,
    q_lo: ArrayLike,
    q_hi: ArrayLike,
) -> float:
    """
    Empirical coverage: fraction of actuals falling within [q_lo, q_hi].
    """
    y_true = np.asarray(y_true, dtype=float)
    q_lo = np.asarray(q_lo, dtype=float)
    q_hi = np.asarray(q_hi, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(q_lo) & ~np.isnan(q_hi)
    in_interval = (y_true[mask] >= q_lo[mask]) & (y_true[mask] <= q_hi[mask])
    return float(np.mean(in_interval))


def winkler_score(
    y_true: ArrayLike,
    q_lo: ArrayLike,
    q_hi: ArrayLike,
    alpha: float = 0.1,
) -> float:
    """
    Winkler interval score for a (1-alpha) prediction interval.

    W = (q_hi - q_lo) + (2/alpha) * max(q_lo - y, 0) + (2/alpha) * max(y - q_hi, 0)
    """
    y_true = np.asarray(y_true, dtype=float)
    q_lo = np.asarray(q_lo, dtype=float)
    q_hi = np.asarray(q_hi, dtype=float)

    width = q_hi - q_lo
    penalty_lo = (2.0 / alpha) * np.maximum(q_lo - y_true, 0.0)
    penalty_hi = (2.0 / alpha) * np.maximum(y_true - q_hi, 0.0)
    scores = width + penalty_lo + penalty_hi
    return float(np.mean(scores))


def pinball_loss(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    tau: float,
) -> float:
    """
    Pinball (quantile) loss for quantile level *tau* in (0, 1).

    L_tau(y, q) = (y - q) * tau       if y >= q
                = (q - y) * (1 - tau) if y <  q
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    losses = np.where(errors >= 0, tau * errors, (tau - 1.0) * errors)
    return float(np.mean(losses))


def crps(y_true: ArrayLike, samples: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score (CRPS) estimated from posterior samples.

    Uses the energy-score decomposition:
        CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|

    where X, X' are i.i.d. draws from the predictive distribution F.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Observed values.
    samples : np.ndarray, shape (n, S) or (S,) for a single observation.
        Posterior predictive samples. S = number of samples.

    Returns
    -------
    float
        Mean CRPS over all observations.
    """
    y_true = np.asarray(y_true, dtype=float)
    samples = np.asarray(samples, dtype=float)

    if samples.ndim == 1:
        # Single observation: samples has shape (S,)
        samples = samples[np.newaxis, :]  # -> (1, S)
        y_true = y_true.reshape(1)

    n, S = samples.shape
    # E[|X - y|] term
    e_abs = np.mean(np.abs(samples - y_true[:, np.newaxis]), axis=1)  # (n,)
    # E[|X - X'|] term: computed as 2 * mean over ordered pairs
    # Equivalent: mean of pairwise differences approximated via sorted samples
    sorted_samples = np.sort(samples, axis=1)
    weights = (2.0 * np.arange(1, S + 1) - S - 1) / (S * S)
    e_pair = np.sum(sorted_samples * weights[np.newaxis, :], axis=1)  # (n,)

    crps_per_obs = e_abs - e_pair
    return float(np.mean(crps_per_obs))


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_forecast(
    y_true: pd.Series,
    y_pred: pd.Series,
    q_lo: pd.Series,
    q_hi: pd.Series,
    y_train: pd.Series,
    seasonality: int = 7,
    alpha_90: float = 0.10,
    alpha_80: float = 0.20,
) -> ForecastMetrics:
    """
    Compute all forecasting metrics and return a ForecastMetrics instance.

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth values for the forecast horizon.
    y_pred : pd.Series
        Point forecast (median or mean).
    q_lo, q_hi : pd.Series
        Lower/upper bounds of the 90% prediction interval.
    y_train : pd.Series
        Historical training values (used as MASE denominator).
    seasonality : int
        Seasonality for MASE computation.
    alpha_90, alpha_80 : float
        Miscoverage levels for Winkler scoring.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    qlo = np.asarray(q_lo, dtype=float)
    qhi = np.asarray(q_hi, dtype=float)
    ytrain = np.asarray(y_train, dtype=float)

    # 80% PI derived by contracting the 90% interval by 1/9 on each side
    interval_half = (qhi - qlo) / 2.0
    qlo_80 = yp - 0.8 * interval_half
    qhi_80 = yp + 0.8 * interval_half

    # CRPS: use point forecast as a degenerate single sample
    crps_val = crps(yt, yp[:, np.newaxis] if yp.ndim == 1 else yp)

    return ForecastMetrics(
        mae=mae(yt, yp),
        mse=mse(yt, yp),
        rmse=rmse(yt, yp),
        mase=mase(yt, yp, ytrain, seasonality=seasonality),
        smape=smape(yt, yp),
        coverage_90=coverage(yt, qlo, qhi),
        coverage_80=coverage(yt, qlo_80, qhi_80),
        winkler_90=winkler_score(yt, qlo, qhi, alpha=alpha_90),
        pinball_10=pinball_loss(yt, qlo, tau=0.10),
        pinball_90=pinball_loss(yt, qhi, tau=0.90),
        crps=crps_val,
    )


# ---------------------------------------------------------------------------
# Rolling-origin backtest
# ---------------------------------------------------------------------------

def rolling_origin_backtest(
    forecaster,
    df: pd.DataFrame,
    n_windows: int = 5,
    horizon: int = 28,
) -> pd.DataFrame:
    """
    Expanding-window rolling-origin evaluation.

    The series in *df* must have columns: ``unique_id``, ``ds``, ``y``.
    Each window trains on all data up to the cutoff date and evaluates on
    the next *horizon* steps.

    Parameters
    ----------
    forecaster : NeuralForecast-compatible forecaster
        Must implement ``.fit(train_df)`` and ``.predict() -> pd.DataFrame``.
    df : pd.DataFrame
        Full dataset with columns [unique_id, ds, y].
    n_windows : int
        Number of rolling-origin windows to evaluate.
    horizon : int
        Forecast horizon in time steps.

    Returns
    -------
    pd.DataFrame
        Columns: window_id, cutoff_date, unique_id, MAE, MASE, coverage_90.
    """
    df = df.copy().sort_values(["unique_id", "ds"]).reset_index(drop=True)
    dates = sorted(df["ds"].unique())
    n_dates = len(dates)

    if n_dates < horizon * 2:
        raise ValueError(
            f"Not enough time steps ({n_dates}) for {n_windows} windows "
            f"with horizon {horizon}."
        )

    # Place cutoff dates evenly in the middle portion of the timeline
    # so each window has at least `horizon` observations before and after.
    first_cutoff_idx = n_dates - horizon * (n_windows + 1)
    if first_cutoff_idx < horizon:
        first_cutoff_idx = horizon

    cutoff_indices = np.linspace(
        first_cutoff_idx,
        n_dates - horizon - 1,
        num=n_windows,
        dtype=int,
    )

    records = []
    for window_id, cutoff_idx in enumerate(cutoff_indices):
        cutoff_date = dates[cutoff_idx]
        train_mask = df["ds"] <= cutoff_date
        val_mask = (df["ds"] > cutoff_date) & (
            df["ds"] <= dates[min(cutoff_idx + horizon, n_dates - 1)]
        )

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()

        if val_df.empty:
            continue

        forecaster.fit(train_df)
        preds_df = forecaster.predict()

        merged = preds_df.merge(
            val_df[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner"
        )
        if merged.empty:
            continue

        # Identify forecast column (first non-identifier column)
        forecast_col = [
            c for c in merged.columns if c not in ("unique_id", "ds", "y")
        ][0]

        for uid, grp in merged.groupby("unique_id"):
            yt = grp["y"].values
            yp = grp[forecast_col].values
            ytrain = train_df[train_df["unique_id"] == uid]["y"].values

            # lo/hi columns if present, else dummy ±1
            lo_col = f"{forecast_col}-lo-90"
            hi_col = f"{forecast_col}-hi-90"
            qlo = grp[lo_col].values if lo_col in grp.columns else yp - 1.0
            qhi = grp[hi_col].values if hi_col in grp.columns else yp + 1.0

            records.append({
                "window_id": window_id,
                "cutoff_date": cutoff_date,
                "unique_id": uid,
                "MAE": mae(yt, yp),
                "MASE": mase(yt, yp, ytrain),
                "coverage_90": coverage(yt, qlo, qhi),
            })

    return pd.DataFrame(records)
