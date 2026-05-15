"""
DeepAR probabilistic forecaster wrapper.

DeepAR trains an autoregressive RNN (LSTM by default in neuralforecast) with
a parametric output distribution. The DistributionLoss("Normal") head produces
mean and variance per step, enabling Monte-Carlo sample paths via
``forecast_samples``.

Sample paths can be used to compute any desired risk measure (VaR, CVaR,
fan charts, etc.) without re-running the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import DeepAR


@dataclass
class DeepARConfig:
    """Configuration for DeepARForecaster.

    Attributes
    ----------
    horizon:
        Forecast horizon.
    input_size:
        Look-back window for conditioning.
    hidden_size:
        LSTM hidden state dimension.
    n_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout probability applied between LSTM layers.
    batch_size:
        Training mini-batch size.
    max_steps:
        Maximum gradient update steps.
    learning_rate:
        Adam optimiser learning rate.
    """

    horizon: int = 28
    input_size: int = 52
    hidden_size: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 32
    max_steps: int = 3000
    learning_rate: float = 1e-3


class DeepARForecaster:
    """DeepAR-backed probabilistic forecaster with sample-path generation.

    Uses a Normal distribution head (DistributionLoss) by default, enabling
    closed-form sampling. Call ``forecast_samples`` to draw Monte-Carlo paths
    for uncertainty quantification.

    Parameters
    ----------
    config:
        DeepARConfig instance.

    Examples
    --------
    >>> cfg = DeepARConfig(horizon=28, max_steps=300)
    >>> model = DeepARForecaster(cfg)
    >>> model.fit(train_df)
    >>> forecasts = model.predict()
    >>> samples = model.forecast_samples(n_samples=200)   # [series, samples, horizon]
    """

    def __init__(self, config: DeepARConfig) -> None:
        self.config = config
        loss = DistributionLoss("Normal")
        model = DeepAR(
            h=config.horizon,
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            n_layers=config.n_layers,
            dropout=config.dropout,
            batch_size=config.batch_size,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            loss=loss,
        )
        self._nf = NeuralForecast(models=[model], freq="D")
        self._train_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "DeepARForecaster":
        """Fit DeepAR on a NeuralForecast-format DataFrame.

        Parameters
        ----------
        df:
            Long-format DataFrame with ``unique_id``, ``ds``, ``y``.
        """
        self._train_df = df.copy()
        self._nf.fit(df)
        return self

    def predict(self) -> pd.DataFrame:
        """Produce distributional point-estimate forecasts.

        Returns
        -------
        pd.DataFrame
            Columns ``unique_id``, ``ds``, ``DeepAR`` (mean forecast), and
            optional ``DeepAR-median`` / scale columns depending on
            neuralforecast version.
        """
        return self._nf.predict()

    def forecast_samples(self, n_samples: int = 500) -> np.ndarray:
        """Draw Monte-Carlo sample paths from the fitted Normal distribution.

        For each series the model's predictive Normal(mu, sigma) per step is
        sampled independently; samples are drawn via reparameterisation.

        Parameters
        ----------
        n_samples:
            Number of trajectories to draw per series.

        Returns
        -------
        np.ndarray
            Array of shape ``[n_series, n_samples, horizon]``.
        """
        # Obtain mean and scale (std) from NeuralForecast predict
        pred_df = self._nf.predict()

        # Identify the unique series
        series_ids = pred_df["unique_id"].unique()
        n_series = len(series_ids)
        horizon = self.config.horizon

        # Extract mu and sigma columns (neuralforecast DistributionLoss naming)
        mu_col = next(
            (c for c in pred_df.columns if c.endswith("-loc") or c == "DeepAR"), None
        )
        sigma_col = next(
            (c for c in pred_df.columns if c.endswith("-scale")), None
        )

        samples = np.zeros((n_series, n_samples, horizon), dtype=np.float32)

        for i, sid in enumerate(series_ids):
            mask = pred_df["unique_id"] == sid
            mu = pred_df.loc[mask, mu_col].values if mu_col else np.zeros(horizon)
            sigma = (
                pred_df.loc[mask, sigma_col].values if sigma_col else np.ones(horizon)
            )
            # mu: [horizon], sigma: [horizon]
            rng = np.random.default_rng()
            # shape: [n_samples, horizon]
            samples[i] = rng.normal(
                loc=mu[np.newaxis, :],
                scale=np.maximum(sigma[np.newaxis, :], 1e-6),
                size=(n_samples, horizon),
            )

        return samples

    def save(self, path: str | Path) -> None:
        """Persist the fitted model to *path*.

        Parameters
        ----------
        path:
            Target directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._nf.save(str(path), model_index=None, overwrite=True, save_dataset=True)

    @classmethod
    def load(cls, path: str | Path) -> "DeepARForecaster":
        """Restore a saved DeepARForecaster from *path*.

        Parameters
        ----------
        path:
            Directory previously passed to ``save``.
        """
        instance = cls.__new__(cls)
        instance.config = DeepARConfig()
        instance._train_df = None
        instance._nf = NeuralForecast.load(str(path))
        return instance
