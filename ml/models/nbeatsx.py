"""
N-BEATSx forecaster wrapper.

N-BEATSx extends N-BEATS with exogenous covariate support. It decomposes
the forecast into interpretable trend, seasonality, and identity (residual)
stacks — each stack being a stack of fully-connected blocks with basis
expansion.

NOTE: N-BEATSx is CPU-friendly and does NOT require a GPU. Training is
efficient even on modest hardware thanks to the MLP-only architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE, MQLoss
from neuralforecast.models import NBEATSx


@dataclass
class NBEATSxConfig:
    """Configuration for NBEATSxForecaster.

    Attributes
    ----------
    horizon:
        Forecast horizon (number of future steps).
    input_size:
        Look-back window length.
    n_harmonics:
        Number of Fourier harmonics used in the seasonality stack.
    n_polynomials:
        Degree of polynomial used in the trend stack.
    stack_types:
        Ordered list of stack types. Recognised values: "identity", "trend",
        "seasonality".
    n_blocks:
        Number of blocks per stack.
    mlp_units:
        List of hidden-layer width lists for each block. e.g. [[512, 512]]
        means one block with two hidden layers of width 512.
    dropout_prob_theta:
        Dropout probability on the basis-expansion (theta) output.
    batch_size:
        Mini-batch size during training.
    max_steps:
        Maximum gradient update steps.
    learning_rate:
        Adam optimiser learning rate.
    loss:
        "MAE" for point forecasting; "MQLoss" for quantile mode.
    quantiles:
        Quantile levels. Activates MQLoss when non-empty.
    """

    horizon: int = 28
    input_size: int = 104
    n_harmonics: int = 2
    n_polynomials: int = 2
    stack_types: list = field(
        default_factory=lambda: ["identity", "trend", "seasonality"]
    )
    n_blocks: int = 1
    mlp_units: list = field(default_factory=lambda: [[512, 512]])
    dropout_prob_theta: float = 0.0
    batch_size: int = 32
    max_steps: int = 5000
    learning_rate: float = 1e-3
    loss: str = "MAE"
    quantiles: list = field(default_factory=list)


class NBEATSxForecaster:
    """N-BEATSx-backed forecaster.

    Wraps neuralforecast's NBEATSx inside a NeuralForecast container.
    CPU-friendly — no GPU required.

    Parameters
    ----------
    config:
        NBEATSxConfig instance.

    Examples
    --------
    >>> cfg = NBEATSxConfig(horizon=28, max_steps=500)
    >>> model = NBEATSxForecaster(cfg)
    >>> model.fit(train_df)
    >>> forecasts = model.predict()
    """

    def __init__(self, config: NBEATSxConfig) -> None:
        self.config = config
        loss = (
            MQLoss(quantiles=config.quantiles)
            if config.quantiles
            else MAE()
        )
        n_blocks_list = [config.n_blocks] * len(config.stack_types)
        mlp_units_per_stack = config.mlp_units * len(config.stack_types)

        model = NBEATSx(
            h=config.horizon,
            input_size=config.input_size,
            n_harmonics=config.n_harmonics,
            n_polynomials=config.n_polynomials,
            stack_types=config.stack_types,
            n_blocks=n_blocks_list,
            mlp_units=mlp_units_per_stack,
            dropout_prob_theta=config.dropout_prob_theta,
            batch_size=config.batch_size,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            loss=loss,
        )
        self._nf = NeuralForecast(models=[model], freq="D")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "NBEATSxForecaster":
        """Fit N-BEATSx on a NeuralForecast-format DataFrame.

        Parameters
        ----------
        df:
            Long-format DataFrame with columns ``unique_id``, ``ds``, ``y``.
        """
        self._nf.fit(df)
        return self

    def predict(self) -> pd.DataFrame:
        """Produce multi-horizon forecasts.

        Returns
        -------
        pd.DataFrame
            Forecast DataFrame with ``unique_id``, ``ds``, and forecast
            columns (quantile-suffixed when MQLoss was used).
        """
        return self._nf.predict()

    def save(self, path: str | Path) -> None:
        """Persist the fitted model to *path*.

        Parameters
        ----------
        path:
            Target directory for NeuralForecast artefacts.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._nf.save(str(path), model_index=None, overwrite=True, save_dataset=True)

    @classmethod
    def load(cls, path: str | Path) -> "NBEATSxForecaster":
        """Restore a saved NBEATSxForecaster from *path*.

        Parameters
        ----------
        path:
            Directory previously passed to ``save``.
        """
        instance = cls.__new__(cls)
        instance.config = NBEATSxConfig()
        instance._nf = NeuralForecast.load(str(path))
        return instance
