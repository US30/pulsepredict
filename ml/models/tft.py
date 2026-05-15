"""
Temporal Fusion Transformer (TFT) probabilistic forecaster wrapper.

TFT uses gated residual networks, variable-selection networks, and multi-head
attention to handle static metadata, known future inputs, and observed
time-varying covariates in a single unified architecture.

Quantile outputs (q0.1, q0.5, q0.9) are produced via MQLoss, giving a 80 %
prediction interval with the median as the point forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE, MQLoss
from neuralforecast.models import TemporalFusionTransformer


@dataclass
class TFTConfig:
    """Configuration for TFTForecaster.

    Attributes
    ----------
    horizon:
        Number of future steps to forecast.
    input_size:
        Look-back window (number of past time steps).
    hidden_size:
        Width of all hidden layers inside TFT.
    n_heads:
        Number of multi-head attention heads.
    attn_dropout:
        Dropout applied to the attention weights.
    dropout:
        General dropout probability for all other layers.
    batch_size:
        Mini-batch size during training.
    max_steps:
        Maximum gradient update steps.
    learning_rate:
        Adam optimiser learning rate.
    static_covs:
        List of static (time-invariant) covariate column names.
    time_varying_known:
        List of future-known covariate column names (e.g. calendar features).
    quantiles:
        Quantile levels. MQLoss is used when this list is non-empty.
    """

    horizon: int = 28
    input_size: int = 104
    hidden_size: int = 256
    n_heads: int = 4
    attn_dropout: float = 0.1
    dropout: float = 0.1
    batch_size: int = 16
    max_steps: int = 5000
    learning_rate: float = 5e-4
    static_covs: Optional[list] = None
    time_varying_known: Optional[list] = None
    quantiles: list = field(default_factory=lambda: [0.1, 0.5, 0.9])


class TFTForecaster:
    """TFT-backed probabilistic multi-horizon forecaster.

    Wraps neuralforecast's TemporalFusionTransformer inside a NeuralForecast
    container. Static and future-known covariates are forwarded to the
    underlying model when provided.

    Parameters
    ----------
    config:
        TFTConfig controlling all model hyper-parameters.

    Examples
    --------
    >>> cfg = TFTConfig(horizon=28, max_steps=200)
    >>> model = TFTForecaster(cfg)
    >>> model.fit(train_df)
    >>> forecasts = model.predict()
    """

    def __init__(self, config: TFTConfig) -> None:
        self.config = config
        loss = MQLoss(quantiles=config.quantiles) if config.quantiles else MAE()

        tft_kwargs: dict = dict(
            h=config.horizon,
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            n_head=config.n_heads,
            attn_dropout=config.attn_dropout,
            dropout=config.dropout,
            batch_size=config.batch_size,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            loss=loss,
        )
        if config.static_covs:
            tft_kwargs["stat_exog_list"] = config.static_covs
        if config.time_varying_known:
            tft_kwargs["futr_exog_list"] = config.time_varying_known

        self._model = TemporalFusionTransformer(**tft_kwargs)
        self._nf = NeuralForecast(models=[self._model], freq="D")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        static_df: Optional[pd.DataFrame] = None,
    ) -> "TFTForecaster":
        """Fit TFT on training data.

        Parameters
        ----------
        df:
            Long-format DataFrame with columns ``unique_id``, ``ds``, ``y``
            plus any time-varying covariates declared in ``config``.
        static_df:
            Optional DataFrame with columns ``unique_id`` plus static
            covariate columns declared in ``config.static_covs``.
        """
        fit_kwargs: dict = {"df": df}
        if static_df is not None:
            fit_kwargs["static_df"] = static_df
        self._nf.fit(**fit_kwargs)
        return self

    def predict(self) -> pd.DataFrame:
        """Produce multi-horizon quantile forecasts.

        Returns
        -------
        pd.DataFrame
            Columns ``unique_id``, ``ds``, and quantile forecast columns.
        """
        return self._nf.predict()

    def save(self, path: str | Path) -> None:
        """Save the fitted model to *path*.

        Parameters
        ----------
        path:
            Directory where NeuralForecast artefacts are written.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._nf.save(str(path), model_index=None, overwrite=True, save_dataset=True)

    @classmethod
    def load(cls, path: str | Path) -> "TFTForecaster":
        """Restore a fitted TFTForecaster from *path*.

        Parameters
        ----------
        path:
            Directory previously passed to ``save``.
        """
        instance = cls.__new__(cls)
        instance.config = TFTConfig()
        instance._nf = NeuralForecast.load(str(path))
        return instance

    def attention_weights(self) -> dict:
        """Return attention weight tensors from the last forward pass.

        Currently a stub — full implementation requires hooking into TFT's
        internal multi-head attention modules post-inference. Returns an empty
        dict as placeholder.

        Returns
        -------
        dict
            Keys would be layer indices; values would be np.ndarray attention
            matrices of shape [batch, heads, seq, seq].
        """
        return {}
