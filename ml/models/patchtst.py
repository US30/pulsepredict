"""
PatchTST probabilistic forecaster wrapper around neuralforecast.

PatchTST divides the input sequence into non-overlapping (or strided) patches
and processes them with a Transformer encoder. When ``quantiles`` are provided,
training uses MQLoss so the model produces a full predictive distribution.
The 90 % prediction interval is obtained from the q0.1 (lower) and q0.9 (upper)
quantile outputs, while q0.5 is the median point forecast.

Zero-shot usage is NOT supported — call ``fit`` before ``predict``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE, MQLoss
from neuralforecast.models import PatchTST as _PatchTST


@dataclass
class PatchTSTConfig:
    """Configuration for PatchTSTForecaster.

    Attributes
    ----------
    horizon:
        Number of future steps to forecast.
    input_size:
        Number of past time steps fed as context (look-back window).
    patch_len:
        Length of each patch extracted from the input sequence.
    stride:
        Step size between consecutive patches.
    d_model:
        Dimension of the Transformer embedding.
    n_heads:
        Number of attention heads in each Transformer layer.
    n_layers:
        Number of Transformer encoder layers.
    dropout:
        Dropout probability applied inside the Transformer.
    batch_size:
        Mini-batch size during training.
    max_steps:
        Maximum number of gradient update steps.
    learning_rate:
        Adam optimiser learning rate.
    loss:
        Loss identifier; "MAE" uses point loss, "MQLoss" enables quantile mode.
        Overridden automatically to "MQLoss" when ``quantiles`` is non-empty.
    quantiles:
        List of quantile levels to predict. Defaults to [0.1, 0.5, 0.9] which
        gives a symmetric 80 % PI plus the median.
    """

    horizon: int = 28
    input_size: int = 104
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    n_heads: int = 16
    n_layers: int = 3
    dropout: float = 0.1
    batch_size: int = 32
    max_steps: int = 5000
    learning_rate: float = 1e-4
    loss: str = "MAE"
    quantiles: list = field(default_factory=lambda: [0.1, 0.5, 0.9])


class PatchTSTForecaster:
    """Probabilistic time-series forecaster backed by PatchTST.

    The 90 % prediction interval is constructed from the q0.1 and q0.9 outputs
    of MQLoss. q0.5 is the median point forecast. Training is end-to-end via
    NeuralForecast's built-in training loop.

    Parameters
    ----------
    config:
        PatchTSTConfig instance controlling all hyper-parameters.

    Examples
    --------
    >>> cfg = PatchTSTConfig(horizon=28, max_steps=100)
    >>> model = PatchTSTForecaster(cfg)
    >>> model.fit(train_df)           # train_df: pd.DataFrame [unique_id, ds, y]
    >>> forecasts = model.predict()   # returns pd.DataFrame with quantile cols
    """

    def __init__(self, config: PatchTSTConfig) -> None:
        self.config = config
        loss = self._build_loss()
        model = _PatchTST(
            h=config.horizon,
            input_size=config.input_size,
            patch_len=config.patch_len,
            stride=config.stride,
            hidden_size=config.d_model,
            n_heads=config.n_heads,
            encoder_layers=config.n_layers,
            dropout=config.dropout,
            batch_size=config.batch_size,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            loss=loss,
        )
        self._nf = NeuralForecast(models=[model], freq="D")

    def _build_loss(self):
        if self.config.quantiles:
            return MQLoss(quantiles=self.config.quantiles)
        return MAE()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "PatchTSTForecaster":
        """Fit the model on a NeuralForecast-format DataFrame.

        Parameters
        ----------
        df:
            Long-format DataFrame with columns ``unique_id``, ``ds``, ``y``.
        """
        self._nf.fit(df)
        return self

    def predict(self) -> pd.DataFrame:
        """Generate multi-horizon forecasts.

        Returns
        -------
        pd.DataFrame
            Forecast DataFrame indexed by ``unique_id`` and ``ds``.
            Quantile columns are named ``PatchTST-q0.10``, ``PatchTST-q0.50``,
            ``PatchTST-q0.90`` (or ``PatchTST`` for point forecasts).
        """
        return self._nf.predict()

    def save(self, path: str | Path) -> None:
        """Persist the fitted NeuralForecast object to *path*.

        Parameters
        ----------
        path:
            Directory path where model artefacts are saved.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._nf.save(str(path), model_index=None, overwrite=True, save_dataset=True)

    @classmethod
    def load(cls, path: str | Path) -> "PatchTSTForecaster":
        """Restore a saved PatchTSTForecaster from *path*.

        Parameters
        ----------
        path:
            Directory previously passed to ``save``.

        Returns
        -------
        PatchTSTForecaster
            Instance with ``_nf`` ready for inference (no re-training needed).
        """
        instance = cls.__new__(cls)
        instance.config = PatchTSTConfig()  # placeholder; not used for inference
        instance._nf = NeuralForecast.load(str(path))
        return instance

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def prediction_interval(
        self, forecasts: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract lower / upper PI columns from a quantile forecast DataFrame.

        Parameters
        ----------
        forecasts:
            Output of ``predict()``.

        Returns
        -------
        pd.DataFrame
            Columns ``unique_id``, ``ds``, ``lower``, ``median``, ``upper``.
        """
        q_cols = {c: c for c in forecasts.columns if "q0" in c}
        lower_col = next((c for c in q_cols if "0.10" in c), None)
        median_col = next((c for c in q_cols if "0.50" in c), None)
        upper_col = next((c for c in q_cols if "0.90" in c), None)

        out = forecasts[["unique_id", "ds"]].copy()
        if lower_col:
            out["lower"] = forecasts[lower_col].values
        if median_col:
            out["median"] = forecasts[median_col].values
        if upper_col:
            out["upper"] = forecasts[upper_col].values
        return out
