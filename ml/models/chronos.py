"""
Chronos-T5 zero-shot foundation model wrapper.

Chronos is a family of pretrained language models for time-series forecasting
developed by Amazon. It tokenises time series values and performs autoregressive
generation — no task-specific training is required (zero-shot inference).

Usage (zero-shot — no ``fit`` step):

    >>> cfg = ChronosConfig(model_id="amazon/chronos-t5-small", prediction_length=28)
    >>> forecaster = ChronosForecaster(cfg)
    >>> # Single series (numpy)
    >>> samples = forecaster.predict_series(history_array)  # [num_samples, horizon]
    >>> # Many series (pandas long-format)
    >>> forecast_df = forecaster.predict_df(long_df)

Model size / VRAM guide:
    - chronos-t5-tiny   ~ 1.5 GB VRAM
    - chronos-t5-mini   ~ 2.0 GB VRAM
    - chronos-t5-small  ~ 3.0 GB VRAM
    - chronos-t5-base   ~ 6.0 GB VRAM
    - chronos-t5-large  ~ 14.0 GB VRAM
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline


@dataclass
class ChronosConfig:
    """Configuration for ChronosForecaster.

    Attributes
    ----------
    model_id:
        HuggingFace model identifier for the Chronos checkpoint.
    device:
        Compute device: "cuda", "cpu", or "mps".
    torch_dtype:
        String name of the torch dtype used for model weights.
        "bfloat16" is recommended for CUDA; use "float32" for CPU.
    prediction_length:
        Number of future steps to forecast.
    num_samples:
        Number of Monte-Carlo trajectories drawn per prediction call.
    temperature:
        Softmax temperature for token sampling. 1.0 = unscaled.
    top_k:
        Top-K sampling truncation (0 = disabled).
    top_p:
        Nucleus sampling probability threshold (1.0 = disabled).
    """

    model_id: str = "amazon/chronos-t5-small"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    prediction_length: int = 28
    num_samples: int = 20
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0


class ChronosForecaster:
    """Zero-shot time-series forecaster using Chronos-T5.

    No training is required. The model is loaded from HuggingFace on
    ``__init__`` and is immediately ready for inference.

    Parameters
    ----------
    config:
        ChronosConfig controlling the checkpoint and sampling parameters.
    """

    def __init__(self, config: ChronosConfig) -> None:
        self.config = config
        self._pipeline = ChronosPipeline.from_pretrained(
            config.model_id,
            device_map=config.device,
            torch_dtype=getattr(torch, config.torch_dtype),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_series(self, history: np.ndarray) -> np.ndarray:
        """Forecast a single time series by sampling from Chronos.

        Parameters
        ----------
        history:
            1-D array of observed values (the look-back context). Length
            should be at least a few multiples of ``prediction_length``.

        Returns
        -------
        np.ndarray
            Sample trajectories of shape ``[num_samples, prediction_length]``.
        """
        context = torch.tensor(history, dtype=torch.float32).unsqueeze(0)
        forecast = self._pipeline.predict(
            context=context,
            prediction_length=self.config.prediction_length,
            num_samples=self.config.num_samples,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
        )
        # forecast shape: [1, num_samples, prediction_length]
        return forecast.squeeze(0).numpy()  # [num_samples, prediction_length]

    def predict_df(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.DataFrame:
        """Batch-forecast all series in a long-format DataFrame.

        Iterates over unique series IDs, calls ``predict_series`` for each,
        and returns a tidy DataFrame with quantile columns.

        Parameters
        ----------
        df:
            Long-format DataFrame with at least ``id_col``, ``time_col``,
            ``target_col`` columns.
        id_col:
            Column name for series identifier.
        time_col:
            Column name for timestamps.
        target_col:
            Column name for target values.

        Returns
        -------
        pd.DataFrame
            Columns: ``unique_id``, ``ds``, ``q0.1``, ``q0.5``, ``q0.9``.
            ``ds`` values are extrapolated beyond the last observed date using
            the inferred frequency.
        """
        records = []
        for uid, grp in df.groupby(id_col, sort=False):
            grp = grp.sort_values(time_col)
            history = grp[target_col].values.astype(np.float32)

            samples = self.predict_series(history)  # [num_samples, horizon]

            # Infer future dates
            last_date = pd.Timestamp(grp[time_col].iloc[-1])
            try:
                freq = pd.infer_freq(grp[time_col])
            except Exception:
                freq = "D"
            future_dates = pd.date_range(
                start=last_date,
                periods=self.config.prediction_length + 1,
                freq=freq,
            )[1:]

            q10 = np.quantile(samples, 0.1, axis=0)
            q50 = np.quantile(samples, 0.5, axis=0)
            q90 = np.quantile(samples, 0.9, axis=0)

            for t, (lo, med, hi) in enumerate(zip(q10, q50, q90)):
                records.append(
                    {
                        "unique_id": uid,
                        "ds": future_dates[t],
                        "q0.1": float(lo),
                        "q0.5": float(med),
                        "q0.9": float(hi),
                    }
                )

        return pd.DataFrame(records)

    def vram_gb(self) -> float:
        """Return approximate VRAM consumption for the loaded checkpoint.

        Returns
        -------
        float
            Estimated VRAM in gigabytes. Values are rule-of-thumb estimates
            for bfloat16 weights; actual usage depends on batch size and
            sequence length.
        """
        _vram_map = {
            "tiny": 1.5,
            "mini": 2.0,
            "small": 3.0,
            "base": 6.0,
            "large": 14.0,
        }
        model_id_lower = self.config.model_id.lower()
        for key, gb in _vram_map.items():
            if key in model_id_lower:
                return gb
        # Default fallback
        return 3.0
