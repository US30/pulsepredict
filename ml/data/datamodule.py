"""
Lightning DataModule for PulsePredict.

NOTE: NeuralForecast manages its own DataLoaders internally. This module's
role is purely to orchestrate data download, feature engineering, and
train/val/test splitting in a structured, reproducible way.
``train_dataloader`` / ``val_dataloader`` / ``test_dataloader`` are NOT
implemented here — NeuralForecast models receive ``self.train_df`` etc.
directly.

Typical usage with a NeuralForecast model:

    dm = ForecastDataModule(DatasetConfig(), FeatureConfig())
    dm.prepare_data()
    dm.setup()
    nf_model.fit(dm.train)
    forecasts = nf_model.predict()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightning as L
import pandas as pd

from ml.data.dataset import DatasetConfig, M5Dataset
from ml.data.feature_lib import FeatureConfig, TimeSeriesFeatureLib


class ForecastDataModule(L.LightningDataModule):
    """Manages dataset preparation and splitting for NeuralForecast models.

    NOTE: NeuralForecast handles its own DataLoaders internally. This module
    just manages train / val / test splits as plain pandas DataFrames and
    exposes them via properties. Pass ``dm.train`` directly to
    ``NeuralForecast.fit()``.

    Parameters
    ----------
    config:
        DatasetConfig controlling dataset selection and date cutoffs.
    feature_config:
        FeatureConfig controlling lag, rolling, and calendar feature generation.
    add_features:
        Whether to run feature engineering before returning splits.
        Set to ``False`` when the model handles its own feature extraction
        (e.g. PatchTST, TFT with built-in covariates).
    """

    def __init__(
        self,
        config: DatasetConfig,
        feature_config: FeatureConfig,
        add_features: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.feature_config = feature_config
        self.add_features = add_features

        self._raw_df = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

        self._dataset = M5Dataset(config)

    # ------------------------------------------------------------------
    # LightningDataModule interface
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        """Download or validate raw data files.

        Checks that the required CSV files exist in ``config.data_dir``.
        If files are missing, raises ``FileNotFoundError`` with guidance on
        where to obtain the M5 dataset from Kaggle.

        This method is called on the main process only (DDP-safe).
        """
        data_dir = Path(self.config.data_dir)
        required = [
            "sales_train_evaluation.csv",
            "calendar.csv",
            "sell_prices.csv",
        ]
        missing = [f for f in required if not (data_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing M5 data files in {data_dir}: {missing}.\n"
                "Download from: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data\n"
                "Then place CSV files in the data_dir directory."
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and create splits.

        Runs on every process in DDP. Loads raw CSVs, optionally applies
        feature engineering, then splits into train / val / test.

        Parameters
        ----------
        stage:
            ``"fit"``, ``"validate"``, ``"test"``, or ``None`` (all stages).
            Currently all splits are always computed regardless of stage.
        """
        import polars as pl

        data_dir = Path(self.config.data_dir)
        raw_pl = self._dataset.load_raw(data_dir)
        self._raw_df = raw_pl

        if self.add_features:
            lib = TimeSeriesFeatureLib()
            raw_pl = lib.build_features(raw_pl, self.feature_config)

        train_df, val_df, test_df = self._dataset.get_splits(raw_pl)
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def train(self) -> pd.DataFrame:
        """Training split as a pandas DataFrame (unique_id, ds, y)."""
        if self.train_df is None:
            raise RuntimeError("Call setup() before accessing train.")
        return self.train_df

    @property
    def val(self) -> pd.DataFrame:
        """Validation split as a pandas DataFrame (unique_id, ds, y)."""
        if self.val_df is None:
            raise RuntimeError("Call setup() before accessing val.")
        return self.val_df

    @property
    def test(self) -> pd.DataFrame:
        """Test split as a pandas DataFrame (unique_id, ds, y)."""
        if self.test_df is None:
            raise RuntimeError("Call setup() before accessing test.")
        return self.test_df

    @property
    def n_series(self) -> int:
        """Number of unique time series in the training set."""
        if self.train_df is None:
            raise RuntimeError("Call setup() before accessing n_series.")
        return int(self.train_df["unique_id"].nunique())

    @property
    def horizon(self) -> int:
        """Forecast horizon from config."""
        return self.config.horizon

    @property
    def freq(self) -> str:
        """Pandas frequency string inferred from config dataset type."""
        # M5 is daily; ETT variants are hourly or 15-minute
        if self.config.dataset == "m5":
            return "D"
        elif "h" in self.config.dataset.lower():
            return "H"
        elif "m" in self.config.dataset.lower():
            return "15T"
        return "D"

    # ------------------------------------------------------------------
    # Unused DataLoader methods — NeuralForecast manages its own loaders
    # ------------------------------------------------------------------

    def train_dataloader(self):
        raise NotImplementedError(
            "NeuralForecast manages DataLoaders internally. "
            "Pass dm.train directly to NeuralForecast.fit()."
        )

    def val_dataloader(self):
        raise NotImplementedError(
            "NeuralForecast manages DataLoaders internally. "
            "Pass dm.val directly to NeuralForecast.fit()."
        )

    def test_dataloader(self):
        raise NotImplementedError(
            "NeuralForecast manages DataLoaders internally. "
            "Pass dm.test directly to NeuralForecast.predict()."
        )
