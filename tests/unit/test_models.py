"""
Unit tests for all PulsePredict model classes.

Runs on CPU only — no real data required. NeuralForecast and heavy ML
dependencies are mocked so the suite stays fast and self-contained.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_forecast_df(unique_ids, horizon, model_col="NBEATSx-median"):
    """Return a synthetic forecast DataFrame in NeuralForecast format."""
    dates = pd.date_range("2023-01-01", periods=horizon, freq="D")
    rows = []
    for uid in unique_ids:
        for d in dates:
            rows.append({"unique_id": uid, "ds": d, model_col: float(np.random.rand())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PatchTST
# ---------------------------------------------------------------------------

class TestPatchTST:
    """Tests for PatchTSTConfig and PatchTSTForecaster."""

    def test_config_defaults(self):
        from ml.models.patchtst import PatchTSTConfig

        cfg = PatchTSTConfig()
        assert cfg.horizon == 28, "Default horizon should be 28"
        assert cfg.input_size >= 1, "input_size must be positive"
        assert cfg.max_steps > 0, "max_steps must be positive"
        assert 0 < cfg.learning_rate < 1, "learning_rate should be in (0, 1)"
        assert cfg.batch_size > 0

    def test_forecaster_init(self):
        from ml.models.patchtst import PatchTSTConfig, PatchTSTForecaster

        with patch("ml.models.patchtst.NeuralForecast") as MockNF:
            MockNF.return_value = MagicMock()
            cfg = PatchTSTConfig(max_steps=5)
            forecaster = PatchTSTForecaster(cfg)
            assert forecaster is not None
            # NeuralForecast should have been constructed
            assert MockNF.called

    def test_predict_shape(self):
        from ml.models.patchtst import PatchTSTConfig, PatchTSTForecaster

        horizon = 28
        unique_ids = ["ITEM_1", "ITEM_2"]
        fake_pred = _make_forecast_df(unique_ids, horizon, model_col="PatchTST-median")

        with patch("ml.models.patchtst.NeuralForecast") as MockNF:
            mock_nf_instance = MagicMock()
            mock_nf_instance.predict.return_value = fake_pred
            MockNF.return_value = mock_nf_instance

            cfg = PatchTSTConfig(horizon=horizon, max_steps=5)
            forecaster = PatchTSTForecaster(cfg)
            result = forecaster.predict()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(unique_ids) * horizon
        assert "unique_id" in result.columns
        assert "ds" in result.columns


# ---------------------------------------------------------------------------
# TFT
# ---------------------------------------------------------------------------

class TestTFT:
    """Tests for TFTConfig and TFTForecaster."""

    def test_config_defaults(self):
        from ml.models.tft import TFTConfig

        cfg = TFTConfig()
        assert cfg.horizon == 28
        assert cfg.hidden_size > 0
        assert cfg.num_heads >= 1
        assert cfg.dropout >= 0.0
        assert cfg.max_steps > 0

    def test_forecaster_init(self):
        from ml.models.tft import TFTConfig, TFTForecaster

        with patch("ml.models.tft.NeuralForecast") as MockNF:
            MockNF.return_value = MagicMock()
            cfg = TFTConfig(max_steps=5)
            forecaster = TFTForecaster(cfg)
            assert forecaster is not None
            assert MockNF.called


# ---------------------------------------------------------------------------
# NBEATSx
# ---------------------------------------------------------------------------

class TestNBEATSx:
    """Tests for NBEATSxConfig and NBEATSxForecaster."""

    def test_config_defaults(self):
        from ml.models.nbeatsx import NBEATSxConfig

        cfg = NBEATSxConfig()
        assert cfg.horizon == 28
        assert cfg.n_harmonics >= 1
        assert cfg.n_polynomials >= 1
        assert cfg.max_steps > 0
        assert cfg.batch_size > 0

    def test_forecaster_init(self):
        from ml.models.nbeatsx import NBEATSxConfig, NBEATSxForecaster

        with patch("ml.models.nbeatsx.NeuralForecast") as MockNF:
            MockNF.return_value = MagicMock()
            cfg = NBEATSxConfig(max_steps=5)
            forecaster = NBEATSxForecaster(cfg)
            assert forecaster is not None
            assert MockNF.called


# ---------------------------------------------------------------------------
# DeepAR
# ---------------------------------------------------------------------------

class TestDeepAR:
    """Tests for DeepARConfig and DeepARForecaster."""

    def test_config_defaults(self):
        from ml.models.deepar import DeepARConfig

        cfg = DeepARConfig()
        assert cfg.horizon == 28
        assert cfg.hidden_size > 0
        assert cfg.n_layers >= 1
        assert cfg.max_steps > 0

    def test_forecaster_init(self):
        from ml.models.deepar import DeepARConfig, DeepARForecaster

        with patch("ml.models.deepar.NeuralForecast") as MockNF:
            MockNF.return_value = MagicMock()
            cfg = DeepARConfig(max_steps=5)
            forecaster = DeepARForecaster(cfg)
            assert forecaster is not None
            assert MockNF.called


# ---------------------------------------------------------------------------
# Chronos
# ---------------------------------------------------------------------------

class TestChronos:
    """Tests for ChronosConfig and ChronosForecaster."""

    def test_config_defaults(self):
        from ml.models.chronos import ChronosConfig

        cfg = ChronosConfig()
        assert cfg.horizon == 28
        assert cfg.model_id.startswith("amazon/chronos")
        assert cfg.num_samples > 0

    def test_forecaster_skips_on_no_gpu(self):
        """When ChronosPipeline.from_pretrained raises (e.g. no GPU / no model),
        the forecaster should propagate a clear RuntimeError."""
        from ml.models.chronos import ChronosConfig, ChronosForecaster

        with patch(
            "ml.models.chronos.ChronosPipeline.from_pretrained",
            side_effect=RuntimeError("CUDA out of memory or model unavailable"),
        ):
            cfg = ChronosConfig()
            with pytest.raises(RuntimeError):
                forecaster = ChronosForecaster(cfg)

    def test_vram_gb(self):
        from ml.models.chronos import ChronosConfig

        cfg = ChronosConfig(model_id="amazon/chronos-t5-small")
        assert cfg.vram_gb == pytest.approx(3.0, abs=0.1), (
            "chronos-t5-small should require ~3 GB VRAM"
        )


# ---------------------------------------------------------------------------
# FeatureLib
# ---------------------------------------------------------------------------

class TestFeatureLib:
    """Tests for TimeSeriesFeatureLib."""

    @pytest.fixture()
    def synthetic_polars_df(self):
        """Two series × 100 daily observations in Polars format."""
        import polars as pl

        dates = pd.date_range("2022-01-01", periods=100, freq="D")
        rows = []
        rng = np.random.default_rng(42)
        for uid in ["ITEM_A", "ITEM_B"]:
            for d in dates:
                rows.append({
                    "unique_id": uid,
                    "ds": d.date(),
                    "y": float(rng.poisson(10)),
                })
        return pl.DataFrame(rows)

    def test_build_features_shape(self, synthetic_polars_df):
        from ml.features.feature_lib import TimeSeriesFeatureLib, FeatureConfig

        lib = TimeSeriesFeatureLib()
        cfg = FeatureConfig()
        result = lib.build_features(synthetic_polars_df, cfg)

        # Must have at least one new column beyond the original 3
        assert result.shape[1] > synthetic_polars_df.shape[1], (
            "build_features must add new feature columns"
        )

    def test_calendar_features_present(self, synthetic_polars_df):
        from ml.features.feature_lib import TimeSeriesFeatureLib, FeatureConfig

        lib = TimeSeriesFeatureLib()
        cfg = FeatureConfig()
        result = lib.build_features(synthetic_polars_df, cfg)

        cols = result.columns
        assert "day_of_week" in cols, "day_of_week calendar feature missing"
        assert "month" in cols, "month calendar feature missing"
        assert "week_of_year" in cols, "week_of_year calendar feature missing"

    def test_lag_features_present(self, synthetic_polars_df):
        from ml.features.feature_lib import TimeSeriesFeatureLib, FeatureConfig

        lib = TimeSeriesFeatureLib()
        cfg = FeatureConfig()
        result = lib.build_features(synthetic_polars_df, cfg)

        cols = result.columns
        assert "y_lag_1" in cols, "y_lag_1 lag feature missing"
        assert "y_lag_7" in cols, "y_lag_7 lag feature missing"
