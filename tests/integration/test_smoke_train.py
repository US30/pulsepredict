"""
Integration smoke tests: 1-step training on synthetic data.

These tests are deliberately lightweight (max_steps=5, small batch, short
horizon) so they finish quickly without a GPU.  They verify that the full
training → prediction round-trip produces output of the correct shape and
schema, not that the model has converged.

Mark: integration (excluded from the default unit-test run).
Run with: pytest -m integration tests/integration/
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_train_df() -> pd.DataFrame:
    """M5-format training DataFrame.

    Returns:
        pd.DataFrame with columns [unique_id, ds, y].
        - 10 unique series (unique_id = "ITEM_001" … "ITEM_010")
        - 200 daily observations per series
        - y ~ Uniform(0, 100) integers
    """
    rng = np.random.default_rng(seed=2024)
    n_series = 10
    n_obs = 200
    start_date = pd.Timestamp("2022-01-01")

    rows = []
    for i in range(1, n_series + 1):
        uid = f"ITEM_{i:03d}"
        dates = pd.date_range(start_date, periods=n_obs, freq="D")
        y_values = rng.integers(0, 101, size=n_obs).astype(float)
        for d, y in zip(dates, y_values):
            rows.append({"unique_id": uid, "ds": d, "y": y})

    df = pd.DataFrame(rows)
    assert set(df.columns) == {"unique_id", "ds", "y"}
    assert df["unique_id"].nunique() == n_series
    assert len(df) == n_series * n_obs
    return df


# ---------------------------------------------------------------------------
# NBEATSx smoke
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestNBEATSxSmoke:
    """Smoke test: NBEATSx trains and predicts without errors."""

    @pytest.mark.timeout(60)
    def test_smoke_train_predict(self, synthetic_train_df):
        from ml.models.nbeatsx import NBEATSxConfig, NBEATSxForecaster

        horizon = 7
        cfg = NBEATSxConfig(
            max_steps=5,
            batch_size=4,
            horizon=horizon,
        )
        forecaster = NBEATSxForecaster(cfg)
        forecaster.fit(synthetic_train_df)
        preds = forecaster.predict()

        # --- schema checks ---
        assert isinstance(preds, pd.DataFrame), "predict() must return pd.DataFrame"
        assert "unique_id" in preds.columns, "unique_id column missing"
        assert "ds" in preds.columns, "ds column missing"

        # Must have at least one forecast column
        forecast_cols = [c for c in preds.columns if c not in {"unique_id", "ds"}]
        assert len(forecast_cols) >= 1, "No forecast value columns returned"

        # Check shape: 10 series × 7 horizon steps
        assert len(preds) == synthetic_train_df["unique_id"].nunique() * horizon, (
            f"Expected {synthetic_train_df['unique_id'].nunique() * horizon} rows, "
            f"got {len(preds)}"
        )

        # No NaNs in forecast columns
        assert not preds[forecast_cols].isna().any().any(), (
            "NaN values found in forecast output"
        )


# ---------------------------------------------------------------------------
# DeepAR smoke
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDeepARSmoke:
    """Smoke test: DeepAR trains and predicts without errors."""

    @pytest.mark.timeout(60)
    def test_smoke_train_predict(self, synthetic_train_df):
        from ml.models.deepar import DeepARConfig, DeepARForecaster

        horizon = 7
        cfg = DeepARConfig(
            max_steps=5,
            horizon=horizon,
        )
        forecaster = DeepARForecaster(cfg)
        forecaster.fit(synthetic_train_df)
        preds = forecaster.predict()

        # --- schema checks ---
        assert isinstance(preds, pd.DataFrame)
        assert "unique_id" in preds.columns
        assert "ds" in preds.columns

        forecast_cols = [c for c in preds.columns if c not in {"unique_id", "ds"}]
        assert len(forecast_cols) >= 1

        n_expected = synthetic_train_df["unique_id"].nunique() * horizon
        assert len(preds) == n_expected, (
            f"Expected {n_expected} rows, got {len(preds)}"
        )

        assert not preds[forecast_cols].isna().any().any()


# ---------------------------------------------------------------------------
# SplitConformal smoke
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSplitConformalSmoke:
    """End-to-end smoke: calibrate → predict → check coverage."""

    def test_calibrate_predict_on_synthetic(self):
        from ml.conformal.split_conformal import SplitConformalPredictor, SplitConformalConfig

        rng = np.random.default_rng(99)

        # Calibration data: absolute residuals
        cal_y_true = rng.normal(loc=5.0, scale=1.0, size=300)
        cal_y_hat = cal_y_true + rng.normal(loc=0.0, scale=0.5, size=300)
        cal_residuals = np.abs(cal_y_true - cal_y_hat)

        cfg = SplitConformalConfig(alpha=0.1)
        predictor = SplitConformalPredictor(cfg)
        predictor.calibrate(cal_residuals)
        assert predictor.q_hat is not None and predictor.q_hat > 0

        # Test predictions
        test_y_hat = rng.normal(loc=5.0, scale=1.0, size=100)
        test_y_true = test_y_hat + rng.normal(loc=0.0, scale=0.5, size=100)

        lo, hi = predictor.predict(test_y_hat)
        assert lo.shape == test_y_hat.shape
        assert hi.shape == test_y_hat.shape

        cov = float(np.mean((test_y_true >= lo) & (test_y_true <= hi)))
        assert 0.0 <= cov <= 1.0, "Coverage must be in [0, 1]"
        # Very loose bound — this is a smoke test, not a statistical guarantee
        assert cov > 0.5, f"Coverage {cov:.2f} unexpectedly low"


# ---------------------------------------------------------------------------
# FeatureLib smoke
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFeatureLibSmoke:
    """End-to-end smoke: raw polars df → feature pipeline → output validation."""

    def test_feature_pipeline_end_to_end(self):
        import polars as pl
        from ml.features.feature_lib import TimeSeriesFeatureLib, FeatureConfig

        rng = np.random.default_rng(7)
        n_obs = 120

        dates = pd.date_range("2022-01-01", periods=n_obs, freq="D")
        rows = []
        for uid in ["ITEM_A", "ITEM_B", "ITEM_C"]:
            for d in dates:
                rows.append({
                    "unique_id": uid,
                    "ds": d.date(),
                    "y": float(rng.poisson(15)),
                })
        pl_df = pl.DataFrame(rows)

        lib = TimeSeriesFeatureLib()
        cfg = FeatureConfig()
        result = lib.build_features(pl_df, cfg)

        assert isinstance(result, pl.DataFrame), "build_features must return pl.DataFrame"

        # More columns than input
        assert result.shape[1] > pl_df.shape[1], (
            "Output must have more columns than raw input"
        )

        # Required feature columns
        required_cols = {"day_of_week", "month", "week_of_year", "y_lag_1", "y_lag_7"}
        missing = required_cols - set(result.columns)
        assert not missing, f"Missing feature columns: {missing}"

        # No NaNs except potentially the first few rows of each series (lag edges)
        # We drop the first 7 rows of each series and check the remainder
        n_series = result["unique_id"].n_unique()
        # Rows per series after dropping the 7-row lag warm-up
        warm_up = 7
        df_pandas = result.to_pandas().sort_values(["unique_id", "ds"])
        df_trimmed = (
            df_pandas
            .groupby("unique_id", group_keys=False)
            .apply(lambda g: g.iloc[warm_up:])
        )
        feature_cols = [c for c in result.columns if c not in {"unique_id", "ds", "y"}]
        nan_counts = df_trimmed[feature_cols].isna().sum()
        cols_with_nans = nan_counts[nan_counts > 0].index.tolist()
        assert not cols_with_nans, (
            f"NaN values found in non-edge feature rows for columns: {cols_with_nans}"
        )
