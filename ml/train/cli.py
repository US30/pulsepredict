from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import mlflow
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _get_model_class(model_key: str):
    """Return (ForecasterClass, ConfigClass) for the given model key."""
    if model_key == "patchtst":
        from ml.models.patchtst import PatchTSTForecaster, PatchTSTConfig
        return PatchTSTForecaster, PatchTSTConfig
    elif model_key == "tft":
        from ml.models.tft import TFTForecaster, TFTConfig
        return TFTForecaster, TFTConfig
    elif model_key == "nbeatsx":
        from ml.models.nbeatsx import NBEATSxForecaster, NBEATSxConfig
        return NBEATSxForecaster, NBEATSxConfig
    elif model_key == "deepar":
        from ml.models.deepar import DeepARForecaster, DeepARConfig
        return DeepARForecaster, DeepARConfig
    elif model_key == "chronos":
        from ml.models.chronos import ChronosForecaster, ChronosConfig
        return ChronosForecaster, ChronosConfig
    else:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            "Supported: patchtst, tft, nbeatsx, deepar, chronos."
        )


def _build_config(ConfigClass, config_dict: dict):
    """Instantiate ConfigClass from a flat config dict, ignoring unknown keys."""
    import dataclasses
    known = {f.name for f in dataclasses.fields(ConfigClass)}
    filtered = {k: v for k, v in config_dict.items() if k in known}
    return ConfigClass(**filtered)


# ---------------------------------------------------------------------------
# fit command
# ---------------------------------------------------------------------------

def fit(args: argparse.Namespace) -> None:
    """Load config, train a forecaster, evaluate on validation, log to MLflow."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = args.experiment_name or config.get("experiment_name", "pulsepredict")
    mlflow.set_experiment(experiment_name)

    model_key = config["model"]
    ForecasterClass, ConfigClass = _get_model_class(model_key)

    # Build typed config from YAML (unknown keys silently dropped)
    model_config = _build_config(ConfigClass, config)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    from ml.data.dataset import M5Dataset, DatasetConfig

    ds_cfg_keys = {"dataset", "data_dir", "horizon", "train_cutoff", "val_cutoff", "test_cutoff"}
    ds_cfg_dict = {k: v for k, v in config.items() if k in ds_cfg_keys}
    dataset_cfg = DatasetConfig(**ds_cfg_dict)
    dataset = M5Dataset(dataset_cfg)

    data_dir = Path(dataset_cfg.data_dir)
    logger.info("Loading M5 data from %s …", data_dir)
    try:
        raw = dataset.load_raw(data_dir)
        train_df, val_df, _ = dataset.get_splits(raw)
    except FileNotFoundError:
        logger.warning("M5 data not found — generating synthetic data for smoke run")
        import numpy as np, pandas as pd
        n_series, n_obs = 10, 200
        horizon = getattr(model_config, "horizon", 28)
        rows = []
        for i in range(n_series):
            uid = f"SYNTH_{i:03d}"
            for t, d in enumerate(pd.date_range("2014-01-01", periods=n_obs)):
                rows.append({"unique_id": uid, "ds": d, "y": float(np.random.poisson(10))})
        full = pd.DataFrame(rows)
        split = n_obs - horizon
        train_df = full[full.ds < full.ds.unique()[split]].copy()
        val_df = full[full.ds >= full.ds.unique()[split]].copy()

    # Optional feature engineering
    if config.get("features"):
        from ml.data.feature_lib import TimeSeriesFeatureLib, FeatureConfig
        import pandas as pd, polars as pl
        feat_cfg = FeatureConfig(**config["features"])
        lib = TimeSeriesFeatureLib()
        train_df = lib.build_features(pl.from_pandas(train_df), feat_cfg).to_pandas()
        val_df = lib.build_features(pl.from_pandas(val_df), feat_cfg).to_pandas()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    forecaster = ForecasterClass(model_config)
    run_name = f"{model_key}_h{getattr(model_config, 'horizon', config.get('horizon', 28))}"

    with mlflow.start_run(run_name=run_name):
        # Log all scalar config params
        flat_params: dict = {}
        for k, v in config.items():
            if isinstance(v, (int, float, str, bool)):
                flat_params[k] = v
            elif isinstance(v, list):
                flat_params[k] = str(v)
        mlflow.log_params(flat_params)

        logger.info("Fitting %s …", model_key)
        if model_key == "chronos":
            # Chronos is zero-shot — no fit needed; predict directly
            logger.info("Chronos is zero-shot — skipping fit, predicting on val series")
            preds_df = forecaster.predict_df(val_df)
        else:
            forecaster.fit(train_df)
            logger.info("Generating validation forecasts …")
            preds_df = forecaster.predict()

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------
        from ml.eval.metrics import evaluate_forecast
        import pandas as pd

        val_dates = preds_df["ds"].unique() if hasattr(preds_df["ds"], "unique") else set(preds_df["ds"])
        val_trimmed = val_df[val_df["ds"].isin(val_dates)].copy()
        merged = preds_df.merge(val_trimmed[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner")

        y_true = merged["y"]
        # Pick median / main forecast column
        forecast_cols = [c for c in merged.columns if c not in ("unique_id", "ds", "y")]
        y_pred_col = next((c for c in forecast_cols if "0.5" in c or "median" in c.lower()), forecast_cols[0] if forecast_cols else None)
        if y_pred_col is None:
            logger.warning("No forecast column found — skipping metrics")
        else:
            y_pred = merged[y_pred_col]
            lo_col = next((c for c in forecast_cols if "0.1" in c or "lo" in c.lower()), None)
            hi_col = next((c for c in forecast_cols if "0.9" in c or "hi" in c.lower()), None)
            q_lo = merged[lo_col] if lo_col else y_pred - 1.0
            q_hi = merged[hi_col] if hi_col else y_pred + 1.0

            metrics = evaluate_forecast(
                y_true=y_true,
                y_pred=y_pred,
                q_lo=q_lo,
                q_hi=q_hi,
                y_train=train_df["y"],
            )
            metric_dict = {
                "val/mae": metrics.mae,
                "val/rmse": metrics.rmse,
                "val/mase": metrics.mase,
                "val/smape": metrics.smape,
                "val/coverage_90": metrics.coverage_90,
                "val/winkler_90": metrics.winkler_90,
            }
            mlflow.log_metrics(metric_dict)
            logger.info("Validation metrics: %s", metric_dict)

        # ------------------------------------------------------------------
        # Artifact saving
        # ------------------------------------------------------------------
        model_artifact_dir = output_dir / model_key
        model_artifact_dir.mkdir(parents=True, exist_ok=True)

        preds_csv = model_artifact_dir / "val_predictions.csv"
        preds_df.to_csv(preds_csv, index=False)
        shutil.copy(config_path, model_artifact_dir / "config.yaml")

        if model_key != "chronos" and hasattr(forecaster, "save"):
            forecaster.save(str(model_artifact_dir / "model"))

        mlflow.log_artifacts(str(model_artifact_dir))
        logger.info("Artifacts saved → %s", model_artifact_dir)

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# predict command
# ---------------------------------------------------------------------------

def predict(args: argparse.Namespace) -> None:
    """Load a saved forecaster and generate predictions."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_key = config["model"]
    ForecasterClass, ConfigClass = _get_model_class(model_key)
    model_config = _build_config(ConfigClass, config)

    model_path = Path(args.model_path)
    saved_model_dir = model_path / "model"
    if saved_model_dir.exists():
        forecaster = ForecasterClass.load(str(saved_model_dir))
    else:
        import joblib
        forecaster = joblib.load(model_path / "model.pkl")

    logger.info("Generating predictions …")
    preds_df = forecaster.predict()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(output_path, index=False)
    logger.info("Predictions saved → %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PulsePredict NeuralForecast training CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Train a forecasting model")
    fit_parser.add_argument("--config", required=True, help="Path to YAML config")
    fit_parser.add_argument("--output-dir", default="artifacts", help="Artifact output directory")
    fit_parser.add_argument("--experiment-name", default=None, help="MLflow experiment name")
    fit_parser.set_defaults(func=fit)

    pred_parser = subparsers.add_parser("predict", help="Inference with a saved model")
    pred_parser.add_argument("--config", required=True, help="Path to YAML config")
    pred_parser.add_argument("--model-path", required=True, help="Directory with saved artifacts")
    pred_parser.add_argument("--output", default="reports/predictions.csv")
    pred_parser.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
