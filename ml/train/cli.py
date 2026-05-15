import argparse
import yaml
import mlflow
import logging
from pathlib import Path

from ml.data.dataset import M5Dataset, DatasetConfig
from ml.data.feature_lib import TimeSeriesFeatureLib, FeatureConfig

logger = logging.getLogger(__name__)

_MODEL_REGISTRY = {}


def _get_model_class(model_key: str):
    """Lazily import and return the forecaster class for the given model key."""
    if model_key == "patchtst":
        from ml.models.patchtst import PatchTSTForecaster
        return PatchTSTForecaster
    elif model_key == "tft":
        from ml.models.tft import TFTForecaster
        return TFTForecaster
    elif model_key == "nhits":
        from ml.models.nhits import NHITSForecaster
        return NHITSForecaster
    elif model_key == "nbeats":
        from ml.models.nbeats import NBEATSForecaster
        return NBEATSForecaster
    elif model_key == "timesnet":
        from ml.models.timesnet import TimesNetForecaster
        return TimesNetForecaster
    else:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            "Supported: patchtst, tft, nhits, nbeats, timesnet."
        )


def fit(args):
    """Load config, train a forecaster, evaluate on validation, log to MLflow."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = args.experiment_name or config.get("experiment_name", "pulsepredict")
    mlflow.set_experiment(experiment_name)

    model_key = config["model"]
    horizon = config["horizon"]
    ForecasterClass = _get_model_class(model_key)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    logger.info("Loading M5Dataset …")
    dataset_cfg = DatasetConfig(**config.get("dataset", {}))
    dataset = M5Dataset(dataset_cfg)
    train_df, val_df = dataset.get_train_val_split()

    # Optional feature engineering
    if config.get("features"):
        feat_cfg = FeatureConfig(**config["features"])
        feat_lib = TimeSeriesFeatureLib(feat_cfg)
        train_df = feat_lib.transform(train_df)
        val_df = feat_lib.transform(val_df)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    model_params = config.get("model_params", {})
    forecaster = ForecasterClass(h=horizon, **model_params)

    # ------------------------------------------------------------------
    # MLflow run
    # ------------------------------------------------------------------
    run_name = f"{model_key}_h{horizon}"
    with mlflow.start_run(run_name=run_name):
        # Log all config params (flatten nested dicts one level)
        flat_params = {}
        for k, v in config.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat_params[f"{k}.{kk}"] = vv
            else:
                flat_params[k] = v
        mlflow.log_params(flat_params)

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        logger.info(f"Fitting {model_key} …")
        forecaster.fit(train_df)

        # ------------------------------------------------------------------
        # Validation predictions
        # ------------------------------------------------------------------
        logger.info("Generating validation forecasts …")
        preds_df = forecaster.predict()

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------
        from ml.eval.metrics import evaluate_forecast

        # Merge preds with val actuals on (unique_id, ds)
        val_df_trimmed = val_df[val_df["ds"].isin(preds_df["ds"].unique())]
        merged = preds_df.merge(
            val_df_trimmed[["unique_id", "ds", "y"]],
            on=["unique_id", "ds"],
            how="inner",
        )

        y_true = merged["y"]
        y_pred = merged.get(model_key, merged.iloc[:, 2])  # first forecast column

        # Quantile columns (lo/hi) may be present
        q_lo = merged.get(f"{model_key}-lo-90", y_pred - 1.0)
        q_hi = merged.get(f"{model_key}-hi-90", y_pred + 1.0)

        # Training target for MASE denominator
        y_train_all = train_df.groupby("unique_id")["y"].apply(list)
        y_train_flat = train_df["y"]

        metrics = evaluate_forecast(
            y_true=y_true,
            y_pred=y_pred,
            q_lo=q_lo,
            q_hi=q_hi,
            y_train=y_train_flat,
        )

        metric_dict = {
            "val/mae": metrics.mae,
            "val/mse": metrics.mse,
            "val/rmse": metrics.rmse,
            "val/mase": metrics.mase,
            "val/smape": metrics.smape,
            "val/coverage_90": metrics.coverage_90,
            "val/coverage_80": metrics.coverage_80,
            "val/winkler_90": metrics.winkler_90,
            "val/crps": metrics.crps,
        }
        mlflow.log_metrics(metric_dict)
        logger.info(f"Validation metrics: {metric_dict}")

        # ------------------------------------------------------------------
        # Artifact saving
        # ------------------------------------------------------------------
        model_artifact_dir = output_dir / model_key
        model_artifact_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions CSV
        preds_csv = model_artifact_dir / "val_predictions.csv"
        preds_df.to_csv(preds_csv, index=False)

        # Save config copy alongside the model
        import shutil
        shutil.copy(config_path, model_artifact_dir / "config.yaml")

        # Persist model (NeuralForecast .save convention)
        if hasattr(forecaster, "save"):
            forecaster.save(str(model_artifact_dir / "model"))
        else:
            import joblib
            joblib.dump(forecaster, model_artifact_dir / "model.pkl")

        mlflow.log_artifacts(str(model_artifact_dir))
        logger.info(f"Artifacts saved to {model_artifact_dir}")

    logger.info("Training complete.")


def predict(args):
    """Load a saved forecaster and generate predictions."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_key = config["model"]
    horizon = config["horizon"]
    ForecasterClass = _get_model_class(model_key)

    model_path = Path(args.model_path)
    if (model_path / "model").exists():
        forecaster = ForecasterClass.load(str(model_path / "model"))
    else:
        import joblib
        forecaster = joblib.load(model_path / "model.pkl")

    logger.info("Generating predictions …")
    preds_df = forecaster.predict()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="PulsePredict NeuralForecast training CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- fit subcommand ------------------------------------------------
    fit_parser = subparsers.add_parser("fit", help="Train a forecasting model")
    fit_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    fit_parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory to save model artifacts (default: artifacts/)",
    )
    fit_parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name (overrides config value)",
    )
    fit_parser.set_defaults(func=fit)

    # ---- predict subcommand --------------------------------------------
    pred_parser = subparsers.add_parser("predict", help="Run inference with a saved model")
    pred_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (for model type + horizon)",
    )
    pred_parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Directory containing saved model artifacts",
    )
    pred_parser.add_argument(
        "--output",
        type=str,
        default="reports/predictions.csv",
        help="Output CSV path for predictions",
    )
    pred_parser.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
