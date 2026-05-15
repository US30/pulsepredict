from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import optuna
from optuna.integration import MLflowCallback
from optuna.pruners import HyperbandPruner

logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, base_config: dict) -> float:
    """
    Optuna objective function.

    Suggests hyperparameters based on the model name stored in *base_config*,
    trains for 1/5 of the full budget (multi-fidelity / early stopping), and
    returns the validation MAE so Optuna can minimise it.
    """
    import copy
    import mlflow

    config = copy.deepcopy(base_config)
    model_name = config.get("model", "patchtst")

    # ------------------------------------------------------------------ #
    # Hyperparameter search spaces                                         #
    # ------------------------------------------------------------------ #
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    config.setdefault("model_params", {})

    if model_name == "patchtst":
        config["model_params"]["d_model"] = trial.suggest_categorical(
            "d_model", [64, 128, 256]
        )
        config["model_params"]["n_heads"] = trial.suggest_categorical(
            "n_heads", [4, 8, 16]
        )
        config["model_params"]["patch_len"] = trial.suggest_categorical(
            "patch_len", [8, 16, 32]
        )
        config["model_params"]["dropout"] = trial.suggest_float(
            "dropout", 0.0, 0.3
        )
        config["model_params"]["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True
        )

    elif model_name == "tft":
        config["model_params"]["hidden_size"] = trial.suggest_categorical(
            "hidden_size", [64, 128, 256, 512]
        )
        config["model_params"]["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True
        )

    else:
        # Generic fallback for nhits / nbeats / timesnet
        config["model_params"]["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True
        )
        config["model_params"]["hidden_size"] = trial.suggest_categorical(
            "hidden_size", [64, 128, 256, 512]
        )

    config["model_params"]["batch_size"] = batch_size

    # ------------------------------------------------------------------ #
    # Reduced-fidelity training (1/5 of full steps for pruning purposes)  #
    # ------------------------------------------------------------------ #
    max_steps = config.get("max_steps", 500)
    fidelity_steps = max(1, max_steps // 5)
    config["model_params"]["max_steps"] = fidelity_steps

    # ------------------------------------------------------------------ #
    # Lazy imports so the module remains importable without heavy deps     #
    # ------------------------------------------------------------------ #
    from ml.data.dataset import M5Dataset, DatasetConfig
    from ml.eval.metrics import evaluate_forecast
    from ml.train.cli import _get_model_class, _build_config

    dataset_cfg = DatasetConfig(**config.get("dataset", {}))
    dataset = M5Dataset(dataset_cfg)
    data_dir = Path(dataset_cfg.data_dir)
    try:
        raw = dataset.load_raw(data_dir)
        train_df, val_df, _ = dataset.get_splits(raw)
    except FileNotFoundError:
        import numpy as np
        import pandas as pd
        horizon = config.get("horizon", 28)
        n_obs = 200
        rows = [{"unique_id": "SYNTH", "ds": d, "y": float(np.random.poisson(10))}
                for d in pd.date_range("2014-01-01", periods=n_obs)]
        full = pd.DataFrame(rows)
        train_df = full.iloc[: n_obs - horizon].copy()
        val_df = full.iloc[n_obs - horizon :].copy()

    ForecasterClass, ConfigClass = _get_model_class(model_name)
    merged_cfg = {**config, **config.get("model_params", {})}
    model_config = _build_config(ConfigClass, merged_cfg)
    forecaster = ForecasterClass(model_config)

    forecaster.fit(train_df)
    preds_df = forecaster.predict()

    val_trimmed = val_df[val_df["ds"].isin(preds_df["ds"].unique())]
    merged = preds_df.merge(
        val_trimmed[["unique_id", "ds", "y"]],
        on=["unique_id", "ds"],
        how="inner",
    )

    y_true = merged["y"]
    y_pred_col = [c for c in merged.columns if c not in ("unique_id", "ds", "y")][0]
    y_pred = merged[y_pred_col]

    q_lo = y_pred - 1.0
    q_hi = y_pred + 1.0

    metrics = evaluate_forecast(
        y_true=y_true,
        y_pred=y_pred,
        q_lo=q_lo,
        q_hi=q_hi,
        y_train=train_df["y"],
    )

    val_mae = metrics.mae

    # Report intermediate value so Hyperband can prune bad trials
    trial.report(val_mae, step=fidelity_steps)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_mae


def run_hpo(
    model_name: str,
    base_config: Optional[dict] = None,
    n_trials: int = 50,
    experiment_name: str = "pulsepredict-hpo",
    data_dir: str = "data/raw/m5",
) -> dict:
    """
    Create an Optuna study with HyperbandPruner, run *n_trials* trials, and
    return the best hyperparameters found.

    Parameters
    ----------
    model_name:
        One of "patchtst", "tft", "nhits", "nbeats", "timesnet".
    base_config:
        Base YAML config dict. ``model_name`` is injected automatically.
    n_trials:
        Number of Optuna trials.
    experiment_name:
        MLflow experiment name used by the MLflowCallback.

    Returns
    -------
    dict
        ``best_trial.params`` from the Optuna study.
    """
    if base_config is None:
        base_config = {}

    base_config = {**base_config, "model": model_name, "dataset": {"data_dir": data_dir}}

    pruner = HyperbandPruner(
        min_resource=1,
        max_resource=base_config.get("max_steps", 500) // 5,
        reduction_factor=3,
    )

    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name=f"{experiment_name}_{model_name}",
    )

    mlflow_callback = MLflowCallback(
        tracking_uri=None,  # uses MLFLOW_TRACKING_URI env var or local
        metric_name="val_mae",
        mlflow_kwargs={"experiment_name": experiment_name},
    )

    logger.info(
        f"Starting HPO: model={model_name}, n_trials={n_trials}, "
        f"experiment={experiment_name}"
    )

    study.optimize(
        lambda trial: objective(trial, base_config),
        n_trials=n_trials,
        callbacks=[mlflow_callback],
        gc_after_trial=True,
    )

    best_params = study.best_trial.params
    logger.info(f"Best trial params: {best_params}")
    logger.info(f"Best val MAE: {study.best_trial.value:.4f}")

    return best_params


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="PulsePredict Optuna HPO runner"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="patchtst",
        choices=["patchtst", "tft", "nhits", "nbeats", "timesnet"],
        help="Model to tune",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="pulsepredict-hpo",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional base YAML config (provides dataset/horizon settings)",
    )
    args = parser.parse_args()

    base_config: dict = {}
    if args.config:
        import yaml
        with open(args.config) as f:
            base_config = yaml.safe_load(f) or {}

    best_params = run_hpo(
        model_name=args.model,
        base_config=base_config,
        n_trials=args.n_trials,
        experiment_name=args.experiment_name,
    )

    print("\nBest hyperparameters found:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
