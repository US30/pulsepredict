from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from ml.data.dataset import DatasetConfig, M5Dataset
from ml.eval.backtest import BacktestRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling-origin backtest across all trained models")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/m5"))
    parser.add_argument("--n-windows", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--output", type=Path, default=Path("reports/backtest"))
    parser.add_argument("--experiment-name", default="pulsepredict-backtest")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    log.info("Loading data from %s ...", args.data_dir)
    try:
        ds = DatasetConfig(data_dir=str(args.data_dir), horizon=args.horizon)
        m5 = M5Dataset()
        raw = m5.load_raw(args.data_dir)
        train_df, val_df, test_df = m5.get_splits(raw)
        full_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    except FileNotFoundError:
        log.warning("M5 data not found; generating synthetic data for smoke test")
        import numpy as np
        n_series, n_obs = 10, 200
        full_df = pd.DataFrame(
            {
                "unique_id": [f"ITEM_{i:03d}" for i in range(n_series) for _ in range(n_obs)],
                "ds": pd.date_range("2014-01-01", periods=n_obs).tolist() * n_series,
                "y": np.random.poisson(20, n_series * n_obs).astype(float),
            }
        )

    forecasters: dict = {}
    for model_dir in sorted(args.artifacts_dir.glob("*")):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        try:
            if model_name == "patchtst":
                from ml.models.patchtst import PatchTSTForecaster
                forecasters[model_name] = PatchTSTForecaster.load(model_dir)
            elif model_name == "tft":
                from ml.models.tft import TFTForecaster
                forecasters[model_name] = TFTForecaster.load(model_dir)
            elif model_name == "nbeatsx":
                from ml.models.nbeatsx import NBEATSxForecaster
                forecasters[model_name] = NBEATSxForecaster.load(model_dir)
            elif model_name == "deepar":
                from ml.models.deepar import DeepARForecaster
                forecasters[model_name] = DeepARForecaster.load(model_dir)
            log.info("Loaded %s from %s", model_name, model_dir)
        except Exception as e:
            log.warning("Could not load %s: %s", model_name, e)

    if not forecasters:
        log.warning("No trained models found in %s — backtest skipped", args.artifacts_dir)
        placeholder = {"status": "no_models_found", "artifacts_dir": str(args.artifacts_dir)}
        with open(args.output / "metrics.json", "w") as f:
            json.dump(placeholder, f, indent=2)
        return

    runner = BacktestRunner(
        forecasters=forecasters,
        df=full_df,
        horizon=args.horizon,
        n_windows=args.n_windows,
    )
    results = runner.run()
    runner.log_to_mlflow(results, args.experiment_name)

    results.to_csv(args.output / "results.csv", index=False)
    metrics = results.groupby("model")[["MAE", "MASE", "coverage_90"]].mean().to_dict()
    with open(args.output / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    runner.plot_comparison(results, args.output / "comparison.png")
    log.info("Backtest complete → %s", args.output)


if __name__ == "__main__":
    main()
