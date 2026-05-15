"""
Rolling-origin backtest runner for PulsePredict.

Compares multiple trained forecasters on the same test data using
expanding-window cross-validation, logs results to MLflow, and
generates a grouped bar chart comparing MAE and coverage.
"""

import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from ml.eval.metrics import rolling_origin_backtest, evaluate_forecast

logger = logging.getLogger(__name__)


class BacktestRunner:
    """
    Run rolling-origin backtests across multiple forecasters and aggregate results.

    Parameters
    ----------
    forecasters : dict
        Mapping of {model_name: forecaster_instance}. Each forecaster must
        implement ``.fit(train_df)`` and ``.predict() -> pd.DataFrame``.
    df : pd.DataFrame
        Full dataset with columns [unique_id, ds, y].
    horizon : int
        Forecast horizon in time steps.
    n_windows : int
        Number of rolling-origin windows.
    """

    def __init__(
        self,
        forecasters: dict,
        df: pd.DataFrame,
        horizon: int = 28,
        n_windows: int = 5,
    ):
        self.forecasters = forecasters
        self.df = df.copy()
        self.horizon = horizon
        self.n_windows = n_windows

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Run each forecaster through rolling-origin evaluation.

        Returns
        -------
        pd.DataFrame
            Comparison table with columns:
            model, window_id, cutoff_date, unique_id, MAE, MASE, coverage_90.
        """
        all_results = []

        for model_name, forecaster in self.forecasters.items():
            logger.info(f"Backtesting model: {model_name}")
            try:
                results = rolling_origin_backtest(
                    forecaster=forecaster,
                    df=self.df,
                    n_windows=self.n_windows,
                    horizon=self.horizon,
                )
                results.insert(0, "model", model_name)
                all_results.append(results)
                logger.info(
                    f"  {model_name}: mean MAE={results['MAE'].mean():.4f}, "
                    f"mean coverage_90={results['coverage_90'].mean():.3f}"
                )
            except Exception as exc:
                logger.error(f"  {model_name} failed: {exc}")

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_comparison(self, results: pd.DataFrame, output_path: Path) -> None:
        """
        Produce a grouped bar chart comparing MAE and coverage_90 per model.

        Parameters
        ----------
        results : pd.DataFrame
            Output of :meth:`run`.
        output_path : Path
            File path for the saved PNG.
        """
        import matplotlib.pyplot as plt

        if results.empty:
            logger.warning("No results to plot.")
            return

        summary = (
            results.groupby("model")[["MAE", "MASE", "coverage_90"]]
            .mean()
            .reset_index()
        )

        models = summary["model"].tolist()
        x = np.arange(len(models))
        bar_width = 0.25

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: MAE
        ax0 = axes[0]
        ax0.bar(x, summary["MAE"], width=bar_width, label="MAE", color="steelblue")
        ax0.bar(
            x + bar_width, summary["MASE"], width=bar_width, label="MASE", color="coral"
        )
        ax0.set_xticks(x + bar_width / 2)
        ax0.set_xticklabels(models, rotation=15)
        ax0.set_ylabel("Error")
        ax0.set_title("Error Metrics by Model")
        ax0.legend()
        ax0.grid(axis="y", linestyle="--", alpha=0.6)

        # Right: Coverage
        ax1 = axes[1]
        ax1.bar(x, summary["coverage_90"], width=bar_width * 1.5, color="mediumseagreen")
        ax1.axhline(0.90, color="red", linestyle="--", linewidth=1.2, label="Target 90%")
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=15)
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel("Coverage")
        ax1.set_title("90% PI Coverage by Model")
        ax1.legend()
        ax1.grid(axis="y", linestyle="--", alpha=0.6)

        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Comparison chart saved to {output_path}")

    # ------------------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------------------

    def log_to_mlflow(
        self,
        results: pd.DataFrame,
        experiment_name: str = "pulsepredict-backtest",
    ) -> None:
        """
        Log each model's backtest metrics as a separate MLflow run.

        Parameters
        ----------
        results : pd.DataFrame
            Output of :meth:`run`.
        experiment_name : str
            MLflow experiment to use.
        """
        if results.empty:
            logger.warning("No backtest results to log.")
            return

        mlflow.set_experiment(experiment_name)

        for model_name, grp in results.groupby("model"):
            run_name = f"backtest_{model_name}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("model", model_name)
                mlflow.log_param("n_windows", self.n_windows)
                mlflow.log_param("horizon", self.horizon)

                mlflow.log_metric("mean_mae", float(grp["MAE"].mean()))
                mlflow.log_metric("mean_mase", float(grp["MASE"].mean()))
                mlflow.log_metric("mean_coverage_90", float(grp["coverage_90"].mean()))
                mlflow.log_metric("std_mae", float(grp["MAE"].std()))

                # Per-window metrics
                for _, row in grp.iterrows():
                    step = int(row["window_id"])
                    mlflow.log_metric("window_mae", row["MAE"], step=step)
                    mlflow.log_metric("window_coverage_90", row["coverage_90"], step=step)

            logger.info(f"Logged MLflow run: {run_name}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from ml.data.dataset import M5Dataset, DatasetConfig

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="PulsePredict backtest runner")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--output-dir", type=str, default="reports/backtest")
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--n-windows", type=int, default=5)
    parser.add_argument("--experiment-name", type=str, default="pulsepredict-backtest")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    logger.info("Loading dataset …")
    dataset = M5Dataset(DatasetConfig())
    _, test_df = dataset.get_train_val_split()

    # ------------------------------------------------------------------
    # Load trained forecasters from artifacts/
    # ------------------------------------------------------------------
    forecasters: dict = {}
    for model_dir in sorted(artifacts_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        model_path = model_dir / "model"
        pkl_path = model_dir / "model.pkl"

        try:
            from ml.train.cli import _get_model_class
            ForecasterClass = _get_model_class(model_name)
            if model_path.exists():
                forecaster = ForecasterClass.load(str(model_path))
            elif pkl_path.exists():
                import joblib
                forecaster = joblib.load(pkl_path)
            else:
                logger.warning(f"No model file found in {model_dir}, skipping.")
                continue
            forecasters[model_name] = forecaster
            logger.info(f"Loaded model: {model_name}")
        except Exception as exc:
            logger.warning(f"Could not load {model_name}: {exc}")

    if not forecasters:
        logger.error("No models loaded. Exiting.")
        return

    # ------------------------------------------------------------------
    # Run backtest
    # ------------------------------------------------------------------
    runner = BacktestRunner(
        forecasters=forecasters,
        df=test_df,
        horizon=args.horizon,
        n_windows=args.n_windows,
    )

    results = runner.run()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if not results.empty:
        csv_path = output_dir / "metrics.csv"
        results.to_csv(csv_path, index=False)
        logger.info(f"Metrics CSV saved to {csv_path}")

        summary = results.groupby("model")[["MAE", "MASE", "coverage_90"]].mean()
        json_path = output_dir / "metrics.json"
        json_path.write_text(
            json.dumps(summary.reset_index().to_dict(orient="records"), indent=2)
        )
        logger.info(f"Metrics JSON saved to {json_path}")

        chart_path = output_dir / "comparison.png"
        runner.plot_comparison(results, chart_path)

        runner.log_to_mlflow(results, experiment_name=args.experiment_name)
    else:
        logger.warning("Backtest produced no results.")


if __name__ == "__main__":
    main()
