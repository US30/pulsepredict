from __future__ import annotations

import argparse
import json

from ml.train.hpo import run_hpo


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO runner for PulsePredict models")
    parser.add_argument(
        "--model",
        choices=["patchtst", "tft", "nbeatsx", "deepar"],
        required=True,
        help="Model to tune",
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--experiment-name", default="pulsepredict-hpo")
    parser.add_argument("--data-dir", default="data/raw/m5")
    args = parser.parse_args()

    best_params = run_hpo(
        model_name=args.model,
        n_trials=args.n_trials,
        experiment_name=args.experiment_name,
        data_dir=args.data_dir,
    )
    print(f"\nBest params for {args.model}:")
    print(json.dumps(best_params, indent=2))


if __name__ == "__main__":
    main()
