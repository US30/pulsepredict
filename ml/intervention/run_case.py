from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from ml.intervention.causal_impact import BayesianCausalImpact, InterventionConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_promotion_case(
    series_id: str = "HOBBIES_1_001_CA_1",
    intervention_date: str = "2016-01-01",
    data_dir: Path = Path("data/raw/m5"),
    output_dir: Path = Path("reports/intervention"),
    experiment_name: str = "pulsepredict-intervention",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        sales = pd.read_csv(data_dir / "sales_train_evaluation.csv")
        cal = pd.read_csv(data_dir / "calendar.csv")
        row = sales[sales["id"] == series_id + "_evaluation"]
        if row.empty:
            row = sales[sales["id"].str.startswith(series_id.split("_CA_1")[0])]
        if row.empty:
            raise FileNotFoundError(f"Series {series_id} not found")
        day_cols = [c for c in row.columns if c.startswith("d_")]
        y = row[day_cols].values.flatten().astype(float)
        dates = pd.to_datetime(cal["date"].values[: len(y)])
        df = pd.DataFrame({"ds": dates, "y": y})
    except FileNotFoundError:
        log.warning("M5 data not found; using synthetic series (smoke mode)")
        n_total = 200
        rng = np.random.default_rng(42)
        t = np.arange(n_total)
        y_pre_true = 50 + 0.05 * t[:100] + rng.normal(0, 3, 100)
        effect = 0.15
        y_post = 50 + 0.05 * t[100:] * (1 + effect) + rng.normal(0, 3, 100)
        dates = pd.date_range("2015-01-01", periods=n_total)
        y = np.concatenate([y_pre_true, y_post])
        df = pd.DataFrame({"ds": dates, "y": y})
        intervention_date = str(dates[100].date())

    intervention_ts = pd.Timestamp(intervention_date)
    pre_mask = df["ds"] < intervention_ts
    post_mask = df["ds"] >= intervention_ts

    y_pre = df.loc[pre_mask, "y"].values
    y_post = df.loc[post_mask, "y"].values
    y_full = np.concatenate([y_pre, y_post])

    log.info("Fitting BayesianCausalImpact on %d pre-period obs...", len(y_pre))
    config = InterventionConfig(mcmc_samples=500, tune=300)
    bci = BayesianCausalImpact(config)
    bci.fit(y_pre)

    log.info("Predicting counterfactual for %d post-period steps...", len(y_post))
    idata = bci.predict_counterfactual(len(y_post))

    result = bci.estimate_effect(y_post, idata)
    result["series_id"] = series_id
    result["intervention_date"] = intervention_date
    result["n_pre"] = int(len(y_pre))
    result["n_post"] = int(len(y_post))

    log.info("Effect: %+.1f%% (95%% CI: [%.1f, %.1f])",
             result["relative_effect_pct"],
             result["credible_interval_95"][0],
             result["credible_interval_95"][1])

    plot_path = str(output_dir / f"{series_id}_impact.png")
    bci.plot(y_full, idata, intervention_date, plot_path)
    log.info("Plot saved → %s", plot_path)

    report_path = output_dir / f"{series_id}_report.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"intervention_{series_id}"):
        mlflow.log_params({"series_id": series_id, "intervention_date": intervention_date})
        mlflow.log_metric("relative_effect_pct", result["relative_effect_pct"])
        mlflow.log_metric("cumulative_effect", result["cumulative_effect"])
        mlflow.log_metric("posterior_prob_positive", result["posterior_prob_positive"])
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(plot_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian CausalImpact case study runner")
    parser.add_argument("--series-id", default="HOBBIES_1_001_CA_1")
    parser.add_argument("--intervention-date", default="2016-01-01")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/m5"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/intervention"))
    parser.add_argument("--experiment-name", default="pulsepredict-intervention")
    args = parser.parse_args()

    result = run_promotion_case(
        series_id=args.series_id,
        intervention_date=args.intervention_date,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
