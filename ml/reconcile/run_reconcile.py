from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import pandas as pd

from ml.data.dataset import DatasetConfig, M5Dataset
from ml.reconcile.reconciler import HierarchicalReconciler, ReconcileConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_reconcile(
    artifacts_dir: Path = Path("artifacts"),
    data_dir: Path = Path("data/raw/m5"),
    output_dir: Path = Path("reports/reconciled"),
    method: str = "mint_shrink",
    experiment_name: str = "pulsepredict-reconcile",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_config = DatasetConfig(data_dir=str(data_dir))
    m5 = M5Dataset()

    log.info("Loading M5 data...")
    try:
        raw = m5.load_raw(data_dir)
        _, _, test_df = m5.get_splits(raw)
        hierarchy = m5.hierarchy(test_df)
    except FileNotFoundError:
        log.warning("M5 data not found; using synthetic hierarchy for smoke test")
        n = 100
        test_df = pd.DataFrame(
            {
                "unique_id": ["item_001"] * n + ["item_002"] * n,
                "ds": pd.date_range("2016-04-25", periods=n).tolist() * 2,
                "y": list(range(n)) + list(range(n, 2 * n)),
            }
        )
        hierarchy = {
            "Y_df": test_df,
            "S_df": pd.DataFrame(
                {"item_001": [1, 1], "item_002": [1, 0]},
                index=["total", "store_1"],
            ),
            "tags": {"Level0": ["total"], "Level1": ["store_1"]},
        }

    log.info("Loading base forecasts from %s ...", artifacts_dir)
    forecast_path = artifacts_dir / "patchtst" / "forecasts.parquet"
    if forecast_path.exists():
        base_forecasts = pd.read_parquet(forecast_path)
    else:
        log.warning("No saved forecasts found; using actuals as base (smoke mode)")
        base_forecasts = hierarchy["Y_df"].copy()
        base_forecasts["PatchTST"] = base_forecasts["y"] * 1.05

    config = ReconcileConfig(method=method)
    reconciler = HierarchicalReconciler(config)

    log.info("Reconciling with method=%s ...", method)
    try:
        reconciled = reconciler.reconcile(
            Y_df=hierarchy["Y_df"],
            S_df=hierarchy["S_df"],
            tags=hierarchy["tags"],
        )
        comparison = reconciler.compare_base_vs_reconciled(
            base=base_forecasts,
            reconciled=reconciled,
            actuals=hierarchy["Y_df"],
        )
    except Exception as exc:
        log.warning("Reconciliation failed (%s); writing placeholder report", exc)
        reconciled = base_forecasts
        comparison = {"error": str(exc)}

    out_csv = output_dir / "reconciled_forecasts.csv"
    reconciled.to_csv(out_csv, index=False)
    log.info("Saved reconciled forecasts → %s", out_csv)

    report = {"method": method, "comparison": comparison}
    report_path = output_dir / "reconcile_metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Saved metrics → %s", report_path)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"reconcile_{method}"):
        mlflow.log_params({"method": method})
        if isinstance(comparison, dict):
            for k, v in comparison.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
        mlflow.log_artifact(str(report_path))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Hierarchical reconciliation runner")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/m5"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/reconciled"))
    parser.add_argument(
        "--method",
        choices=["mint_shrink", "bottomup", "topdown_proportion"],
        default="mint_shrink",
    )
    parser.add_argument("--experiment-name", default="pulsepredict-reconcile")
    args = parser.parse_args()

    report = run_reconcile(
        artifacts_dir=args.artifacts_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        method=args.method,
        experiment_name=args.experiment_name,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
