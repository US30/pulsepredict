"""
Week 6: Hierarchical reconciliation with MinT, BottomUp, TopDown.

Builds M5 hierarchy (item → dept → cat → state → total), generates base
forecasts at all levels using StatsForecast Naive/SeasonalNaive, then
reconciles with hierarchicalforecast and compares accuracy.
"""

import json
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5001")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports/reconciled")


def load_m5_hierarchy():
    """Load M5 data and build hierarchy at all aggregation levels."""
    from ml.data.dataset import M5Dataset, DatasetConfig

    cfg = DatasetConfig(
        data_dir="data/raw/m5",
        max_series=1000,
        train_cutoff="2016-03-27",
        val_cutoff="2016-04-24",
        test_cutoff="2016-05-22",
    )
    ds = M5Dataset(cfg)
    raw = ds.load_raw(Path(cfg.data_dir))

    raw_pd = raw.to_pandas()
    raw_pd["ds"] = pd.to_datetime(raw_pd["ds"])

    # Extract hierarchy components from unique_id
    # M5 unique_id format: CATEGORY_DEPT_ITEM_STATE_STORE
    # e.g., FOODS_1_001_WI_3
    def parse_hierarchy(uid):
        parts = uid.split("_")
        if len(parts) >= 5:
            cat = parts[0]
            dept = f"{parts[0]}_{parts[1]}"
            state = parts[3]
            return cat, dept, state
        return "UNKNOWN", "UNKNOWN", "UNKNOWN"

    unique_ids = raw_pd["unique_id"].unique()
    hierarchy_map = {}
    for uid in unique_ids:
        cat, dept, state = parse_hierarchy(uid)
        hierarchy_map[uid] = {"cat": cat, "dept": dept, "state": state}

    # Build aggregated series at each level
    core = raw_pd[["unique_id", "ds", "y"]].copy()
    core["cat"] = core["unique_id"].map(lambda x: hierarchy_map.get(x, {}).get("cat", "UNK"))
    core["dept"] = core["unique_id"].map(lambda x: hierarchy_map.get(x, {}).get("dept", "UNK"))
    core["state"] = core["unique_id"].map(lambda x: hierarchy_map.get(x, {}).get("state", "UNK"))

    # Bottom level
    bottom_df = core[["unique_id", "ds", "y"]].copy()

    # Aggregate levels
    agg_frames = []

    # Total
    total = core.groupby("ds")["y"].sum().reset_index()
    total["unique_id"] = "Total"
    agg_frames.append(total[["unique_id", "ds", "y"]])

    # State
    for state, grp in core.groupby("state"):
        agg = grp.groupby("ds")["y"].sum().reset_index()
        agg["unique_id"] = state
        agg_frames.append(agg[["unique_id", "ds", "y"]])

    # Category
    for cat, grp in core.groupby("cat"):
        agg = grp.groupby("ds")["y"].sum().reset_index()
        agg["unique_id"] = cat
        agg_frames.append(agg[["unique_id", "ds", "y"]])

    # Department
    for dept, grp in core.groupby("dept"):
        agg = grp.groupby("ds")["y"].sum().reset_index()
        agg["unique_id"] = dept
        agg_frames.append(agg[["unique_id", "ds", "y"]])

    # Combine all levels
    Y_df = pd.concat([bottom_df[["unique_id", "ds", "y"]]] + agg_frames, ignore_index=True)
    Y_df = Y_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # Build tags (ordered from top to bottom)
    states = sorted(core["state"].unique().tolist())
    cats = sorted(core["cat"].unique().tolist())
    depts = sorted(core["dept"].unique().tolist())
    items = sorted(unique_ids.tolist())

    tags = OrderedDict({
        "Total": np.array(["Total"]),
        "State": np.array(states),
        "Category": np.array(cats),
        "Department": np.array(depts),
        "Item": np.array(items),
    })

    # Build S matrix
    all_series = ["Total"] + states + cats + depts + items
    n_all = len(all_series)
    n_bottom = len(items)

    S = np.zeros((n_all, n_bottom), dtype=int)
    item_to_idx = {item: i for i, item in enumerate(items)}
    series_to_row = {s: i for i, s in enumerate(all_series)}

    for uid in items:
        cat, dept, state = parse_hierarchy(uid)
        col = item_to_idx[uid]
        S[series_to_row[uid], col] = 1
        S[series_to_row["Total"], col] = 1
        if state in series_to_row:
            S[series_to_row[state], col] = 1
        if cat in series_to_row:
            S[series_to_row[cat], col] = 1
        if dept in series_to_row:
            S[series_to_row[dept], col] = 1

    S_df = pd.DataFrame(S, columns=items)
    S_df.insert(0, "unique_id", all_series)

    logger.info(f"Hierarchy built: {n_all} total series ({n_bottom} bottom, "
                f"{len(states)} states, {len(cats)} cats, {len(depts)} depts)")

    return Y_df, S_df, tags, cfg


def generate_base_forecasts(Y_df, train_cutoff, val_cutoff):
    """Generate base forecasts at all hierarchy levels using StatsForecast.

    Returns (Y_hat_df, val_df, Y_fitted) where Y_fitted contains insample
    predictions needed for MinTrace covariance estimation.
    """
    from statsforecast import StatsForecast
    from statsforecast.models import Naive, SeasonalNaive

    Y_df = Y_df.copy()
    Y_df["ds"] = pd.to_datetime(Y_df["ds"])

    train = Y_df[Y_df["ds"] <= pd.Timestamp(train_cutoff)].copy()
    val = Y_df[(Y_df["ds"] > pd.Timestamp(train_cutoff)) &
               (Y_df["ds"] <= pd.Timestamp(val_cutoff))].copy()

    logger.info(f"Fitting StatsForecast on {train['unique_id'].nunique()} series...")

    sf = StatsForecast(
        models=[Naive(), SeasonalNaive(season_length=7)],
        freq="D",
        n_jobs=1,
    )

    forecasts = sf.forecast(df=train, h=28, fitted=True)
    forecasts = forecasts.reset_index()
    if "index" in forecasts.columns:
        forecasts = forecasts.drop(columns=["index"])
    forecasts["ds"] = pd.to_datetime(forecasts["ds"])

    # Insample fitted values for MinTrace covariance
    Y_fitted = sf.forecast_fitted_values()
    Y_fitted = Y_fitted.reset_index()
    if "index" in Y_fitted.columns:
        Y_fitted = Y_fitted.drop(columns=["index"])
    Y_fitted["ds"] = pd.to_datetime(Y_fitted["ds"])
    Y_fitted = Y_fitted.dropna()

    logger.info(f"Generated {len(forecasts)} base forecasts, {len(Y_fitted)} insample fitted values")
    return forecasts, val, Y_fitted


def run_reconciliation(Y_hat_df, Y_fitted, S_df, tags, val_df):
    """Run BottomUp and MinTrace reconciliation and compare."""
    from hierarchicalforecast.methods import MinTrace, BottomUp
    from hierarchicalforecast.core import HierarchicalReconciliation

    results = {}

    for method_name, method_obj in [
        ("BottomUp", BottomUp()),
        ("MinTrace_shrink", MinTrace(method="mint_shrink")),
    ]:
        logger.info(f"\n  Reconciling with {method_name}...")
        try:
            hrec = HierarchicalReconciliation(reconcilers=[method_obj])
            reconciled = hrec.reconcile(
                Y_hat_df=Y_hat_df,
                S_df=S_df,
                tags=tags,
                Y_df=Y_fitted,
            )

            # Evaluate: merge reconciled with val actuals
            # Reconciled columns have format "ModelName/MethodName"
            rec_cols = [c for c in reconciled.columns if "/" in c]
            # Prefer SeasonalNaive reconciled over Naive reconciled
            rec_main_col = next(
                (c for c in rec_cols if "SeasonalNaive" in c),
                rec_cols[0] if rec_cols else None,
            )

            if rec_main_col and len(val_df) > 0:
                merged = reconciled.merge(
                    val_df[["unique_id", "ds", "y"]],
                    on=["unique_id", "ds"],
                    how="inner",
                )

                if len(merged) > 0:
                    mae = float(np.mean(np.abs(merged["y"] - merged[rec_main_col])))
                    rmse = float(np.sqrt(np.mean((merged["y"] - merged[rec_main_col]) ** 2)))

                    # Per-level MAE
                    level_maes = {}
                    for level_name, level_ids in tags.items():
                        level_data = merged[merged["unique_id"].isin(level_ids)]
                        if len(level_data) > 0:
                            level_mae = float(np.mean(np.abs(level_data["y"] - level_data[rec_main_col])))
                            level_maes[level_name] = round(level_mae, 4)

                    results[method_name] = {
                        "overall_mae": round(mae, 4),
                        "overall_rmse": round(rmse, 4),
                        "n_matched": len(merged),
                        "reconciled_col": rec_main_col,
                        "per_level_mae": level_maes,
                    }
                    logger.info(f"  {method_name}: MAE={mae:.4f}, RMSE={rmse:.4f}")
                    for lev, lmae in level_maes.items():
                        logger.info(f"    {lev}: MAE={lmae}")

            reconciled.to_csv(
                REPORTS_DIR / f"reconciled_{method_name}.csv", index=False
            )

        except Exception as e:
            logger.error(f"  {method_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[method_name] = {"error": str(e)}

    return results


def evaluate_base_forecasts(Y_hat_df, val_df, tags):
    """Evaluate base (unreconciled) forecasts for comparison."""
    val_df = val_df.copy()
    val_df["ds"] = pd.to_datetime(val_df["ds"])
    Y_hat_df = Y_hat_df.copy()
    Y_hat_df["ds"] = pd.to_datetime(Y_hat_df["ds"])

    merged = Y_hat_df.merge(val_df[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner")

    results = {}
    forecast_cols = [c for c in merged.columns if c not in ("unique_id", "ds", "y", "index")]

    for col in forecast_cols:
        valid = merged[["y", col]].dropna()
        if len(valid) == 0:
            continue
        mae = float(np.mean(np.abs(valid["y"] - valid[col])))
        rmse = float(np.sqrt(np.mean((valid["y"] - valid[col]) ** 2)))

        level_maes = {}
        for level_name, level_ids in tags.items():
            level_data = merged[merged["unique_id"].isin(level_ids)][["y", col]].dropna()
            if len(level_data) > 0:
                level_maes[level_name] = round(float(np.mean(np.abs(level_data["y"] - level_data[col]))), 4)

        results[col] = {
            "overall_mae": round(mae, 4),
            "overall_rmse": round(rmse, 4),
            "per_level_mae": level_maes,
        }

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Building M5 hierarchy...")
    Y_df, S_df, tags, cfg = load_m5_hierarchy()

    logger.info("\nGenerating base forecasts (StatsForecast Naive + SeasonalNaive)...")
    Y_hat_df, val_df, Y_fitted = generate_base_forecasts(Y_df, cfg.train_cutoff, cfg.val_cutoff)

    logger.info("\nEvaluating base forecasts...")
    base_results = evaluate_base_forecasts(Y_hat_df, val_df, tags)
    for col, res in base_results.items():
        logger.info(f"  Base {col}: MAE={res['overall_mae']:.4f}")

    logger.info("\nRunning hierarchical reconciliation...")
    rec_results = run_reconciliation(Y_hat_df, Y_fitted, S_df, tags, val_df)

    # Combined report
    report = {
        "base_forecasts": base_results,
        "reconciliation": rec_results,
        "hierarchy": {
            "n_total_series": sum(len(v) for v in tags.values()),
            "n_bottom_series": len(tags["Item"]),
            "levels": {k: len(v) for k, v in tags.items()},
        },
    }

    report_path = REPORTS_DIR / "reconciliation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"\nReport saved → {report_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("HIERARCHICAL RECONCILIATION SUMMARY")
    print(f"{'='*80}")
    print(f"\nHierarchy: {report['hierarchy']['levels']}")

    print("\n--- Base Forecast MAE ---")
    for col, res in base_results.items():
        print(f"  {col}: {res['overall_mae']:.4f}")

    print("\n--- Reconciled MAE (overall) ---")
    for method, res in rec_results.items():
        if "error" in res:
            print(f"  {method}: FAILED — {res['error']}")
        else:
            print(f"  {method}: {res['overall_mae']:.4f}")

    print("\n--- Per-Level MAE Comparison ---")
    all_methods = list(rec_results.keys())
    header = f"  {'Level':<15}"
    for col in base_results:
        header += f" {'Base_'+col:<15}"
    for m in all_methods:
        header += f" {m:<20}"
    print(header)

    for level_name in tags.keys():
        row = f"  {level_name:<15}"
        for col, res in base_results.items():
            val = res.get("per_level_mae", {}).get(level_name, "—")
            row += f" {str(val):<15}"
        for m in all_methods:
            if "per_level_mae" in rec_results.get(m, {}):
                val = rec_results[m]["per_level_mae"].get(level_name, "—")
                row += f" {str(val):<20}"
            else:
                row += f" {'ERR':<20}"
        print(row)

    # Log to MLflow
    try:
        import mlflow
        mlflow.set_experiment("pulsepredict-reconcile")
        with mlflow.start_run(run_name="hierarchical_reconciliation"):
            for method, res in rec_results.items():
                if "overall_mae" in res:
                    mlflow.log_metric(f"{method}/mae", res["overall_mae"])
                    mlflow.log_metric(f"{method}/rmse", res["overall_rmse"])
            mlflow.log_artifacts(str(REPORTS_DIR))
        logger.info("Logged to MLflow")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")


if __name__ == "__main__":
    main()
