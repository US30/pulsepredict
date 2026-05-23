"""
Hierarchical forecast reconciliation for PulsePredict.

Uses the `hierarchicalforecast` library (Nixtla) to reconcile base forecasts
so that they are coherent across the M5 aggregation hierarchy:
    item → dept → cat → state → national (sales channel similarly)

Supported methods:
    mint_shrink             — MinTrace with shrinkage covariance estimator
    bottomup                — Bottom-up aggregation
    topdown_proportion      — Top-down using historical proportions
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal

from hierarchicalforecast.methods import MinTrace, BottomUp, TopDown
from hierarchicalforecast.core import HierarchicalReconciliation


@dataclass
class ReconcileConfig:
    """
    Configuration for :class:`HierarchicalReconciler`.

    Parameters
    ----------
    method : str
        Reconciliation method.  One of:
        ``"mint_shrink"``, ``"bottomup"``, ``"topdown_proportion"``.
    level : list[int]
        Prediction-interval levels to include in the reconciled output
        (e.g. [80, 90]).
    """

    method: Literal["mint_shrink", "bottomup", "topdown_proportion"] = "mint_shrink"
    level: list = field(default_factory=lambda: [80, 90])


class HierarchicalReconciler:
    """
    Wraps `hierarchicalforecast.core.HierarchicalReconciliation` with a
    simpler interface tuned for the PulsePredict / M5 use case.

    Parameters
    ----------
    config : ReconcileConfig
    """

    def __init__(self, config: ReconcileConfig):
        self.config = config
        self._reconciliation: HierarchicalReconciliation | None = None

    # ------------------------------------------------------------------
    # Method factory
    # ------------------------------------------------------------------

    def _build_method(self):
        """Instantiate the chosen reconciliation method object."""
        method = self.config.method
        if method == "mint_shrink":
            return MinTrace(method="mint_shrink")
        elif method == "bottomup":
            return BottomUp()
        elif method == "topdown_proportion":
            return TopDown(method="forecast_proportions")
        else:
            raise ValueError(
                f"Unknown reconciliation method '{method}'. "
                "Choose from: mint_shrink, bottomup, topdown_proportion."
            )

    # ------------------------------------------------------------------
    # Core reconciliation
    # ------------------------------------------------------------------

    def reconcile(
        self,
        Y_df: pd.DataFrame,
        S_df: pd.DataFrame,
        tags: dict,
    ) -> pd.DataFrame:
        """
        Reconcile base forecasts so they are hierarchically coherent.

        Parameters
        ----------
        Y_df : pd.DataFrame
            Base forecasts in the NeuralForecast long format:
            columns = [``unique_id``, ``ds``, ``<model_name>``].
            All hierarchy levels must be present in ``unique_id``.
        S_df : pd.DataFrame
            Summing / aggregation matrix of shape (n_series_all, n_bottom).
            Rows = all aggregated series; columns = bottom-level series.
            Index = unique_id for each series.
        tags : dict
            Mapping of aggregation level name → list of unique_ids at that level.
            E.g. ``{"Level1": ["Total"], "Level2": ["HOBBIES", "FOOD", ...], ...}``.

        Returns
        -------
        pd.DataFrame
            Reconciled forecasts in the same long format as ``Y_df``.
        """
        rec_method = self._build_method()
        hrec = HierarchicalReconciliation(reconcilers=[rec_method])

        reconciled_df = hrec.reconcile(
            Y_df=Y_df,
            S=S_df,
            tags=tags,
            level=self.config.level,
        )

        self._reconciliation = hrec
        return reconciled_df

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_base_vs_reconciled(
        self,
        base: pd.DataFrame,
        reconciled: pd.DataFrame,
        actuals: pd.DataFrame,
    ) -> dict:
        """
        Compare MAE of base vs reconciled forecasts at each hierarchy level.

        Parameters
        ----------
        base : pd.DataFrame
            Base (unreconciled) forecasts with columns [unique_id, ds, <forecast_col>].
        reconciled : pd.DataFrame
            Reconciled forecasts with the same schema.
        actuals : pd.DataFrame
            Ground-truth actuals with columns [unique_id, ds, y].

        Returns
        -------
        dict
            ``{level_tag: {"base_mae": float, "reconciled_mae": float, "improvement_pct": float}}``
            plus an ``"overall"`` key.
        """
        # Identify forecast columns (exclude identifier columns)
        id_cols = {"unique_id", "ds"}
        base_fcst_col = [c for c in base.columns if c not in id_cols][0]
        rec_fcst_col = [c for c in reconciled.columns if c not in id_cols][0]

        # Merge with actuals
        base_merged = base.merge(
            actuals[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner"
        )
        rec_merged = reconciled.merge(
            actuals[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner"
        )

        def _mae(df, fcst_col):
            return float(np.mean(np.abs(df["y"].values - df[fcst_col].values)))

        results: dict = {}

        # Overall
        results["overall"] = {
            "base_mae": _mae(base_merged, base_fcst_col),
            "reconciled_mae": _mae(rec_merged, rec_fcst_col),
        }
        base_mae = results["overall"]["base_mae"]
        rec_mae = results["overall"]["reconciled_mae"]
        results["overall"]["improvement_pct"] = round(
            100.0 * (base_mae - rec_mae) / (base_mae + 1e-12), 2
        )

        # Per unique_id prefix (hierarchy level inferred from id structure)
        # M5 ids have the form DEPT_CAT_ITEM_STATE_STORE
        # We group by the number of underscore-separated segments as a proxy for level.
        for df_name, merged, fcst_col in [
            ("base", base_merged, base_fcst_col),
            ("reconciled", rec_merged, rec_fcst_col),
        ]:
            merged["_level"] = merged["unique_id"].apply(
                lambda uid: len(str(uid).split("_"))
            )
            for level, grp in merged.groupby("_level"):
                level_key = f"depth_{level}"
                if level_key not in results:
                    results[level_key] = {}
                results[level_key][f"{df_name}_mae"] = _mae(grp, fcst_col)

        # Compute improvement per level
        for key, val in results.items():
            if "base_mae" in val and "reconciled_mae" in val and "improvement_pct" not in val:
                b = val["base_mae"]
                r = val["reconciled_mae"]
                val["improvement_pct"] = round(100.0 * (b - r) / (b + 1e-12), 2)

        return results

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HierarchicalReconciler("
            f"method={self.config.method!r}, "
            f"level={self.config.level})"
        )
