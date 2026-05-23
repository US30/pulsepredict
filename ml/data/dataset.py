"""
M5 and ETT dataset loaders for PulsePredict.

M5 (Walmart hierarchical sales) and ETT (Electricity Transformer Temperature)
are canonical benchmarks for multi-horizon time-series forecasting.

Both loaders return data in NeuralForecast's long-format: a pandas DataFrame
with at minimum the columns ``unique_id``, ``ds``, ``y``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and splitting.

    Attributes
    ----------
    dataset:
        Dataset identifier: ``"m5"`` or ``"ett"``.
    data_dir:
        Path to the directory containing raw CSV files.
    horizon:
        Forecast horizon used to determine the test window size.
    train_cutoff:
        Last date (inclusive) of the training split (ISO format).
    val_cutoff:
        Last date (inclusive) of the validation split (ISO format).
    test_cutoff:
        Last date (inclusive) of the test split (ISO format).
    """

    dataset: str = "m5"
    data_dir: str = "data/raw/m5"
    horizon: int = 28
    max_series: int = 0
    train_cutoff: str = "2016-04-24"
    val_cutoff: str = "2016-04-24"
    test_cutoff: str = "2016-05-22"


class M5Dataset:
    """Loader for the Kaggle M5 Forecasting competition dataset.

    Expected files in ``data_dir``:
        - ``sales_train_evaluation.csv``
        - ``calendar.csv``
        - ``sell_prices.csv``

    All files are available at:
    https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
    """

    def __init__(self, config: Optional[DatasetConfig] = None) -> None:
        self.config = config or DatasetConfig()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_raw(self, data_dir: Path) -> pl.DataFrame:
        """Read M5 CSV files and return a long-format Polars DataFrame.

        Parameters
        ----------
        data_dir:
            Directory containing the three M5 CSV files.

        Returns
        -------
        pl.DataFrame
            Columns: ``unique_id`` (item_id + "_" + store_id), ``ds`` (date),
            ``y`` (unit sales), ``wm_yr_wk`` (week), ``event_name_1``,
            ``event_type_1``, ``snap_CA``, ``snap_TX``, ``snap_WI``,
            ``sell_price``.
        """
        data_dir = Path(data_dir)
        sales_path = data_dir / "sales_train_evaluation.csv"
        calendar_path = data_dir / "calendar.csv"
        prices_path = data_dir / "sell_prices.csv"

        # ---- Sales: wide -> long ------------------------------------------
        sales_wide = pl.read_csv(sales_path)

        if self.config.max_series > 0:
            sales_wide = sales_wide.sample(
                n=min(self.config.max_series, len(sales_wide)),
                seed=42,
            )

        id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        d_cols = [c for c in sales_wide.columns if c.startswith("d_")]

        sales_long = sales_wide.unpivot(
            on=d_cols,
            index=id_cols,
            variable_name="d",
            value_name="y",
        )
        sales_long = sales_long.with_columns(
            (pl.col("item_id") + "_" + pl.col("store_id")).alias("unique_id")
        )

        # ---- Calendar: add date mapping -----------------------------------
        calendar = pl.read_csv(calendar_path, try_parse_dates=True)
        calendar = calendar.select(
            ["d", "date", "wm_yr_wk", "event_name_1", "event_type_1",
             "snap_CA", "snap_TX", "snap_WI"]
        ).rename({"date": "ds"})

        # ---- Merge sales + calendar ---------------------------------------
        df = sales_long.join(calendar, on="d", how="left")

        # ---- Prices -------------------------------------------------------
        prices = pl.read_csv(prices_path)
        prices = prices.rename({"sell_price": "sell_price"})
        df = df.join(
            prices,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
        )

        # ---- Final column selection and sort ------------------------------
        df = df.select(
            [
                "unique_id", "ds", "y",
                "item_id", "dept_id", "cat_id", "store_id", "state_id",
                "wm_yr_wk", "event_name_1", "event_type_1",
                "snap_CA", "snap_TX", "snap_WI", "sell_price",
            ]
        ).sort(["unique_id", "ds"])

        return df

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def get_splits(
        self,
        df: pl.DataFrame,
        train_cutoff: Optional[str] = None,
        val_cutoff: Optional[str] = None,
        test_cutoff: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split the dataset into train / validation / test partitions.

        All cutoffs are inclusive on the right boundary. The ``ds`` column is
        cast to date for comparison.

        Parameters
        ----------
        df:
            Output of ``load_raw``.
        train_cutoff, val_cutoff, test_cutoff:
            ISO date strings overriding ``config`` values.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            ``(train_df, val_df, test_df)`` in NeuralForecast format
            (columns: ``unique_id``, ``ds``, ``y``).
        """
        tc = train_cutoff or self.config.train_cutoff
        vc = val_cutoff or self.config.val_cutoff
        ec = test_cutoff or self.config.test_cutoff

        # Ensure ds is date type for comparison
        if df["ds"].dtype != pl.Date:
            df = df.with_columns(pl.col("ds").cast(pl.Date))

        core_cols = ["unique_id", "ds", "y"]

        train_df = (
            df.filter(pl.col("ds") <= pl.lit(tc).str.to_date())
            .select(core_cols)
            .to_pandas()
        )
        val_df = (
            df.filter(
                (pl.col("ds") > pl.lit(tc).str.to_date())
                & (pl.col("ds") <= pl.lit(vc).str.to_date())
            )
            .select(core_cols)
            .to_pandas()
        )
        test_df = (
            df.filter(
                (pl.col("ds") > pl.lit(vc).str.to_date())
                & (pl.col("ds") <= pl.lit(ec).str.to_date())
            )
            .select(core_cols)
            .to_pandas()
        )

        # Convert ds to datetime for NeuralForecast compatibility
        for part in (train_df, val_df, test_df):
            part["ds"] = pd.to_datetime(part["ds"])

        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Hierarchy
    # ------------------------------------------------------------------

    def hierarchy(self, df: pd.DataFrame) -> dict:
        """Build hierarchy metadata for HierarchicalForecast.

        Constructs the ``S_df`` summing matrix and ``tags`` dict covering four
        aggregation levels: state, dept, cat, item.

        Parameters
        ----------
        df:
            Long-format pandas DataFrame including ``unique_id``, ``ds``,
            ``y``, ``state_id``, ``dept_id``, ``cat_id``.

        Returns
        -------
        dict
            Keys:
            - ``"Y_df"``: NeuralForecast-format DataFrame with all levels.
            - ``"S_df"``: pd.DataFrame summing matrix (items x aggregates).
            - ``"tags"``: dict mapping level names to lists of series IDs.
        """
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Parse unique_id: "item_id_store_id" -> item/store/dept/cat/state
        # M5 unique_id format: {FOODS/HOBBIES/HOUSEHOLD}_{dept_num}_{item_num}_{state}_{store_num}
        # Build aggregate series by summing within each level
        records = []
        for uid, grp in df.groupby("unique_id", sort=False):
            parts = uid.split("_")
            # Determine hierarchical IDs
            # M5: FOODS_1_001_CA_1 -> cat=FOODS, dept=FOODS_1, state=CA, item=uid
            if len(parts) >= 5:
                cat_id = parts[0]
                dept_id = "_".join(parts[:2])
                state_id = parts[3]
            else:
                cat_id = dept_id = state_id = "UNKNOWN"

            for _, row in grp.iterrows():
                records.append({
                    "unique_id": uid,
                    "ds": row["ds"],
                    "y": row["y"],
                    "cat_id": cat_id,
                    "dept_id": dept_id,
                    "state_id": state_id,
                })

        full_df = pd.DataFrame(records)

        # Build aggregate DataFrames for each level
        agg_frames = []
        tags: dict = {"item": [], "state": [], "dept": [], "cat": [], "total": ["Total"]}

        for lvl, group_col in [("state", "state_id"), ("dept", "dept_id"), ("cat", "cat_id")]:
            for gid, grp in full_df.groupby(group_col, sort=False):
                agg = grp.groupby("ds")["y"].sum().reset_index()
                agg["unique_id"] = str(gid)
                agg_frames.append(agg[["unique_id", "ds", "y"]])
                tags[lvl].append(str(gid))

        # Total
        total = full_df.groupby("ds")["y"].sum().reset_index()
        total["unique_id"] = "Total"
        agg_frames.append(total[["unique_id", "ds", "y"]])

        tags["item"] = df["unique_id"].unique().tolist()

        Y_df = pd.concat(
            [df[["unique_id", "ds", "y"]]] + agg_frames, ignore_index=True
        )

        # Build summing matrix S_df (items as rows, all aggregates as columns)
        all_series = Y_df["unique_id"].unique().tolist()
        item_series = tags["item"]

        S_df = pd.DataFrame(0, index=item_series, columns=all_series)
        for item in item_series:
            parts = item.split("_")
            cat_id = parts[0] if len(parts) >= 1 else "UNKNOWN"
            dept_id = "_".join(parts[:2]) if len(parts) >= 2 else "UNKNOWN"
            state_id = parts[3] if len(parts) >= 4 else "UNKNOWN"

            S_df.loc[item, item] = 1
            if state_id in S_df.columns:
                S_df.loc[item, state_id] = 1
            if dept_id in S_df.columns:
                S_df.loc[item, dept_id] = 1
            if cat_id in S_df.columns:
                S_df.loc[item, cat_id] = 1
            if "Total" in S_df.columns:
                S_df.loc[item, "Total"] = 1

        return {"Y_df": Y_df, "S_df": S_df, "tags": tags}


class ETTDataset:
    """Loader for the ETT (Electricity Transformer Temperature) dataset.

    Supports ETTh1, ETTh2 (hourly) and ETTm1, ETTm2 (15-minute) variants.
    The CSV files are available at:
    https://github.com/zhouhaoyi/ETDataset

    Expected columns in the CSV:
        ``date``, ``HUFL``, ``HULL``, ``MUFL``, ``MULL``, ``LUFL``, ``LULL``, ``OT``
    """

    def load(self, path: Path, freq: str = "H") -> pd.DataFrame:
        """Load an ETT CSV and return a long-format NeuralForecast DataFrame.

        Each multivariate column is treated as a separate time series with a
        ``unique_id`` equal to the column name.

        Parameters
        ----------
        path:
            Path to an ETT CSV file (e.g. ``ETTh1.csv``).
        freq:
            Pandas frequency string: ``"H"`` for hourly variants,
            ``"15T"`` for 15-minute variants.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns ``unique_id``, ``ds``, ``y``
            and additional multivariate columns available as separate series.
        """
        path = Path(path)
        raw = pd.read_csv(path, parse_dates=["date"])
        raw = raw.rename(columns={"date": "ds"})
        raw["ds"] = pd.to_datetime(raw["ds"])
        raw = raw.sort_values("ds").reset_index(drop=True)

        value_cols = [c for c in raw.columns if c != "ds"]

        records = []
        for col in value_cols:
            sub = raw[["ds", col]].copy()
            sub["unique_id"] = col
            sub = sub.rename(columns={col: "y"})
            records.append(sub[["unique_id", "ds", "y"]])

        long_df = pd.concat(records, ignore_index=True)
        long_df["ds"] = pd.to_datetime(long_df["ds"])
        return long_df
