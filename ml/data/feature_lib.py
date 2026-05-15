"""
Time-series feature engineering library using Polars and DuckDB.

Features are computed in a lazy Polars pipeline grouped by ``unique_id`` so
every operation is series-local. DuckDB is used for large out-of-core parquet
workflows where the dataset does not fit in RAM.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import duckdb
import polars as pl


@dataclass
class FeatureConfig:
    """Controls which features are computed by TimeSeriesFeatureLib.

    Attributes
    ----------
    lags:
        Lag offsets (in time steps) for creating y_lag_{k} features.
    rolling_windows:
        Window sizes for rolling mean, std, and max features.
    add_calendar:
        Whether to add calendar-based features (day of week, month, etc.).
    add_price:
        Whether to add price-based features (requires a ``price`` column).
    """

    lags: list = field(default_factory=lambda: [1, 7, 14, 28, 56])
    rolling_windows: list = field(default_factory=lambda: [7, 14, 28, 56])
    add_calendar: bool = True
    add_price: bool = False


class TimeSeriesFeatureLib:
    """Feature engineering for long-format time-series DataFrames.

    Operates on DataFrames with at minimum the columns:
    ``unique_id`` (series identifier), ``ds`` (date/datetime), ``y`` (target).

    All feature computations are performed within group (per ``unique_id``)
    to prevent information leakage across series.
    """

    # ------------------------------------------------------------------
    # Core feature builder
    # ------------------------------------------------------------------

    def build_features(
        self, df: pl.DataFrame, config: FeatureConfig
    ) -> pl.DataFrame:
        """Compute lag, rolling, calendar, and log features.

        Parameters
        ----------
        df:
            Long-format Polars DataFrame. Required columns: ``unique_id``,
            ``ds``, ``y``. Additional covariate columns are preserved.
        config:
            FeatureConfig specifying which features to compute.

        Returns
        -------
        pl.DataFrame
            Input DataFrame enriched with all requested features.
            Rows with NaN lag/rolling values (leading window) are retained
            but will contain nulls — callers should forward-fill or drop as
            appropriate.
        """
        # Work in lazy mode for efficient query planning
        lf = df.lazy().sort(["unique_id", "ds"])

        # ---- Lag features ------------------------------------------------
        lag_exprs = [
            pl.col("y")
            .shift(k)
            .over("unique_id")
            .alias(f"y_lag_{k}")
            for k in config.lags
        ]
        lf = lf.with_columns(lag_exprs)

        # ---- Rolling statistics -------------------------------------------
        rolling_exprs = []
        for w in config.rolling_windows:
            rolling_exprs += [
                pl.col("y")
                .rolling_mean(window_size=w)
                .over("unique_id")
                .alias(f"y_roll_mean_{w}"),
                pl.col("y")
                .rolling_std(window_size=w)
                .over("unique_id")
                .alias(f"y_roll_std_{w}"),
                pl.col("y")
                .rolling_max(window_size=w)
                .over("unique_id")
                .alias(f"y_roll_max_{w}"),
            ]
        lf = lf.with_columns(rolling_exprs)

        # ---- Calendar features -------------------------------------------
        if config.add_calendar:
            lf = lf.with_columns(
                [
                    pl.col("ds").dt.weekday().alias("day_of_week"),
                    pl.col("ds").dt.month().alias("month"),
                    pl.col("ds").dt.week().alias("week_of_year"),
                    pl.col("ds").dt.day().alias("day_of_month"),
                    (pl.col("ds").dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),
                    pl.col("ds").dt.quarter().alias("quarter"),
                ]
            )

        # ---- Log transform -----------------------------------------------
        lf = lf.with_columns(
            pl.col("y").log1p().alias("y_log")
        )

        return lf.collect()

    # ------------------------------------------------------------------
    # DuckDB large-scale pipeline
    # ------------------------------------------------------------------

    def build_with_duckdb(
        self,
        parquet_path: str,
        output_path: str,
        config: Optional[FeatureConfig] = None,
    ) -> None:
        """Run SQL-based feature extraction on parquet files via DuckDB.

        Suitable for datasets that do not fit in memory. DuckDB reads the
        parquet file lazily and writes results incrementally.

        Parameters
        ----------
        parquet_path:
            Glob-compatible path to input parquet file(s),
            e.g. ``"data/raw/sales/*.parquet"``.
        output_path:
            Destination parquet path for the feature-enriched dataset.
        config:
            FeatureConfig; uses defaults if None.
        """
        if config is None:
            config = FeatureConfig()

        lag_sql = ", ".join(
            f"LAG(y, {k}) OVER (PARTITION BY unique_id ORDER BY ds) AS y_lag_{k}"
            for k in config.lags
        )
        rolling_sql_parts = []
        for w in config.rolling_windows:
            rolling_sql_parts += [
                f"AVG(y) OVER (PARTITION BY unique_id ORDER BY ds ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW) AS y_roll_mean_{w}",
                f"STDDEV(y) OVER (PARTITION BY unique_id ORDER BY ds ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW) AS y_roll_std_{w}",
                f"MAX(y) OVER (PARTITION BY unique_id ORDER BY ds ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW) AS y_roll_max_{w}",
            ]
        rolling_sql = ", ".join(rolling_sql_parts)

        calendar_sql = ""
        if config.add_calendar:
            calendar_sql = (
                ", DAYOFWEEK(ds) AS day_of_week"
                ", MONTH(ds) AS month"
                ", WEEKOFYEAR(ds) AS week_of_year"
                ", DAY(ds) AS day_of_month"
                ", CASE WHEN DAYOFWEEK(ds) IN (6,7) THEN 1 ELSE 0 END AS is_weekend"
                ", QUARTER(ds) AS quarter"
            )

        select_parts = ["*", lag_sql, rolling_sql]
        if config.add_calendar:
            select_parts.append(
                "DAYOFWEEK(ds) AS day_of_week"
                ", MONTH(ds) AS month"
                ", WEEKOFYEAR(ds) AS week_of_year"
                ", DAY(ds) AS day_of_month"
                ", CASE WHEN DAYOFWEEK(ds) IN (6,7) THEN 1 ELSE 0 END AS is_weekend"
                ", QUARTER(ds) AS quarter"
            )

        full_select = ", ".join(
            [
                "*",
                lag_sql,
                rolling_sql,
            ]
        )
        if config.add_calendar:
            full_select += calendar_sql
        full_select += ", LN(y + 1) AS y_log"

        query = f"""
        COPY (
            SELECT {full_select}
            FROM read_parquet('{parquet_path}')
        ) TO '{output_path}' (FORMAT PARQUET)
        """

        con = duckdb.connect()
        con.execute(query)
        con.close()

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize(
        self,
        df: pl.DataFrame,
        method: str = "minmax",
        cols: Optional[list] = None,
    ) -> tuple[pl.DataFrame, dict]:
        """Normalise numeric feature columns.

        Parameters
        ----------
        df:
            Feature DataFrame (output of ``build_features``).
        method:
            Normalisation strategy: ``"minmax"`` scales to [0, 1];
            ``"zscore"`` standardises to zero mean and unit variance.
        cols:
            Subset of columns to normalise. Defaults to all numeric columns
            except ``unique_id``, ``ds``, and ``y``.

        Returns
        -------
        tuple[pl.DataFrame, dict]
            ``(normalised_df, stats_dict)`` where ``stats_dict`` maps each
            column name to its normalisation parameters, enabling inverse
            transform at inference time.

            For minmax: ``{"col": {"min": float, "max": float}}``
            For zscore: ``{"col": {"mean": float, "std": float}}``
        """
        exclude = {"unique_id", "ds", "y"}
        if cols is None:
            cols = [
                c
                for c in df.columns
                if c not in exclude and df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
            ]

        stats: dict = {}
        exprs = []

        if method == "minmax":
            for col in cols:
                col_min = df[col].min()
                col_max = df[col].max()
                stats[col] = {"min": col_min, "max": col_max}
                denom = col_max - col_min if col_max != col_min else 1.0
                exprs.append(
                    ((pl.col(col) - col_min) / denom).alias(col)
                )
        elif method == "zscore":
            for col in cols:
                col_mean = df[col].mean()
                col_std = df[col].std()
                stats[col] = {"mean": col_mean, "std": col_std}
                denom = col_std if col_std and col_std > 0 else 1.0
                exprs.append(
                    ((pl.col(col) - col_mean) / denom).alias(col)
                )
        else:
            raise ValueError(f"Unknown normalisation method: {method!r}. Use 'minmax' or 'zscore'.")

        normalised_df = df.with_columns(exprs) if exprs else df
        return normalised_df, stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line interface for batch feature engineering.

    Usage
    -----
    python -m ml.data.feature_lib \\
        --input data/raw/sales.parquet \\
        --output data/features/sales_features.parquet \\
        --lags 1 7 14 28 \\
        --windows 7 14 28 \\
        --no-calendar
    """
    parser = argparse.ArgumentParser(
        description="Build time-series features from a parquet file."
    )
    parser.add_argument("--input", required=True, help="Input parquet path (glob ok).")
    parser.add_argument("--output", required=True, help="Output parquet path.")
    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=[1, 7, 14, 28, 56],
        help="Lag offsets.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[7, 14, 28, 56],
        help="Rolling window sizes.",
    )
    parser.add_argument(
        "--no-calendar",
        action="store_false",
        dest="calendar",
        help="Disable calendar features.",
    )
    parser.add_argument(
        "--add-price",
        action="store_true",
        help="Include price features (requires 'price' column).",
    )
    args = parser.parse_args()

    config = FeatureConfig(
        lags=args.lags,
        rolling_windows=args.windows,
        add_calendar=args.calendar,
        add_price=args.add_price,
    )

    lib = TimeSeriesFeatureLib()

    input_path = Path(args.input)
    if input_path.suffix == ".parquet" or "*" in args.input:
        print(f"Running DuckDB pipeline: {args.input} -> {args.output}")
        lib.build_with_duckdb(args.input, args.output, config)
    else:
        print(f"Loading CSV: {args.input}")
        df = pl.read_csv(args.input, try_parse_dates=True)
        features_df = lib.build_features(df, config)
        features_df.write_parquet(args.output)
        print(f"Features written to {args.output} ({len(features_df)} rows).")


if __name__ == "__main__":
    main()
