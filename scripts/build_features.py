"""Build M5 features from raw CSVs — memory-efficient version.

Uses DuckDB for the heavy wide-to-long pivot and join, then builds
features in chunks to avoid OOM on machines with limited RAM.
"""
from pathlib import Path

import duckdb
import polars as pl

from ml.data.feature_lib import TimeSeriesFeatureLib, FeatureConfig


def main():
    Path("data/features").mkdir(parents=True, exist_ok=True)
    data_dir = Path("data/raw/m5")

    con = duckdb.connect()

    print("Pivoting M5 sales wide->long via DuckDB...")
    con.execute(f"""
        CREATE OR REPLACE TABLE sales_long AS
        WITH wide AS (
            SELECT * FROM read_csv_auto('{data_dir / "sales_train_evaluation.csv"}')
        )
        SELECT
            item_id || '_' || store_id AS unique_id,
            item_id, dept_id, cat_id, store_id, state_id,
            UNNEST(list_value({', '.join(
                f"struct_pack(d := 'd_{i}', y := CAST(d_{i} AS DOUBLE))"
                for i in range(1, 1942)
            )})) AS s
        FROM wide
    """)

    con.execute("""
        CREATE OR REPLACE TABLE sales_flat AS
        SELECT unique_id, item_id, dept_id, cat_id, store_id, state_id,
               s.d AS d, s.y AS y
        FROM sales_long
    """)

    print("Joining with calendar...")
    con.execute(f"""
        CREATE OR REPLACE TABLE calendar AS
        SELECT d, date AS ds, wm_yr_wk, event_name_1, event_type_1,
               snap_CA, snap_TX, snap_WI
        FROM read_csv_auto('{data_dir / "calendar.csv"}')
    """)

    con.execute("""
        CREATE OR REPLACE TABLE merged AS
        SELECT s.unique_id, c.ds, s.y,
               s.item_id, s.dept_id, s.cat_id, s.store_id, s.state_id
        FROM sales_flat s
        JOIN calendar c ON s.d = c.d
        ORDER BY s.unique_id, c.ds
    """)

    row_count = con.execute("SELECT COUNT(*) FROM merged").fetchone()[0]
    n_series = con.execute("SELECT COUNT(DISTINCT unique_id) FROM merged").fetchone()[0]
    print(f"Merged: {row_count:,} rows, {n_series:,} series")

    # Get unique series list
    series_list = [r[0] for r in con.execute(
        "SELECT DISTINCT unique_id FROM merged ORDER BY unique_id"
    ).fetchall()]

    # Process in chunks of 1000 series
    chunk_size = 1000
    lib = TimeSeriesFeatureLib()
    config = FeatureConfig()
    parquet_files = []

    for i in range(0, len(series_list), chunk_size):
        chunk_ids = series_list[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        total_chunks = (len(series_list) + chunk_size - 1) // chunk_size
        print(f"  Chunk {chunk_num}/{total_chunks} ({len(chunk_ids)} series)...")

        placeholders = ", ".join(f"'{sid}'" for sid in chunk_ids)
        chunk_df = con.execute(f"""
            SELECT unique_id, CAST(ds AS DATE) AS ds, y
            FROM merged
            WHERE unique_id IN ({placeholders})
            ORDER BY unique_id, ds
        """).pl()

        feat_df = lib.build_features(chunk_df, config)

        out_path = f"data/features/m5_chunk_{chunk_num:03d}.parquet"
        feat_df.write_parquet(out_path)
        parquet_files.append(out_path)

    con.close()

    # Combine chunk info
    print(f"\nDone! {len(parquet_files)} parquet chunks in data/features/")
    total_size = sum(Path(f).stat().st_size for f in parquet_files) / 1_048_576
    print(f"Total size: {total_size:.1f} MB")


if __name__ == "__main__":
    main()
