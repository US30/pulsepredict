from __future__ import annotations

import argparse
import logging
import os
import subprocess
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REQUIRED_FILES = [
    "sales_train_evaluation.csv",
    "calendar.csv",
    "sell_prices.csv",
]


def download_m5(output_dir: Path = Path("data/raw/m5")) -> None:
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not username or not key:
        raise EnvironmentError(
            "Set KAGGLE_USERNAME and KAGGLE_KEY env vars. "
            "Get them at https://www.kaggle.com/settings → API → Create New Token."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading M5 competition data to %s ...", output_dir)

    result = subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", "m5-forecasting-accuracy",
            "-p", str(output_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"kaggle CLI failed:\n{result.stderr}")

    zip_path = output_dir / "m5-forecasting-accuracy.zip"
    if zip_path.exists():
        log.info("Extracting %s ...", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        zip_path.unlink()

    for fname in REQUIRED_FILES:
        fpath = output_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Expected file not found after extraction: {fpath}")
        size_mb = fpath.stat().st_size / 1_048_576
        log.info("  ✓ %s  (%.1f MB)", fname, size_mb)

    log.info("M5 data ready at %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download M5 competition data via Kaggle API")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw/m5"),
        help="Destination directory (default: data/raw/m5)",
    )
    args = parser.parse_args()
    download_m5(args.output_dir)


if __name__ == "__main__":
    main()
