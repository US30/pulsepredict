from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{filename}"
)
FILES = ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]


def download_ett(output_dir: Path = Path("data/raw/ett")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        url = BASE_URL.format(filename=fname)
        dest = output_dir / fname
        if dest.exists():
            log.info("Already exists, skipping: %s", dest)
            continue
        log.info("Downloading %s ...", fname)
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size / 1024
        log.info("  ✓ %s  (%.1f KB)", fname, size_kb)
    log.info("ETT data ready at %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ETT datasets from GitHub")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/ett"))
    args = parser.parse_args()
    download_ett(args.output_dir)


if __name__ == "__main__":
    main()
