"""Train N-BEATSx, PatchTST, and TFT sequentially."""
import os
import subprocess
import sys

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5001")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

MODELS = [
    ("nbeatsx", "configs/nbeatsx.yaml"),
    ("patchtst", "configs/patchtst.yaml"),
    ("tft", "configs/tft.yaml"),
]

for name, config in MODELS:
    print(f"\n{'='*60}")
    print(f"  Training {name.upper()}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        [
            sys.executable, "-m", "ml.train.cli", "fit",
            "--config", config,
            "--output-dir", "artifacts",
        ],
        env=os.environ,
    )
    if result.returncode != 0:
        print(f"\n*** {name} FAILED (exit code {result.returncode}) ***\n")
    else:
        print(f"\n*** {name} DONE ***\n")

print("\nAll training runs complete.")
