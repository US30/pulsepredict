# PulsePredict — How to Run

End-to-end guide for setting up and running the full probabilistic
time-series forecasting pipeline on a machine with an NVIDIA RTX 2070 Super.

---

## 1. Prerequisites

| Requirement | Minimum version | Notes |
|---|---|---|
| Python | 3.11 | `pyenv` or system package |
| Docker + Docker Compose | 24 + | `docker compose` (v2 plugin) |
| NVIDIA driver | 535 | Supports CUDA 12.1 |
| CUDA toolkit | 12.1 | Needed for PyTorch GPU kernels |
| Kaggle account | — | Required to download M5 dataset |
| ~40 GB free disk | — | M5 raw + features + artifacts |

Verify your GPU is visible:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 2. Clone and install

```bash
git clone https://github.com/<your-username>/pulsepredict.git
cd pulsepredict

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package and all production dependencies
pip install -e .

# Install dev extras (linting, tests, pre-commit)
make dev-install
```

---

## 3. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in:

- `KAGGLE_USERNAME` — your Kaggle username
- `KAGGLE_KEY` — your Kaggle API key (from https://www.kaggle.com/settings → API → Create Token)

All other defaults work out of the box with the local Docker stack.

---

## 4. Start the Docker stack

```bash
make up
```

After ~60 seconds the following services will be available:

| Service | URL | Credentials |
|---|---|---|
| MinIO console | http://localhost:9001 | minioadmin / minioadmin |
| MLflow UI | http://localhost:5001 | — |
| Grafana | http://localhost:3001 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| FastAPI docs | http://localhost:8000/docs | — |
| Streamlit UI | http://localhost:8501 | — |

Run `make logs` to tail all service logs.

Create the MinIO buckets on first start:

```bash
# Inside the MinIO console (http://localhost:9001) create two buckets:
#   pulsepredict-data
#   pulsepredict-mlflow
# Or via the mc CLI:
docker run --rm --network host minio/mc \
  alias set local http://localhost:9000 minioadmin minioadmin
docker run --rm --network host minio/mc mb local/pulsepredict-data
docker run --rm --network host minio/mc mb local/pulsepredict-mlflow
```

---

## 5. Download M5 Competition data

The M5 Accuracy / Uncertainty Competition dataset is hosted on Kaggle.

```bash
make download-m5
```

This runs `scripts/download_m5.py`, which calls the Kaggle API and saves files
to `data/raw/m5/`. Expect ~200 MB compressed, ~500 MB extracted.

To also download the ETT benchmark (ETTh1/ETTh2/ETTm1/ETTm2):

```bash
make download-ett
```

---

## 6. Feature engineering

Build lag features, rolling statistics, and calendar covariates:

```bash
python -m ml.data.feature_lib build \
  --input  data/raw/m5/ \
  --output data/features/
```

Output lands in `data/features/` as Parquet shards (~2 GB).

---

## 7. Train DeepAR baseline (≈ 20 min)

```bash
make train-deepar
```

Runs `python -m ml.train.cli fit --config configs/deepar.yaml`.
Experiment is logged to MLflow automatically. Check progress at
http://localhost:5001.

---

## 8. Train PatchTST (≈ 1.5 hr on RTX 2070 Super)

```bash
make train-patchtst
```

GPU memory: ~3 GB. If you hit OOM, reduce `batch_size` in
`configs/patchtst.yaml` (try 32 → 16).

---

## 9. Train Temporal Fusion Transformer (≈ 2 hr on RTX 2070 Super)

```bash
make train-tft
```

GPU memory: ~4 GB. If you hit OOM, reduce `batch_size` in
`configs/tft.yaml` (try 64 → 32).

To train all four models sequentially (deepar + nbeatsx + patchtst + tft):

```bash
make train-all
```

---

## 10. Run walk-forward backtest

```bash
make backtest
```

Produces `reports/backtest/metrics.json` with RMSSE, WRMSSE, CRPS, and
pinball-loss across all models and horizons. Results are also pushed to MLflow.

---

## 11. Conformal prediction coverage

```bash
make conformal
```

Evaluates empirical coverage of prediction intervals at α ∈ {0.05, 0.10, 0.20}
against nominal coverage. Output: `reports/conformal_coverage.json`.

---

## 12. Hierarchical reconciliation

```bash
make reconcile
```

Applies MinT (Mint Shrinkage), BottomUp, and TopDown reconciliation using
`hierarchicalforecast`. Reconciled forecasts are saved to `reports/reconciled/`.

---

## 13. Bayesian causal intervention

```bash
make intervention
```

Fits a PyMC structural time-series model to detect the causal effect of a
step intervention in the series. Results (trace, posterior predictive plots)
are saved to `reports/intervention/` and logged to MLflow as artefacts.

---

## 14. View results

| Tool | URL | What to look at |
|---|---|---|
| MLflow | http://localhost:5001 | Metrics, params, artefacts for every run |
| Grafana | http://localhost:3001 | Real-time API latency and prediction dashboards |
| Streamlit | http://localhost:8501 | Interactive forecast explorer and coverage plots |

---

## GPU memory reference

| Model | Approx. VRAM | Config to tune if OOM |
|---|---|---|
| DeepAR | ~1.5 GB | `batch_size`, `input_size` |
| N-BEATSx | ~1 GB | `batch_size`, `stacks` |
| PatchTST | ~3 GB | `batch_size`, `patch_len`, `d_model` |
| TFT | ~4 GB | `batch_size`, `hidden_size` |

---

## DVC pipeline (optional)

Instead of running steps manually you can reproduce the full pipeline with
DVC's DAG runner:

```bash
dvc repro
```

DVC will skip stages whose dependencies have not changed (content-hash based).

---

## Stopping the stack

```bash
make down
```

Data volumes (`postgres_data`, `minio_data`, `mlflow_data`, `grafana_data`)
are preserved. Pass `-v` to also delete volumes:

```bash
docker compose -f docker-compose.yml down -v
```
