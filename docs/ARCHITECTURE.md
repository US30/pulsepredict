# PulsePredict — Architecture

## 1. Overview

PulsePredict is a research-grade probabilistic time-series forecasting platform designed to:

1. **Benchmark** modern deep-learning forecasters (PatchTST, TFT, N-BEATSx, DeepAR) and a zero-shot foundation model (Chronos-T5) on a common dataset.
2. **Calibrate** prediction intervals with theoretical coverage guarantees using split and adaptive conformal prediction.
3. **Reconcile** base forecasts hierarchically to ensure coherence across aggregation levels.
4. **Estimate** causal effects of interventions using Bayesian structural time-series.
5. **Serve** forecasts at low latency via FastAPI with full observability.

Design goals: reproducible (DVC-tracked), observable (MLflow + Grafana), GPU-efficient (RTX 2070 Super 8 GB), and runnable with `make train-patchtst` on a single machine.

---

## 2. Data Layer

### Ingestion
- **M5 dataset**: 30 490 item-store daily sales series from Walmart (2011–2016). Downloaded via Kaggle API (`scripts/download_m5.py`). Files: `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`.
- **ETT dataset**: 7 transformer-temperature multivariate hourly/15-min series. Downloaded from GitHub (`scripts/download_ett.py`).

### Feature Engineering (`ml/data/feature_lib.py`)
Implemented in **Polars** (lazy evaluation) for speed; **DuckDB** SQL for out-of-core parquet datasets.

Features per series:
- **Lag features**: `y_lag_{1,7,14,28,56}`
- **Rolling stats**: `y_roll_mean_{7,14,28,56}`, `y_roll_std_*`, `y_roll_max_*`
- **Calendar**: day_of_week, month, week_of_year, day_of_month, is_weekend, quarter
- **Transforms**: `y_log = log1p(y)` for zero-inflated series

### Storage
- Raw CSVs → DVC-tracked in `data/raw/`
- Processed features → `data/features/` (Parquet, DVC-tracked)
- Model artifacts → `artifacts/{model_name}/` → MLflow artifact store (MinIO S3)
- Reports → `reports/{backtest,conformal,reconciled,intervention}/` (JSON + PNG)

---

## 3. Modelling Layer

All models use [**NeuralForecast**](https://nixtlaverse.nixtla.io/neuralforecast/) for a unified `fit(df) → predict()` API in long-format DataFrames.

### 3.1 Seasonal Naive (baseline)
Repeats the value from the same day-of-week `s` periods ago. Sets a lower bound on what deep learning must beat.

### 3.2 DeepAR
LSTM-based autoregressive model with Normal distribution head. `DistributionLoss("Normal")` enables sample-based probabilistic forecasting. ~1 M params, CPU-friendly.

### 3.3 N-BEATSx
Interpretable basis expansion architecture. Separate trend (polynomial), seasonality (Fourier), and identity stacks. No attention — very fast on CPU (~15 min).

### 3.4 PatchTST
Treats the time series as a sequence of non-overlapping **patches** (default 16 steps) fed to a standard Transformer encoder. Channel-independence assumption avoids cross-series contamination. Trained with `MQLoss` on quantiles `[0.1, 0.5, 0.9]`. ~7 M params, 3 GB VRAM.

### 3.5 Temporal Fusion Transformer (TFT)
Multi-head attention with variable selection networks, gated residuals, and interpretable attention over past/future covariates. Can ingest static metadata and known future covariates (calendar). ~15 M params, 4 GB VRAM.

### 3.6 Chronos-T5 (zero-shot)
Amazon's pre-trained language-model-style forecaster. Tokenises time-series values into 4096 bins and autoregressively predicts future tokens. No fine-tuning required — evaluated directly on M5 as a zero-shot baseline. `chronos-t5-small` = 46 M params, ~3 GB VRAM.

### Training configuration
- **Framework**: NeuralForecast with LightningTrainer backend
- **Optimiser**: AdamW with cosine LR decay
- **Loss**: MQLoss (quantile regression) for probabilistic heads
- **Tracking**: MLflow autolog + explicit `log_metric` per epoch
- **HPO**: Optuna `HyperbandPruner` (multi-fidelity ASHA), 50 trials per model, MLflowCallback

---

## 4. Probabilistic Calibration (`ml/conformal/`)

### 4.1 Split Conformal Prediction
Given a calibration set {(y_i, ŷ_i)}:
1. Compute nonconformity scores: `sᵢ = |yᵢ − ŷᵢ|`
2. Compute `q̂ = ⌈(1−α)(1 + 1/n)⌉`-th empirical quantile of scores
3. Prediction interval: `[ŷ − q̂, ŷ + q̂]`

Guarantee: marginal coverage ≥ 1 − α on exchangeable data.

### 4.2 Adaptive Conformal Inference (ACI)
For non-stationary or distribution-shifted series, ACI (Gibbs & Candès 2021) updates α online:

```
α_{t+1} = α_t + γ · (α − 1{y_t ∉ CI_t})
```

- If actual is outside CI: α_t increases → wider next interval
- If actual is inside CI: α_t decreases slightly → narrower next interval

`γ = 0.005` (step size). α_t is clamped to (0, 1). This achieves long-run coverage guarantee even under distribution shift.

### Evaluation Metrics
- **Empirical coverage**: fraction of test points inside PI
- **Winkler score**: `(hi − lo) + (2/α)·max(lo − y, 0) + (2/α)·max(y − hi, 0)`
- **PI width**: mean(hi − lo)

---

## 5. Hierarchical Reconciliation (`ml/reconcile/`)

M5 hierarchy (bottom-up):
```
item × store (30 490 series)
      │
  dept × store (70 series)
      │
  category × store (30 series)
      │
  state (10 series)
      │
   national (1 series)
```

### MinT-Shrink (Wickramasuriya et al. 2019)
MinT minimises the trace of the MSE matrix of reconciled forecasts:

`P = (S'Ŵ⁻¹S)⁻¹ S'Ŵ⁻¹`

where `S` is the summation matrix and `Ŵ` is a diagonal estimate of the base forecast residual covariance (shrinkage estimator). Implemented via `hierarchicalforecast.methods.MinTrace(method="mint_shrink")`.

Also implemented: **BottomUp** (sum item forecasts up) and **TopDown** (distribute proportionally from national).

---

## 6. Observability

### MLflow (http://localhost:5001)
- Experiment per model: `pulsepredict-patchtst`, `pulsepredict-tft`, ...
- Params logged: all config keys (horizon, batch_size, max_steps, lr, ...)
- Metrics per step: train_loss, val_loss, MAE, MASE, coverage_90
- Artifacts: model checkpoint, forecast CSV, backtest results

### Prometheus + Grafana (http://localhost:3001)
Metrics exposed at `GET /metrics` (prometheus-client):

| Metric | Type | Labels |
|--------|------|--------|
| `forecast_requests_total` | Counter | model, status |
| `forecast_latency_seconds` | Histogram | model |
| `batch_jobs_total` | Counter | status |
| `batch_job_duration_seconds` | Histogram | — |

Grafana panels: request rate, P99/P50 latency, request count by model, error rate.

### Evidently (drift monitor)
- `evidently.metrics.RegressionPresetMetrics` on rolling 7-day residual windows
- Alerts: MASE drift > 10% from baseline → Prometheus counter `drift_alerts_total`
