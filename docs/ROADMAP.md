# PulsePredict — Roadmap

## 8-Week Build Plan

| Week | Milestone | Status |
|------|-----------|--------|
| **W1** | Scaffold: full repo, Docker Compose stack (Postgres, Redis, MinIO, MLflow, Prometheus, Grafana, FastAPI, Streamlit), directory tree, pyproject.toml, Makefile, DVC pipeline, CI workflow, model wrappers, data loaders, conformal + reconcile + intervention stubs | ✅ Done |
| **W2** | Data pipeline: `make download-m5`, feature lib (lags/rolling/calendar via Polars), DuckDB out-of-core, verify `data/features/` parquet, DVC commit | ⏳ |
| **W3** | Baselines: SeasonalNaive + DeepAR + N-BEATSx, MLflow tracked, backtest runner, first metrics.json | ⏳ |
| **W4** | Transformer models: PatchTST (RTX 2070S, ~1.5 hr), TFT (~2 hr), Optuna HPO for both, best checkpoint saved to MinIO | ⏳ |
| **W5** | Conformal: split conformal calibrated on val set, adaptive CI online simulation, PI coverage + Winkler report, `reports/conformal_coverage.json` | ⏳ |
| **W6** | Chronos-T5: zero-shot eval on M5, ensemble (mean of PatchTST + TFT + Chronos), hierarchical MinT reconciliation, `reports/reconciled/` | ⏳ |
| **W7** | Bayesian intervention: CausalImpact case study on M5 item HOBBIES_1_001_CA_1, 5-replicate simulation, plots, report, Streamlit explorer all 5 pages | ⏳ |
| **W8** | Polish: Evidently residual drift monitor, Grafana 4-panel dashboard live, README final, demo video (OBS screen record), push to GitHub, HuggingFace Space (Streamlit) | ⏳ |

---

## Known Limitations

- **M5 private test**: true M5 competition test labels are not public. Evaluation uses the provided evaluation split only.
- **Chronos hallucinations**: Chronos-T5 may produce unrealistic forecasts for highly irregular or zero-inflated series (common in M5 bottom level). Filter by `y.mean() > 1`.
- **PyMC MCMC speed**: NUTS sampler on long pre-periods (>500 obs) is slow. Mitigate with JAX backend (`pymc.sampling.jax.sample_numpyro_nuts`) or reduce `mcmc_samples`.
- **TFT attention interpretability**: TFT attention weights are not fully interpretable without the original implementation's variable-selection normalisation. Treat as approximate.
- **DuckDB memory**: out-of-core feature building requires DuckDB ≥ 1.0.0 with `memory_limit = '8GB'` set explicitly for M5 full dataset.

---

## Stretch Goals

| Goal | Effort | Impact |
|------|--------|--------|
| Chronos fine-tune on M5 (LoRA adapter) | High | High — shows fine-tuning foundation forecasters |
| N-HiTS (Neural Hierarchical Interpolation) | Medium | Adds another SOTA architecture |
| TimesNet on ETT multivariate | Medium | Covers multivariate forecasting |
| Kaggle M5 submission (public leaderboard) | Low | Concrete score for resume |
| GluonTS DeepState (SSM baseline) | Medium | Classic Bayesian baseline |
| ONNX export + Triton serving | Medium | Production latency demo |
| K8s HPA autoscaling demo | High | MLE infra signal |
