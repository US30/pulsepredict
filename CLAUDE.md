# PulsePredict

Probabilistic multi-horizon time-series forecasting platform — PatchTST/TFT/Chronos + conformal prediction + MinT reconciliation + Bayesian CausalImpact.

## Owner
Utkarsh Sinha (sinha.utkarshsinha30@gmail.com) — MTech Data Science student building resume portfolio.

## Purpose
Third resume project. Fills time-series forecasting, probabilistic/Bayesian stats, conformal prediction, hierarchical reconciliation gaps. Target: quant, supply-chain, energy, product DS roles.

## Use Case
Benchmark PatchTST, TFT, N-BEATSx, DeepAR, and Chronos-T5 foundation model on M5 Walmart dataset (30k series), produce calibrated 90% PIs via adaptive conformal prediction, reconcile hierarchically with MinT, estimate promo-lift via Bayesian CausalImpact.

## ML Stack
- **Models**: PatchTST, TFT, N-BEATSx, DeepAR, Chronos-T5-small (zero-shot)
- **Framework**: NeuralForecast (unified fit/predict API)
- **Probabilistic**: MQLoss quantile regression + split conformal + Adaptive CI (Gibbs & Candes 2021)
- **Hierarchical**: hierarchicalforecast MinT-shrink + BottomUp + TopDown
- **Intervention**: PyMC local-level BSTS + BayesianCausalImpact
- **HPO**: Optuna multi-fidelity ASHA + MLflowCallback
- **Feature engineering**: Polars lazy API + DuckDB SQL (out-of-core)

## Infra Stack
FastAPI + Celery + Redis + PostgreSQL + MinIO + MLflow + Prometheus + Grafana + Docker Compose + GitHub Actions CI + DVC + Helm/k3s + Streamlit UI

## GPU Budget (RTX 2070 Super 8 GB)
| Model | VRAM | Train time |
|---|---|---|
| DeepAR | CPU | ~20 min |
| N-BEATSx | CPU | ~15 min |
| PatchTST | ~3 GB | ~1.5 hr |
| TFT | ~4 GB | ~2 hr |
| Chronos-T5-small | ~3 GB | 0 (zero-shot) |

## 8-Week Roadmap
| Week | Milestone |
|---|---|
| 1 | Scaffold (commit 655d736, 2026-05-16) |
| 2 | Feature lib (Polars/DuckDB) + M5 download |
| 3 | SeasonalNaive + DeepAR + N-BEATSx baselines |
| 4 | PatchTST + TFT + Optuna HPO |
| 5 | Conformal prediction (split + ACI) |
| 6 | Chronos-T5 zero-shot + MinT reconciliation |
| 7 | Bayesian CausalImpact + Streamlit explorer |
| 8 | Drift monitor + Grafana dashboards + demo video |

## Datasets
M5 (Walmart, 30k series, Kaggle free), ETT (Electricity Transformer Temp, GitHub free).

## Ablation Questions
1. Which forecasting architecture performs best at 28-day horizon?
2. Does ACI improve calibration over split conformal under distribution shift?
3. Does MinT reconciliation improve aggregate accuracy?
4. Can Bayesian CausalImpact reliably estimate promotion lift?

## Portfolio Context
Project 3 of 4. Others: KhetSAR (satellite EO), DocuMind (multimodal doc RAG), GraphPulse (streaming fraud). All share infra patterns.
