# =============================================================================
#  PulsePredict — Makefile
#  Probabilistic time-series forecasting · MTech DS portfolio project
# =============================================================================
.DEFAULT_GOAL := help
SHELL         := /bin/bash
COMPOSE       := docker compose -f docker-compose.yml

# ── Colours ──────────────────────────────────────────────────────────────────
RESET  := \033[0m
BOLD   := \033[1m
GREEN  := \033[32m
YELLOW := \033[33m
CYAN   := \033[36m

.PHONY: help \
        install dev-install \
        lint lint-fix typecheck \
        test smoke \
        download-m5 download-ett dvc-repro \
        train-naive train-deepar train-nbeatsx train-patchtst train-tft train-all hpo \
        backtest conformal reconcile intervention \
        up down logs db-migrate \
        mlflow-ui \
        k3s-apply

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@printf "$(BOLD)PulsePredict targets$(RESET)\n\n"
	@printf "$(CYAN)Environment$(RESET)\n"
	@printf "  install       Install production dependencies\n"
	@printf "  dev-install   Install dev extras + pre-commit hooks\n\n"
	@printf "$(CYAN)Code quality$(RESET)\n"
	@printf "  lint          Run ruff check (read-only)\n"
	@printf "  lint-fix      Run ruff check --fix + ruff format\n"
	@printf "  typecheck     Run mypy strict\n\n"
	@printf "$(CYAN)Tests$(RESET)\n"
	@printf "  test          Full pytest suite with coverage\n"
	@printf "  smoke         Run integration smoke-train test\n\n"
	@printf "$(CYAN)Data$(RESET)\n"
	@printf "  download-m5   Download M5 Competition dataset via Kaggle API\n"
	@printf "  download-ett  Download ETDataset (ETTh1/ETTh2/ETTm1/ETTm2)\n"
	@printf "  dvc-repro     Reproduce full DVC pipeline\n\n"
	@printf "$(CYAN)Training$(RESET)\n"
	@printf "  train-naive   Seasonal Naïve baseline (StatsForecast)\n"
	@printf "  train-deepar  DeepAR probabilistic model\n"
	@printf "  train-nbeatsx N-BEATSx interpretable model\n"
	@printf "  train-patchtst PatchTST transformer (~1.5 hr on RTX 2070 Super)\n"
	@printf "  train-tft     Temporal Fusion Transformer (~2 hr on RTX 2070 Super)\n"
	@printf "  train-all     Run all four models sequentially\n"
	@printf "  hpo           Optuna hyperparameter optimisation\n\n"
	@printf "$(CYAN)Evaluation$(RESET)\n"
	@printf "  backtest      Walk-forward backtest across all models\n"
	@printf "  conformal     Conformal prediction coverage evaluation\n"
	@printf "  reconcile     Hierarchical MinT reconciliation\n"
	@printf "  intervention  Bayesian causal intervention analysis\n\n"
	@printf "$(CYAN)Serving$(RESET)\n"
	@printf "  up            Start full Docker stack\n"
	@printf "  down          Stop Docker stack\n"
	@printf "  logs          Tail all service logs\n"
	@printf "  db-migrate    Run Alembic migrations inside api container\n\n"
	@printf "$(CYAN)MLflow$(RESET)\n"
	@printf "  mlflow-ui     Open MLflow at http://localhost:5001\n\n"
	@printf "$(CYAN)Kubernetes$(RESET)\n"
	@printf "  k3s-apply     Helm upgrade/install onto local k3s cluster\n"

# ── Environment ───────────────────────────────────────────────────────────────
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pre-commit install
	@printf "$(GREEN)Dev environment ready. pre-commit hooks installed.$(RESET)\n"

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	ruff check ml/ apps/ services/ scripts/ tests/

lint-fix:
	ruff check --fix ml/ apps/ services/ scripts/ tests/
	ruff format ml/ apps/ services/ scripts/ tests/

typecheck:
	mypy ml/ apps/ services/

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest

smoke:
	pytest tests/integration/test_smoke_train.py -v

# ── Data ──────────────────────────────────────────────────────────────────────
download-m5:
	@printf "$(YELLOW)Downloading M5 Competition data via Kaggle API...$(RESET)\n"
	python scripts/download_m5.py

download-ett:
	@printf "$(YELLOW)Downloading ETDataset (ETTh1/ETTh2/ETTm1/ETTm2)...$(RESET)\n"
	python scripts/download_ett.py

dvc-repro:
	dvc repro

# ── Training ──────────────────────────────────────────────────────────────────
train-naive:
	@printf "$(YELLOW)Training Seasonal Naïve baseline...$(RESET)\n"
	python -m ml.train.cli fit --config configs/seasonal_naive.yaml

train-deepar:
	@printf "$(YELLOW)Training DeepAR probabilistic model...$(RESET)\n"
	python -m ml.train.cli fit --config configs/deepar.yaml

train-nbeatsx:
	@printf "$(YELLOW)Training N-BEATSx interpretable model...$(RESET)\n"
	python -m ml.train.cli fit --config configs/nbeatsx.yaml

train-patchtst:
	@printf "$(YELLOW)Training PatchTST transformer (~1.5 hr on RTX 2070 Super)...$(RESET)\n"
	python -m ml.train.cli fit --config configs/patchtst.yaml

train-tft:
	@printf "$(YELLOW)Training Temporal Fusion Transformer (~2 hr on RTX 2070 Super)...$(RESET)\n"
	python -m ml.train.cli fit --config configs/tft.yaml

train-all: train-deepar train-nbeatsx train-patchtst train-tft
	@printf "$(GREEN)All models trained.$(RESET)\n"

hpo:
	@printf "$(YELLOW)Running Optuna HPO sweep...$(RESET)\n"
	python scripts/run_hpo.py

# ── Evaluation ────────────────────────────────────────────────────────────────
backtest:
	@printf "$(YELLOW)Running walk-forward backtest...$(RESET)\n"
	python scripts/run_backtest.py

conformal:
	@printf "$(YELLOW)Evaluating conformal prediction coverage...$(RESET)\n"
	python -m ml.conformal.evaluate

reconcile:
	@printf "$(YELLOW)Running hierarchical MinT reconciliation...$(RESET)\n"
	python -m ml.reconcile.run_reconcile

intervention:
	@printf "$(YELLOW)Running Bayesian causal intervention analysis...$(RESET)\n"
	python -m ml.intervention.run_case

# ── Serving ───────────────────────────────────────────────────────────────────
up:
	@printf "$(BOLD)Starting PulsePredict stack...$(RESET)\n"
	$(COMPOSE) up -d --build
	@printf "\n$(GREEN)Stack is up!$(RESET)\n"
	@printf "  MinIO console  → $(CYAN)http://localhost:9001$(RESET)  (minioadmin / minioadmin)\n"
	@printf "  MLflow UI      → $(CYAN)http://localhost:5001$(RESET)\n"
	@printf "  Grafana        → $(CYAN)http://localhost:3001$(RESET)  (admin / admin)\n"
	@printf "  Prometheus     → $(CYAN)http://localhost:9090$(RESET)\n"
	@printf "  FastAPI docs   → $(CYAN)http://localhost:8000/docs$(RESET)\n"
	@printf "  Streamlit UI   → $(CYAN)http://localhost:8501$(RESET)\n"

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

db-migrate:
	$(COMPOSE) exec api alembic upgrade head

# ── MLflow ────────────────────────────────────────────────────────────────────
mlflow-ui:
	@printf "$(CYAN)Opening MLflow at http://localhost:5001$(RESET)\n"
	open http://localhost:5001 || xdg-open http://localhost:5001

# ── Kubernetes ───────────────────────────────────────────────────────────────
k3s-apply:
	@printf "$(YELLOW)Deploying to local k3s cluster via Helm...$(RESET)\n"
	helm upgrade --install pulsepredict ./infra/helm
