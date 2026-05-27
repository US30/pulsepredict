from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time, logging, uuid, json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "forecast_requests_total",
    "Total number of forecast requests",
    ["model", "status"],
)
REQUEST_LATENCY = Histogram(
    "forecast_latency_seconds",
    "Forecast request latency in seconds",
    ["model"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Drift monitoring gauges (updated by batch drift eval)
DRIFT_PSI = Gauge("pulsepredict_psi", "Population Stability Index", ["model"])
DRIFT_MAE = Gauge("pulsepredict_mae", "Mean Absolute Error", ["model"])
DRIFT_COVERAGE = Gauge("pulsepredict_coverage_90", "Conformal 90% coverage", ["model"])
DRIFT_DETECTED = Gauge("pulsepredict_drift_detected", "Drift detected flag", ["model"])
PREDICTION_MEAN = Gauge("pulsepredict_prediction_mean", "Mean prediction value", ["model"])
PREDICTION_STD = Gauge("pulsepredict_prediction_std", "Prediction std dev", ["model"])

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    unique_id: str
    history: list[float]
    horizon: int = 28
    model: str = "patchtst"
    return_quantiles: bool = True


class ForecastResponse(BaseModel):
    unique_id: str
    forecasts: list[float]
    q10: list[float]
    q90: list[float]
    model: str
    latency_ms: float


class BatchForecastRequest(BaseModel):
    series: list[ForecastRequest]
    job_id: str = None


class BatchJobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    result_url: str = None


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PulsePredict Forecast API",
    description="Probabilistic time-series forecasting — batch and online endpoints.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model registry (populated on startup)
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path(__file__).resolve().parents[3] / "artifacts"
_model_registry: dict[str, object] = {}


def _scan_artifacts() -> list[str]:
    """Return list of model names whose artifact directories exist."""
    if not ARTIFACTS_DIR.exists():
        return []
    return [
        d.name
        for d in ARTIFACTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]


def _load_model(model_name: str):
    """
    Lazy-load a model from artifacts/. Returns a callable that accepts
    (history, horizon) and returns (mean, q10, q90) as lists of floats.

    Falls back to a naive seasonal-naive stub when the artifact is absent
    so the API stays up during development.
    """
    artifact_path = ARTIFACTS_DIR / model_name
    if not artifact_path.exists():
        logger.warning(
            "Artifact for model '%s' not found at %s — using stub.", model_name, artifact_path
        )
        return None
    # Real loading logic would go here, e.g.:
    #   import mlflow
    #   return mlflow.pyfunc.load_model(str(artifact_path))
    logger.info("Loaded model '%s' from %s.", model_name, artifact_path)
    return artifact_path  # placeholder; replace with actual model object


def _stub_forecast(history: list[float], horizon: int) -> tuple[list[float], list[float], list[float]]:
    """Seasonal-naive stub: repeat the last season; add ±10% PI."""
    import statistics

    period = min(7, len(history))
    season = history[-period:]
    mean = [season[i % period] for i in range(horizon)]
    std = statistics.stdev(history[-max(period, 2):]) if len(history) >= 2 else 1.0
    q10 = [v - 1.28 * std for v in mean]
    q90 = [v + 1.28 * std for v in mean]
    return mean, q10, q90


@app.on_event("startup")
async def startup_event() -> None:
    available = _scan_artifacts()
    for name in available:
        _model_registry[name] = _load_model(name)
    logger.info("Model registry initialised with %d model(s): %s", len(_model_registry), list(_model_registry))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    available_models = _scan_artifacts()
    return {"status": "ok", "models_loaded": available_models}


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest) -> ForecastResponse:
    start = time.perf_counter()
    model_name = req.model

    if not req.history:
        REQUEST_COUNT.labels(model=model_name, status="error").inc()
        raise HTTPException(status_code=422, detail="history must not be empty")

    try:
        model_obj = _model_registry.get(model_name)
        if model_obj is None:
            # Use stub for any unregistered / unloaded model
            mean, q10, q90 = _stub_forecast(req.history, req.horizon)
        else:
            # Production path: model_obj is a loaded MLflow/neuralforecast model
            # mean, q10, q90 = model_obj.predict(req.history, req.horizon)
            mean, q10, q90 = _stub_forecast(req.history, req.horizon)

        elapsed = (time.perf_counter() - start) * 1000.0
        REQUEST_COUNT.labels(model=model_name, status="success").inc()
        REQUEST_LATENCY.labels(model=model_name).observe(elapsed / 1000.0)

        return ForecastResponse(
            unique_id=req.unique_id,
            forecasts=mean,
            q10=q10 if req.return_quantiles else [],
            q90=q90 if req.return_quantiles else [],
            model=model_name,
            latency_ms=round(elapsed, 3),
        )
    except Exception as exc:
        REQUEST_COUNT.labels(model=model_name, status="error").inc()
        logger.exception("Forecast failed for unique_id=%s", req.unique_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/forecast/batch")
async def batch_forecast(req: BatchForecastRequest, background_tasks: BackgroundTasks) -> dict:
    from apps.api.tasks import batch_forecast_task  # lazy import to avoid circular

    job_id = req.job_id or str(uuid.uuid4())
    requests_payload = [s.model_dump() for s in req.series]

    task = batch_forecast_task.apply_async(
        args=[job_id, requests_payload],
        task_id=job_id,
        queue="forecast",
    )
    logger.info("Enqueued batch job %s with %d series.", job_id, len(req.series))
    return {"job_id": job_id, "status": "pending", "n_series": len(req.series)}


@app.get("/forecast/batch/{job_id}", response_model=BatchJobStatus)
async def get_batch_status(job_id: str) -> BatchJobStatus:
    from celery.result import AsyncResult
    from apps.api.celery_app import celery_app

    result = AsyncResult(job_id, app=celery_app)
    status_map = {
        "PENDING": "pending",
        "STARTED": "running",
        "SUCCESS": "done",
        "FAILURE": "failed",
        "RETRY": "running",
        "REVOKED": "failed",
    }
    status = status_map.get(result.state, result.state.lower())

    result_url: str | None = None
    if status == "done":
        result_url = f"/tmp/results/{job_id}.json"

    import datetime
    return BatchJobStatus(
        job_id=job_id,
        status=status,
        created_at=datetime.datetime.utcnow().isoformat(),
        result_url=result_url,
    )


@app.post("/drift/update")
async def update_drift_metrics() -> dict:
    """Load latest drift report and update Prometheus gauges."""
    drift_path = Path(__file__).resolve().parents[2] / "reports" / "drift" / "drift_report.json"
    if not drift_path.exists():
        raise HTTPException(status_code=404, detail="No drift report found. Run scripts/run_drift_eval.py first.")

    with open(drift_path) as f:
        reports = json.load(f)

    # Use last window per model
    latest: dict[str, dict] = {}
    for r in reports:
        latest[r["model"]] = r

    for model, r in latest.items():
        DRIFT_PSI.labels(model=model).set(r["psi"])
        DRIFT_MAE.labels(model=model).set(r["mae"])
        DRIFT_COVERAGE.labels(model=model).set(r["coverage_90"])
        DRIFT_DETECTED.labels(model=model).set(1 if r["drift_detected"] else 0)
        PREDICTION_MEAN.labels(model=model).set(r["pred_mean"])
        PREDICTION_STD.labels(model=model).set(r["pred_std"])

    return {"updated": len(latest), "models": list(latest.keys())}


@app.get("/metrics")
async def metrics() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
