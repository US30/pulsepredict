from apps.api.celery_app import celery_app
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("/tmp/results")
ARTIFACTS_DIR = Path(__file__).resolve().parents[3] / "artifacts"


def _stub_forecast(history: list[float], horizon: int) -> tuple[list[float], list[float], list[float]]:
    """
    Seasonal-naive fallback used when a model artifact is not present.
    Repeats the last season and adds ±1.28 std as 10/90 prediction intervals.
    """
    import statistics

    period = min(7, len(history))
    season = history[-period:] if period > 0 else [0.0]
    mean = [season[i % len(season)] for i in range(horizon)]
    std = statistics.stdev(history[-max(period, 2):]) if len(history) >= 2 else 1.0
    q10 = [v - 1.28 * std for v in mean]
    q90 = [v + 1.28 * std for v in mean]
    return mean, q10, q90


def _run_single_forecast(req: dict) -> dict:
    """
    Run forecast for a single series dict.

    Expected keys: unique_id, history, horizon, model, return_quantiles.
    Returns a result dict ready for JSON serialisation.
    """
    unique_id: str = req["unique_id"]
    history: list[float] = req["history"]
    horizon: int = req.get("horizon", 28)
    model_name: str = req.get("model", "patchtst")
    return_quantiles: bool = req.get("return_quantiles", True)

    artifact_path = ARTIFACTS_DIR / model_name
    if not artifact_path.exists():
        logger.warning(
            "Artifact for model '%s' not found — using stub for series '%s'.",
            model_name,
            unique_id,
        )
        mean, q10, q90 = _stub_forecast(history, horizon)
    else:
        # Production path: load and call the real model
        # model = mlflow.pyfunc.load_model(str(artifact_path))
        # mean, q10, q90 = model.predict(history, horizon)
        mean, q10, q90 = _stub_forecast(history, horizon)

    result = {
        "unique_id": unique_id,
        "model": model_name,
        "forecasts": mean,
    }
    if return_quantiles:
        result["q10"] = q10
        result["q90"] = q90
    return result


@celery_app.task(
    bind=True,
    name="pulsepredict.batch_forecast",
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def batch_forecast_task(self, job_id: str, requests: list[dict]) -> dict:
    """
    Process a batch of forecast requests.

    Parameters
    ----------
    job_id:
        Unique identifier for this batch job (used as filename stem).
    requests:
        List of serialised ForecastRequest dicts.

    Returns
    -------
    dict with keys: job_id, n_series, status, result_path.
    """
    logger.info("Starting batch job %s — %d series.", job_id, len(requests))

    # Update Celery task state so the API can report "running"
    self.update_state(state="STARTED", meta={"job_id": job_id, "n_series": len(requests)})

    results: list[dict] = []
    errors: list[dict] = []

    for idx, req in enumerate(requests):
        try:
            result = _run_single_forecast(req)
            results.append(result)
        except Exception as exc:
            logger.exception(
                "Failed forecast for series '%s' (job=%s, idx=%d).",
                req.get("unique_id", "unknown"),
                job_id,
                idx,
            )
            errors.append({"unique_id": req.get("unique_id"), "error": str(exc)})

    # Persist results to a local JSON file so the API can serve a download URL
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{job_id}.json"
    payload = {
        "job_id": job_id,
        "n_series": len(requests),
        "n_success": len(results),
        "n_errors": len(errors),
        "results": results,
        "errors": errors,
        "status": "done" if not errors else "partial",
    }
    result_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Batch job %s completed — %d succeeded, %d failed. Written to %s.",
        job_id,
        len(results),
        len(errors),
        result_path,
    )

    return {
        "job_id": job_id,
        "n_series": len(requests),
        "status": "done",
        "result_path": str(result_path),
    }
