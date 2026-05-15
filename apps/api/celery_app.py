from celery import Celery
import os


def make_celery() -> Celery:
    """
    Factory that creates and configures the Celery application.

    All settings can be overridden through environment variables so that
    the same image works in dev (local Redis), staging, and production.
    """
    app = Celery("pulsepredict")
    app.config_from_object(
        {
            "broker_url": os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"),
            "result_backend": os.getenv(
                "CELERY_RESULT_BACKEND", "redis://localhost:6379/2"
            ),
            "task_serializer": "json",
            "result_serializer": "json",
            "accept_content": ["json"],
            # Acknowledge the task only after it has completed successfully.
            # This prevents silent message loss when the worker crashes mid-task.
            "task_acks_late": True,
            # Fetch one task at a time so long-running GPU forecasts don't
            # starve other tasks in the queue.
            "worker_prefetch_multiplier": 1,
            # Route all forecast work to the dedicated queue.
            "task_routes": {
                "pulsepredict.batch_forecast": {"queue": "forecast"},
            },
            # Reasonable time limits — override per-task if needed.
            "task_soft_time_limit": 3600,   # 1 hour soft limit → raises SoftTimeLimitExceeded
            "task_time_limit": 3900,        # 5-min grace, then SIGKILL
            "result_expires": 86400,        # keep results for 24 h
        }
    )
    return app


celery_app = make_celery()
