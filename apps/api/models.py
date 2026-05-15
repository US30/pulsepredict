from sqlalchemy import Column, String, DateTime, Text, Enum as SAEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid
import datetime
import enum

Base = declarative_base()


class JobStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ForecastJob(Base):
    """Persists every batch forecast job for audit, retry, and result retrieval."""

    __tablename__ = "forecast_jobs"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    status = Column(
        SAEnum(JobStatus, name="job_status_enum"),
        nullable=False,
        default=JobStatus.PENDING,
    )
    model = Column(String(64), nullable=False)
    request_json = Column(Text, nullable=False)
    result_json = Column(Text, nullable=True)
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
    )
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    def __repr__(self) -> str:
        return (
            f"<ForecastJob id={self.id} model={self.model!r} status={self.status.value}>"
        )
