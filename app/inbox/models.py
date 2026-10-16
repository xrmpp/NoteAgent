from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from app.storage.database import Base


CandidateStatus = Literal["new", "triaged", "queued", "archived", "dropped", "started", "ingested"]
STATUSES = ("new", "triaged", "queued", "archived", "dropped", "started", "ingested")
STATUS_SQL = "'new', 'triaged', 'queued', 'archived', 'dropped', 'started', 'ingested'"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """SQLite loses offsets; all inbox timestamps are written and read as UTC."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Inbox timestamps must include a timezone.")
        return value.astimezone(timezone.utc)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class CandidateInput(BaseModel):
    external_id: str = Field(min_length=1, max_length=255)
    url: str
    aihot_url: str
    title: str = Field(min_length=1)
    original_title: str | None = None
    summary: str
    source_name: str
    category: Literal["tip", "paper"]
    published_at: datetime | None
    discovered_at: datetime
    aihot_score: float | None = None
    aihot_reason: str | None = None
    raw: dict[str, Any]

    @field_validator("published_at", "discovered_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None:
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("AIHOT timestamps must include a timezone.")
            return value.astimezone(timezone.utc)
        return None


class TriageOutput(BaseModel):
    model_config = ConfigDict(strict=True)

    usage: Literal["practice", "know", "skip"]
    quality: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    reason: str
    try_what: str
    est_minutes: int | None
    prerequisites: list[str]
    unknowns: list[str]

    @model_validator(mode="after")
    def practice_requires_action(self) -> TriageOutput:
        if self.usage == "practice" and not self.try_what.strip():
            raise ValueError("practice requires non-empty try_what.")
        return self


class InboxCandidateORM(Base):
    __tablename__ = "inbox_candidates"
    __table_args__ = (CheckConstraint(f"status IN ({STATUS_SQL})"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    url_normalized: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    aihot_url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    original_title: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    source_name: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    discovered_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    aihot_score: Mapped[float | None] = mapped_column(Float)
    aihot_reason: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    raw_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="new", index=True)
    user_reason: Mapped[str | None] = mapped_column(Text)
    drop_reason: Mapped[str | None] = mapped_column(Text)
    note_source: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    status_updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)


class InboxTriageResultORM(Base):
    __tablename__ = "inbox_triage_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("inbox_candidates.id"), nullable=False, index=True)
    prompt_version: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    usage: Mapped[str | None] = mapped_column(String(32))
    quality: Mapped[int | None] = mapped_column(Integer)
    relevance: Mapped[int | None] = mapped_column(Integer)
    reason: Mapped[str | None] = mapped_column(Text)
    try_what: Mapped[str | None] = mapped_column(Text)
    est_minutes: Mapped[int | None] = mapped_column(Integer)
    prerequisites_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    unknowns_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    similar_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    raw_output: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)


class InboxStatusEventORM(Base):
    __tablename__ = "inbox_status_events"
    __table_args__ = (
        CheckConstraint(f"from_status IS NULL OR from_status IN ({STATUS_SQL})"),
        CheckConstraint(f"to_status IN ({STATUS_SQL})"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("inbox_candidates.id"), nullable=False, index=True)
    from_status: Mapped[str | None] = mapped_column(String(32))
    to_status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
