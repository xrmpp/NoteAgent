from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent.models import AgentRun, AgentRunResult, AgentStep, ToolCall, run_metadata


class PublicInfoResponse(BaseModel):
    app_title: str
    provider: str
    model: str


class ThreadCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=255)

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("title cannot be empty")
        return cleaned


class ThreadTitleRequest(ThreadCreateRequest):
    title: str = Field(min_length=1, max_length=255)


class ThreadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str | None
    created_at: datetime
    updated_at: datetime


class ThreadListResponse(BaseModel):
    threads: list[ThreadResponse]


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int | None
    role: str
    content: str
    created_at: datetime


class MessageListResponse(BaseModel):
    messages: list[MessageResponse]


PublicRunStatus = Literal["running", "completed", "failed", "paused"]


class RunRequest(BaseModel):
    mode: Literal["rag", "agent"] = "agent"
    thread_id: int | None = Field(default=None, gt=0)
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question cannot be empty")
        return cleaned


class DocumentIngestRequest(BaseModel):
    source: str = Field(min_length=1, max_length=512)
    title: str | None = Field(default=None, max_length=255)
    content: str = Field(min_length=1)

    @field_validator("source")
    @classmethod
    def clean_source(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("source cannot be empty")
        return cleaned

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content cannot be empty")
        return value

    def validate_source_extension(self) -> None:
        if Path(self.source).suffix.lower() not in {".md", ".txt"}:
            raise ValueError("source must end with .md or .txt")


class AgentStepResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    step_order: int
    name: str
    input: dict[str, Any] | None
    output: dict[str, Any] | None
    latency_ms: float | None
    error: str | None
    created_at: datetime

    @classmethod
    def from_domain(cls, step: AgentStep) -> Self:
        return cls.model_validate(step)


class ToolCallResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    step_id: int | None
    tool_name: str
    arguments: dict[str, Any] | None
    result: dict[str, Any] | None
    latency_ms: float | None
    error: str | None
    created_at: datetime

    @classmethod
    def from_domain(cls, call: ToolCall) -> Self:
        return cls.model_validate(call)


class RunResponse(BaseModel):
    mode: Literal["rag", "agent"] = "agent"
    stop_reason: str | None = None
    run_id: int | None
    thread_id: int | None = None
    question: str | None = None
    status: PublicRunStatus
    final_answer: str
    error: str | None = None
    total_latency_ms: float | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    approval: dict[str, Any] | None = None
    steps: list[AgentStepResponse] = Field(default_factory=list)
    tool_calls: list[ToolCallResponse] = Field(default_factory=list)

    @classmethod
    def from_result(cls, result: AgentRunResult, run: AgentRun | None = None) -> Self:
        steps = [AgentStepResponse.from_domain(step) for step in result.steps]
        return cls(
            run_id=result.run_id,
            mode=result.mode,
            stop_reason=result.stop_reason,
            thread_id=run.thread_id if run else None,
            question=run.question if run else None,
            status=to_public_status(result.status),
            final_answer=result.final_answer,
            error=result.error,
            total_latency_ms=result.total_latency_ms,
            started_at=run.started_at if run else None,
            completed_at=run.completed_at if run else None,
            approval=extract_approval(result.steps),
            steps=steps,
            tool_calls=[ToolCallResponse.from_domain(call) for call in result.tool_calls],
        )

    @classmethod
    def from_trace(
        cls,
        run: AgentRun,
        steps: list[AgentStep],
        tool_calls: list[ToolCall],
    ) -> Self:
        return cls(
            run_id=run.id,
            **run_metadata(steps),
            thread_id=run.thread_id,
            question=run.question,
            status=to_public_status(run.status),
            final_answer=run.final_answer or "",
            error=run.error,
            total_latency_ms=run.total_latency_ms,
            started_at=run.started_at,
            completed_at=run.completed_at,
            approval=extract_approval(steps),
            steps=[AgentStepResponse.from_domain(step) for step in steps],
            tool_calls=[ToolCallResponse.from_domain(call) for call in tool_calls],
        )


class HealthResponse(BaseModel):
    status: Literal["ok"]
    database: Literal["ok"]
    provider: str


class DocumentIngestResponse(BaseModel):
    document_id: int
    source: str
    status: Literal["ingested"] = "ingested"


def to_public_status(status: str) -> PublicRunStatus:
    if status == "waiting_approval":
        return "paused"
    if status in {"running", "completed", "failed"}:
        return status
    return "failed"


def extract_approval(steps: list[AgentStep]) -> dict[str, Any] | None:
    # Old persisted runs may still contain approvals; there is no active approval API.
    for step in reversed(steps):
        if step.name == "sql_approval_requested" and step.output:
            return step.output
    return None
