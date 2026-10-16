from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Union


JsonDict = dict[str, Any]


@dataclass(slots=True)
class AgentRun:
    id: int
    thread_id: int
    question: str
    status: str
    final_answer: str | None = None
    total_latency_ms: float | None = None
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


@dataclass(slots=True)
class AgentStep:
    id: int
    run_id: int
    step_order: int
    name: str
    input: JsonDict | None = None
    output: JsonDict | None = None
    latency_ms: float | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class ToolCall:
    id: int
    run_id: int
    tool_name: str
    step_id: int | None = None
    arguments: JsonDict | None = None
    result: JsonDict | None = None
    latency_ms: float | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))



@dataclass(slots=True)
class AgentRunResult:
    run_id: int | None
    final_answer: str
    status: str
    steps: list[AgentStep] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    error: str | None = None
    mode: str = "agent"
    stop_reason: str | None = None
    total_latency_ms: float | None = None


@dataclass(slots=True)
class AgentStreamEvent:
    run_id: int | None
    event_type: str
    node: str | None = None
    status: str = "running"
    payload: JsonDict = field(default_factory=dict)
    result: AgentRunResult | None = None



@dataclass(slots=True)
class RagCitation:
    source: str
    chunk_id: int
    score: float
    evidence_id: str | None = None


@dataclass(slots=True)
class RagEvidence:
    evidence_id: str
    source: str
    chunk_id: int
    content: str
    score: float


@dataclass(slots=True)
class GeneratedRagAnswer:
    answer: str
    citation_ids: list[str]
    insufficient_context: bool


@dataclass(slots=True)
class CitationValidationResult:
    valid: bool
    citation_ids: list[str]
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RagAnswer:
    answer: str
    citations: list[RagCitation]
    retrieved_chunks: list[JsonDict]
    recall_scores: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class KnowledgeStepEvent:
    """A non-tool knowledge-loop step (initial retrieval, model call, validation)."""
    name: str
    input: JsonDict | None
    output: JsonDict
    latency_ms: float
    error: str | None = None

    def to_payload(self) -> JsonDict:
        return {"name": self.name, "input": self.input, "output": self.output,
                "latency_ms": self.latency_ms, "error": self.error}


@dataclass(slots=True)
class KnowledgeToolEvent:
    """A single tool call and its model-visible result."""
    name: str
    tool_call_id: str
    arguments: JsonDict
    output: JsonDict
    latency_ms: float
    error: str | None = None

    def to_payload(self) -> JsonDict:
        return {"name": "knowledge_tool", "output": self.output, "arguments": self.arguments,
                "tool_name": self.name, "tool_call_id": self.tool_call_id,
                "error": self.error, "latency_ms": self.latency_ms}


@dataclass(slots=True)
class KnowledgeFinishedEvent:
    """Terminal event carrying the answer, stop reason and run summary."""
    answer: str
    stop_reason: str
    limit_reason: str | None
    coverage: list[JsonDict]
    summary: JsonDict
    evidence_chars: int

    def to_payload(self) -> JsonDict:
        success = self.stop_reason in {"answer", "partial_answer", "insufficient_context", "clarification"}
        return {"name": "knowledge_finished", "output": {
            "stop_reason": self.stop_reason, "limit_reason": self.limit_reason,
            "coverage": self.coverage, **self.summary, "evidence_chars": self.evidence_chars,
            "final_answer": self.answer if success else f"本次任务未完成:{self.stop_reason}。执行记录已保留。",
        }, "error": None if success else self.stop_reason, "latency_ms": 0.0}


KnowledgeEvent = Union[KnowledgeStepEvent, KnowledgeToolEvent, KnowledgeFinishedEvent]


def run_metadata(steps: list[AgentStep]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"mode": "agent", "stop_reason": None}
    for step in steps:
        if step.name == "knowledge_run" and step.output:
            metadata["mode"] = step.output["mode"]
        elif step.name == "knowledge_finished" and step.output:
            metadata["stop_reason"] = step.output["stop_reason"]
    return metadata
