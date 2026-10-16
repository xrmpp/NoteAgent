from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str
    id: int | None = None
    run_id: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class ChatThread:
    id: int
    title: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True)
class ModelResponse:
    content: str
    provider: str
    model: str
    error: str | None = None
