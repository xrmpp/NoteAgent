"""Knowledge-tool definitions: one place for schema, validation, execution and evidence."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.agent.evidence import EvidenceStore, model_tool_payload
from app.agent.rag import RagService
from app.llm.base import ProviderExecutionError


INVALID_ARGUMENTS = "invalid_arguments_or_unknown_tool"
DOCUMENT_UNAVAILABLE = "invalid_arguments_or_document_unavailable"


class SearchKnowledgeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=5)


class ReadDocumentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    document_id: int = Field(gt=0)
    start_chunk: int = Field(default=0, ge=0)
    start_offset: int = Field(default=0, ge=0)
    max_chars: int = Field(default=3000, ge=1000, le=16000)


def query_signature(query: str) -> str:
    return " ".join(query.split()).casefold()


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    arguments: type[BaseModel]
    run: Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(slots=True)
class PreparedToolCall:
    ok: bool
    name: str
    arguments: dict[str, Any]
    signature: str
    error: str | None = None


@dataclass(slots=True)
class ToolExecution:
    output: dict[str, Any]
    model_payload: dict[str, Any]
    executed: bool
    error: str | None = None
    fatal_reason: str | None = None
    evidence_exhausted: bool = False


class KnowledgeToolExecutor:
    """Per-run read-only allowlist; never exposes the legacy tool registry."""

    def __init__(self, rag_service: RagService, evidence: EvidenceStore, *, deadline: float | None = None) -> None:
        self._rag_service = rag_service
        self._evidence = evidence
        self._deadline = deadline
        self._specs = [
            ToolSpec("search_knowledge",
                     "Locate documents with short previews and chunk positions. Use read_document for full text.",
                     SearchKnowledgeArgs, self._search),
            ToolSpec("read_document",
                     "Read continuous text from a discovered document. Default 3000 chars. For a specific fact start at the matching chunk_index and stop when sufficient. For a whole-document overview use 12000–16000 chars from chunk 0 and follow next_cursor.",
                     ReadDocumentArgs, self._read),
        ]
        self._by_name = {spec.name: spec for spec in self._specs}

    def specs(self) -> list[ToolSpec]:
        return list(self._specs)

    def schemas(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": spec.name,
                 "description": spec.description, "parameters": spec.arguments.model_json_schema()}}
                for spec in self._specs]

    def prepare(self, name: str, raw_arguments: str) -> PreparedToolCall:
        spec = self._by_name.get(name)
        if spec is None:
            return PreparedToolCall(False, name, {"raw_arguments": raw_arguments[:1000]}, "", INVALID_ARGUMENTS)
        try:
            loaded = json.loads(raw_arguments)
            parsed = spec.arguments.model_validate(loaded).model_dump()
        except (ValueError, TypeError, ValidationError):
            return PreparedToolCall(False, name, {"raw_arguments": raw_arguments[:1000]}, "", INVALID_ARGUMENTS)
        return PreparedToolCall(True, name, parsed, self._signature(name, parsed))

    def execute(self, prepared: PreparedToolCall) -> ToolExecution:
        spec = self._by_name[prepared.name]
        try:
            raw_output = spec.run(prepared.arguments)
        except ValueError:
            return ToolExecution({}, {}, True, DOCUMENT_UNAVAILABLE)
        except TimeoutError:
            return ToolExecution({}, {}, True, "tool_timeout", "timeout")
        except ProviderExecutionError:
            return ToolExecution({}, {}, True, "embedding_provider_error", "provider_error")
        except SQLAlchemyError as exc:
            reason = "timeout" if getattr(getattr(exc, "orig", None), "sqlstate", None) == "57014" else "tool_error"
            return ToolExecution({}, {}, True, "database_" + reason, reason)
        reading = prepared.name == "read_document"
        output = self._evidence.present(raw_output, reading)
        return ToolExecution(output, model_tool_payload(output), True,
                             evidence_exhausted=self._evidence.exhausted)

    @staticmethod
    def _signature(name: str, arguments: dict[str, Any]) -> str:
        if name == "search_knowledge":
            return "search_knowledge:" + query_signature(str(arguments["query"]))
        return name + ":" + json.dumps(arguments, sort_keys=True)

    def _search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._rag_service.search_knowledge(**arguments, **({"deadline": self._deadline} if self._deadline else {}))

    def _read(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments["document_id"] not in self._evidence.discovered:
            raise ValueError("document_not_discovered: search first")
        return self._rag_service.read_document_page(**arguments, **({"deadline": self._deadline} if self._deadline else {}))


def build_knowledge_tool_executor(rag_service: RagService, evidence: EvidenceStore,
                                  *, deadline: float | None = None) -> KnowledgeToolExecutor:
    return KnowledgeToolExecutor(rag_service, evidence, deadline=deadline)
