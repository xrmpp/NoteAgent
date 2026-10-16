"""Knowledge loop with bounded conversation context and run-local evidence."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Iterator

from sqlalchemy.exc import SQLAlchemyError

from app.agent.evidence import EvidenceStore, model_tool_payload
from app.agent.knowledge_prompts import CONTEXT_PROMPT, SYSTEM_PROMPT, build_system_prompt
from app.agent.models import (
    KnowledgeEvent,
    KnowledgeFinishedEvent,
    KnowledgeStepEvent,
    KnowledgeToolEvent,
)
from app.agent.rag import RagService
from app.agent.rag_answer import CitationValidator, RagAnswerGenerator
from app.agent.tools import build_knowledge_tool_executor, query_signature
from app.llm.base import ProviderExecutionError, ToolCallingProvider, provider_error_details


@dataclass
class KnowledgeRunState:
    """All state for one knowledge run; never shared between requests."""

    question: str
    allow_tools: bool
    evidence: EvidenceStore
    deadline: float
    tool_deadline: float
    executor: Any
    schemas: list[dict[str, Any]]
    summary: dict[str, Any]
    messages: list[dict[str, Any]] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    tool_count: int = 0
    correcting: bool = False
    limit_reason: str | None = None
    tokens_unknown: bool = False


class KnowledgeLoop:
    MAX_MODEL_CALLS = 8
    MAX_TOOL_CALLS = 6
    TASK_SECONDS = 120.0
    FINAL_RESERVE_SECONDS = 30.0
    EVIDENCE_CHARS = 20000

    def __init__(self, provider: ToolCallingProvider, rag_service: RagService) -> None:
        self.provider = provider
        self.rag_service = rag_service

    def stream(self, question: str, *, allow_tools: bool = True,
               history: list[dict[str, Any]] | None = None) -> Iterator[KnowledgeEvent]:
        state = self._new_state(question, allow_tools)
        history = history or []
        query = question
        if history:
            query = yield from self._resolve_question(state, history)
            if query is None:
                return
        started = time.perf_counter()
        try:
            initial_event, model_payload = self._initial_retrieval(state, query, started)
        except (TimeoutError, ProviderExecutionError, SQLAlchemyError) as exc:
            reason = ("timeout" if isinstance(exc, TimeoutError) or
                      getattr(getattr(exc, "orig", None), "sqlstate", None) == "57014" else
                      "provider_error" if isinstance(exc, ProviderExecutionError) else "tool_error")
            yield KnowledgeStepEvent(
                "knowledge_initial_retrieval", {"query": query, "top_k": 3},
                {"error": reason, "detail": str(exc)[:120] if isinstance(exc, TimeoutError) else None,
                 **({"error_details": provider_error_details(exc)}
                    if isinstance(exc, (TimeoutError, ProviderExecutionError)) else {})},
                (time.perf_counter() - started) * 1000, reason)
            yield self._finish(state, reason)
            return
        yield initial_event
        state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
             "<conversation_context>\n" + json.dumps(history, ensure_ascii=False) +
             "\n</conversation_context>\n当前问题:" + question + "\n检索问题:" + query +
             "\n\n<untrusted_initial_evidence>\n" +
             json.dumps(model_payload, ensure_ascii=False) + "\n</untrusted_initial_evidence>"},
        ]
        state.seen = {"search_knowledge:" + query_signature(query)}

        for model_count in range(state.summary["model_calls"] + 1, self.MAX_MODEL_CALLS + 1):
            remaining = state.deadline - time.monotonic()
            if remaining <= 0:
                yield self._finish(state, "timeout")
                return
            if state.allow_tools and not state.limit_reason:
                if state.tool_count >= self.MAX_TOOL_CALLS:
                    state.limit_reason = "tool_limit"
                elif model_count >= self.MAX_MODEL_CALLS - 1:
                    state.limit_reason = "model_limit"
                elif time.monotonic() >= state.tool_deadline:
                    state.limit_reason = "time_reserve"
            finalizing = state.correcting or state.limit_reason is not None or not state.allow_tools
            state.messages[0]["content"] = build_system_prompt(
                tools_remaining=max(0, self.MAX_TOOL_CALLS - state.tool_count),
                models_remaining=self.MAX_MODEL_CALLS - model_count + 1,
                coverage=state.evidence.coverage(),
                tools_closed=state.correcting or state.limit_reason is not None,
                tools_disabled=not state.allow_tools,
            )
            started = time.perf_counter()
            try:
                state.summary["model_calls"] += 1
                response = self.provider.generate_tools(
                    state.messages, [] if finalizing else state.schemas,
                    timeout_seconds=min(30.0, remaining))
            except (TimeoutError, ProviderExecutionError, AttributeError) as exc:
                reason = "timeout" if isinstance(exc, TimeoutError) else "provider_error"
                yield KnowledgeStepEvent("knowledge_model", None, {
                    "model_call": model_count, "decision": "error", "tool_names": [], "usage": {},
                    "finalizing": finalizing, "tools_remaining": max(0, self.MAX_TOOL_CALLS - state.tool_count),
                    "error_details": provider_error_details(exc),
                }, (time.perf_counter() - started) * 1000, reason)
                yield self._finish(state, reason)
                return
            message = response["message"]
            calls = message.get("tool_calls") or []
            state.summary["tool_requests"] += len(calls)
            self._accumulate_tokens(state, response.get("usage") or {})
            yield KnowledgeStepEvent("knowledge_model", None, {
                "model_call": model_count, "decision": "tools" if calls else "final",
                "tool_calls": [{"id": c["id"], "name": c["function"]["name"]} for c in calls],
                "tool_names": [c["function"]["name"] for c in calls],
                "usage": response.get("usage", {}), "finalizing": finalizing,
                "tools_remaining": max(0, self.MAX_TOOL_CALLS - state.tool_count),
            }, (time.perf_counter() - started) * 1000)
            state.messages.append(message)  # Native reasoning fields remain in memory only.
            if time.monotonic() >= state.deadline:
                yield self._finish(state, "timeout")
                return
            if not calls:
                if (yield from self._handle_final_answer(state, model_count)):
                    return
                continue
            fatal = "provider_error" if finalizing else None
            fatal = yield from self._execute_tool_batch(state, calls, fatal)
            if fatal:
                yield self._finish(state, fatal)
                return
        yield self._finish(state, "budget")

    def _new_state(self, question: str, allow_tools: bool) -> KnowledgeRunState:
        evidence = EvidenceStore(self.EVIDENCE_CHARS)
        deadline = time.monotonic() + self.TASK_SECONDS
        tool_deadline = deadline - min(self.FINAL_RESERVE_SECONDS, self.TASK_SECONDS / 2)
        executor = build_knowledge_tool_executor(self.rag_service, evidence, deadline=tool_deadline)
        schemas = executor.schemas() if allow_tools else []
        summary: dict[str, Any] = {"initial_retrieval_count": 0, "model_calls": 0,
                                   "tool_requests": 0, "tool_executions": 0,
                                   "delivered_text_chars": 0, "tokens": None}
        return KnowledgeRunState(question=question, allow_tools=allow_tools, evidence=evidence,
                                 deadline=deadline, tool_deadline=tool_deadline, executor=executor,
                                 schemas=schemas, summary=summary)

    def _resolve_question(self, state: KnowledgeRunState,
                          history: list[dict[str, Any]]) -> Generator[KnowledgeEvent, None, str | None]:
        started = time.perf_counter()
        step_input = {"question": state.question, "history_message_ids": [m["id"] for m in history],
                      "history_chars": sum(len(m["content"]) for m in history)}
        usage: dict[str, Any] = {}
        error = None
        error_details = None
        try:
            state.summary["model_calls"] += 1
            response = self.provider.generate_tools([
                {"role": "system", "content": CONTEXT_PROMPT},
                {"role": "user", "content": json.dumps(
                    {"history": history, "question": state.question}, ensure_ascii=False)},
            ], [], timeout_seconds=min(30.0, max(0.001, state.deadline - time.monotonic())))
            usage = response.get("usage") or {}
            self._accumulate_tokens(state, usage)
            message = response["message"]
            resolved = json.loads(message.get("content") or "")
            query, clarification = resolved["query"], resolved["clarification"]
            if (message.get("tool_calls") or not isinstance(query, str) or
                    not isinstance(clarification, str) or len(query) > 2000 or len(clarification) > 300):
                raise ValueError("Invalid context resolution")
            query, clarification = query.strip(), clarification.strip()
            if bool(query) == bool(clarification):
                raise ValueError("Expected a query or a clarification")
            if time.monotonic() >= state.deadline:
                raise TimeoutError()
        except (TimeoutError, ProviderExecutionError, ValueError, TypeError, KeyError, AttributeError) as exc:
            error = "timeout" if isinstance(exc, TimeoutError) else "context_resolution_error"
            error_details = provider_error_details(exc)
        output = ({"usage": usage, "error_details": error_details} if error else
                  {"query": query, "clarification": clarification, "usage": usage})
        yield KnowledgeStepEvent("knowledge_context", step_input, output,
                                 (time.perf_counter() - started) * 1000, error)
        if error:
            yield self._finish(state, error)
            return None
        if clarification:
            yield self._finish(state, "clarification", clarification)
            return None
        return query

    def _initial_retrieval(self, state: KnowledgeRunState, query: str,
                           started: float) -> tuple[KnowledgeStepEvent, dict[str, Any]]:
        raw = self.rag_service.search_knowledge(query, top_k=3, deadline=state.tool_deadline)
        initial = state.evidence.present(raw, reading=False)
        model_payload = model_tool_payload(initial)
        state.summary["initial_retrieval_count"] = 1
        state.summary["delivered_text_chars"] += sum(len(row["content"]) for row in model_payload["chunks"])
        event = KnowledgeStepEvent(
            "knowledge_initial_retrieval", {"query": query, "top_k": 3},
            {**initial, "model_payload": model_payload}, (time.perf_counter() - started) * 1000)
        return event, model_payload

    def _accumulate_tokens(self, state: KnowledgeRunState, usage: dict[str, Any]) -> None:
        if usage.get("total_tokens") is None:
            state.tokens_unknown = True
            state.summary["tokens"] = None
        elif not state.tokens_unknown:
            state.summary["tokens"] = (state.summary["tokens"] or 0) + usage["total_tokens"]

    def _handle_final_answer(self, state: KnowledgeRunState, model_count: int) -> Iterator[KnowledgeEvent]:
        """Validate a model's final JSON; return True when the run is finished."""
        content = state.messages[-1].get("content")
        try:
            payload = json.loads(content or "")
            if not isinstance(payload, dict):
                raise ValueError("Expected JSON object")
            answer = RagAnswerGenerator.parse(payload)
            check = CitationValidator().validate(answer, list(state.evidence.items.values()))
            if not check.valid:
                raise ValueError("; ".join(check.errors))
        except (ValueError, TypeError) as exc:
            yield KnowledgeStepEvent("knowledge_validation", None, {"valid": False, "detail": str(exc)[:300]}, 0.0,
                                     "invalid_answer_or_citation")
            if state.correcting or model_count == self.MAX_MODEL_CALLS:
                yield self._finish(state, "citation_error")
                return True
            state.correcting = True
            state.messages.append({"role": "user",
                                   "content": "只修正最终 JSON 和引用,不调用工具:" + str(exc)[:300]})
            return False
        yield KnowledgeStepEvent("knowledge_validation", None, {"valid": True,
               "answer": answer.answer, "citation_ids": answer.citation_ids,
               "insufficient_context": answer.insufficient_context}, 0.0)
        reason = "insufficient_context" if answer.insufficient_context else "answer"
        rendered = answer.answer
        if state.limit_reason and not answer.insufficient_context:
            reason = "partial_answer"
            rendered = "本次已停止进一步检索,以下仅依据已取得的资料整理,不保证覆盖全部笔记。\n\n" + rendered
        sources = [item for item in state.evidence.items.values() if item.evidence_id in answer.citation_ids]
        if sources:
            rendered += "\n\n来源:\n" + "\n".join(
                f"- [{item.evidence_id}] {item.source}(片段 {item.chunk_id})" for item in sources)
        yield self._finish(state, reason, rendered, state.limit_reason)
        return True

    def _execute_tool_batch(self, state: KnowledgeRunState, calls: list[dict[str, Any]],
                            fatal: str | None) -> Iterator[KnowledgeEvent]:
        """Run the tool calls from one assistant message; return the (possibly updated) fatal reason."""
        for call in calls:
            name = call["function"]["name"]
            prepared = state.executor.prepare(name, call["function"]["arguments"])
            args, error, executed = prepared.arguments, None, False
            started = time.perf_counter()
            if not prepared.ok:
                error = prepared.error
            if fatal:
                error = "not_executed_after_fatal_error"
            elif state.limit_reason or time.monotonic() >= state.tool_deadline or state.tool_count >= self.MAX_TOOL_CALLS:
                state.limit_reason = state.limit_reason or (
                    "tool_limit" if state.tool_count >= self.MAX_TOOL_CALLS else "time_reserve")
                error = "not_executed_" + state.limit_reason
            else:
                state.tool_count += 1  # Invalid attempts also consume the budget.
                if prepared.ok and prepared.signature in state.seen:
                    state.limit_reason, error = "repeat", "not_executed_repeat"
                elif prepared.ok and not error:
                    state.seen.add(prepared.signature)
            output: dict[str, Any] = {}
            if not error:
                executed = True
                state.summary["tool_executions"] += 1
                execution = state.executor.execute(prepared)
                if execution.fatal_reason:
                    fatal, error = execution.fatal_reason, execution.error
                elif execution.error:
                    error = execution.error
                else:
                    output = execution.output
                    if execution.evidence_exhausted:
                        state.limit_reason = "evidence_limit"
            if error:
                output = {"error": error, "executed": executed}
            model_payload = model_tool_payload(output)
            state.summary["delivered_text_chars"] += sum(len(row["content"]) for row in model_payload["chunks"])
            yield KnowledgeToolEvent(name, call["id"], args,
                                     {**output, "model_payload": model_payload, "executed": executed},
                                     (time.perf_counter() - started) * 1000, error)
            state.messages.append({"role": "tool", "tool_call_id": call["id"],
                                   "content": json.dumps(model_payload, ensure_ascii=False)})
        return fatal

    @staticmethod
    def _finish(state: KnowledgeRunState, reason: str, answer: str = "",
                limit_reason: str | None = None) -> KnowledgeFinishedEvent:
        success = reason in {"answer", "partial_answer", "insufficient_context", "clarification"}
        final_answer = answer if success else f"本次任务未完成:{reason}。执行记录已保留。"
        return KnowledgeFinishedEvent(final_answer, reason, limit_reason,
                                      state.evidence.coverage(), dict(state.summary),
                                      state.evidence.chars)
