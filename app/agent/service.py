"""Unified Agent entry point.

Runs :class:`KnowledgeLoop` for knowledge base interaction (mode="agent" or mode="rag").
"""
from __future__ import annotations

import re
import time
from typing import Any, Iterator

from app.agent.knowledge_prompts import KNOWLEDGE_PROMPT_VERSION
from app.agent.loop import KnowledgeLoop
from app.agent.models import (
    AgentRun,
    AgentRunResult,
    AgentStep,
    AgentStreamEvent,
    KnowledgeEvent,
    KnowledgeFinishedEvent,
    KnowledgeToolEvent,
    ToolCall,
    run_metadata,
)
from app.agent.rag import RagService
from app.llm.base import LLMProvider
from app.storage.agent_repository import AgentRepository
from app.storage.repository import ChatRepository


class AgentService:
    def __init__(
        self,
        chat_repository: ChatRepository,
        agent_repository: AgentRepository,
        provider: LLMProvider,
        rag_service: RagService,
    ) -> None:
        self._chat_repository = chat_repository
        self._agent_repository = agent_repository
        self._provider = provider
        self._rag_service = rag_service

    def run(self, thread_id: int, question: str, mode: str = "agent") -> AgentRunResult:
        for event in self.stream(thread_id, question, mode=mode):
            if event.result is not None:
                return event.result
        raise RuntimeError("Knowledge run did not finish")

    def stream(self, thread_id: int, question: str, mode: str = "agent") -> Iterator[AgentStreamEvent]:
        yield from self._stream_knowledge(thread_id, question, mode)

    def list_recent_runs(self, thread_id: int, limit: int = 5):
        return self._agent_repository.list_recent_runs(thread_id, limit)

    def list_steps(self, run_id: int) -> list[AgentStep]:
        return self._agent_repository.list_steps(run_id)

    def list_tool_calls(self, run_id: int) -> list[ToolCall]:
        return self._agent_repository.list_tool_calls(run_id)

    def get_run(self, run_id: int) -> AgentRunResult | None:
        run = self._agent_repository.get_run(run_id)
        if run is None:
            return None
        return self._result_from_repository(run)

    def _stream_knowledge(self, thread_id: int, question: str, mode: str) -> Iterator[AgentStreamEvent]:
        if mode not in {"rag", "agent"}:
            raise ValueError(f"Unknown knowledge mode: {mode}")
        cleaned_question = question.strip()
        if not cleaned_question:
            yield AgentStreamEvent(None, "final", status="failed", result=AgentRunResult(
                run_id=None, final_answer="", status="failed", error="Prompt cannot be empty.", mode=mode))
            return
        created = self._agent_repository.create_run(thread_id=thread_id, question=cleaned_question)
        run_id = created.id
        self._chat_repository.append_message(thread_id, "user", cleaned_question, run_id=run_id)
        started = time.perf_counter()
        self._agent_repository.record_step(
            run_id=run_id, name="knowledge_run",
            input_payload=None,
            output_payload={"mode": mode, "context_turns": 3, "context_chars_limit": 6000,
                            "strategy": "retrieval_first", "trace_version": 3,
                            "prompt_version": KNOWLEDGE_PROMPT_VERSION,
                            "allow_tools": mode == "agent"},
            latency_ms=0.0, error=None,
        )
        final_answer = ""
        error: str | None = None
        try:
            yield AgentStreamEvent(run_id, "started", payload={"mode": mode})
            history = []
            for message in self._chat_repository.recent_conversation(thread_id):
                content = message.content
                if message.role == "assistant":
                    # Citation IDs belong to one run and must not leak into the next.
                    content = re.sub(r"\[C\d+\]", "", content.partition("\n\n来源:\n")[0])
                history.append({"id": message.id, "role": message.role,
                                "content": content if len(content) <= 1000 else content[:996] + "[截断]"})
            for event in KnowledgeLoop(self._provider, self._rag_service).stream(
                cleaned_question, allow_tools=mode == "agent", history=history
            ):
                payload = event.to_payload()
                step = self._record_event_step(run_id, event, payload)
                if isinstance(event, KnowledgeToolEvent):
                    self._agent_repository.record_tool_call(
                        run_id=run_id, step_id=step.id, tool_name=event.name,
                        arguments=event.arguments, result=event.output,
                        latency_ms=event.latency_ms, error=event.error)
                    yield AgentStreamEvent(run_id, "node_completed", node="knowledge_tool",
                                           payload={**payload["output"], "tool_name": event.name,
                                                    "error": event.error})
                elif isinstance(event, KnowledgeFinishedEvent):
                    final_answer = event.answer
                    error = payload["error"]
                    yield AgentStreamEvent(run_id, "node_completed", node="knowledge_finished",
                                           payload=payload["output"])
                else:
                    yield AgentStreamEvent(run_id, "node_completed", node=event.name,
                                           payload={**event.output, "error": event.error})
            result = self._finalize_knowledge(thread_id, run_id, started, final_answer, error)
        except GeneratorExit:
            self._agent_repository.record_step(
                run_id=run_id, name="knowledge_finished", input_payload=None,
                output_payload={"stop_reason": "cancelled"}, latency_ms=0.0, error=None)
            self._finalize_knowledge(thread_id, run_id, started, "本次任务已取消。", "cancelled")
            raise
        except Exception:
            # Raw SDK errors may contain credentials or response bodies; keep trace bounded and safe.
            self._agent_repository.record_step(
                run_id=run_id, name="knowledge_finished", input_payload=None,
                output_payload={"stop_reason": "tool_error"}, latency_ms=0.0, error=None)
            result = self._fail_knowledge(thread_id, run_id, started)
        yield AgentStreamEvent(run_id, "final", status=result.status, result=result)

    def _record_event_step(self, run_id: int, event: KnowledgeEvent,
                           payload: dict[str, Any]) -> AgentStep:
        if isinstance(event, KnowledgeToolEvent):
            input_payload: dict[str, Any] | None = {"tool_call_id": event.tool_call_id,
                                                    "tool_name": event.name, "arguments": event.arguments}
        elif isinstance(event, KnowledgeFinishedEvent):
            input_payload = None
        else:
            input_payload = event.input
        return self._agent_repository.record_step(
            run_id=run_id, name=payload["name"], input_payload=input_payload,
            output_payload=payload["output"], latency_ms=payload["latency_ms"], error=payload["error"])

    def _finalize_knowledge(self, thread_id: int, run_id: int, started: float,
                            final_answer: str, error: str | None) -> AgentRunResult:
        status = "failed" if error else "completed"
        answer = final_answer or "没有生成回答。"
        run = self._agent_repository.complete_run(
            run_id=run_id, final_answer=answer, status=status,
            total_latency_ms=(time.perf_counter() - started) * 1000, error=error)
        if status == "completed":
            self._chat_repository.append_message(thread_id, "assistant", answer, run_id=run_id)
        return self._result_from_repository(run)

    def _fail_knowledge(self, thread_id: int, run_id: int, started: float) -> AgentRunResult:
        error = "knowledge_execution_error"
        run = self._agent_repository.complete_run(
            run_id=run_id, final_answer="Agent 执行失败:knowledge_execution_error",
            status="failed", total_latency_ms=(time.perf_counter() - started) * 1000, error=error)
        return self._result_from_repository(run)

    def _result_from_repository(self, run: AgentRun) -> AgentRunResult:
        steps = self._agent_repository.list_steps(run.id)
        return AgentRunResult(
            run_id=run.id, final_answer=run.final_answer or "", status=run.status,
            **run_metadata(steps), steps=steps,
            tool_calls=self._agent_repository.list_tool_calls(run.id), error=run.error,
            total_latency_ms=run.total_latency_ms)
