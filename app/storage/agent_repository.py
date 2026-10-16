from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app.agent.models import AgentRun, AgentStep, ToolCall
from app.storage.database import AgentRunORM, AgentStepORM, ToolCallORM


class AgentRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def create_run(self, thread_id: int, question: str) -> AgentRun:
        with self._session_factory() as session:
            run = AgentRunORM(thread_id=thread_id, question=question, status="running")
            session.add(run)
            session.commit()
            session.refresh(run)
            return self._to_run(run)

    def complete_run(
        self,
        run_id: int,
        final_answer: str,
        status: str,
        total_latency_ms: float,
        error: str | None = None,
    ) -> AgentRun:
        with self._session_factory() as session:
            run = session.get(AgentRunORM, run_id)
            if run is None:
                raise ValueError(f"Agent run {run_id} does not exist.")
            run.final_answer = final_answer
            run.status = status
            run.total_latency_ms = total_latency_ms
            run.error = error
            run.completed_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(run)
            return self._to_run(run)

    def record_step(
        self,
        run_id: int,
        name: str,
        input_payload: dict[str, Any] | None,
        output_payload: dict[str, Any] | None,
        latency_ms: float | None,
        error: str | None = None,
    ) -> AgentStep:
        with self._session_factory() as session:
            current_max = session.scalar(
                select(func.max(AgentStepORM.step_order)).where(AgentStepORM.run_id == run_id)
            )
            step = AgentStepORM(
                run_id=run_id,
                step_order=int(current_max or 0) + 1,
                name=name,
                input_json=self._dump(input_payload),
                output_json=self._dump(output_payload),
                latency_ms=latency_ms,
                error=error,
            )
            session.add(step)
            session.commit()
            session.refresh(step)
            return self._to_step(step)

    def record_tool_call(
        self,
        run_id: int,
        step_id: int | None,
        tool_name: str,
        arguments: dict[str, Any] | None,
        result: dict[str, Any] | None,
        latency_ms: float | None,
        error: str | None = None,
    ) -> ToolCall:
        with self._session_factory() as session:
            call = ToolCallORM(
                run_id=run_id,
                step_id=step_id,
                tool_name=tool_name,
                arguments_json=self._dump(arguments),
                result_json=self._dump(result),
                latency_ms=latency_ms,
                error=error,
            )
            session.add(call)
            session.commit()
            session.refresh(call)
            return self._to_tool_call(call)

    def get_run(self, run_id: int) -> AgentRun | None:
        with self._session_factory() as session:
            run = session.get(AgentRunORM, run_id)
            return self._to_run(run) if run is not None else None

    def list_steps(self, run_id: int) -> list[AgentStep]:
        with self._session_factory() as session:
            steps = session.scalars(
                select(AgentStepORM)
                .where(AgentStepORM.run_id == run_id)
                .order_by(AgentStepORM.step_order.asc(), AgentStepORM.id.asc())
            ).all()
            return [self._to_step(step) for step in steps]

    def list_tool_calls(self, run_id: int) -> list[ToolCall]:
        with self._session_factory() as session:
            calls = session.scalars(
                select(ToolCallORM)
                .where(ToolCallORM.run_id == run_id)
                .order_by(ToolCallORM.id.asc())
            ).all()
            return [self._to_tool_call(call) for call in calls]

    def list_recent_runs(self, thread_id: int, limit: int = 5) -> list[AgentRun]:
        with self._session_factory() as session:
            runs = session.scalars(
                select(AgentRunORM)
                .where(AgentRunORM.thread_id == thread_id)
                .order_by(AgentRunORM.started_at.desc(), AgentRunORM.id.desc())
                .limit(limit)
            ).all()
            return [self._to_run(run) for run in runs]

    @staticmethod
    def _dump(payload: dict[str, Any] | None) -> str | None:
        if payload is None:
            return None
        return json.dumps(payload, ensure_ascii=False, default=str)

    @staticmethod
    def _load(payload: str | None) -> dict[str, Any] | None:
        if not payload:
            return None
        loaded = json.loads(payload)
        return loaded if isinstance(loaded, dict) else {"value": loaded}

    @classmethod
    def _to_run(cls, run: AgentRunORM) -> AgentRun:
        return AgentRun(
            id=run.id,
            thread_id=run.thread_id,
            question=run.question,
            status=run.status,
            final_answer=run.final_answer,
            total_latency_ms=run.total_latency_ms,
            error=run.error,
            started_at=run.started_at,
            completed_at=run.completed_at,
        )

    @classmethod
    def _to_step(cls, step: AgentStepORM) -> AgentStep:
        return AgentStep(
            id=step.id,
            run_id=step.run_id,
            step_order=step.step_order,
            name=step.name,
            input=cls._load(step.input_json),
            output=cls._load(step.output_json),
            latency_ms=step.latency_ms,
            error=step.error,
            created_at=step.created_at,
        )

    @classmethod
    def _to_tool_call(cls, call: ToolCallORM) -> ToolCall:
        return ToolCall(
            id=call.id,
            run_id=call.run_id,
            step_id=call.step_id,
            tool_name=call.tool_name,
            arguments=cls._load(call.arguments_json),
            result=cls._load(call.result_json),
            latency_ms=call.latency_ms,
            error=call.error,
            created_at=call.created_at,
        )

