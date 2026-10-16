from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Callable

import httpx
from pydantic import ValidationError

from app.domain.models import ChatMessage
from app.inbox.models import InboxCandidateORM, InboxTriageResultORM, TriageOutput, utc_now
from app.inbox.store import InboxStore
from app.llm.base import LLMProvider, ProviderExecutionError


DAILY_CALL_LIMIT = 50
PROMPT_PATH = Path("data/inbox/prompts/triage.md")
PROFILE_PATH = Path("data/inbox/profile.md")


@dataclass
class TriageBatch:
    candidates: int = 0
    calls: int = 0
    succeeded: int = 0
    failed: int = 0
    quota_reached: bool = False
    results: list[tuple[InboxCandidateORM, InboxTriageResultORM]] = field(default_factory=list)


def load_prompt(prompt_path: Path, profile_path: Path) -> tuple[str, str]:
    content = prompt_path.read_bytes()
    prompt = content.decode("utf-8")
    if "{{profile}}" not in prompt:
        raise ValueError("triage prompt must contain {{profile}}.")
    profile = profile_path.read_text(encoding="utf-8")
    version = f"{prompt_path.name}@{hashlib.sha256(content).hexdigest()[:8]}"
    return prompt.replace("{{profile}}", profile), version


def build_messages(
    system: str, candidate: InboxCandidateORM, *, exclude_aihot_reason: bool = False,
) -> list[ChatMessage]:
    material = {
        "title": candidate.title, "originalTitle": candidate.original_title,
        "summary": candidate.summary, "source": candidate.source_name,
        "category": candidate.category, "reason": candidate.aihot_reason,
    }
    if exclude_aihot_reason:
        del material["reason"]
    # Escape tag delimiters so material cannot close its own data boundary.
    payload = escape(json.dumps(material, ensure_ascii=False), quote=False)
    return [ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=f"<material>\n{payload}\n</material>")]


def _provider_failure(exc: Exception) -> tuple[str, str | None, bool]:
    cause = exc
    while cause.__cause__ is not None:
        cause = cause.__cause__
    if isinstance(cause, httpx.HTTPStatusError):
        return f"Model HTTP {cause.response.status_code}.", None, False
    if isinstance(cause, (httpx.HTTPError, TimeoutError)):
        return f"Model connection or timeout error ({type(cause).__name__}).", None, False
    if isinstance(cause, json.JSONDecodeError):
        return "Model returned invalid JSON.", cause.doc, True
    if str(exc) in {
        "Model JSON response must be an object.",
        "Model JSON response content is not a string.",
    }:
        return "Model JSON output has an invalid type.", None, True
    # Do not persist provider exception strings: they can include response bodies or credentials.
    return f"Model provider error ({type(exc).__name__}).", None, False


def triage_candidates(
    store: InboxStore, provider: LLMProvider, *, model: str, limit: int = 20,
    candidate_id: int | None = None, retriage: bool = False,
    exclude_aihot_reason: bool = False,
    prompt_path: Path = PROMPT_PATH, profile_path: Path = PROFILE_PATH,
    clock: Callable[[], datetime] = utc_now,
) -> TriageBatch:
    system, version = load_prompt(prompt_path, profile_path)
    if exclude_aihot_reason:
        version += "+noreason"
    candidates = store.triage_candidates(limit=limit, candidate_id=candidate_id, retriage=retriage)
    batch = TriageBatch()
    for candidate in candidates:
        messages = build_messages(system, candidate, exclude_aihot_reason=exclude_aihot_reason)
        result = None
        for attempt in range(2):
            started_at = clock()
            if store.daily_triage_count(started_at) >= DAILY_CALL_LIMIT:
                batch.quota_reached = True
                break
            if attempt == 0:
                batch.candidates += 1
            batch.calls += 1
            output = None
            raw_output = None
            error = None
            retryable = False
            try:
                raw = provider.generate_json(messages, schema_name="inbox_triage")
                raw_output = json.dumps(raw, ensure_ascii=False)
                output = TriageOutput.model_validate(raw)
            except ValidationError as exc:
                error = "Output validation failed: " + json.dumps(
                    exc.errors(include_input=False, include_context=False, include_url=False),
                    ensure_ascii=False,
                )
                retryable = True
            except (ProviderExecutionError, httpx.HTTPError, json.JSONDecodeError, TimeoutError) as exc:
                error, raw_output, retryable = _provider_failure(exc)
            # One append per physical call makes failed/retried calls count toward the UTC quota.
            result = store.append_triage_result(
                candidate.id, prompt_version=version, model=model, output=output,
                raw_output=raw_output, error=error, mark_triaged=True, created_at=started_at,
            )
            if output is not None or not retryable:
                break
        if result is not None:
            candidate = store.get_candidate(candidate.id)
            batch.results.append((candidate, result))
            if result.error:
                batch.failed += 1
            else:
                batch.succeeded += 1
        if batch.quota_reached:
            break
    return batch


def format_results(rows: list[tuple[InboxCandidateORM, InboxTriageResultORM]]) -> str:
    def cell(value) -> str:
        return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    lines = ["| id | 标题 | usage | quality | relevance | reason | 我的决定 | 错误 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for candidate, result in rows:
        decision = "未决定" if candidate.status in {"new", "triaged"} else candidate.status
        if candidate.user_reason:
            decision += f"：{candidate.user_reason}"
        if candidate.drop_reason:
            decision += f"（{candidate.drop_reason}）"
        values = (candidate.id, candidate.title, result.usage, result.quality, result.relevance,
                  result.reason, decision, result.error)
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    return "\n".join(lines)
