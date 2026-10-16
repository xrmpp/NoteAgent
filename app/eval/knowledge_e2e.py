"""One repeatable HTTP smoke run against the existing knowledge corpus."""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.agent.knowledge import KNOWLEDGE_PROMPT_VERSION
from app.api.schemas import RunResponse
from app.storage.database import create_session_factory
from app.storage.rag_repository import RagRepository
from app.ui.api_backend import ApiAgentBackend, ApiBackendError


CASES = (
    {
        "question": "文档重新导入时，系统如何避免产生重复的过期切片？",
        "source": "document_ingestion.md",
        "expected": "相同 source 的文档再次导入时，系统会更新 document 记录、删除旧 chunk 并插入新 chunk。",
    },
    {
        "question": "在多路召回阶段，向量召回与关键词召回分别负责什么，结果如何融合？",
        "source": "rag_pipeline.md",
        "expected": "向量召回负责语义相近内容，关键词召回负责精确术语、表名和配置项；两路结果使用 RRF 融合。",
    },
    {
        "question": "为什么不用固定权重相加？",
        "source": "rag_pipeline.md",
        "expected": "承接上一轮多路召回融合机制，说明因为不同检索器的分数尺度不可比，所以不能直接用固定权重相加。",
        "followup": True,
        "followup_pattern": r"(?:权重|尺度|不可比|rrf|融合)",
    },
    {
        "question": "根据现有文档，系统在 2035 年计划支持的量子向量计算引擎名称是什么？",
        "source": None,
        "expected": "现有文档未记载任何 2035 年量子向量计算引擎，应明确说明资料不足并拒答。",
        "insufficient": True,
    },
)


def inspect_case(
    post: dict,
    persisted: dict,
    repository: RagRepository,
    insufficient: bool,
    expected_source: str | None = None,
) -> list[str]:
    errors: list[str] = []
    steps = persisted.get("steps") or []
    names = [step["name"] for step in steps]
    initial = [step for step in steps if step["name"] == "knowledge_initial_retrieval"]
    models = [step for step in steps if step["name"] == "knowledge_model"]
    finished = next((step.get("output") or {} for step in reversed(steps)
                     if step["name"] == "knowledge_finished"), {})
    if persisted.get("status") != "completed" or finished.get("stop_reason") not in {"answer", "partial_answer", "insufficient_context"}:
        errors.append("no_successful_terminal_state")
    if len(initial) != 1 or not models or names.index("knowledge_initial_retrieval") > names.index("knowledge_model"):
        errors.append("initial_retrieval_order_or_count")
    if any(call["tool_name"] not in {"search_knowledge", "read_document"} for call in persisted.get("tool_calls", [])):
        errors.append("unexpected_tool")
    if finished.get("limit_reason") or finished.get("stop_reason") in {"citation_error", "timeout"}:
        errors.append("limit_or_validation_failure")
    if any(post.get(key) != persisted.get(key) for key in ("final_answer", "mode", "stop_reason")):
        errors.append("post_get_mismatch")
    validations = [step.get("output") or {} for step in steps if step["name"] == "knowledge_validation"]
    valid = next((item for item in reversed(validations) if item.get("valid")), None)
    if valid is None:
        errors.append("missing_validated_answer")
    elif insufficient:
        if not valid.get("insufficient_context") or valid.get("citation_ids"):
            errors.append("insufficient_answer_has_citations")
    else:
        delivered = {}
        for step in steps:
            if step["name"] in {"knowledge_initial_retrieval", "knowledge_tool"}:
                for row in (step.get("output") or {}).get("model_payload", {}).get("chunks", []):
                    delivered[row["evidence_id"]] = row
        chunks = {int(row["id"]): row for row in repository.list_chunks()}
        if not valid.get("citation_ids"):
            errors.append("missing_citation")
        citation_sources: set[str] = set()
        for citation in valid.get("citation_ids", []):
            row = delivered.get(citation)
            original = chunks.get(int(row["chunk_id"])) if row else None
            if not original or row["content"] != original["content"][row["start_offset"]:row["end_offset"]]:
                errors.append("citation_not_delivered_verbatim")
            if original and original.get("source"):
                citation_sources.add(original["source"])
        if expected_source and not any(expected_source in s for s in citation_sources):
            errors.append("citation_source_mismatch")
    return errors


def inspect_thread_api(client: httpx.Client, backend: ApiAgentBackend) -> dict:
    report: dict = {"status": "failed", "thread_id": None, "checks": [], "errors": []}

    def check(name: str, passed: bool) -> None:
        report["checks"].append({"name": name, "passed": passed})
        if not passed:
            report["errors"].append(name)

    thread_id = None
    deleted = False
    try:
        info = client.get("/v1/info")
        check("public_info_whitelist", info.status_code == 200 and
              set(info.json()) == {"app_title", "provider", "model"})
        backend.get_info()
        created = client.post("/v1/threads", json={})
        check("create_201", created.status_code == 201)
        created.raise_for_status()
        thread_id = int(created.json()["id"])
        report["thread_id"] = thread_id
        check("list_contains_created", any(t.id == thread_id for t in backend.list_threads()))
        check("get_created", backend.get_thread(thread_id).id == thread_id)
        welcome = backend.load_messages(thread_id)
        check("created_welcome", len(welcome) == 1 and welcome[0].role == "assistant" and
              welcome[0].run_id is None and bool(welcome[0].content))
        renamed = backend.update_thread_title(thread_id, " API 契约验收 ")
        check("rename_trimmed", renamed.title == "API 契约验收" and
              backend.get_thread(thread_id).title == renamed.title)
        for name, method, path, body in (
            ("empty_title_422", "PATCH", f"/v1/threads/{thread_id}", {"title": "  "}),
            ("missing_title_422", "PATCH", f"/v1/threads/{thread_id}", {}),
            ("empty_create_title_422", "POST", "/v1/threads", {"title": "  "}),
            ("limit_zero_422", "GET", "/v1/threads?limit=0", None),
            ("limit_large_422", "GET", "/v1/threads?limit=101", None),
            ("invalid_id_422", "GET", "/v1/threads/0", None),
        ):
            check(name, client.request(method, path, json=body).status_code == 422)
        cleared = client.delete(f"/v1/threads/{thread_id}/messages")
        check("clear_204_empty_body", cleared.status_code == 204 and not cleared.content)
        backend.clear_thread(thread_id)
        messages = backend.load_messages(thread_id)
        check("clear_restores_one_welcome", len(messages) == 1 and bool(welcome) and
              messages[0].content == welcome[0].content and messages[0].role == "assistant")
        removed = client.delete(f"/v1/threads/{thread_id}")
        deleted = removed.status_code == 204
        check("delete_204_empty_body", deleted and not removed.content)
        for name, method, path, body in (
            ("get_deleted_404", "GET", f"/v1/threads/{thread_id}", None),
            ("messages_deleted_404", "GET", f"/v1/threads/{thread_id}/messages", None),
            ("rename_deleted_404", "PATCH", f"/v1/threads/{thread_id}", {"title": "不存在"}),
            ("clear_deleted_404", "DELETE", f"/v1/threads/{thread_id}/messages", None),
            ("delete_deleted_404", "DELETE", f"/v1/threads/{thread_id}", None),
            ("run_deleted_404", "POST", "/v1/agent/runs", {"thread_id": thread_id, "question": "无效会话"}),
            ("stream_deleted_404", "POST", "/v1/agent/runs/stream", {"thread_id": thread_id, "question": "无效会话"}),
        ):
            check(name, client.request(method, path, json=body).status_code == 404)
    except (httpx.HTTPError, ApiBackendError, ValueError, KeyError, TypeError) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if thread_id is not None and not deleted:
            try:
                backend.delete_thread(thread_id)
            except ApiBackendError as exc:
                report["errors"].append(f"cleanup_failed: {exc}")
    report["status"] = "failed" if report["errors"] else "pass"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/knowledge-e2e.json"))
    args = parser.parse_args()
    settings = get_settings()
    session_factory = create_session_factory(settings.database_url, embedding_dimensions=settings.embedding_dimensions)
    repository = RagRepository(session_factory)
    artifact = {"executed_at": datetime.now(timezone.utc).isoformat(),
                "model": settings.model_name, "embedding_model": settings.embedding_model,
                "prompt_version": KNOWLEDGE_PROMPT_VERSION,
                "thread_id": None, "status": "blocked", "cases": [],
                "uncovered": ["UI interaction", "ambiguous clarification", "stream interruption",
                              "context truncation stress", "fingerprint invalidation"]}
    try:
        with httpx.Client(base_url="http://127.0.0.1:8000", timeout=httpx.Timeout(180.0, connect=10.0)) as client:
            backend = ApiAgentBackend("http://127.0.0.1:8000", client=client)
            artifact["thread_api"] = inspect_thread_api(client, backend)
            thread = backend.create_thread()
            artifact["thread_id"] = thread.id
            blocked = False
            for index, case in enumerate(CASES):
                entry = {**case, "post": None, "persisted": None, "errors": [], "status": "not_run"}
                artifact["cases"].append(entry)
                if blocked:
                    continue
                try:
                    messages = backend.load_messages(thread.id)
                    completed_ids = {m.run_id for m in messages if m.role == "assistant" and m.run_id is not None}
                    history_ids = [m.id for m in messages if m.run_id in completed_ids and
                                   m.role in {"user", "assistant"}][-6:]
                    if index == 1:
                        backend.update_thread_title(thread.id, "API-only 对话与进度验收")
                    request = {"question": case["question"], "thread_id": thread.id}
                    if index:
                        entry["events"] = []
                        started = time.monotonic()
                        for event in backend.stream(thread.id, case["question"]):
                            entry["events"].append({"type": event.event_type, "node": event.node,
                                                    "elapsed_ms": (time.monotonic() - started) * 1000})
                            if event.result:
                                entry["post"] = RunResponse.from_result(event.result).model_dump(mode="json")
                        if not any(e["type"] == "node_completed" for e in entry["events"][:-1]):
                            entry["errors"].append("missing_intermediate_stream_events")
                    else:
                        response = client.post("/v1/agent/runs", json=request)
                        response.raise_for_status()
                        entry["post"] = response.json()
                    title = backend.get_thread(thread.id).title
                    expected_title = (case["question"][:14] + ("..." if len(case["question"]) > 14 else "")
                                      if index == 0 else "API-only 对话与进度验收")
                    if title != expected_title:
                        entry["errors"].append("thread_auto_title_or_custom_title_changed")
                    run_id = entry["post"].get("run_id")
                    if run_id is not None:
                        stored = client.get(f"/v1/agent/runs/{run_id}")
                        stored.raise_for_status()
                        entry["persisted"] = stored.json()
                        entry["errors"].extend(inspect_case(entry["post"], entry["persisted"], repository,
                                                            case.get("insufficient", False),
                                                            expected_source=case.get("source")))
                        steps = entry["persisted"]["steps"]
                        context = next((s for s in steps if s["name"] == "knowledge_context"), None)
                        if history_ids:
                            if (not context or context["input"]["history_message_ids"] != history_ids or
                                    context["input"]["history_chars"] > 6000):
                                entry["errors"].append("conversation_context_mismatch")
                        elif context:
                            entry["errors"].append("new_thread_inherited_context")
                        finished = next((step.get("output") or {} for step in reversed(entry["persisted"]["steps"])
                                         if step["name"] == "knowledge_finished"), {})
                        entry["summary"] = {key: finished.get(key) for key in
                                            ("initial_retrieval_count", "model_calls", "tool_requests",
                                             "tool_executions", "delivered_text_chars", "evidence_chars",
                                             "tokens", "stop_reason", "limit_reason")}
                        entry["summary"]["total_latency_ms"] = entry["persisted"].get("total_latency_ms")
                        initial = next((step for step in entry["persisted"]["steps"]
                                        if step["name"] == "knowledge_initial_retrieval"), None)
                        if initial:
                            artifact["corpus_fingerprint"] = (initial.get("output") or {}).get("retrieval", {}).get("corpus_fingerprint")
                            query = initial["input"]["query"]
                            if context and context["output"].get("query") != query:
                                entry["errors"].append("resolved_query_not_used")
                            if case.get("followup"):
                                pattern = case.get("followup_pattern", r"(?:权重|尺度|不可比|rrf|融合)")
                                if not re.search(pattern, query, re.IGNORECASE):
                                    entry["errors"].append("followup_subject_or_negation_lost")
                        if finished.get("model_calls") != sum(s["name"] in {"knowledge_context", "knowledge_model"} for s in steps):
                            entry["errors"].append("model_calls_accounting_mismatch")
                        if entry["post"].get("total_latency_ms") != entry["persisted"].get("total_latency_ms"):
                            entry["errors"].append("latency_contract_mismatch")
                    else:
                        entry["errors"].append("missing_run_id")
                    entry["status"] = "pass" if not entry["errors"] else "failed"
                    if (entry.get("summary") or {}).get("stop_reason") == "provider_error" or (
                        (entry.get("summary") or {}).get("stop_reason") == "timeout"
                        and (entry.get("summary") or {}).get("model_calls") == 0
                    ):
                        entry["status"] = "blocked"
                        blocked = True
                except (httpx.HTTPError, ApiBackendError, OSError) as exc:
                    entry["errors"].append(str(exc))
                    entry["status"] = "blocked"
                    blocked = True
                finally:
                    artifact["status"] = ("blocked" if blocked else "failed" if
                        artifact["thread_api"]["status"] != "pass" or any(
                            item["status"] == "failed" for item in artifact["cases"]) else "pass")
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        artifact["status"] = "blocked"
        artifact["error"] = str(exc)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"status={artifact['status']} output={args.output}")
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
