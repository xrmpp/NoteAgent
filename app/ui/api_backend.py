from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterator

import httpx

from app.agent.models import AgentRunResult, AgentStep, AgentStreamEvent, ToolCall
from app.api.schemas import MessageListResponse, PublicInfoResponse, RunResponse, ThreadListResponse, ThreadResponse
from app.domain.models import ChatMessage, ChatThread


class ApiBackendError(RuntimeError):
    pass


class ApiAgentBackend:
    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 150.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)

    def get_info(self) -> PublicInfoResponse:
        try:
            return PublicInfoResponse.model_validate(self._request("GET", "/v1/info"))
        except ValueError as exc:
            raise ApiBackendError("运行信息响应格式错误。") from exc

    def list_threads(self, limit: int = 20) -> list[ChatThread]:
        try:
            payload = self._request("GET", "/v1/threads", params={"limit": limit})
            response = ThreadListResponse.model_validate(payload)
            return [ChatThread(**thread.model_dump()) for thread in response.threads]
        except ValueError as exc:
            raise ApiBackendError("会话列表响应格式错误。") from exc

    def get_thread(self, thread_id: int) -> ChatThread:
        return self._to_thread(self._request("GET", f"/v1/threads/{thread_id}"))

    def create_thread(self, title: str | None = None) -> ChatThread:
        return self._to_thread(self._request("POST", "/v1/threads", json={"title": title}))

    def update_thread_title(self, thread_id: int, title: str) -> ChatThread:
        return self._to_thread(self._request("PATCH", f"/v1/threads/{thread_id}", json={"title": title}))

    def delete_thread(self, thread_id: int) -> None:
        self._request("DELETE", f"/v1/threads/{thread_id}")

    def load_messages(self, thread_id: int) -> list[ChatMessage]:
        try:
            payload = self._request("GET", f"/v1/threads/{thread_id}/messages")
            response = MessageListResponse.model_validate(payload)
            return [ChatMessage(**message.model_dump()) for message in response.messages]
        except ValueError as exc:
            raise ApiBackendError("历史消息响应格式错误。") from exc

    def clear_thread(self, thread_id: int) -> None:
        self._request("DELETE", f"/v1/threads/{thread_id}/messages")

    def run(self, thread_id: int, question: str, mode: str = "agent") -> AgentRunResult:
        payload = self._request(
            "POST",
            "/v1/agent/runs",
            json={"thread_id": thread_id, "question": question, "mode": mode},
        )
        return self._to_result(payload)

    def get_run(self, run_id: int) -> AgentRunResult:
        return self._to_result(self._request("GET", f"/v1/agent/runs/{run_id}"))

    def stream(self, thread_id: int, question: str, mode: str = "agent") -> Iterator[AgentStreamEvent]:
        try:
            with self._client.stream(
                "POST", f"{self._base_url}/v1/agent/runs/stream",
                json={"thread_id": thread_id, "question": question, "mode": mode},
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()
                data: list[str] = []
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        data.append(line[5:].lstrip(" "))
                    elif not line and data:
                        payload = json.loads("\n".join(data))
                        data.clear()
                        kind = payload.get("type")
                        if kind not in {"started", "node_completed", "result"}:
                            raise ApiBackendError("运行事件异常，请查看服务端执行记录。")
                        result = self._to_result(payload["result"]) if kind == "result" else None
                        yield AgentStreamEvent(
                            run_id=payload.get("run_id"), event_type="final" if result else kind,
                            node=payload.get("node"), status=payload.get("status", "running"),
                            payload=payload.get("payload") or {}, result=result)
                        if result is not None:
                            return
        except httpx.HTTPStatusError as exc:
            raise ApiBackendError(
                f"后端 API 返回 HTTP {exc.response.status_code}: {_response_error_detail(exc.response)}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiBackendError("进度连接失败或中断，请查看已有运行记录；问题未自动重发。") from exc
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ApiBackendError("运行事件格式错误，请查看已有运行记录。") from exc
        raise ApiBackendError("进度连接已结束，但未收到最终结果；问题未自动重发。")

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, f"{self._base_url}{path}", **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ApiBackendError(
                f"后端 API 返回 HTTP {exc.response.status_code}: {_response_error_detail(exc.response)}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiBackendError("无法连接后端 API，请检查服务与 API 地址；请求未自动重试。") from exc
        if response.status_code == 204:
            return {}
        try:
            data = response.json()
        except ValueError as exc:
            raise ApiBackendError("后端 API 返回了无效的 JSON。") from exc
        if not isinstance(data, dict):
            raise ApiBackendError("后端 API 返回了无效的数据格式。")
        return data

    @staticmethod
    def _to_thread(payload: dict[str, Any]) -> ChatThread:
        try:
            return ChatThread(**ThreadResponse.model_validate(payload).model_dump())
        except ValueError as exc:
            raise ApiBackendError("会话响应格式错误。") from exc

    @staticmethod
    def _to_result(payload: dict[str, Any]) -> AgentRunResult:
        try:
            validated = RunResponse.model_validate(payload).model_dump()
        except ValueError as exc:
            raise ApiBackendError("运行结果响应格式错误。") from exc
        return AgentRunResult(
            run_id=validated["run_id"],
            final_answer=validated["final_answer"],
            status="waiting_approval" if validated["status"] == "paused" else validated["status"],
            error=validated["error"],
            mode=validated["mode"],
            stop_reason=validated["stop_reason"],
            total_latency_ms=validated["total_latency_ms"],
            steps=[_to_step(item) for item in validated["steps"]],
            tool_calls=[_to_tool_call(item) for item in validated["tool_calls"]],
        )


def _to_step(payload: dict[str, Any]) -> AgentStep:
    return AgentStep(
        id=int(payload["id"]),
        run_id=int(payload["run_id"]),
        step_order=int(payload["step_order"]),
        name=str(payload["name"]),
        input=payload.get("input"),
        output=payload.get("output"),
        latency_ms=payload.get("latency_ms"),
        error=payload.get("error"),
        created_at=_parse_datetime(payload.get("created_at")),
    )


def _to_tool_call(payload: dict[str, Any]) -> ToolCall:
    return ToolCall(
        id=int(payload["id"]),
        run_id=int(payload["run_id"]),
        step_id=payload.get("step_id"),
        tool_name=str(payload["tool_name"]),
        arguments=payload.get("arguments"),
        result=payload.get("result"),
        latency_ms=payload.get("latency_ms"),
        error=payload.get("error"),
        created_at=_parse_datetime(payload.get("created_at")),
    )


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return datetime.now().astimezone()


def _response_error_detail(response: httpx.Response) -> str:
    if response.status_code >= 500:
        return "服务端处理失败，请检查后端运行记录。"
    try:
        payload = response.json()
    except ValueError:
        return "请求未被接受。"
    if isinstance(payload, dict) and isinstance(payload.get("detail"), str):
        return payload["detail"][:300]
    return "请求参数无效或操作未被接受。"
