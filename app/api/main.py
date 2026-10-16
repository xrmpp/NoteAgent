from __future__ import annotations

import json

import anyio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Path, Query, Request, Response, status
from sse_starlette.sse import EventSourceResponse
from sqlalchemy import text
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from app.agent.models import AgentRunResult, AgentStreamEvent
from app.api.schemas import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    HealthResponse,
    MessageListResponse,
    MessageResponse,
    PublicInfoResponse,
    RunRequest,
    RunResponse,
    ThreadCreateRequest,
    ThreadListResponse,
    ThreadResponse,
    ThreadTitleRequest,
)
from app.core.config import Settings, get_settings
from app.core.services import AppServices, build_app_services

MAX_DOCUMENT_BYTES = 1024 * 1024
ServicesFactory = Callable[[Settings], AppServices]


def create_app(services_factory: ServicesFactory = build_app_services) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.services = services_factory(get_settings())
        try:
            yield
        finally:
            application.state.services.close()

    application = FastAPI(title="NoteAgent API", version="1.0.0", lifespan=lifespan)

    @application.get("/healthz", response_model=HealthResponse)
    def healthz(request: Request) -> HealthResponse:
        services = _services(request)
        try:
            with services.session_factory() as session:
                session.execute(text("SELECT 1"))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="database unavailable",
            ) from exc
        return HealthResponse(
            status="ok",
            database="ok",
            provider=services.provider.provider_name,
        )

    @application.get("/v1/info", response_model=PublicInfoResponse)
    def public_info(request: Request) -> PublicInfoResponse:
        services = _services(request)
        return PublicInfoResponse(
            app_title=services.settings.app_title,
            provider=services.provider.provider_name,
            model=services.settings.model_name,
        )

    @application.get("/v1/threads", response_model=ThreadListResponse)
    def list_threads(request: Request, limit: Annotated[int, Query(ge=1, le=100)] = 20) -> ThreadListResponse:
        return ThreadListResponse(threads=[
            ThreadResponse.model_validate(thread) for thread in _services(request).chat.list_threads(limit)
        ])

    @application.post("/v1/threads", response_model=ThreadResponse, status_code=status.HTTP_201_CREATED)
    def create_thread(payload: ThreadCreateRequest, request: Request) -> ThreadResponse:
        return ThreadResponse.model_validate(_services(request).chat.create_thread(payload.title))

    @application.get("/v1/threads/{thread_id}", response_model=ThreadResponse)
    def get_thread(thread_id: Annotated[int, Path(gt=0)], request: Request) -> ThreadResponse:
        services = _services(request)
        _resolve_thread_id(services, thread_id)
        return ThreadResponse.model_validate(services.chat.get_thread(thread_id))

    @application.patch("/v1/threads/{thread_id}", response_model=ThreadResponse)
    def update_thread(thread_id: Annotated[int, Path(gt=0)], payload: ThreadTitleRequest,
                      request: Request) -> ThreadResponse:
        services = _services(request)
        _resolve_thread_id(services, thread_id)
        services.chat.update_thread_title(thread_id, payload.title)
        return ThreadResponse.model_validate(services.chat.get_thread(thread_id))

    @application.get("/v1/threads/{thread_id}/messages", response_model=MessageListResponse)
    def list_messages(thread_id: Annotated[int, Path(gt=0)], request: Request) -> MessageListResponse:
        services = _services(request)
        _resolve_thread_id(services, thread_id)
        return MessageListResponse(messages=[
            MessageResponse.model_validate(message) for message in services.chat.load_messages(thread_id)
        ])

    @application.delete("/v1/threads/{thread_id}/messages", status_code=status.HTTP_204_NO_CONTENT)
    def clear_messages(thread_id: Annotated[int, Path(gt=0)], request: Request) -> Response:
        services = _services(request)
        _resolve_thread_id(services, thread_id)
        services.chat.clear_thread(thread_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @application.delete("/v1/threads/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_thread(thread_id: Annotated[int, Path(gt=0)], request: Request) -> Response:
        if not _services(request).chat.delete_thread(thread_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="thread not found")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @application.post("/v1/agent/runs", response_model=RunResponse)
    def create_run(payload: RunRequest, request: Request) -> RunResponse:
        services = _services(request)
        thread_id = _prepare_run(services, payload)
        result = services.agent.run(thread_id, payload.question, mode=payload.mode)
        return _response_from_result(services, result)

    @application.get("/v1/agent/runs/{run_id}", response_model=RunResponse)
    def get_run(run_id: Annotated[int, Path(gt=0)], request: Request) -> RunResponse:
        services = _services(request)
        run = services.agent_repository.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
        return RunResponse.from_trace(
            run,
            services.agent_repository.list_steps(run_id),
            services.agent_repository.list_tool_calls(run_id),
        )

    @application.post("/v1/agent/runs/stream")
    async def stream_run(payload: RunRequest, request: Request) -> EventSourceResponse:
        services = _services(request)
        thread_id = await run_in_threadpool(_prepare_run, services, payload)

        async def generate_events() -> AsyncIterator[dict[str, str]]:
            iterator = iter(services.agent.stream(thread_id, payload.question, mode=payload.mode))
            try:
                async for event in iterate_in_threadpool(iterator):
                    event_name, event_data = _serialize_stream_event(services, event)
                    yield {"event": event_name, "data": json.dumps(event_data, ensure_ascii=False)}
            except Exception as exc:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(exc)}, ensure_ascii=False),
                }

            finally:
                with anyio.CancelScope(shield=True):
                    await run_in_threadpool(iterator.close)

        return EventSourceResponse(generate_events())

    @application.post("/v1/rag/documents", response_model=DocumentIngestResponse)
    def ingest_document(payload: DocumentIngestRequest, request: Request) -> DocumentIngestResponse:
        try:
            payload.validate_source_extension()
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if len(payload.content.encode("utf-8")) > MAX_DOCUMENT_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="document content exceeds 1 MB",
            )
        services = _services(request)
        document_id = services.rag.ingest_text(payload.source, payload.content, payload.title)
        return DocumentIngestResponse(document_id=document_id, source=payload.source)

    return application


def _services(request: Request) -> AppServices:
    return request.app.state.services


def _resolve_thread_id(services: AppServices, requested_thread_id: int | None) -> int:
    if requested_thread_id is None:
        return services.chat.ensure_primary_thread().id
    if services.chat.get_thread(requested_thread_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="thread not found")
    return requested_thread_id


def _prepare_run(services: AppServices, payload: RunRequest) -> int:
    thread_id = _resolve_thread_id(services, payload.thread_id)
    services.chat.name_thread_from_question(thread_id, payload.question)
    return thread_id


def _response_from_result(services: AppServices, result: AgentRunResult) -> RunResponse:
    run = services.agent_repository.get_run(result.run_id) if result.run_id is not None else None
    return RunResponse.from_result(result, run)


def _serialize_stream_event(
    services: AppServices,
    event: AgentStreamEvent,
) -> tuple[str, dict[str, Any]]:
    event_name = "result" if event.event_type == "final" else event.event_type
    payload: dict[str, Any] = {
        "run_id": event.run_id,
        "type": event_name,
        "node": event.node,
        "status": "paused" if event.status == "waiting_approval" else event.status,
        "payload": event.payload,
    }
    if event.result is not None:
        payload["result"] = _response_from_result(services, event.result).model_dump(mode="json")
    return event_name, payload


app = create_app()
