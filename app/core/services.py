from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import sessionmaker

from app.agent.rag import RagService
from app.agent.rag_answer import RagAnswerGenerator
from app.agent.service import AgentService
from app.chat.service import ChatService
from app.core.config import Settings
from app.llm.base import EmbeddingProvider, LLMProvider
from app.llm.factory import create_embedding_provider, create_provider
from app.storage.agent_repository import AgentRepository
from app.storage.database import create_session_factory
from app.storage.rag_repository import RagRepository
from app.storage.repository import ChatRepository


@dataclass(slots=True)
class AppServices:
    settings: Settings
    session_factory: sessionmaker
    provider: LLMProvider
    chat: ChatService
    agent: AgentService
    rag: RagService
    agent_repository: AgentRepository

    def close(self) -> None:
        pass


def build_app_services(
    settings: Settings,
    *,
    provider_override: LLMProvider | None = None,
    embedding_provider_override: EmbeddingProvider | None = None,
    database_url_override: str | None = None,
    initialize_corpus: bool = True,
    **kwargs: Any,
) -> AppServices:
    session_factory = create_session_factory(
        database_url_override or settings.database_url,
        embedding_dimensions=settings.embedding_dimensions,
    )
    chat_repository = ChatRepository(session_factory, settings.primary_thread_title)
    agent_repository = AgentRepository(session_factory)
    rag_repository = RagRepository(session_factory)
    provider = provider_override or create_provider(settings)
    embedding_provider = embedding_provider_override or create_embedding_provider(settings)

    rag_service = RagService(
        rag_repository,
        embedding_provider=embedding_provider,
        rerank_provider=provider,
        rerank_enabled=settings.rerank_enabled,
        answer_generator=(
            RagAnswerGenerator(provider) if settings.rag_generation_enabled else None
        ),
        generation_enabled=settings.rag_generation_enabled,
        generation_max_chunks=settings.rag_generation_max_chunks,
        generation_context_chars=settings.rag_generation_context_chars,
    )
    if initialize_corpus and settings.auto_ingest_demo_corpus:
        rag_service.ensure_demo_corpus()

    chat_service = ChatService(
        repository=chat_repository,
        provider=provider,
        max_context_messages=settings.max_context_messages,
    )

    agent_service = AgentService(
        chat_repository=chat_repository,
        agent_repository=agent_repository,
        provider=provider,
        rag_service=rag_service,
    )

    return AppServices(
        settings=settings,
        session_factory=session_factory,
        provider=provider,
        chat=chat_service,
        agent=agent_service,
        rag=rag_service,
        agent_repository=agent_repository,
    )
