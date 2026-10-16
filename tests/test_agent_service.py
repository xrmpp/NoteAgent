from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.agent.rag import RagService
from app.agent.rag_answer import RagAnswerGenerator
from app.agent.service import AgentService
from app.domain.models import ChatMessage, ModelResponse
from app.storage.agent_repository import AgentRepository
from app.storage.database import create_session_factory
from app.storage.rag_repository import RagRepository
from app.storage.repository import ChatRepository


@dataclass
class FakeProvider:
    provider_name: str = "fake"
    reply_text: str = "general answer"
    json_queue: list[dict[str, Any]] = field(default_factory=list)

    def generate_reply(self, messages: list[ChatMessage]) -> ModelResponse:
        return ModelResponse(content=self.reply_text, provider=self.provider_name, model="fake-model")

    def generate_json(self, messages: list[ChatMessage], schema_name: str) -> dict[str, Any]:
        if self.json_queue:
            return self.json_queue.pop(0)
        raise RuntimeError("No fake JSON response queued.")

    def generate_tools(self, messages: list[dict[str, Any]], schemas: list[dict[str, Any]], timeout_seconds: float = 30.0) -> dict[str, Any]:
        if self.json_queue:
            content = self.json_queue.pop(0)
            return {"message": {"role": "assistant", "content": content if isinstance(content, str) else str(content)}, "usage": {"total_tokens": 10}}
        return {"message": {"role": "assistant", "content": '{"answer": "测试回答", "citation_ids": [], "insufficient_context": true}'}, "usage": {"total_tokens": 10}}


def build_agent_service(
    tmp_path: Path,
    provider: FakeProvider | None = None,
) -> AgentService:
    tmp_path.mkdir(parents=True, exist_ok=True)
    session_factory = create_session_factory(f"sqlite+pysqlite:///{tmp_path / 'agent.db'}")
    chat_repository = ChatRepository(session_factory, "Primary Thread")
    agent_repository = AgentRepository(session_factory)
    provider = provider or FakeProvider()
    rag_repository = RagRepository(session_factory)
    rag_service = RagService(
        rag_repository,
        answer_generator=RagAnswerGenerator(provider),
        generation_enabled=True,
    )
    rag_service.ensure_demo_corpus()

    return AgentService(
        chat_repository=chat_repository,
        agent_repository=agent_repository,
        provider=provider,
        rag_service=rag_service,
    )


def test_agent_run_knowledge_mode(tmp_path: Path) -> None:
    service = build_agent_service(tmp_path)
    thread = service._chat_repository.get_or_create_primary_thread()

    result = service.run(thread.id, "测试知识库问题", mode="agent")

    assert result.status in {"completed", "failed"}
    assert result.run_id is not None
    messages = service._chat_repository.list_messages(thread.id)
    assert len(messages) >= 1
    assert messages[0].role == "user"
