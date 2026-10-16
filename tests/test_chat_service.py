from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

from app.chat.service import ChatService
from app.domain.models import ChatMessage, ModelResponse
from app.storage.database import create_session_factory
from app.storage.repository import ChatRepository


@dataclass
class FakeProvider:
    provider_name: str = "fake"
    reply_text: str = "hello from fake"

    def generate_reply(self, messages: list[ChatMessage]) -> ModelResponse:
        return ModelResponse(
            content=self.reply_text,
            provider=self.provider_name,
            model="fake-model",
        )


def build_service(tmp_path: Path, max_context_messages: int = 20) -> ChatService:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'chat.db'}"
    repository = ChatRepository(create_session_factory(database_url), "Primary Thread")
    return ChatService(repository, FakeProvider(), max_context_messages=max_context_messages)


def test_send_user_message_persists_user_and_assistant_messages(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    thread = service.ensure_primary_thread()
    service.bootstrap_welcome_message(thread.id, "welcome")

    response = service.send_user_message(thread.id, "hi there")

    messages = service.load_messages(thread.id)
    assert response.content == "hello from fake"
    assert [message.role for message in messages] == ["assistant", "user", "assistant"]
    assert messages[-1].content == "hello from fake"


def test_clear_thread_removes_messages_and_preserves_reuse(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    thread = service.ensure_primary_thread()
    service.bootstrap_welcome_message(thread.id, "welcome")
    service.send_user_message(thread.id, "hi")

    service.clear_thread(thread.id)

    assert service.load_messages(thread.id) == []


def test_context_window_keeps_latest_messages(tmp_path: Path) -> None:
    service = build_service(tmp_path, max_context_messages=3)
    messages = [
        ChatMessage(role="user", content="one"),
        ChatMessage(role="assistant", content="two"),
        ChatMessage(role="user", content="three"),
        ChatMessage(role="assistant", content="four"),
    ]

    selected = service.select_context_window(messages)

    assert [message.content for message in selected] == ["two", "three", "four"]


def test_existing_database_adds_nullable_message_run_link(tmp_path: Path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'old.db'}"
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE chat_threads ("
            "id INTEGER PRIMARY KEY, title VARCHAR(255), "
            "created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, "
            "updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        ))
        connection.execute(text(
            "CREATE TABLE chat_messages ("
            "id INTEGER PRIMARY KEY, thread_id INTEGER NOT NULL, "
            "role VARCHAR(32) NOT NULL, content TEXT NOT NULL, "
            "created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, "
            "FOREIGN KEY(thread_id) REFERENCES chat_threads(id))"
        ))
    engine.dispose()

    factory = create_session_factory(database_url)

    columns = {column["name"] for column in inspect(factory.kw["bind"]).get_columns("chat_messages")}
    indexes = {index["name"] for index in inspect(factory.kw["bind"]).get_indexes("chat_messages")}
    assert "run_id" in columns
    assert "ix_chat_messages_run_id" in indexes
