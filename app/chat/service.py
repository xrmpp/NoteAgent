from __future__ import annotations

from app.domain.models import ChatMessage, ChatThread, ModelResponse
from app.llm.base import LLMProvider
from app.storage.repository import ChatRepository


WELCOME_MESSAGE = "你好！我是你的私有知识库助手。请随时向我提问笔记中的内容。"
DEFAULT_THREAD_TITLES = {"", "新会话", "Primary Thread", "默认会话"}


class ChatService:
    def __init__(
        self,
        repository: ChatRepository,
        provider: LLMProvider,
        max_context_messages: int = 20,
    ) -> None:
        self._repository = repository
        self._provider = provider
        self._max_context_messages = max_context_messages

    def ensure_primary_thread(self) -> ChatThread:
        thread = self._repository.get_or_create_primary_thread()
        self.bootstrap_welcome_message(thread.id, WELCOME_MESSAGE)
        return thread

    def get_thread(self, thread_id: int) -> ChatThread | None:
        return self._repository.get_thread(thread_id)

    def list_threads(self, limit: int = 20) -> list[ChatThread]:
        return self._repository.list_threads(limit=limit)

    def create_thread(self, title: str | None = None) -> ChatThread:
        thread = self._repository.create_thread(title=title)
        self.bootstrap_welcome_message(thread.id, WELCOME_MESSAGE)
        return thread

    def update_thread_title(self, thread_id: int, title: str) -> None:
        self._repository.update_thread_title(thread_id, title)

    def name_thread_from_question(self, thread_id: int, question: str) -> None:
        thread = self.get_thread(thread_id)
        if thread is not None and (thread.title or "").strip() in DEFAULT_THREAD_TITLES:
            title = question[:14] + ("..." if len(question) > 14 else "")
            self.update_thread_title(thread_id, title)

    def delete_thread(self, thread_id: int) -> bool:
        return self._repository.delete_thread(thread_id)

    def load_messages(self, thread_id: int) -> list[ChatMessage]:
        return self._repository.list_messages(thread_id=thread_id)

    def get_message_count(self, thread_id: int) -> int:
        return self._repository.get_message_count(thread_id=thread_id)

    def clear_thread(self, thread_id: int) -> None:
        self._repository.clear_thread_messages(thread_id)
        self.bootstrap_welcome_message(thread_id, WELCOME_MESSAGE)

    def send_user_message(self, thread_id: int, prompt: str) -> ModelResponse:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            return ModelResponse(
                content="",
                provider=self._provider.provider_name,
                model="",
                error="Prompt cannot be empty.",
            )

        self._repository.append_message(thread_id, "user", cleaned_prompt)
        messages = self._repository.list_messages(thread_id=thread_id)
        context = self.select_context_window(messages)
        response = self._provider.generate_reply(context)

        if not response.error:
            self._repository.append_message(thread_id, "assistant", response.content)
        return response

    def bootstrap_welcome_message(self, thread_id: int, content: str) -> None:
        if self._repository.get_message_count(thread_id) == 0:
            self._repository.append_message(thread_id, "assistant", content)

    def select_context_window(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        if len(messages) <= self._max_context_messages:
            return messages
        return messages[-self._max_context_messages :]
