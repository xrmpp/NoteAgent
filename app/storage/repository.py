from __future__ import annotations

from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session, aliased, sessionmaker

from app.domain.models import ChatMessage, ChatThread
from app.storage.database import AgentRunORM, ChatMessageORM, ChatThreadORM


class ChatRepository:
    def __init__(self, session_factory: sessionmaker, primary_thread_title: str) -> None:
        self._session_factory = session_factory
        self._primary_thread_title = primary_thread_title

    def get_or_create_primary_thread(self) -> ChatThread:
        with self._session_factory() as session:
            thread = session.scalar(
                select(ChatThreadORM).where(ChatThreadORM.title == self._primary_thread_title)
            )
            if thread is None:
                thread = ChatThreadORM(title=self._primary_thread_title)
                session.add(thread)
                session.commit()
                session.refresh(thread)
            return self._to_thread(thread)

    def get_thread(self, thread_id: int) -> ChatThread | None:
        with self._session_factory() as session:
            thread = session.get(ChatThreadORM, thread_id)
            return self._to_thread(thread) if thread is not None else None

    def list_messages(self, thread_id: int, limit: int | None = None) -> list[ChatMessage]:
        with self._session_factory() as session:
            stmt = (
                select(ChatMessageORM)
                .where(ChatMessageORM.thread_id == thread_id)
                .order_by(ChatMessageORM.created_at.asc(), ChatMessageORM.id.asc())
            )
            if limit is not None:
                stmt = stmt.limit(limit)
            messages = session.scalars(stmt).all()
            return [self._to_message(message) for message in messages]

    def recent_conversation(self, thread_id: int, turns: int = 3) -> list[ChatMessage]:
        """Return recent completed pairs; cleared messages and failed runs stay excluded."""
        user, assistant = aliased(ChatMessageORM), aliased(ChatMessageORM)
        with self._session_factory() as session:
            pairs = session.execute(
                select(user, assistant)
                .join(assistant, assistant.run_id == user.run_id)
                .join(AgentRunORM, AgentRunORM.id == user.run_id)
                .where(user.thread_id == thread_id, assistant.thread_id == thread_id,
                       user.role == "user", assistant.role == "assistant",
                       AgentRunORM.status == "completed")
                .order_by(user.id.desc()).limit(turns)
            ).all()
            return [self._to_message(message) for pair in reversed(pairs) for message in pair]

    def append_message(
        self,
        thread_id: int,
        role: str,
        content: str,
        run_id: int | None = None,
    ) -> ChatMessage:
        with self._session_factory() as session:
            message = ChatMessageORM(
                thread_id=thread_id,
                run_id=run_id,
                role=role,
                content=content,
            )
            session.add(message)

            thread = session.get(ChatThreadORM, thread_id)
            if thread is None:
                raise ValueError(f"Thread {thread_id} does not exist.")
            session.flush()
            thread.updated_at = func.now()
            session.commit()
            session.refresh(message)
            return self._to_message(message)

    def list_threads(self, limit: int = 20) -> list[ChatThread]:
        with self._session_factory() as session:
            stmt = select(ChatThreadORM).order_by(ChatThreadORM.updated_at.desc(), ChatThreadORM.id.desc()).limit(limit)
            threads = session.scalars(stmt).all()
            return [self._to_thread(t) for t in threads]

    def create_thread(self, title: str | None = None) -> ChatThread:
        with self._session_factory() as session:
            thread = ChatThreadORM(title=title or "新会话")
            session.add(thread)
            session.commit()
            session.refresh(thread)
            return self._to_thread(thread)

    def update_thread_title(self, thread_id: int, title: str) -> None:
        with self._session_factory() as session:
            thread = session.get(ChatThreadORM, thread_id)
            if thread is not None:
                thread.title = title
                thread.updated_at = func.now()
                session.commit()

    def delete_thread(self, thread_id: int) -> bool:
        with self._session_factory() as session:
            thread = session.get(ChatThreadORM, thread_id)
            if thread is None:
                return False
            # 严格按外键依赖顺序物理级联清理，杜绝 ForeignKeyViolation 错误
            session.execute(
                text("DELETE FROM tool_calls WHERE run_id IN (SELECT id FROM agent_runs WHERE thread_id = :tid)"),
                {"tid": thread_id},
            )
            session.execute(
                text("DELETE FROM agent_steps WHERE run_id IN (SELECT id FROM agent_runs WHERE thread_id = :tid)"),
                {"tid": thread_id},
            )
            session.execute(
                text("DELETE FROM chat_messages WHERE thread_id = :tid"),
                {"tid": thread_id},
            )
            session.execute(
                text("DELETE FROM agent_runs WHERE thread_id = :tid"),
                {"tid": thread_id},
            )
            session.execute(
                text("DELETE FROM chat_threads WHERE id = :tid"),
                {"tid": thread_id},
            )
            session.commit()
            return True

    def clear_thread_messages(self, thread_id: int) -> None:
        with self._session_factory() as session:
            session.execute(delete(ChatMessageORM).where(ChatMessageORM.thread_id == thread_id))
            thread = session.get(ChatThreadORM, thread_id)
            if thread is not None:
                thread.updated_at = func.now()
            session.commit()

    def get_message_count(self, thread_id: int | None = None) -> int:
        with self._session_factory() as session:
            stmt = select(func.count(ChatMessageORM.id))
            if thread_id is not None:
                stmt = stmt.where(ChatMessageORM.thread_id == thread_id)
            result = session.scalar(stmt)
            return int(result or 0)

    @staticmethod
    def _to_thread(thread: ChatThreadORM) -> ChatThread:
        return ChatThread(
            id=thread.id,
            title=thread.title,
            created_at=thread.created_at,
            updated_at=thread.updated_at,
        )

    @staticmethod
    def _to_message(message: ChatMessageORM) -> ChatMessage:
        return ChatMessage(
            id=message.id,
            run_id=message.run_id,
            role=message.role,
            content=message.content,
            created_at=message.created_at,
        )
