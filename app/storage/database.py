from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    inspect,
    text as sql_text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

DEFAULT_EMBEDDING_DIMENSIONS = 1536


class Base(DeclarativeBase):
    pass


class ChatThreadORM(Base):
    __tablename__ = "chat_threads"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    messages: Mapped[list["ChatMessageORM"]] = relationship(
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="ChatMessageORM.created_at",
    )


class ChatMessageORM(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    thread_id: Mapped[int] = mapped_column(ForeignKey("chat_threads.id"), nullable=False, index=True)
    run_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_runs.id"), nullable=True, index=True
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    thread: Mapped[ChatThreadORM] = relationship(back_populates="messages")


class AgentRunORM(Base):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    thread_id: Mapped[int] = mapped_column(ForeignKey("chat_threads.id"), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running", index=True)
    total_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    steps: Mapped[list["AgentStepORM"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="AgentStepORM.step_order",
    )
    tool_calls: Mapped[list["ToolCallORM"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="ToolCallORM.id",
    )


class AgentStepORM(Base):
    __tablename__ = "agent_steps"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id"), nullable=False, index=True)
    step_order: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    input_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    run: Mapped[AgentRunORM] = relationship(back_populates="steps")
    tool_calls: Mapped[list["ToolCallORM"]] = relationship(back_populates="step")


class ToolCallORM(Base):
    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id"), nullable=False, index=True)
    step_id: Mapped[int | None] = mapped_column(ForeignKey("agent_steps.id"), nullable=True, index=True)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    arguments_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    run: Mapped[AgentRunORM] = relationship(back_populates="tool_calls")
    step: Mapped[AgentStepORM | None] = relationship(back_populates="tool_calls")


class CustomerORM(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    segment: Mapped[str] = mapped_column(String(64), nullable=False, default="standard")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ProductORM(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(128), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)


class ChannelORM(Base):
    __tablename__ = "channels"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    visitor_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    conversion_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ad_spend: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class OrderORM(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), nullable=False, index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False, index=True)
    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id"), nullable=False, index=True)
    ordered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="paid", index=True)
    refunded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    customer: Mapped[CustomerORM] = relationship()
    product: Mapped[ProductORM] = relationship()
    channel: Mapped[ChannelORM] = relationship()


class KnowledgeDocumentORM(Base):
    __tablename__ = "knowledge_documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    index_signature: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    chunks: Mapped[list["KnowledgeChunkORM"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="KnowledgeChunkORM.chunk_index",
    )


class KnowledgeChunkORM(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_documents.id"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_json: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Any | None] = mapped_column(
        VECTOR(DEFAULT_EMBEDDING_DIMENSIONS), nullable=True
    )
    keyword_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}", server_default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped[KnowledgeDocumentORM] = relationship(back_populates="chunks")


def create_session_factory(
    database_url: str,
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> sessionmaker:
    _set_embedding_dimensions(embedding_dimensions)
    engine = create_engine(database_url, future=True)
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    _ensure_knowledge_metadata_schema(engine)
    _ensure_chat_message_run_schema(engine)
    if engine.dialect.name == "postgresql":
        _ensure_postgres_vector_schema(engine, embedding_dimensions)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def _set_embedding_dimensions(embedding_dimensions: int) -> None:
    if embedding_dimensions <= 0:
        raise ValueError("embedding_dimensions must be positive.")
    KnowledgeChunkORM.__table__.c.embedding.type.dim = embedding_dimensions


def _ensure_knowledge_metadata_schema(engine) -> None:
    additions = {
        "knowledge_documents": {"index_signature": "VARCHAR(64)"},
        "knowledge_chunks": {"metadata_json": "TEXT NOT NULL DEFAULT '{}'"},
    }
    for table, fields in additions.items():
        columns = {column["name"] for column in inspect(engine).get_columns(table)}
        with engine.begin() as connection:
            for name, declaration in fields.items():
                if name not in columns:
                    connection.execute(sql_text(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"))


def _ensure_chat_message_run_schema(engine) -> None:
    """Add the nullable message/run link to databases created before this field existed."""
    columns = {column["name"] for column in inspect(engine).get_columns("chat_messages")}
    with engine.begin() as connection:
        if "run_id" not in columns:
            connection.execute(
                sql_text(
                    "ALTER TABLE chat_messages "
                    "ADD COLUMN run_id INTEGER REFERENCES agent_runs(id)"
                )
            )
        connection.execute(
            sql_text(
                "CREATE INDEX IF NOT EXISTS ix_chat_messages_run_id "
                "ON chat_messages (run_id)"
            )
        )
        _backfill_unambiguous_message_runs(connection)


def _backfill_unambiguous_message_runs(connection) -> None:
    """Link old messages only when message/run text matching is one-to-one per role."""
    candidates = connection.execute(
        sql_text(
            "SELECT message.id AS message_id, message.role AS role, run.id AS run_id "
            "FROM chat_messages AS message "
            "JOIN agent_runs AS run ON run.thread_id = message.thread_id "
            "AND ((message.role = 'user' AND message.content = run.question) "
            "OR (message.role = 'assistant' AND run.status = 'completed' "
            "AND message.content = run.final_answer)) "
            "WHERE message.run_id IS NULL"
        )
    ).mappings().all()
    runs_by_message: dict[int, set[int]] = {}
    messages_by_run_role: dict[tuple[int, str], set[int]] = {}
    for row in candidates:
        message_id, run_id, role = int(row["message_id"]), int(row["run_id"]), str(row["role"])
        runs_by_message.setdefault(message_id, set()).add(run_id)
        messages_by_run_role.setdefault((run_id, role), set()).add(message_id)
    for message_id, run_ids in runs_by_message.items():
        if len(run_ids) != 1:
            continue
        run_id = next(iter(run_ids))
        role = next(
            str(row["role"]) for row in candidates
            if int(row["message_id"]) == message_id and int(row["run_id"]) == run_id
        )
        if len(messages_by_run_role[(run_id, role)]) != 1:
            continue
        connection.execute(
            sql_text(
                "UPDATE chat_messages SET run_id = :run_id "
                "WHERE id = :message_id AND run_id IS NULL"
            ),
            {"run_id": run_id, "message_id": message_id},
        )


def _ensure_postgres_vector_schema(engine, embedding_dimensions: int) -> None:
    with engine.begin() as connection:
        connection.execute(
            sql_text(
                "ALTER TABLE knowledge_chunks "
                f"ADD COLUMN IF NOT EXISTS embedding vector({embedding_dimensions})"
            )
        )
        connection.execute(
            sql_text(
                "CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_embedding_hnsw "
                "ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )
