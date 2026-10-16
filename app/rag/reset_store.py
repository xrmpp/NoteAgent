from __future__ import annotations

import argparse

from sqlalchemy import create_engine, text

from app.core.config import get_settings


def reset_postgres_rag_store(
    database_url: str,
    *,
    embedding_dimensions: int,
    include_chat: bool = False,
) -> None:
    if embedding_dimensions <= 0:
        raise ValueError("embedding_dimensions must be positive.")

    engine = create_engine(database_url, future=True)
    if engine.dialect.name != "postgresql":
        raise ValueError("RAG vector-dimension reset requires PostgreSQL with pgvector.")

    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        if include_chat:
            connection.execute(
                text(
                    "TRUNCATE TABLE tool_calls, agent_steps, agent_runs, "
                    "chat_messages, chat_threads RESTART IDENTITY CASCADE"
                )
            )
        connection.execute(
            text(
                "TRUNCATE TABLE knowledge_chunks, knowledge_documents "
                "RESTART IDENTITY CASCADE"
            )
        )
        connection.execute(text("DROP INDEX IF EXISTS ix_knowledge_chunks_embedding_hnsw"))
        connection.execute(text("ALTER TABLE knowledge_chunks DROP COLUMN IF EXISTS embedding"))
        connection.execute(
            text(
                "ALTER TABLE knowledge_chunks "
                f"ADD COLUMN embedding vector({embedding_dimensions})"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX ix_knowledge_chunks_embedding_hnsw "
                "ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clear RAG data and rebuild the PostgreSQL vector column."
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        help="Target vector dimensions. Defaults to EMBEDDING_DIMENSIONS.",
    )
    parser.add_argument(
        "--include-chat",
        action="store_true",
        help="Also clear chat messages and agent execution traces.",
    )
    args = parser.parse_args()

    settings = get_settings()
    dimensions = args.dimensions or settings.embedding_dimensions
    reset_postgres_rag_store(
        settings.database_url,
        embedding_dimensions=dimensions,
        include_chat=args.include_chat,
    )
    print(
        f"Reset RAG store with vector({dimensions}); "
        f"chat_cleared={args.include_chat}."
    )


if __name__ == "__main__":
    main()
