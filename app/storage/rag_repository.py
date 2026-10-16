from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import contextmanager
from typing import Any

from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import sessionmaker

from app.storage.database import KnowledgeChunkORM, KnowledgeDocumentORM


class RagRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def upsert_document(
        self,
        source: str,
        title: str,
        content: str,
        chunks: list[dict[str, Any]],
        index_signature: str | None = None,
    ) -> int:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        use_pgvector = self._dialect_name() == "postgresql"
        with self._session_factory() as session:
            document = session.scalar(
                select(KnowledgeDocumentORM).where(KnowledgeDocumentORM.source == source)
            )
            if document is None:
                document = KnowledgeDocumentORM(
                    source=source,
                    title=title,
                    content_hash=content_hash,
                    index_signature=index_signature,
                )
                session.add(document)
                session.flush()
            else:
                document.title = title
                document.content_hash = content_hash
                document.index_signature = index_signature
                session.execute(
                    delete(KnowledgeChunkORM).where(
                        KnowledgeChunkORM.document_id == document.id
                    )
                )

            for index, chunk in enumerate(chunks):
                session.add(
                    KnowledgeChunkORM(
                        document_id=document.id,
                        chunk_index=index,
                        source=source,
                        content=chunk["content"],
                        embedding_json=json.dumps(chunk["embedding"]),
                        embedding=chunk["embedding"] if use_pgvector else None,
                        keyword_text=chunk.get("keyword_text", ""),
                        metadata_json=json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                    )
                )
            session.commit()
            return int(document.id)

    @staticmethod
    def _document_summary(document: KnowledgeDocumentORM, chunk_count: int) -> dict[str, Any]:
        return {"id": document.id, "source": document.source, "title": document.title,
                "content_hash": document.content_hash, "index_signature": document.index_signature,
                "chunk_count": chunk_count}

    def find_document(self, source: str) -> dict[str, Any] | None:
        with self._session_factory() as session:
            document = session.scalar(select(KnowledgeDocumentORM).where(
                KnowledgeDocumentORM.source == source))
            if document is None:
                return None
            count = int(session.scalar(select(func.count(KnowledgeChunkORM.id)).where(
                KnowledgeChunkORM.document_id == document.id)) or 0)
            return self._document_summary(document, count)

    def list_documents(self) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            rows = session.execute(
                select(KnowledgeDocumentORM, func.count(KnowledgeChunkORM.id))
                .outerjoin(KnowledgeChunkORM, KnowledgeChunkORM.document_id == KnowledgeDocumentORM.id)
                .group_by(KnowledgeDocumentORM.id, KnowledgeDocumentORM.source,
                          KnowledgeDocumentORM.title, KnowledgeDocumentORM.content_hash,
                          KnowledgeDocumentORM.index_signature, KnowledgeDocumentORM.created_at)
                .order_by(KnowledgeDocumentORM.source)
            ).all()
            return [self._document_summary(document, int(count)) for document, count in rows]

    def delete_document(self, source: str) -> bool:
        with self._session_factory() as session:
            document = session.scalar(select(KnowledgeDocumentORM).where(
                KnowledgeDocumentORM.source == source))
            if document is None:
                return False
            session.execute(delete(KnowledgeChunkORM).where(KnowledgeChunkORM.document_id == document.id))
            session.delete(document)
            session.commit()
            return True

    def document_titles(self, document_ids: list[int], *, deadline: float | None = None) -> dict[int, str]:
        with self._read_session(deadline) as session:
            return dict(session.execute(
                select(KnowledgeDocumentORM.id, KnowledgeDocumentORM.title)
                .where(KnowledgeDocumentORM.id.in_(document_ids))
            ).all())

    def read_document_chunks(
        self, document_id: int, start_chunk: int = 0, chunk_count: int = 3,
    ) -> dict[str, Any]:
        if document_id <= 0 or start_chunk < 0 or not 1 <= chunk_count <= 5:
            raise ValueError("Invalid document range")
        with self._session_factory() as session:
            document = session.get(KnowledgeDocumentORM, document_id)
            if document is None:
                raise ValueError("document_not_found")
            rows = session.scalars(
                select(KnowledgeChunkORM)
                .where(KnowledgeChunkORM.document_id == document_id,
                       KnowledgeChunkORM.chunk_index >= start_chunk)
                .order_by(KnowledgeChunkORM.chunk_index).limit(chunk_count + 1)
            ).all()
            return {
                "chunks": [{**self._to_chunk(row), "title": document.title}
                           for row in rows[:chunk_count]],
                "next_start_chunk": rows[chunk_count].chunk_index if len(rows) > chunk_count else None,
            }

    def read_document_page(
        self, document_id: int, start_chunk: int = 0, start_offset: int = 0,
        max_chars: int = 12000, *, deadline: float | None = None,
    ) -> dict[str, Any]:
        """Page stored text without re-ingesting; cursor also handles oversized chunks."""
        if document_id <= 0 or start_chunk < 0 or start_offset < 0 or not 1000 <= max_chars <= 16000:
            raise ValueError("invalid_document_range")
        with self._read_session(deadline) as session:
            document = session.get(KnowledgeDocumentORM, document_id)
            if document is None:
                raise ValueError("document_not_found")
            total = int(session.scalar(select(func.count(KnowledgeChunkORM.id)).where(
                KnowledgeChunkORM.document_id == document_id)) or 0)
            rows = session.execute(select(
                KnowledgeChunkORM.id, KnowledgeChunkORM.chunk_index, KnowledgeChunkORM.content,
                KnowledgeChunkORM.metadata_json,
            ).where(KnowledgeChunkORM.document_id == document_id,
                    KnowledgeChunkORM.chunk_index >= start_chunk)
              .order_by(KnowledgeChunkORM.chunk_index).limit(65)).all()
            if (not rows and start_offset) or (rows and start_offset >= len(rows[0].content) and start_offset):
                raise ValueError("invalid_document_offset")
            chunks, remaining, cursor = [], max_chars, None
            for i, row in enumerate(rows):
                offset = start_offset if i == 0 else 0
                if i == 64 or remaining == 0:
                    cursor = {"start_chunk": row.chunk_index, "start_offset": offset}
                    break
                content = row.content[offset:offset + remaining]
                end = offset + len(content)
                chunks.append({"id": row.id, "document_id": document_id, "title": document.title,
                               "source": document.source, "chunk_index": row.chunk_index,
                               "metadata": json.loads(row.metadata_json or "{}"),
                               "content": content, "start_offset": offset, "end_offset": end,
                               "chunk_chars": len(row.content)})
                remaining -= len(content)
                if end < len(row.content):
                    cursor = {"start_chunk": row.chunk_index, "start_offset": end}
                    break
            return {"chunks": chunks, "next_cursor": cursor, "total_chunks": total,
                    "start_chunk": start_chunk, "start_offset": start_offset,
                    "returned_chars": max_chars - remaining}

    @contextmanager
    def _read_session(self, deadline: float | None = None):
        with self._session_factory() as session:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Knowledge query deadline exceeded")
                if self._dialect_name() == "postgresql":
                    session.execute(text("SELECT set_config('statement_timeout', :timeout, true)"),
                                    {"timeout": str(max(1, int(min(5.0, remaining) * 1000)))})
            yield session
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Knowledge query deadline exceeded")

    def count_documents(self) -> int:
        with self._session_factory() as session:
            return int(session.scalar(select(func.count(KnowledgeDocumentORM.id))) or 0)

    def corpus_fingerprint(self, *, deadline: float | None = None) -> str:
        with self._read_session(deadline) as session:
            rows = session.execute(
                select(KnowledgeDocumentORM.id, KnowledgeDocumentORM.source,
                       KnowledgeDocumentORM.title, KnowledgeDocumentORM.content_hash,
                       KnowledgeDocumentORM.index_signature,
                       KnowledgeChunkORM.id.label("chunk_id"), KnowledgeChunkORM.chunk_index)
                .outerjoin(KnowledgeChunkORM, KnowledgeChunkORM.document_id == KnowledgeDocumentORM.id)
                .order_by(KnowledgeDocumentORM.id, KnowledgeChunkORM.chunk_index, KnowledgeChunkORM.id)
            ).all()
        payload = json.dumps([tuple(row) for row in rows], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def list_lexical_chunks(self, *, deadline: float | None = None) -> list[dict[str, Any]]:
        with self._read_session(deadline) as session:
            rows = session.execute(
                select(
                    KnowledgeChunkORM.id,
                    KnowledgeChunkORM.document_id,
                    KnowledgeChunkORM.chunk_index,
                    KnowledgeChunkORM.source,
                    KnowledgeChunkORM.content,
                    KnowledgeChunkORM.keyword_text,
                    KnowledgeChunkORM.metadata_json,
                ).order_by(KnowledgeChunkORM.id.asc())
            ).all()
        return [
            {
                "id": row.id,
                "document_id": row.document_id,
                "chunk_index": row.chunk_index,
                "source": row.source,
                "content": row.content,
                "keyword_text": row.keyword_text,
                "metadata": json.loads(row.metadata_json or "{}"),
            }
            for row in rows
        ]

    def list_chunks(self, *, deadline: float | None = None) -> list[dict[str, Any]]:
        with self._read_session(deadline) as session:
            rows = session.scalars(
                select(KnowledgeChunkORM).order_by(KnowledgeChunkORM.id.asc())
            ).all()
            return [self._to_chunk(row) for row in rows]

    def search_by_vector(self, query_embedding: list[float], top_k: int, *, deadline: float | None = None) -> list[dict[str, Any]]:
        if not query_embedding or top_k <= 0:
            return []
        if self._dialect_name() == "postgresql":
            return self._search_by_vector_pg(query_embedding, top_k, deadline=deadline)
        return self._search_by_vector_sqlite(query_embedding, top_k, deadline=deadline)

    def _search_by_vector_pg(
        self, query_embedding: list[float], top_k: int, *, deadline: float | None = None
    ) -> list[dict[str, Any]]:
        distance_expr = KnowledgeChunkORM.embedding.cosine_distance(query_embedding).label("distance")
        with self._read_session(deadline) as session:
            rows = session.execute(
                select(KnowledgeChunkORM, distance_expr)
                .where(KnowledgeChunkORM.embedding.is_not(None))
                .order_by(distance_expr.asc(), KnowledgeChunkORM.id.asc())
                .limit(top_k)
            ).all()
            results: list[dict[str, Any]] = []
            for rank, (row, distance) in enumerate(rows, start=1):
                chunk = self._to_chunk(row)
                chunk["vector_distance"] = float(distance)
                chunk["vector_score"] = 1.0 - float(distance)
                chunk["vector_rank"] = rank
                results.append(chunk)
            return results

    def _search_by_vector_sqlite(
        self, query_embedding: list[float], top_k: int, *, deadline: float | None = None
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for chunk in self.list_chunks(deadline=deadline):
            score = cosine_similarity(query_embedding, chunk["embedding"])
            chunk["vector_score"] = score
            scored.append(chunk)
        scored.sort(key=lambda row: row["vector_score"], reverse=True)
        for rank, row in enumerate(scored[:top_k], start=1):
            row["vector_rank"] = rank
        return scored[:top_k]

    def _dialect_name(self) -> str:
        bind = self._session_factory.kw["bind"]
        return bind.dialect.name

    @staticmethod
    def _to_chunk(row: KnowledgeChunkORM) -> dict[str, Any]:
        return {
            "id": row.id,
            "document_id": row.document_id,
            "chunk_index": row.chunk_index,
            "source": row.source,
            "content": row.content,
            "embedding": json.loads(row.embedding_json),
            "keyword_text": row.keyword_text,
            "metadata": json.loads(row.metadata_json or "{}"),
        }


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
