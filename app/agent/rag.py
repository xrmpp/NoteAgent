from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import bm25s

from app.agent.models import (
    CitationValidationResult,
    GeneratedRagAnswer,
    RagAnswer,
    RagCitation,
    RagEvidence,
)
from app.agent.rag_answer import RAG_ANSWER_PROMPT_VERSION
from app.agent.rag_answer import CitationValidator, RagAnswerGenerator
from app.domain.models import ChatMessage
from app.llm.base import EmbeddingProvider, LLMProvider
from app.rag.chunking import Chunker
from app.rag.ingest import MAX_DOCUMENT_BYTES
from app.storage.rag_repository import RagRepository

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"
RRF_K = 60
RECALL_POOL_SIZE = 10


class HashEmbeddingProvider:
    """Deterministic test stub, not a production embedding model."""

    provider_name = "hash"

    def __init__(self, dimensions: int = 1536) -> None:
        self._dimensions = dimensions
        self.index_identity = {"provider": "hash", "dimensions": dimensions}

    def embed_texts(self, texts: list[str], *, deadline: float | None = None) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        tokens = tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self._dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


@dataclass(frozen=True, slots=True)
class DocumentIngestResult:
    document_id: int
    source: str
    status: str
    chunk_count: int
    embedding_texts: int


@dataclass(frozen=True, slots=True)
class _Bm25Snapshot:
    fingerprint: str
    chunks: list[dict[str, Any]]
    retriever: Any | None


class Bm25LexicalIndex:
    """Lazy in-process index; PostgreSQL remains the source of truth."""

    def __init__(self, repository: RagRepository) -> None:
        self._repository = repository
        self._snapshot: _Bm25Snapshot | None = None
        self._lock = threading.Lock()

    def search(
        self,
        query_tokens: list[str],
        top_k: int,
        *,
        deadline: float | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        snapshot = self._ensure_snapshot(deadline)
        if not query_tokens or top_k <= 0 or not snapshot.chunks or snapshot.retriever is None:
            return [], snapshot.fingerprint

        result = snapshot.retriever.retrieve(
            [query_tokens],
            k=min(top_k, len(snapshot.chunks)),
            show_progress=False,
        )
        ranked: list[dict[str, Any]] = []
        for position, score in zip(result.documents[0], result.scores[0]):
            numeric_score = float(score)
            if numeric_score <= 0:
                continue
            row = dict(snapshot.chunks[int(position)])
            row["bm25_score"] = numeric_score
            row["bm25_rank"] = len(ranked) + 1
            ranked.append(row)
        return ranked, snapshot.fingerprint

    def _ensure_snapshot(self, deadline: float | None) -> _Bm25Snapshot:
        fingerprint = self._repository.corpus_fingerprint(deadline=deadline)
        current = self._snapshot
        if current is not None and current.fingerprint == fingerprint:
            return current

        self._acquire(deadline)
        try:
            current = self._snapshot
            fingerprint = self._repository.corpus_fingerprint(deadline=deadline)
            if current is not None and current.fingerprint == fingerprint:
                return current

            chunks: list[dict[str, Any]] = []
            for _ in range(3):
                before = self._repository.corpus_fingerprint(deadline=deadline)
                chunks = self._repository.list_lexical_chunks(deadline=deadline)
                after = self._repository.corpus_fingerprint(deadline=deadline)
                if before == after:
                    fingerprint = after
                    break
            else:
                raise RuntimeError("Knowledge corpus changed repeatedly while building BM25 index")

            self._check_deadline(deadline)
            retriever = None
            if chunks:
                corpus_tokens = [
                    str(row.get("keyword_text") or "").split() or tokenize(str(row["content"]))
                    for row in chunks
                ]
                retriever = bm25s.BM25()
                retriever.index(corpus_tokens, show_progress=False)
            self._check_deadline(deadline)

            snapshot = _Bm25Snapshot(fingerprint, chunks, retriever)
            self._snapshot = snapshot
            return snapshot
        finally:
            self._lock.release()

    def _acquire(self, deadline: float | None) -> None:
        if deadline is None:
            self._lock.acquire()
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._lock.acquire(timeout=remaining):
            raise TimeoutError("Knowledge query deadline exceeded")

    @staticmethod
    def _check_deadline(deadline: float | None) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Knowledge query deadline exceeded")


class RagService:
    def __init__(
        self,
        repository: RagRepository,
        embedding_provider: EmbeddingProvider | None = None,
        rerank_provider: LLMProvider | None = None,
        rerank_enabled: bool = False,
        chunker: Chunker | None = None,
        answer_generator: RagAnswerGenerator | None = None,
        citation_validator: CitationValidator | None = None,
        generation_enabled: bool = False,
        generation_max_chunks: int = 5,
        generation_context_chars: int = 4000,
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider or HashEmbeddingProvider()
        self._rerank_provider = rerank_provider
        self._rerank_enabled = rerank_enabled
        self._chunker = chunker or Chunker()
        self._answer_generator = answer_generator
        self._citation_validator = citation_validator or CitationValidator()
        self._generation_enabled = generation_enabled and answer_generator is not None
        self._generation_max_chunks = max(1, generation_max_chunks)
        self._generation_context_chars = max(200, generation_context_chars)
        self._bm25_index = Bm25LexicalIndex(repository)

    def ingest_text(self, source: str, text: str, title: str | None = None) -> int:
        return self.ingest_document(source, text, title).document_id

    def ingest_document(self, source: str, text: str, title: str | None = None) -> DocumentIngestResult:
        source = source.strip()
        title = (title or Path(source).stem).strip()
        if not source or len(source) > 512 or any(ord(char) < 32 for char in source):
            raise ValueError("Source must be 1-512 characters without control characters.")
        if not title or len(title) > 255:
            raise ValueError("Title must be 1-255 characters.")
        if not text.strip() or "\x00" in text or len(text.encode("utf-8")) > MAX_DOCUMENT_BYTES:
            raise ValueError("Document must contain non-blank UTF-8 text, at most 1 MB.")
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        identity = getattr(self._embedding_provider, "index_identity", {
            "provider": getattr(self._embedding_provider, "provider_name", "custom"),
            "implementation": type(self._embedding_provider).__qualname__,
        })
        config = {"chunker": self._chunker.index_config, "embedding": identity,
                  "markdown": Path(source).suffix.lower() != ".txt", "title": title,
                  "keyword_version": "cjk-bigrams-v1", "input_version": "context-v1"}
        signature = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        previous = self._repository.find_document(source)
        if (previous and previous["content_hash"] == content_hash and
                previous["index_signature"] == signature):
            return DocumentIngestResult(previous["id"], source, "skipped", previous["chunk_count"], 0)

        parts = self._chunker.split_with_metadata(text, markdown=config["markdown"])
        if not parts:
            raise ValueError("Document produced no non-blank chunks.")
        embedding_inputs = [part.retrieval_text(title) for part in parts]
        embeddings = self._embedding_provider.embed_texts(embedding_inputs)
        if len(embeddings) != len(parts):
            raise ValueError(
                f"Embedding count mismatch: expected {len(parts)}, got {len(embeddings)}."
            )
        expected_dimensions = identity.get("dimensions")
        for vector in embeddings:
            if not vector or (expected_dimensions and len(vector) != expected_dimensions):
                raise ValueError("Embedding dimensions do not match the configured index.")
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector):
                raise ValueError("Embedding contains non-finite or non-numeric values.")
        chunks = [
            {
                "content": part.content,
                "metadata": part.metadata,
                "embedding": embedding,
                "keyword_text": " ".join(tokenize(embedding_input)),
            }
            for part, embedding, embedding_input in zip(parts, embeddings, embedding_inputs)
        ]
        document_id = self._repository.upsert_document(
            source=source,
            title=title,
            content=text,
            chunks=chunks,
            index_signature=signature,
        )
        return DocumentIngestResult(document_id, source, "updated" if previous else "created",
                                    len(chunks), len(embedding_inputs))

    def ingest_file(self, path: str | Path) -> int:
        from app.rag.ingest import read_document_file

        file_path = Path(path)
        return self.ingest_text(
            file_path.name,
            read_document_file(file_path),
            file_path.stem,
        )

    def ensure_demo_corpus(self) -> None:
        if self._repository.count_documents() > 0:
            return
        self.ingest_directory(DEFAULT_CORPUS_DIR)

    def ingest_directory(self, path: str | Path) -> list[int]:
        from app.rag.ingest import ingest_directory

        return ingest_directory(self, Path(path))

    def answer(self, question: str, top_k: int = 5) -> dict[str, Any]:
        retrieval_started = time.perf_counter()
        retrieved, retrieval_trace = self._search_with_trace(question, top_k=top_k)
        retrieval_latency_ms = _elapsed_ms(retrieval_started)
        evidence = self._prepare_evidence(retrieved)
        generation = self._base_generation_metadata(evidence, retrieval_latency_ms)

        if not retrieved:
            answer = "知识库没有找到足够相关的内容。"
            citations: list[RagCitation] = []
            generation.update(
                {
                    "mode": "insufficient_context",
                    "validation": {"valid": True, "citation_ids": [], "errors": []},
                }
            )
        elif not self._generation_enabled:
            answer, citations = self._fallback_answer(evidence)
            generation.update(
                {
                    "mode": "retrieval_only",
                    "citation_ids": [item.evidence_id for item in evidence[:2]],
                    "validation": {"valid": True, "citation_ids": [], "errors": []},
                }
            )
        else:
            answer, citations, generation = self._generate_answer(
                question,
                evidence,
                generation,
            )

        rag_answer = RagAnswer(
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved,
            recall_scores={
                "retrieved": len(retrieved),
                "top_score": retrieved[0]["score"] if retrieved else 0.0,
            },
        )
        return {
            "answer": rag_answer.answer,
            "citations": [
                {
                    "source": citation.source,
                    "chunk_id": citation.chunk_id,
                    "score": citation.score,
                    "evidence_id": citation.evidence_id,
                }
                for citation in rag_answer.citations
            ],
            "retrieved_chunks": rag_answer.retrieved_chunks,
            "recall_scores": rag_answer.recall_scores,
            "retrieval": retrieval_trace,
            "generation": generation,
        }

    def _generate_answer(
        self,
        question: str,
        evidence: list[RagEvidence],
        generation: dict[str, Any],
    ) -> tuple[str, list[RagCitation], dict[str, Any]]:
        assert self._answer_generator is not None
        attempts: list[dict[str, Any]] = []
        previous_answer: GeneratedRagAnswer | None = None
        validation_error: str | None = None
        validation = CitationValidationResult(valid=False, citation_ids=[], errors=[])
        generation_started = time.perf_counter()

        for attempt_number in range(2):
            attempt_started = time.perf_counter()
            try:
                generated = self._answer_generator.generate(
                    question,
                    evidence,
                    validation_error=validation_error,
                    previous_answer=previous_answer,
                )
                validation_started = time.perf_counter()
                validation = self._citation_validator.validate(generated, evidence)
                validation_latency_ms = _elapsed_ms(validation_started)
                attempts.append(
                    {
                        "attempt": attempt_number + 1,
                        "model_output": asdict(generated),
                        "validation": asdict(validation),
                        "latency_ms": _elapsed_ms(attempt_started),
                        "validation_latency_ms": validation_latency_ms,
                        "error": None,
                    }
                )
                previous_answer = generated
                if validation.valid:
                    generation.update(
                        {
                            "mode": (
                                "insufficient_context"
                                if generated.insufficient_context
                                else "generated"
                            ),
                            "citation_ids": list(generated.citation_ids),
                            "validation": asdict(validation),
                            "correction_count": attempt_number,
                            "latency_ms": _elapsed_ms(generation_started),
                            "attempts": attempts,
                        }
                    )
                    if generated.insufficient_context:
                        return "知识库现有资料不足以回答这个问题。", [], generation
                    answer, citations = self._render_generated_answer(generated, evidence)
                    return answer, citations, generation
                validation_error = "; ".join(validation.errors)
            except Exception as exc:
                validation_error = str(exc)
                attempts.append(
                    {
                        "attempt": attempt_number + 1,
                        "model_output": None,
                        "validation": None,
                        "latency_ms": _elapsed_ms(attempt_started),
                        "validation_latency_ms": 0.0,
                        "error": validation_error,
                    }
                )

        answer, citations = self._fallback_answer(evidence)
        generation.update(
            {
                "mode": "fallback",
                "citation_ids": [item.evidence_id for item in evidence[:2]],
                "validation": asdict(validation),
                "correction_count": 1,
                "fallback_reason": validation_error or "RAG answer validation failed.",
                "latency_ms": _elapsed_ms(generation_started),
                "attempts": attempts,
            }
        )
        return answer, citations, generation

    def _prepare_evidence(self, retrieved: list[dict[str, Any]]) -> list[RagEvidence]:
        evidence: list[RagEvidence] = []
        seen_content: set[str] = set()
        remaining_chars = self._generation_context_chars
        for row in retrieved:
            content = str(row.get("content") or "").strip()
            normalized = re.sub(r"\s+", " ", content)
            if not normalized or normalized in seen_content or remaining_chars <= 0:
                continue
            seen_content.add(normalized)
            bounded_content = content[:remaining_chars]
            evidence.append(
                RagEvidence(
                    evidence_id=f"C{len(evidence) + 1}",
                    source=str(row["source"]),
                    chunk_id=int(row["id"]),
                    content=bounded_content,
                    score=float(row.get("score") or 0.0),
                )
            )
            remaining_chars -= len(bounded_content)
            if len(evidence) >= self._generation_max_chunks:
                break
        return evidence

    def _base_generation_metadata(
        self,
        evidence: list[RagEvidence],
        retrieval_latency_ms: float,
    ) -> dict[str, Any]:
        return {
            "mode": "retrieval_only",
            "provider": (
                self._answer_generator.provider_name if self._answer_generator is not None else None
            ),
            "prompt_version": RAG_ANSWER_PROMPT_VERSION,
            "evidence_ids": [item.evidence_id for item in evidence],
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "source": item.source,
                    "chunk_id": item.chunk_id,
                    "score": item.score,
                }
                for item in evidence
            ],
            "citation_ids": [],
            "validation": {"valid": True, "citation_ids": [], "errors": []},
            "correction_count": 0,
            "fallback_reason": None,
            "retrieval_latency_ms": retrieval_latency_ms,
            "latency_ms": 0.0,
            "attempts": [],
        }

    @staticmethod
    def _render_generated_answer(
        generated: GeneratedRagAnswer,
        evidence: list[RagEvidence],
    ) -> tuple[str, list[RagCitation]]:
        by_id = {item.evidence_id: item for item in evidence}
        ordered = [by_id[item] for item in generated.citation_ids]
        display_numbers = {
            item.evidence_id: index for index, item in enumerate(ordered, start=1)
        }
        answer = re.sub(
            r"\[(C[1-9]\d*)\]",
            lambda match: f"[{display_numbers[match.group(1)]}]",
            generated.answer,
        )
        source_text = "\n".join(
            f"[{index}] {item.source}#chunk-{item.chunk_id}"
            for index, item in enumerate(ordered, start=1)
        )
        citations = [
            RagCitation(
                source=item.source,
                chunk_id=item.chunk_id,
                score=item.score,
                evidence_id=item.evidence_id,
            )
            for item in ordered
        ]
        return f"{answer}\n\n引用：\n{source_text}", citations

    @staticmethod
    def _fallback_answer(
        evidence: list[RagEvidence],
    ) -> tuple[str, list[RagCitation]]:
        selected = evidence[:2]
        snippets = [re.sub(r"\s+", " ", item.content)[:180] for item in selected]
        source_text = "\n".join(
            f"[{index}] {item.source}#chunk-{item.chunk_id}"
            for index, item in enumerate(selected, start=1)
        )
        answer = "根据知识库检索结果，" + " ".join(snippets)
        if source_text:
            answer += f"\n\n引用：\n{source_text}"
        citations = [
            RagCitation(
                source=item.source,
                chunk_id=item.chunk_id,
                score=item.score,
                evidence_id=item.evidence_id,
            )
            for item in selected
        ]
        return answer, citations

    def search(
        self,
        question: str,
        top_k: int = 5,
        *,
        deadline: float | None = None,
        allow_rerank: bool = True,
    ) -> list[dict[str, Any]]:
        chunks, _ = self._search_with_trace(
            question,
            top_k=top_k,
            deadline=deadline,
            allow_rerank=allow_rerank,
        )
        return chunks

    def _search_with_trace(
        self,
        question: str,
        top_k: int = 5,
        *,
        deadline: float | None = None,
        allow_rerank: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        options = {"deadline": deadline} if deadline is not None else {}
        query_tokens = tokenize(question)
        embeddings = self._embedding_provider.embed_texts([question], **options)
        query_embedding = embeddings[0] if embeddings else []
        vector_results = self._repository.search_by_vector(query_embedding, top_k=RECALL_POOL_SIZE, **options)
        bm25_results, corpus_fingerprint = self._bm25_index.search(query_tokens, RECALL_POOL_SIZE, deadline=deadline)
        ranked = reciprocal_rank_fusion(vector_results, bm25_results)
        if self._rerank_enabled and allow_rerank:
            ranked = self._rerank(question, ranked[:10]) + ranked[10:]
        chunks = [strip_embedding(row) for row in ranked[:top_k]]
        trace = {
            "query": question,
            "mode": "bm25_hybrid",
            "corpus_fingerprint": corpus_fingerprint,
            "vector": _trace_ranking(vector_results, "vector"),
            "bm25": _trace_ranking(bm25_results, "bm25"),
            "rrf": _trace_ranking(ranked[:RECALL_POOL_SIZE], "rrf"),
        }
        return chunks, trace

    def search_knowledge(
        self,
        query: str,
        top_k: int = 3,
        *,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        chunks, retrieval = self._search_with_trace(
            query,
            top_k=top_k,
            deadline=deadline,
            allow_rerank=False,
        )
        titles = self._repository.document_titles(
            [row["document_id"] for row in chunks],
            deadline=deadline,
        )
        return {
            "chunks": [
                {"id": row["id"], "document_id": row["document_id"],
                 "chunk_index": row["chunk_index"], "title": titles.get(row["document_id"], row["source"]),
                 "source": row["source"], "content": excerpt,
                 "metadata": row.get("metadata", {}),
                 "start_offset": start, "end_offset": end, "chunk_chars": len(row["content"]),
                 "vector_rank": row.get("vector_rank"), "bm25_rank": row.get("bm25_rank"),
                 "rrf_rank": row.get("rrf_rank"), "rrf_score": row.get("rrf_score")}
                for row in chunks
                for excerpt, start, end in [_query_excerpt(row["content"], tokenize(query))]
            ],
            "retrieval": retrieval,
        }

    def read_document(self, document_id: int, start_chunk: int = 0, chunk_count: int = 3) -> dict[str, Any]:
        return self._repository.read_document_chunks(document_id, start_chunk, chunk_count)

    def read_document_page(self, document_id: int, start_chunk: int = 0, start_offset: int = 0,
                           max_chars: int = 12000, *, deadline: float | None = None) -> dict[str, Any]:
        return self._repository.read_document_page(document_id, start_chunk, start_offset,
                                                  max_chars, deadline=deadline)

    def _rerank(self, question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates or self._rerank_provider is None:
            return candidates
        candidate_payload = [
            {
                "chunk_id": candidate["id"],
                "source": candidate["source"],
                "content": candidate["content"][:700],
                "rrf_score": candidate.get("rrf_score"),
            }
            for candidate in candidates
        ]
        try:
            payload = self._rerank_provider.generate_json(
                [
                    ChatMessage(
                        role="system",
                        content=(
                            "Rerank retrieved RAG chunks for answering the question. "
                            "Return JSON with key ordered_chunk_ids as a list of chunk_id integers."
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=f"Question: {question}\nCandidates: {candidate_payload}",
                    ),
                ],
                schema_name="rag_rerank",
            )
            ordered_ids = payload.get("ordered_chunk_ids") or payload.get("chunk_ids") or []
            ordered_ids = [int(chunk_id) for chunk_id in ordered_ids]
        except Exception:
            return candidates

        by_id = {candidate["id"]: candidate for candidate in candidates}
        reranked: list[dict[str, Any]] = []
        for rank, chunk_id in enumerate(ordered_ids, start=1):
            candidate = by_id.get(chunk_id)
            if candidate is not None and candidate not in reranked:
                updated = {**candidate, "rerank_rank": rank}
                reranked.append(updated)
        reranked.extend(candidate for candidate in candidates if candidate["id"] not in ordered_ids)
        return reranked


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    ascii_tokens = re.findall(r"[a-z0-9_]+", lowered)
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", lowered)
    short_cjk: list[str] = []
    for token in cjk_tokens:
        short_cjk.extend(token[index : index + 2] for index in range(max(len(token) - 1, 1)))
    return ascii_tokens + short_cjk


def _query_excerpt(content: str, query_tokens: list[str], max_chars: int = 200) -> tuple[str, int, int]:
    if len(content) <= max_chars:
        return content, 0, len(content)
    matches: dict[str, list[tuple[int, int]]] = {}
    for token in dict.fromkeys(query_tokens):
        pattern = (rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])"
                   if token.isascii() else re.escape(token))
        matches[token] = [(m.start(), m.end()) for m in re.finditer(pattern, content, re.IGNORECASE | re.ASCII)]
    positions = [start for spans in matches.values() for start, _ in spans]
    if not positions:
        return content[:max_chars], 0, max_chars
    best_start, best_count = 0, -1
    for position in positions:
        start = min(max(0, position - 60), len(content) - max_chars)
        end = start + max_chars
        count = sum(any(a >= start and b <= end for a, b in spans) for spans in matches.values())
        if count > best_count or (count == best_count and start < best_start):
            best_start, best_count = start, count
    return content[best_start:best_start + max_chars], best_start, best_start + max_chars


def reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]],
    lexical_results: list[dict[str, Any]],
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    lexical_rank_key = "bm25_rank"
    fused: dict[int, dict[str, Any]] = {}
    for source_name, results, rank_key in [
        ("vector", vector_results, "vector_rank"),
        ("bm25", lexical_results, lexical_rank_key),
    ]:
        for rank, row in enumerate(results, start=1):
            chunk_id = int(row["id"])
            existing = fused.setdefault(
                chunk_id,
                {
                    **row,
                    "vector_rank": None,
                    lexical_rank_key: None,
                    "retrieval_sources": [],
                    "rrf_score": 0.0,
                },
            )
            existing[rank_key] = row.get(rank_key) or rank
            for score_key in ("vector_distance", "vector_score", "bm25_score"):
                if score_key in row:
                    existing[score_key] = row[score_key]
            existing["retrieval_sources"].append(source_name)
            existing["rrf_score"] += 1.0 / (k + rank)

    ranked = sorted(
        fused.values(),
        key=lambda row: (
            row["rrf_score"],
            -(row.get("vector_rank") or 9999),
            -(row.get(lexical_rank_key) or 9999),
        ),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["rrf_rank"] = rank
        row["score"] = row["rrf_score"]
    return ranked


def _trace_ranking(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    rank_key = f"{name}_rank"
    score_key = "rrf_score" if name == "rrf" else f"{name}_score"
    return [
        {
            "chunk_id": int(row["id"]),
            "document_id": int(row["document_id"]),
            "source": row["source"],
            "rank": row.get(rank_key) or index,
            "score": float(row.get(score_key) or 0.0),
        }
        for index, row in enumerate(rows, start=1)
    ]


def strip_embedding(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "embedding"}


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
