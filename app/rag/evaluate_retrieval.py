from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.agent.rag import RagService
from app.core.config import get_settings
import time

def evaluate_rag_recall(
    rag_service: RagService,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    recall_at_3 = 0
    recall_at_5 = 0
    reciprocal_rank_total = 0.0
    details: list[dict[str, Any]] = []
    latencies: list[float] = []
    started_total = time.perf_counter()
    for case in cases:
        started = time.perf_counter()
        results = rag_service.search(case["question"], top_k=5)
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        retrieved_sources = [row["source"] for row in results]
        hit_rank = next(
            (
                index
                for index, row in enumerate(results, start=1)
                if row["source"] == case["expected_source"]
            ),
            None,
        )
        if hit_rank is not None and hit_rank <= 3:
            recall_at_3 += 1
        if hit_rank is not None:
            recall_at_5 += 1
            reciprocal_rank_total += 1.0 / hit_rank
        details.append(
            {
                **case,
                "expected": case["expected_source"],
                "actual": retrieved_sources,
                "retrieved_chunks": results,
                "source": "retrieval",
                "passed": hit_rank is not None,
                "hit": hit_rank is not None,
                "rank": hit_rank,
                "latency_ms": latency_ms,
            }
        )
    total_cases = len(cases)
    return {
        "total": total_cases,
        "total_cases": total_cases,
        "recall_at_3": (recall_at_3 / total_cases) if total_cases else 0.0,
        "recall_at_5": (recall_at_5 / total_cases) if total_cases else 0.0,
        "mrr": (reciprocal_rank_total / total_cases) if total_cases else 0.0,
        "mrr_at_5": (reciprocal_rank_total / total_cases) if total_cases else 0.0,
        "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "total_latency_ms": (time.perf_counter() - started_total) * 1000,
        "details": details,
    }
from app.llm.factory import create_embedding_provider
from app.storage.database import create_session_factory
from app.storage.rag_repository import RagRepository


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Evaluation cases must be a non-empty JSON array.")
    for index, case in enumerate(payload, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Case {index} must be a JSON object.")
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"Case {index} requires a non-empty question.")
        if not isinstance(case.get("expected_source"), str) or not case[
            "expected_source"
        ].strip():
            raise ValueError(f"Case {index} requires a non-empty expected_source.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval against a labeled JSON question set."
    )
    parser.add_argument("cases", type=Path, help="JSON file containing labeled cases.")
    parser.add_argument("--output", required=True, type=Path, help="Result JSON path.")
    args = parser.parse_args()

    settings = get_settings()
    cases = load_cases(args.cases)
    session_factory = create_session_factory(
        settings.database_url,
        embedding_dimensions=settings.embedding_dimensions,
    )
    repository = RagRepository(session_factory)
    embedding_provider = create_embedding_provider(settings)
    service = RagService(repository, embedding_provider=embedding_provider)
    result = evaluate_rag_recall(service, cases)
    artifact = {
        "mode": "bm25_hybrid",
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "corpus_fingerprint": repository.corpus_fingerprint(),
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"total={result['total']} recall@3={result['recall_at_3']:.3f} "
        f"recall@5={result['recall_at_5']:.3f} mrr@5={result['mrr_at_5']:.3f}"
    )


if __name__ == "__main__":
    main()
