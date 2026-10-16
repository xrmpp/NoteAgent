"""Run-local evidence ledger: stable citations, preview upgrades and text budget."""
from __future__ import annotations

import json
from typing import Any

from app.agent.models import RagEvidence


def model_tool_payload(output: dict[str, Any]) -> dict[str, Any]:
    """Trim a tool/search result down to the fields the model is allowed to see."""
    fields = ("evidence_id", "document_id", "chunk_id", "chunk_index", "title", "source",
              "content", "start_offset", "end_offset", "chunk_chars", "preview", "metadata")
    navigation = ("already_seen", "next_cursor", "total_chunks", "truncated", "error", "executed")
    return {"chunks": [{key: row[key] for key in fields if key in row} for row in output.get("chunks", [])],
            **{key: output[key] for key in navigation if key in output}}


class EvidenceStore:
    """Assign stable citations and transmit only new or expanded text."""

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars
        self.items: dict[tuple[int, int], RagEvidence] = {}
        self.discovered: set[int] = set()
        self.ranges: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self.chunk_lengths: dict[tuple[int, int], int] = {}
        self.documents: dict[int, dict[str, Any]] = {}
        self.chars = 0
        self.exhausted = False

    def present(self, output: dict[str, Any], reading: bool) -> dict[str, Any]:
        result = {key: value for key, value in output.items() if key != "chunks"}
        result.update(chunks=[], already_seen=[])
        for chunk in output.get("chunks", []):
            doc_id, chunk_id = int(chunk["document_id"]), int(chunk["id"])
            start = int(chunk.get("start_offset", 0))
            content = str(chunk["content"])
            end_offset = int(chunk.get("end_offset", start + len(content)))
            if end_offset - start != len(content):
                raise ValueError("excerpt_offset_mismatch")
            key = (chunk_id, start)
            previous = self.items.get(key)
            containing = next((item for (existing_id, existing_start), item in self.items.items()
                               if existing_id == chunk_id and existing_start <= start
                               and existing_start + len(item.content) >= end_offset
                               and item.content[start - existing_start:end_offset - existing_start] == content), None)
            if containing:
                result["already_seen"].append({"evidence_id": containing.evidence_id,
                    "document_id": doc_id, "chunk_index": chunk["chunk_index"]})
            else:
                allowance = self.max_chars - self.chars + (len(previous.content) if previous else 0)
                if len(content) > allowance:
                    self.exhausted = True
                    result["truncated"] = True
                    if reading:
                        result["next_cursor"] = {"start_chunk": chunk["chunk_index"], "start_offset": start}
                    break  # Do not cite or mark as read text that was not transmitted.
                evidence_id = previous.evidence_id if previous else f"C{len(self.items) + 1}"
                item = {
                    "evidence_id": evidence_id, "document_id": doc_id, "chunk_id": chunk_id,
                    "title": chunk.get("title", chunk["source"]), "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"], "start_offset": start,
                    "end_offset": end_offset, "chunk_chars": int(chunk.get("chunk_chars", end_offset)),
                    "content": content,
                    "preview": not reading,
                    "metadata": chunk.get("metadata", {}),
                }
                if len(json.dumps(model_tool_payload({**result, "chunks": [*result["chunks"], item]}), ensure_ascii=False)) > 23000:
                    result["truncated"] = True
                    if reading:
                        result["next_cursor"] = {"start_chunk": chunk["chunk_index"], "start_offset": start}
                    break
                self.items[key] = RagEvidence(evidence_id, chunk["source"], chunk_id, content,
                                              float(chunk.get("rrf_score", 0)))
                self.chars += len(content) - (len(previous.content) if previous else 0)
                result["chunks"].append(item)
            self.discovered.add(doc_id)
            if reading:
                self.documents[doc_id] = {"source": chunk["source"], "total_chunks": output.get("total_chunks")}
                range_key = (doc_id, int(chunk["chunk_index"]))
                self.ranges.setdefault(range_key, []).append((start, start + len(content)))
                self.chunk_lengths[range_key] = int(chunk.get("chunk_chars", start + len(content)))
        result["new_text_chars"] = sum(len(item["content"]) for item in result["chunks"])
        return result

    def coverage(self) -> list[dict[str, Any]]:
        covered: dict[int, list[int]] = {}
        for (doc_id, chunk_index), ranges in self.ranges.items():
            end = 0
            for start, stop in sorted(ranges):
                if start > end:
                    break
                end = max(end, stop)
            if end >= self.chunk_lengths[(doc_id, chunk_index)]:
                covered.setdefault(doc_id, []).append(chunk_index)
        return [{"document_id": doc_id, **info,
                 "read_chunks": len(covered.get(doc_id, [])),
                 "complete": info["total_chunks"] is not None and
                             len(covered.get(doc_id, [])) == info["total_chunks"]}
                for doc_id, info in self.documents.items()]
