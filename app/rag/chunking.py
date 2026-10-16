"""Structure-aware chunks with source text kept separate from retrieval context."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from markdown_it import MarkdownIt


@dataclass(frozen=True, slots=True)
class TextChunk:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def retrieval_text(self, title: str) -> str:
        context = [title, " > ".join(self.metadata.get("heading_path", []))]
        context.extend(str(self.metadata[key]) for key in ("language", "table_header")
                       if self.metadata.get(key))
        return "\n".join([part for part in context if part] + [self.content])


@dataclass(frozen=True, slots=True)
class _Block:
    content: str
    metadata: dict[str, Any]


class Chunker:
    VERSION = "markdown-structure-v1"

    def __init__(self, max_chars: int = 600, overlap: int = 80,
                 hard_max_chars: int = 2000) -> None:
        if not 0 <= overlap < max_chars <= hard_max_chars:
            raise ValueError("Require 0 <= overlap < max_chars <= hard_max_chars.")
        self._max_chars = max_chars
        self._overlap = overlap
        self._hard_max_chars = hard_max_chars

    @property
    def index_config(self) -> dict[str, Any]:
        return {"version": self.VERSION, "target_chars": self._max_chars,
                "hard_max_chars": self._hard_max_chars, "overlap": self._overlap}

    def split(self, text: str) -> list[str]:
        return [chunk.content for chunk in self.split_with_metadata(text)]

    def split_with_metadata(self, text: str, *, markdown: bool = True) -> list[TextChunk]:
        blocks = self._markdown_blocks(text) if markdown else self._plain_blocks(text)
        chunks: list[TextChunk] = []
        pending = ""
        pending_metadata: dict[str, Any] = {}

        def flush() -> None:
            nonlocal pending
            if pending.strip():
                chunks.append(TextChunk(pending, dict(pending_metadata)))
            pending = ""

        for block in blocks:
            structured = block.metadata["block_type"] in {"code", "table", "list"}
            if (structured or len(block.content) > self._hard_max_chars or
                    (pending and (pending_metadata != block.metadata or
                                  len(pending) + len(block.content) > self._max_chars))):
                flush()
            if len(block.content) > self._hard_max_chars:
                chunks.extend(self._split_long(block))
            elif structured:
                chunks.append(TextChunk(block.content, block.metadata))
            else:
                pending_metadata = block.metadata
                pending += block.content
        flush()
        return chunks

    def _markdown_blocks(self, text: str) -> list[_Block]:
        lines = text.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        tokens = MarkdownIt("commonmark", {"html": False}).enable("table").parse(text)
        headings: list[tuple[int, str]] = []
        blocks: list[_Block] = []
        cursor = 0
        for index, token in enumerate(tokens):
            if token.level != 0 or token.map is None or token.nesting == -1:
                continue
            if token.type == "heading_open":
                level = int(token.tag[1:])
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, tokens[index + 1].content))
                continue
            start, end = (offsets[line] for line in token.map)
            if end <= cursor:
                continue
            kind = ("code" if token.type in {"fence", "code_block"} else
                    "table" if token.type == "table_open" else
                    "list" if token.type in {"bullet_list_open", "ordered_list_open"} else "text")
            metadata: dict[str, Any] = {"heading_path": [title[:300] for _, title in headings],
                                        "block_type": kind}
            if token.type == "fence" and token.info.strip():
                metadata["language"] = token.info.strip().split()[0][:80]
            if kind == "table":
                metadata["table_header"] = text[start:offsets[min(token.map[0] + 2, token.map[1])]][:1000]
            # Include headings, whitespace and reference definitions omitted by the token stream.
            blocks.append(_Block(text[cursor:end], metadata))
            cursor = end
        if cursor < len(text):
            tail = text[cursor:]
            if blocks:
                last = blocks[-1]
                blocks[-1] = _Block(last.content + tail, last.metadata)
            elif tail.strip():
                blocks.append(_Block(tail, {"heading_path": [t[:300] for _, t in headings],
                                            "block_type": "text"}))
        return blocks

    @staticmethod
    def _plain_blocks(text: str) -> list[_Block]:
        parts = re.split(r"(\n[ \t]*\n)", text)
        blocks = []
        for index in range(0, len(parts), 2):
            content = parts[index] + (parts[index + 1] if index + 1 < len(parts) else "")
            if content.strip():
                blocks.append(_Block(content, {"heading_path": [], "block_type": "text"}))
        return blocks

    def _split_long(self, block: _Block) -> list[TextChunk]:
        metadata = {**block.metadata, "fragment": True}
        if metadata["block_type"] == "text":
            return [TextChunk(part, metadata) for part in self._split_paragraph(block.content) if part.strip()]
        if metadata["block_type"] == "list":
            starts = [match.start() for match in re.finditer(
                r"(?m)^ {0,3}(?:[-+*]|[0-9]+[.)])[ \t]+", block.content)]
            boundaries = sorted({0, *starts, len(block.content)})
            units = [block.content[a:b] for a, b in zip(boundaries, boundaries[1:])]
        else:
            units = block.content.splitlines(keepends=True)
        pieces: list[str] = []
        pending = ""
        for unit in units:
            if len(unit) > self._hard_max_chars:
                if pending:
                    pieces.append(pending)
                    pending = ""
                pieces.extend(self._split_lines(unit))
                continue
            if pending and len(pending) + len(unit) > self._max_chars:
                pieces.append(pending)
                pending = ""
            pending += unit
        if pending:
            pieces.append(pending)
        return [TextChunk(part, metadata) for part in pieces if part.strip()]

    def _split_lines(self, content: str) -> list[str]:
        pieces: list[str] = []
        pending = ""
        for line in content.splitlines(keepends=True):
            if pending and len(pending) + len(line) > self._hard_max_chars:
                pieces.append(pending)
                pending = ""
            while len(line) > self._hard_max_chars:
                pieces.append(line[:self._hard_max_chars])
                line = line[self._hard_max_chars:]
            pending += line
        if pending:
            pieces.append(pending)
        return pieces

    def _split_paragraph(self, content: str) -> list[str]:
        pieces = []
        start = 0
        while start < len(content):
            end = min(start + self._max_chars, len(content))
            if end < len(content):
                window = content[start:end]
                for pattern in (r"(?:[.!?;](?=\s|$)|[\u3002\uff01\uff1f\uff1b])\s*",
                                r"\n", r"[ \t]+"):
                    candidates = list(re.finditer(pattern, window))
                    if candidates and candidates[-1].end() >= self._max_chars // 2:
                        end = start + candidates[-1].end()
                        break
            pieces.append(content[start:end])
            if end == len(content):
                break
            start = max(start + 1, end - self._overlap)
        return pieces
