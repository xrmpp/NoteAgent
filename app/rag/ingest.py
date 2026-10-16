from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agent.rag import RagService

MAX_DOCUMENT_BYTES = 1024 * 1024


def source_files(path: Path, source_prefix: str = "") -> list[tuple[Path, str]]:
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("The import path must be a directory.")
    prefix = source_prefix.strip().strip("/")
    if prefix and any(part in {"", ".", ".."} for part in prefix.split("/")):
        raise ValueError("source_prefix must be a stable relative namespace.")
    files = []
    for file_path in sorted(root.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in {".md", ".txt"}:
            if not file_path.resolve(strict=True).is_relative_to(root):
                raise ValueError("Document symlinks must remain inside the import directory.")
            relative = file_path.relative_to(root).as_posix()
            source = f"{prefix}/{relative}" if prefix else relative
            files.append((file_path, source))
    return files


def read_document_file(path: Path) -> str:
    with path.open("rb") as stream:
        content = stream.read(MAX_DOCUMENT_BYTES + 1)
    if len(content) > MAX_DOCUMENT_BYTES:
        raise ValueError("Document exceeds 1 MB.")
    return content.decode("utf-8-sig")


def ingest_directory(
    service: RagService,
    path: Path,
    *,
    source_prefix: str = "",
) -> list[int]:
    return [service.ingest_text(source, read_document_file(file_path), file_path.stem)
            for file_path, source in source_files(path, source_prefix)]


def main() -> int:
    from app.rag.cli import run_ingest

    parser = argparse.ArgumentParser(description="Ingest Markdown/txt files into the RAG corpus.")
    parser.add_argument("path", type=Path, help="Directory containing Markdown or txt files.")
    parser.add_argument(
        "--source-prefix",
        default="",
        help="Optional stable namespace added before each relative source path.",
    )
    args = parser.parse_args()

    return run_ingest(args.path, source_prefix=args.source_prefix)


if __name__ == "__main__":
    raise SystemExit(main())
