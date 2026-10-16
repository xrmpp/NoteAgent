"""Three knowledge-maintenance operations, independent of the chat UI."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from app.core.config import get_settings
from app.llm.base import ProviderConfigurationError, ProviderExecutionError
from app.rag.ingest import read_document_file, source_files
from app.storage.database import create_session_factory
from app.storage.rag_repository import RagRepository


def _failure(exc: Exception) -> dict[str, str]:
    if isinstance(exc, SQLAlchemyError):
        return {"stage": "database", "error": "Database operation failed; current document was not committed."}
    if isinstance(exc, ProviderConfigurationError):
        return {"stage": "embedding", "error": "Embedding provider configuration is invalid."}
    if isinstance(exc, (ProviderExecutionError, TimeoutError)):
        return {"stage": "embedding", "error": "Embedding provider request failed or timed out."}
    if isinstance(exc, UnicodeError):
        return {"stage": "file", "error": "Document is not valid UTF-8."}
    if isinstance(exc, OSError):
        return {"stage": "file", "error": "File or directory is unavailable."}
    return {"stage": "validation", "error": str(exc)}


def run_ingest(path: Path, *, source_prefix: str = "") -> int:
    from app.agent.rag import RagService
    from app.llm.factory import create_embedding_provider

    engine = None
    counts = {"created": 0, "updated": 0, "skipped": 0, "failed": 0, "embedding_texts": 0}
    try:
        files = source_files(path, source_prefix)
        print(json.dumps({"directory": str(path.resolve()), "files": len(files)}, ensure_ascii=False))
        if not files:
            print(json.dumps(counts))
            return 0
        settings = get_settings()
        factory = create_session_factory(settings.database_url, settings.embedding_dimensions)
        engine = factory.kw["bind"]
        service = RagService(RagRepository(factory), embedding_provider=create_embedding_provider(settings))
        for file_path, source in files:
            try:
                result = service.ingest_document(source, read_document_file(file_path), file_path.stem)
                counts[result.status] += 1
                counts["embedding_texts"] += result.embedding_texts
                print(json.dumps(asdict(result), ensure_ascii=False))
            except (OSError, UnicodeError, ValueError, ProviderConfigurationError,
                    ProviderExecutionError, SQLAlchemyError, TimeoutError) as exc:
                counts["failed"] += 1
                print(json.dumps({"source": source, "status": "failed", **_failure(exc)},
                                 ensure_ascii=False), file=sys.stderr)
        print(json.dumps(counts))
        return 1 if counts["failed"] else 0
    except (OSError, UnicodeError, ValueError, ProviderConfigurationError, SQLAlchemyError) as exc:
        print(json.dumps(_failure(exc), ensure_ascii=False), file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()


def _manage(command: str, source: str | None = None) -> int:
    settings = get_settings()
    factory = create_session_factory(settings.database_url, settings.embedding_dimensions)
    engine = factory.kw["bind"]
    try:
        repository = RagRepository(factory)
        if command == "list":
            documents = repository.list_documents()
            for document in documents:
                print(json.dumps(document, ensure_ascii=False))
            print(json.dumps({"documents": len(documents)}))
            return 0
        if source is None or not source.strip():
            raise ValueError("Deletion requires one exact source.")
        source = source.strip()
        document = repository.find_document(source)
        if document is None:
            print("Source not found.", file=sys.stderr)
            return 1
        print(json.dumps(document, ensure_ascii=False))
        try:
            confirmation = input("Delete this document and its chunks? Type DELETE to confirm: ")
        except EOFError:
            confirmation = ""
        if confirmation != "DELETE":
            print("Cancelled. No knowledge content was deleted.")
            return 1
        removed = repository.delete_document(source)
        print(json.dumps({"source": source, "status": "deleted" if removed else "not_found"},
                         ensure_ascii=False))
        return 0 if removed else 1
    finally:
        engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Maintain knowledge documents without changing chat workflows.")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Recursively add/update Markdown and txt files; skip unchanged sources.")
    ingest.add_argument("path", type=Path)
    commands.add_parser("list", help="List stored sources, hashes and chunk counts.")
    delete = commands.add_parser("delete", help="Preview and confirm deletion of one exact source.")
    delete.add_argument("source")
    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            return run_ingest(args.path)
        return _manage(args.command, getattr(args, "source", None))
    except (SQLAlchemyError, ValueError, OSError) as exc:
        print(json.dumps(_failure(exc), ensure_ascii=False), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
