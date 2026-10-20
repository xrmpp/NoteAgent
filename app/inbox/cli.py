from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field

import httpx
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import get_settings
from app.inbox.aihot import AihotClient, InvalidCursorError
from app.inbox.store import InboxStore  # Imports inbox ORM before create_session_factory.
from app.inbox.triage import format_results, triage_candidates
from app.inbox.urlnorm import content_hash, normalize_url
from app.llm.base import ProviderConfigurationError
from app.llm.factory import create_provider
from app.storage.database import create_session_factory


@dataclass
class FetchReport:
    dry_run: bool
    fetched: int = 0
    added: int = 0
    would_add: int = 0
    skipped: int = 0
    skipped_by_reason: dict[str, int] = field(default_factory=lambda: {
        "external_id": 0, "url_normalized": 0, "content_hash": 0,
    })
    pages: int = 0
    requests: int = 0
    retries: int = 0
    stopped_categories: list[str] = field(default_factory=list)


def fetch_candidates(
    store: InboxStore, client: AihotClient, *, categories: list[str],
    window: str = "24h", max_pages: int = 3, dry_run: bool = False,
) -> FetchReport:
    report = FetchReport(dry_run=dry_run)
    seen: dict[str, set[str]] = {key: set() for key in report.skipped_by_reason}
    for category in dict.fromkeys(categories):
        try:
            for page in client.iter_pages(category, window=window, max_pages=max_pages):
                report.pages += 1
                for candidate in page.items:
                    report.fetched += 1
                    if dry_run:
                        keys = dict(
                            external_id=candidate.external_id,
                            url_normalized=normalize_url(candidate.url),
                            content_hash=content_hash(candidate.title, candidate.summary),
                        )
                        existing = set(store.duplicate_reasons(candidate))
                        reasons = [key for key, value in keys.items()
                                   if key in existing or value in seen[key]]
                        if not reasons:
                            report.would_add += 1
                            for key, value in keys.items():
                                seen[key].add(value)
                    else:
                        _, reasons = store.add_candidate(candidate)
                        if not reasons:
                            report.added += 1
                    if reasons:
                        report.skipped += 1
                        for reason in reasons:
                            report.skipped_by_reason[reason] += 1
        except InvalidCursorError:
            # Already yielded pages are kept; a different category starts without a cursor.
            report.stopped_categories.append(category)
    report.requests = client.requests
    report.retries = client.retries
    return report


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Personal knowledge inbox")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Fetch AIHOT selected tip/paper candidates")
    fetch.add_argument("--category", action="append", choices=["tip", "paper"])
    fetch.add_argument("--window", choices=["24h", "7d"], default="24h")
    fetch.add_argument("--max-pages", type=positive_int, default=3)
    fetch.add_argument("--dry-run", action="store_true")
    triage = commands.add_parser("triage", help="Triage new candidates or append a re-triage")
    triage.add_argument("--limit", type=positive_int, default=20)
    triage.add_argument("--candidate-id", type=positive_int)
    triage.add_argument("--retriage", action="store_true")
    triage.add_argument("--exclude-aihot-reason", action="store_true",
                        help="Exclude AIHOT recommendation reason from model material")
    report_parser = commands.add_parser("report", help="Print recent triage results and decisions")
    report_parser.add_argument("--limit", type=positive_int, default=20)
    args = parser.parse_args(argv)

    engine = None
    try:
        settings = get_settings()
        factory = create_session_factory(
            settings.database_url, embedding_dimensions=settings.embedding_dimensions,
        )
        engine = factory.kw["bind"]
        store = InboxStore(factory)
        if args.command == "report":
            print(format_results(store.recent_triage_results(limit=args.limit)))
            return 0
        if args.command == "triage":
            batch = triage_candidates(
                store, create_provider(settings), model=settings.model_name, limit=args.limit,
                candidate_id=args.candidate_id, retriage=args.retriage,
                exclude_aihot_reason=args.exclude_aihot_reason,
            )
            print(json.dumps({key: value for key, value in vars(batch).items() if key != "results"},
                             ensure_ascii=False))
            print(format_results(batch.results))
            return 1 if batch.failed or batch.quota_reached else 0
        with AihotClient() as client:
            report = fetch_candidates(
                store, client, categories=args.category or ["tip", "paper"],
                window=args.window, max_pages=args.max_pages, dry_run=args.dry_run,
            )
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
        if report.stopped_categories:
            print("invalid_cursor：已停止相应分类翻页，保留已取数据。", file=sys.stderr)
        return 0
    except httpx.HTTPStatusError as exc:
        print(f"{args.command} failed: AIHOT HTTP {exc.response.status_code}", file=sys.stderr)
        return 1
    except httpx.HTTPError:
        print(f"{args.command} failed: AIHOT connection or timeout error", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"{args.command} failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, UnicodeError):
        print(f"{args.command} failed: prompt/profile file is missing or unreadable", file=sys.stderr)
        return 1
    except ProviderConfigurationError:
        print(f"{args.command} failed: model provider is not configured", file=sys.stderr)
        return 1
    except SQLAlchemyError:
        print(f"{args.command} failed: database operation error", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
