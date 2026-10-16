from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from app.inbox.models import CandidateInput


AIHOT_ITEMS_URL = "https://aihot.news/api/v1/items"


class InvalidCursorError(ValueError):
    pass


class _Source(BaseModel):
    name: str


class _Links(BaseModel):
    original: str
    aihot: str


class _Item(BaseModel):
    id: str
    title: str
    originalTitle: str | None
    summary: str
    source: _Source
    links: _Links
    publishedAt: datetime | None
    discoveredAt: datetime
    category: Literal["tip", "paper"]
    score: float | None
    selected: Literal[True]
    reason: str | None


class _PageMetadata(BaseModel):
    count: int = Field(ge=0, strict=True)
    hasMore: bool = Field(strict=True)
    nextCursor: str | None = None


class _Payload(BaseModel):
    schemaVersion: Literal[1]
    items: list[dict[str, Any]]
    page: _PageMetadata


@dataclass
class AihotPage:
    items: list[CandidateInput]
    has_more: bool
    next_cursor: str | None


def parse_page(payload: dict[str, Any]) -> AihotPage:
    parsed = _Payload.model_validate(payload)
    if parsed.page.count != len(parsed.items):
        raise ValueError("AIHOT page.count does not match items.")
    candidates = []
    for raw in parsed.items:
        item = _Item.model_validate(raw)
        candidates.append(CandidateInput(
            external_id=item.id, url=item.links.original, aihot_url=item.links.aihot,
            title=item.title, original_title=item.originalTitle, summary=item.summary,
            source_name=item.source.name, category=item.category,
            published_at=item.publishedAt, discovered_at=item.discoveredAt,
            aihot_score=item.score, aihot_reason=item.reason, raw=raw,
        ))
    return AihotPage(candidates, parsed.page.hasMore, parsed.page.nextCursor)


def retry_after_seconds(value: str | None) -> float:
    if not value:
        raise ValueError("AIHOT rate-limit response has no Retry-After header.")
    if value.strip().isdigit():
        return float(value)
    try:
        moment = parsedate_to_datetime(value)
        if moment.tzinfo is None:
            raise ValueError("Retry-After date must include a timezone.")
        return max(0.0, (moment - datetime.now(timezone.utc)).total_seconds())
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("AIHOT returned an invalid Retry-After header.") from exc


class AihotClient:
    def __init__(
        self, *, transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = httpx.Client(
            transport=transport, timeout=15.0,
            headers={"User-Agent": "ai_agent-inbox/0.1"},
        )
        self._sleep = sleep
        self.requests = 0
        self.retries = 0

    def __enter__(self) -> AihotClient:
        return self

    def __exit__(self, *_exc) -> None:
        self._client.close()

    def _get_page(self, params: dict[str, str | int]) -> AihotPage:
        for attempt in range(3):
            self.requests += 1
            response = self._client.get(AIHOT_ITEMS_URL, params=params)
            if response.status_code in {429, 503} and attempt < 2:
                delay = retry_after_seconds(response.headers.get("Retry-After"))
                self.retries += 1
                self._sleep(delay)
                continue
            if response.status_code == 400:
                problem = response.json()
                if isinstance(problem, dict) and problem.get("code") == "invalid_cursor":
                    raise InvalidCursorError("AIHOT rejected the query cursor.")
            response.raise_for_status()
            return parse_page(response.json())
        raise RuntimeError("Unreachable retry state.")

    def iter_pages(
        self, category: str, *, window: str = "24h", max_pages: int = 3,
    ) -> Iterator[AihotPage]:
        if category not in {"tip", "paper"} or window not in {"24h", "7d"} or max_pages < 1:
            raise ValueError("Invalid AIHOT category, window or max_pages.")
        # Match the public API's default limit; no additional CLI option in V1.
        params: dict[str, str | int] = dict(mode="selected", category=category, window=window, limit=50)
        seen_cursors = set()
        for index in range(max_pages):
            page = self._get_page(params)
            if any(candidate.category != category for candidate in page.items):
                raise ValueError("AIHOT returned a category outside the requested category.")
            yield page
            if not page.has_more or index + 1 == max_pages:
                return
            if not page.next_cursor or page.next_cursor in seen_cursors:
                raise ValueError("AIHOT pagination has a missing or repeated cursor.")
            seen_cursors.add(page.next_cursor)
            params["cursor"] = page.next_cursor
