import copy
import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from app.inbox.aihot import AihotClient, parse_page, retry_after_seconds
from app.inbox.cli import fetch_candidates, main
from app.inbox.store import InboxStore
from app.storage.database import create_session_factory


FIXTURES = Path(__file__).parent / "fixtures"


def payload(category="tip"):
    return json.loads((FIXTURES / f"aihot_{category}_24h.json").read_text(encoding="utf-8"))


@pytest.fixture
def store(tmp_path):
    return InboxStore(create_session_factory(f"sqlite+pysqlite:///{tmp_path / 'fetch.db'}"))


@pytest.mark.parametrize(("category", "count"), [("tip", 3), ("paper", 2)])
def test_parses_real_sanitized_fixtures(category, count):
    raw = payload(category)
    assert "cursor" not in json.dumps(raw).lower()
    page = parse_page(raw)
    assert len(page.items) == count
    assert page.items[0].external_id == raw["items"][0]["id"]
    assert page.items[0].source_name == raw["items"][0]["source"]["name"]
    assert page.items[0].url == raw["items"][0]["links"]["original"]
    assert page.items[0].raw == raw["items"][0]
    assert page.items[0].discovered_at.utcoffset() == timedelta(0)


def test_request_headers_params_timeout_pagination_and_max_pages():
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["User-Agent"] == "ai_agent-inbox/0.1"
        assert all(value == 15.0 for value in request.extensions["timeout"].values())
        assert request.url.params["mode"] == "selected"
        assert request.url.params["category"] == "tip"
        assert request.url.params["window"] == "7d"
        assert request.url.params["limit"] == "50"
        raw = payload()
        raw["page"]["nextCursor"] = f"cursor-{len(requests)}"
        return httpx.Response(200, json=raw)

    with AihotClient(transport=httpx.MockTransport(handler)) as client:
        assert len(list(client.iter_pages("tip", window="7d", max_pages=2))) == 2
        assert client.requests == 2
    assert "cursor" not in requests[0].url.params
    assert requests[1].url.params["cursor"] == "cursor-1"


@pytest.mark.parametrize("status", [429, 503])
def test_retry_after_then_success(status):
    calls, waits = [], []

    def handler(request):
        calls.append(request)
        if len(calls) <= 2:
            return httpx.Response(status, headers={"Retry-After": "2"})
        return httpx.Response(200, json=payload("paper"))

    with AihotClient(transport=httpx.MockTransport(handler), sleep=waits.append) as client:
        assert len(list(client.iter_pages("paper"))) == 1
        assert client.retries == 2
        assert client.requests == 3
    assert waits == [2.0, 2.0]


@pytest.mark.parametrize("status", [429, 503])
def test_retry_limit_is_two(status):
    waits = []
    transport = httpx.MockTransport(lambda request: httpx.Response(status, headers={"Retry-After": "1"}))
    with AihotClient(transport=transport, sleep=waits.append) as client:
        with pytest.raises(httpx.HTTPStatusError):
            list(client.iter_pages("tip"))
        assert client.requests == 3
        assert client.retries == 2
    assert waits == [1.0, 1.0]


def test_retry_after_http_date_and_invalid_header():
    moment = datetime.now(timezone.utc) + timedelta(seconds=30)
    assert 28 <= retry_after_seconds(format_datetime(moment, usegmt=True)) <= 30
    for value in (None, "invalid", "-1"):
        with pytest.raises(ValueError):
            retry_after_seconds(value)


@pytest.mark.parametrize("status", [401, 404, 500])
def test_other_http_errors_do_not_retry(status):
    waits = []
    with AihotClient(transport=httpx.MockTransport(lambda request: httpx.Response(status)), sleep=waits.append) as client:
        with pytest.raises(httpx.HTTPStatusError):
            list(client.iter_pages("tip"))
        assert client.requests == 1
    assert waits == []


def test_timeout_does_not_retry():
    def handler(request):
        raise httpx.ReadTimeout("timeout", request=request)

    with AihotClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.ReadTimeout):
            list(client.iter_pages("tip"))
        assert client.requests == 1


def test_invalid_cursor_keeps_first_page_and_resets_next_category(store):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.params["category"] == "paper":
            assert "cursor" not in request.url.params
            return httpx.Response(200, json=payload("paper"))
        if "cursor" in request.url.params:
            return httpx.Response(400, json={"status": 400, "code": "invalid_cursor", "detail": "Invalid cursor"})
        raw = payload()
        raw["page"]["nextCursor"] = "query-bound-cursor"
        return httpx.Response(200, json=raw)

    with AihotClient(transport=httpx.MockTransport(handler)) as client:
        report = fetch_candidates(store, client, categories=["tip", "paper"])
    assert report.added == 5
    assert report.pages == 2
    assert report.requests == 3
    assert report.stopped_categories == ["tip"]
    assert len(store.list_candidates()) == 5


def test_dry_run_and_real_fetch_have_same_dedup_counts(store):
    raw = payload()
    base = raw["items"][0]
    store.add_candidate(parse_page(raw).items[0])
    raw["items"].append(copy.deepcopy(base))
    raw["page"] = dict(count=4, hasMore=False)

    def run(dry_run):
        with AihotClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=raw))) as client:
            return fetch_candidates(store, client, categories=["tip"], max_pages=1, dry_run=dry_run)

    dry = run(True)
    assert dry.added == 0 and dry.would_add == 2 and dry.skipped == 2
    assert len(store.list_candidates()) == 1
    assert len(store.list_status_events(1)) == 1
    real = run(False)
    assert real.added == 2 and real.skipped == 2
    assert real.skipped_by_reason == dry.skipped_by_reason == {
        "external_id": 2, "url_normalized": 2, "content_hash": 2,
    }
    again = run(False)
    assert again.added == 0 and again.skipped == 4
    assert all(value == 4 for value in again.skipped_by_reason.values())
    assert len(store.list_candidates()) == 3


def test_dry_run_deduplicates_new_items_across_pages(store):
    requests = []

    def handler(request):
        requests.append(request)
        raw = payload()
        raw["page"]["nextCursor"] = f"cursor-{len(requests)}"
        return httpx.Response(200, json=raw)

    with AihotClient(transport=httpx.MockTransport(handler)) as client:
        report = fetch_candidates(store, client, categories=["tip"], max_pages=2, dry_run=True)
    assert report.would_add == 3 and report.skipped == 3
    assert report.skipped_by_reason == {"external_id": 3, "url_normalized": 3, "content_hash": 3}
    assert store.list_candidates() == []


@pytest.mark.parametrize("change", ["category", "timestamp", "count", "selected"])
def test_malformed_page_fails_without_writes(store, change):
    raw = payload()
    if change == "category":
        raw["items"][0]["category"] = "industry"
    elif change == "timestamp":
        raw["items"][0]["discoveredAt"] = "2026-10-02T00:00:00"
    elif change == "count":
        raw["page"]["count"] = 999
    else:
        raw["items"][0]["selected"] = False
    with AihotClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=raw))) as client:
        with pytest.raises((ValidationError, ValueError)):
            fetch_candidates(store, client, categories=["tip"])
    assert store.list_candidates() == []


def test_empty_page_and_cli_help_do_not_write_or_call_models(store, capsys):
    raw = payload()
    raw["items"] = []
    raw["page"] = dict(count=0, hasMore=False)
    with AihotClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=raw))) as client:
        report = fetch_candidates(store, client, categories=["tip"])
    assert report.fetched == report.added == report.skipped == 0
    with pytest.raises(SystemExit) as exc:
        main(["fetch", "--help"])
    assert exc.value.code == 0
    assert "--dry-run" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["fetch", "--max-pages", "0"])
    assert exc.value.code == 2
