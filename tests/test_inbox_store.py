import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from sqlalchemy import Text, inspect, select

from app.inbox.models import CandidateInput, InboxCandidateORM, InboxStatusEventORM, TriageOutput
from app.inbox.store import InboxStore
from app.inbox.urlnorm import content_hash, normalize_url
from app.storage.database import Base, create_session_factory


@pytest.fixture
def store(tmp_path):
    factory = create_session_factory(f"sqlite+pysqlite:///{tmp_path / 'inbox.db'}")
    return InboxStore(factory)


def candidate(**changes):
    values = dict(
        external_id="item-1", url="https://example.com/one", aihot_url="https://aihot.news/items/one",
        title="Useful tip", original_title=None, summary="Try this technique.",
        source_name="Example", category="tip", published_at=None,
        discovered_at=datetime(2026, 10, 2, 8, tzinfo=timezone(timedelta(hours=8))),
        aihot_score=90, aihot_reason="Useful", raw={"id": "item-1", "selected": True},
    )
    values.update(changes)
    return CandidateInput(**values)


def triage(**changes):
    values = dict(usage="practice", quality=4, relevance=5, reason="Matches my work",
                  try_what="Try a local example", est_minutes=20, prerequisites=["Python"], unknowns=[])
    values.update(changes)
    return TriageOutput(**values)


@pytest.mark.parametrize(("url", "expected"), [
    ("HTTPS://Example.COM/path/?utm_source=a&ref=b&fbclid=c&gclid=d&keep=1#part", "https://example.com/path?keep=1"),
    ("https://twitter.com/user/status/1/", "https://x.com/user/status/1"),
    ("http://EXAMPLE.com:8080/?UTM_medium=x&q=&q=two", "http://example.com:8080?q=&q=two"),
    ("https://example.com/a/b?reference=ok", "https://example.com/a/b?reference=ok"),
    ("http://[::1]:8000/path/", "http://[::1]:8000/path"),
])
def test_url_normalization(url, expected):
    assert normalize_url(url) == expected
    assert normalize_url(expected) == expected


@pytest.mark.parametrize("url", ["/relative", "file:///tmp/a", "https://user:pass@example.com/a"])
def test_url_rejects_invalid_inputs(url):
    with pytest.raises(ValueError):
        normalize_url(url)


def test_hash_normalizes_case_whitespace_and_preserves_field_boundary():
    assert content_hash(" Tip  Title ", "Summary\ntext") == content_hash("tip title", "summary text")
    assert content_hash("a b", "c") != content_hash("a", "b c")


def test_shared_metadata_text_json_and_utc_roundtrip(store):
    tables = set(inspect(store._session_factory.kw["bind"]).get_table_names())
    assert {"inbox_candidates", "inbox_triage_results", "inbox_status_events"} <= tables
    assert InboxCandidateORM.metadata is Base.metadata
    assert isinstance(InboxCandidateORM.__table__.c.raw_json.type, Text)
    row, reasons = store.add_candidate(candidate())
    saved = store.get_candidate(row.id)
    assert reasons == []
    assert saved.discovered_at == datetime(2026, 10, 2, tzinfo=timezone.utc)
    assert saved.created_at.utcoffset() == timedelta(0)
    assert saved.status_updated_at.utcoffset() == timedelta(0)
    assert json.loads(saved.raw_json)["selected"] is True


@pytest.mark.parametrize(("changes", "reason"), [
    ({"url": "https://example.com/two", "title": "Other"}, "external_id"),
    ({"external_id": "item-2", "url": "https://example.com/one/?utm_source=new", "title": "Other"}, "url_normalized"),
    ({"external_id": "item-2", "url": "https://example.com/two", "title": " useful TIP "}, "content_hash"),
])
def test_dedup_each_key(store, changes, reason):
    store.add_candidate(candidate())
    row, reasons = store.add_candidate(candidate(**changes))
    assert row is None
    assert reasons == [reason]
    assert len(store.list_candidates()) == 1


def test_dedup_reports_all_matches(store):
    store.add_candidate(candidate())
    assert store.add_candidate(candidate())[1] == ["external_id", "url_normalized", "content_hash"]


def test_status_events_allow_any_valid_transition_and_ignore_noop(store):
    row, _ = store.add_candidate(candidate())
    store.change_status(row.id, "ingested", note_source="practice/note.md")
    before = store.get_candidate(row.id).status_updated_at
    store.change_status(row.id, "ingested", note_source="practice/note.md")
    assert store.get_candidate(row.id).status_updated_at == before
    store.change_status(row.id, "queued", user_reason="Want to try it")
    assert store.get_candidate(row.id).user_reason == "Want to try it"
    events = store.list_status_events(row.id)
    assert [(e.from_status, e.to_status) for e in events] == [
        (None, "new"), ("new", "ingested"), ("ingested", "queued"),
    ]
    assert all(e.created_at.utcoffset() == timedelta(0) for e in events)
    with pytest.raises(ValueError):
        store.change_status(row.id, "not-a-status")
    with pytest.raises(ValueError):
        store.change_status(9999, "queued")
    assert len(store.list_status_events(row.id)) == 3
    assert store.list_candidates(status="new") == []


def test_triage_is_append_only_and_delete_removes_children(store):
    row, _ = store.add_candidate(candidate())
    first = store.append_triage_result(row.id, prompt_version="triage.md@12345678", model="fake",
                                       output=None, raw_output="invalid", error="validation failed")
    second = store.append_triage_result(row.id, prompt_version="triage.md@abcdef12", model="fake",
                                        output=triage(), raw_output='{"usage":"practice"}')
    results = store.list_triage_results(row.id)
    assert [r.id for r in results] == [first.id, second.id]
    assert results[0].error == "validation failed"
    assert results[1].usage == "practice"
    assert json.loads(results[1].prerequisites_json) == ["Python"]
    assert results[1].similar_json == "[]"
    assert results[1].created_at.utcoffset() == timedelta(0)
    store.delete_candidate(row.id)
    assert store.get_candidate(row.id) is None
    assert store.list_triage_results(row.id) == []
    assert store.list_status_events(row.id) == []


@pytest.mark.parametrize("changes", [
    {"usage": "other"}, {"quality": "4"}, {"quality": True}, {"quality": 0},
    {"relevance": 6}, {"try_what": "  "}, {"est_minutes": "20"},
    {"prerequisites": "Python"}, {"unknowns": [1]}, {"reason": 4},
])
def test_triage_output_strict_validation(changes):
    with pytest.raises(ValidationError):
        triage(**changes)


def test_nonpractice_allows_empty_action_and_null_estimate():
    assert triage(usage="know", try_what="", est_minutes=None).est_minutes is None
    with pytest.raises(ValidationError):
        candidate(discovered_at=datetime(2026, 10, 2))


def test_candidate_and_event_rollback_together(store, monkeypatch):
    # An event-write failure must not leave a candidate without its history.
    from sqlalchemy.orm import Session
    original_flush = Session.flush

    def fail_event_flush(session, objects=None):
        if any(isinstance(obj, InboxStatusEventORM) for obj in session.new):
            raise RuntimeError("event write failed")
        return original_flush(session, objects)

    monkeypatch.setattr(Session, "flush", fail_event_flush)
    with pytest.raises(RuntimeError, match="event write failed"):
        store.add_candidate(candidate())
    with store._session_factory() as session:
        assert list(session.scalars(select(InboxCandidateORM))) == []
