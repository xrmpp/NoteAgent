import json
from datetime import datetime, timezone
from html import unescape
from types import SimpleNamespace

import httpx
import pytest

from app.inbox import cli
from app.inbox.models import CandidateInput, TriageOutput
from app.inbox.store import InboxStore
from app.inbox.triage import build_messages, load_prompt
from app.storage.database import create_session_factory


@pytest.mark.parametrize("exclude", [False, True], ids=["default", "noreason"])
def test_cli_retriage_changes_only_aihot_reason_and_version(tmp_path, monkeypatch, exclude):
    def no_network(*args, **kwargs):
        raise AssertionError("Network access is forbidden")

    monkeypatch.setattr(httpx, "post", no_network)
    monkeypatch.setattr(httpx.Client, "send", no_network)
    monkeypatch.chdir(tmp_path)
    prompt = tmp_path / "data/inbox/prompts/triage.md"
    profile = tmp_path / "data/inbox/profile.md"
    prompt.parent.mkdir(parents=True)
    prompt.write_bytes("系统任务\r\n{{profile}}\r\n输出 JSON".encode())
    profile.write_text("私人档案", encoding="utf-8")
    before = (prompt.read_bytes(), profile.read_bytes())
    system, version = load_prompt(prompt, profile)
    factory = create_session_factory(f"sqlite+pysqlite:///{tmp_path / 'noreason.db'}")
    store = InboxStore(factory)
    candidate, _ = store.add_candidate(CandidateInput(
        external_id="one", url="https://example.com/one", aihot_url="https://aihot.news/items/one",
        title="标题", published_at=None,
        original_title="Original", summary="摘要", source_name="Example", category="tip",
        discovered_at=datetime.now(timezone.utc), aihot_score=99.99,
        aihot_reason="AIHOT 独有推荐理由", raw={"reason": "AIHOT 独有推荐理由"},
    ))
    output = dict(usage="practice", quality=4, relevance=5, reason="模型自身判断",
                  try_what="验证一条用例", est_minutes=30, prerequisites=[], unknowns=[])
    baseline = store.append_triage_result(
        candidate.id, prompt_version=version, model="fake-model", output=TriageOutput(**output),
        raw_output=json.dumps(output, ensure_ascii=False), mark_triaged=True,
    )

    class FakeProvider:
        messages = []

        def generate_json(self, messages, schema_name):
            assert schema_name == "inbox_triage"
            self.messages.append(messages)
            return output

    provider = FakeProvider()
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(
        database_url="unused", embedding_dimensions=1536, model_name="fake-model",
    ))
    monkeypatch.setattr(cli, "create_session_factory", lambda *args, **kwargs: factory)
    monkeypatch.setattr(cli, "create_provider", lambda settings: provider)
    args = ["triage", "--candidate-id", str(candidate.id), "--retriage"]
    if exclude:
        args.append("--exclude-aihot-reason")
    assert cli.main(args) == 0
    assert len(provider.messages) == 1
    messages = provider.messages[0]
    assert messages[0].content == system
    expected = json.loads(unescape(build_messages(system, candidate)[1].content
                                  .removeprefix("<material>\n").removesuffix("\n</material>")))
    if exclude:
        del expected["reason"]
        assert "AIHOT 独有推荐理由" not in messages[1].content
    else:
        assert "AIHOT 独有推荐理由" in messages[1].content
    material = json.loads(unescape(messages[1].content.removeprefix("<material>\n")
                                   .removesuffix("\n</material>")))
    assert material == expected
    results = store.list_triage_results(candidate.id)
    assert len(results) == 2 and results[0].id == baseline.id
    assert results[0].prompt_version == version
    assert results[1].prompt_version == version + ("+noreason" if exclude else "")
    assert results[1].model == "fake-model" and results[1].reason == output["reason"]
    assert store.get_candidate(candidate.id).aihot_reason == "AIHOT 独有推荐理由"
    assert len(store.list_status_events(candidate.id)) == 2
    assert before == (prompt.read_bytes(), profile.read_bytes())
