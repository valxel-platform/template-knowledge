"""The tools over real MCP calls: hybrid search with citations, access per tenant and team,
documents, sources and their freshness, scopes, and the audit trail."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx

from .conftest import Live


def result(res: Any) -> Any:
    assert not res.is_error, res.content
    return res.structured_content or json.loads(res.content[0].text)


def ids(res: Any) -> list[str]:
    return [p["source_id"] for p in result(res)["passages"]]


def test_search_finds_the_passage_and_cites_it(live: Live) -> None:
    out = result(
        live.call(
            "nurse", "search", {"question": "How was the warfarin dose changed for atrial fibrillation?"}
        )
    )
    first = out["passages"][0]
    assert first["source_id"] == "notes:n-1001" and first["title"] == "Anticoagulation review"
    assert first["link"] == "https://emr.example/notes/n-1001" and first["date"] == "2026-10-02"
    assert out["index_version"] == 1
    # The scanned forms are files the index can't search: the result says so.
    assert any("Scanned referral forms" in n for n in out["notes"])


def test_keyword_search_catches_names_and_codes(live: Live) -> None:
    assert ids(live.call("nurse", "search", {"question": "amlodipine"}))[0] == "notes:n-1010"


def test_a_search_sees_only_the_callers_clinic_and_teams(live: Live) -> None:
    question = {"question": "warfarin dose INR atrial fibrillation", "k": 20}
    nurse = ids(live.call("nurse", "search", question))
    assert "notes:n-1001" in nurse and "messages:m-201" in nurse
    assert "notes:n-1005" not in nurse  # the south clinic's
    ortho = ids(live.call("ortho", "search", question))
    assert "notes:n-1001" not in ortho and "messages:m-201" not in ortho  # cardiology's notes
    assert "policies:pol-1" in ortho  # a clinic-wide policy
    south = ids(live.call("south", "search", question))
    assert set(south) <= {"notes:n-1005", "messages:m-203", "policies:pol-3"}


def test_filters_narrow_a_search(live: Live) -> None:
    got = ids(live.call("nurse", "search", {"question": "warfarin", "sources": ["policies"]}))
    assert got[0] == "policies:pol-1" and all(g.startswith("policies:") for g in got)
    got = ids(live.call("nurse", "search", {"question": "warfarin INR", "since": "2026-10-05"}))
    assert "notes:n-1001" not in got and "messages:m-201" in got
    res = live.call("nurse", "search", {"question": "warfarin", "sources": ["billing"]})
    assert res.is_error and "list_sources" in res.content[0].text


def test_get_document_returns_what_the_caller_may_see(live: Live) -> None:
    doc = result(live.call("nurse", "get_document", {"source_id": "policies:pol-1"}))
    assert (
        doc["title"] == "Warfarin monitoring protocol"
        and "INR above 5" in doc["text"]
        and not doc["truncated"]
    )
    # Another team's note reads exactly like one that doesn't exist.
    hidden = live.call("ortho", "get_document", {"source_id": "notes:n-1001"})
    missing = live.call("ortho", "get_document", {"source_id": "notes:n-9999"})
    assert hidden.is_error and missing.is_error
    assert hidden.content[0].text.replace("n-1001", "X") == missing.content[0].text.replace("n-9999", "X")
    assert live.call("nurse", "get_document", {"source_id": "billing:b-1"}).is_error


def test_excluded_fields_never_reach_the_index(live: Live) -> None:
    doc = result(live.call("nurse", "get_document", {"source_id": "notes:n-1001"}))
    assert "555-0100" not in doc["text"]


def test_list_sources_counts_only_what_the_caller_may_see(live: Live) -> None:
    nurse = {s["name"]: s for s in result(live.call("nurse", "list_sources", {}))["sources"]}
    ortho = {s["name"]: s for s in result(live.call("ortho", "list_sources", {}))["sources"]}
    assert nurse["notes"]["records"] == 6 and ortho["notes"]["records"] == 2
    assert (
        nurse["scanned-forms"]["indexed"] is False and "can't be searched" in nurse["scanned-forms"]["note"]
    )
    assert nurse["notes"]["indexed"] is True


def test_every_tool_is_read_only_and_scoped(live: Live) -> None:
    tools = {t.name: t for t in live.tools("nurse")}
    assert set(tools) == {"search", "get_document", "list_sources"}
    assert all(t.annotations.read_only_hint for t in tools.values())
    rpc = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    accept = {"Accept": "application/json, text/event-stream"}
    assert httpx.post(live.url, json=rpc, headers=accept).status_code == 401
    assert (
        httpx.post(live.url, json=rpc, headers={**accept, "Authorization": "Bearer no-scopes"}).status_code
        == 403
    )
    no_org = live.call("no-org", "search", {"question": "warfarin"})
    assert no_org.is_error and "no organization" in no_org.content[0].text


def test_the_audit_trail_has_source_ids_and_a_hash_of_the_question(live: Live) -> None:
    question = "How was the warfarin dose changed for atrial fibrillation?"
    found = ids(live.call("nurse", "search", {"question": question}))
    [event] = [e for e in live.audit.events if e["data"]["action"] == "tool.search"]
    data = event["data"]
    assert data["resource"]["type"] == "document" and found[0] in data["resource"]["ids"]
    assert data["details"]["question_sha256"] == hashlib.sha256(question.encode()).hexdigest()
    assert "warfarin" not in json.dumps(event)


def test_rate_limit_per_caller(live: Live) -> None:
    results = [live.call("south", "list_sources", {}) for _ in range(7)]
    assert any(r.is_error and "Rate limit" in r.content[0].text for r in results)
    assert not live.call("nurse", "list_sources", {}).is_error


def test_list_sources_says_when_a_topic_isnt_compacted(live: Live) -> None:
    live.store.set_source_state(
        "messages@topic", topic="emr.messages.v1", cleanup_policy="delete", compacted=False
    )
    sources = {s["name"]: s for s in result(live.call("nurse", "list_sources", {}))["sources"]}
    assert "isn't compacted" in sources["messages"]["note"] and sources["notes"]["note"] == ""


def test_the_example_feature_is_behind_its_flag(live: Live) -> None:
    """The pattern every feature follows: its new behaviour reads its flag, and off is today's."""
    from openfeature.provider.in_memory_provider import InMemoryFlag, InMemoryProvider

    from knowledge import flags
    from knowledge.server import example_flag
    from knowledge.settings import Settings

    assert example_flag(Settings(product="emr")) == "emr.example.sources-used"
    assert result(live.call("nurse", "search", {"question": "amlodipine"}))["sources_used"] is None
    on = InMemoryFlag("on", {"on": True, "off": False})
    keys = {"example.example.sources-used": on}  # the test server names no product
    flags.use_provider(InMemoryProvider(keys))
    try:
        used = result(live.call("nurse", "search", {"question": "amlodipine"}))["sources_used"]
        assert used and "notes" in used
    finally:
        flags.use_provider(InMemoryProvider({}))
