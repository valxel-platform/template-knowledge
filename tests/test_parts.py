"""The parts on their own: knowledge.yaml, passages and access, the gateway's models, hybrid search
and reranking, ingestion from events and scans, and the admin commands."""

from __future__ import annotations

import io
import json
from typing import Any

import httpx
import pytest

from knowledge import admin
from knowledge.app import make_models, seed
from knowledge.config import ConfigError, load, parse
from knowledge.ingest import (
    EventsRunner,
    Ingestor,
    ScanRunner,
    Worker,
    check_topics,
    lag_seconds,
    not_compacted,
    state_key,
)
from knowledge.models import GatewayEmbedder, GatewayReranker, HashEmbedder, ModelUnavailable
from knowledge.passages import Filters, Passage, Viewer, from_record, redact_card_numbers, split
from knowledge.retriever import NoIndex, Retriever
from knowledge.security import principals
from knowledge.settings import Settings
from knowledge.store import MemoryStore, active

from .conftest import CONFIG, CORPUS

CFG = load(CONFIG)
SEEDED_POL1 = next(
    json.loads(ln)["record"]["body"] for ln in CORPUS.read_text().splitlines() if '"pol-1"' in ln
)
NORTH_CARDIO = Viewer(frozenset({"north-clinic"}), frozenset({"user:n-17", "team:cardio"}))


def embedder_for(v: Any) -> HashEmbedder:
    return HashEmbedder(v.dimensions, v.model)


def seeded() -> tuple[MemoryStore, Any]:
    store = MemoryStore()
    return store, seed(store, CFG, embedder_for, CORPUS)


# ---------------------------------------------------------------------- knowledge.yaml
BASE = {
    "index": {"embedding_model": "m", "dimensions": 64},
    "sources": [
        {
            "name": "notes",
            "from": {"events": {"topic": "t", "compacted": True}},
            "text": ["body"],
            "tenant": "clinic",
        }
    ],
}


@pytest.mark.parametrize(
    "change, problem",
    [
        ({"index": {}}, "embedding_model is required"),
        ({"index": {"embedding_model": "m", "dimensions": 3}}, "dimensions must be"),
        ({"index": {"embedding_model": "m", "dimensions": 64, "chunking": {"max_chars": 50}}}, "max_chars"),
        ({"sources": []}, "at least one source"),
        (
            {
                "sources": [
                    {
                        "name": "Bad Name",
                        "from": {"events": {"topic": "t", "compacted": True}},
                        "text": ["b"],
                        "tenant": "c",
                    }
                ]
            },
            "lowercase",
        ),
        ({"sources": [{"name": "a1", "from": {}, "text": ["b"]}]}, "exactly one of events"),
        ({"sources": [{"name": "a1", "from": {"events": {}}, "text": ["b"], "tenant": "c"}]}, "need a topic"),
        (
            {"sources": [{"name": "a1", "from": {"scan": {"query": "q"}}, "text": ["b"], "tenant": "c"}]},
            "dsn_env and query",
        ),
        (
            {
                "sources": [
                    {"name": "a1", "from": {"events": {"topic": "t", "compacted": True}}, "tenant": "c"}
                ]
            },
            "whose text is indexed",
        ),
        (
            {
                "sources": [
                    {"name": "a1", "from": {"events": {"topic": "t", "compacted": True}}, "text": ["b"]}
                ]
            },
            "tenant field",
        ),
        (
            {
                "sources": [
                    {
                        "name": "a1",
                        "from": {"events": {"topic": "t", "compacted": True}},
                        "text": ["b"],
                        "tenant": "c",
                        "exclude": ["b"],
                    }
                ]
            },
            "both indexed and excluded",
        ),
        (
            {
                "sources": [
                    {
                        "name": "a1",
                        "from": {"events": {"topic": "t", "compacted": True}},
                        "text": ["b"],
                        "tenant": "c",
                        "allow": ["x"],
                    }
                ]
            },
            "start with user:",
        ),
        ({"sources": [BASE["sources"][0], BASE["sources"][0]]}, "two sources are named notes"),
        (
            {"sources": [{"name": "a1", "from": {"events": {"topic": "t"}}, "text": ["b"], "tenant": "c"}]},
            "set compacted: true once its events carry whole records",
        ),
    ],
)
def test_knowledge_yaml_says_what_is_wrong(change: dict[str, Any], problem: str) -> None:
    with pytest.raises(ConfigError, match=problem):
        parse({**BASE, **change})


def test_a_payments_product_never_indexes_card_fields() -> None:
    raw = {**BASE, "sources": [{**BASE["sources"][0], "text": ["card_number", "memo"]}]}
    with pytest.raises(ConfigError, match="never indexes card data"):
        parse(raw, data_class="payments")
    assert parse(raw).source("notes").text == ("card_number", "memo")  # other data classes may
    single = {**BASE, "sources": [{k: v for k, v in BASE["sources"][0].items() if k != "tenant"}]}
    assert parse(single, multi_tenant=False).sources[0].tenant == ""


def test_the_example_config_reads() -> None:
    assert [s.name for s in CFG.sources] == ["notes", "messages", "policies", "scanned-forms"]
    assert not CFG.source("scanned-forms").indexed and CFG.source("policies").kind == "scan"
    assert CFG.source("notes").is_delete("note.deleted") and not CFG.source("notes").is_delete("note.signed")
    custom = parse(
        {
            **BASE,
            "sources": [
                {
                    **BASE["sources"][0],
                    "from": {"events": {"topic": "t", "compacted": True, "delete_types": ["gone"]}},
                }
            ],
        }
    )
    assert custom.sources[0].is_delete("gone") and not custom.sources[0].is_delete("x.deleted")
    with pytest.raises(KeyError):
        CFG.source("billing")


# ---------------------------------------------------------------------- passages
def test_records_split_at_headings_paragraphs_and_sentences() -> None:
    text = "# One\n\nFirst para.\n\n# Two\n\nSecond para."
    assert split(text, 1000) == ["# One\n\nFirst para.", "# Two\n\nSecond para."]
    long = " ".join(f"Sentence {i} is here." for i in range(40))
    pieces = split(long, 200)
    assert all(len(p) <= 200 for p in pieces) and "Sentence 39" in pieces[-1]
    assert split("x" * 450, 200) == ["x" * 200, "x" * 200, "x" * 50]
    assert split("a\n\nb", 1000) == ["a\n\nb"] and split("", 100) == []


def test_a_passage_keeps_access_and_fails_closed() -> None:
    notes = CFG.source("notes")
    [p] = from_record(
        notes,
        {
            "note_id": "n1",
            "clinic_id": "north",
            "care_team": "cardio",
            "author": "d-2",
            "body": "Text",
            "title": "T",
            "signed_at": "2026-10-02T09:00:00Z",
        },
        max_chars=500,
    )
    assert p.allow == ("team:cardio", "user:d-2") and p.tenant == "north" and p.dated == "2026-10-02"
    assert p.source_id == "notes:n1" and p.link == "https://emr.example/notes/n1"
    # No access fields at all: nobody may see it, rather than the whole clinic.
    [q] = from_record(
        CFG.source("messages"), {"message_id": "m1", "clinic_id": "north", "body": "x"}, max_chars=500
    )
    assert q.allow == ("user:nobody",)
    with pytest.raises(ValueError, match="without its id"):
        from_record(notes, {"body": "x"}, max_chars=500)
    assert from_record(notes, {"note_id": "n2", "body": ""}, max_chars=500) == []


def test_card_numbers_are_removed_for_payments() -> None:
    assert (
        redact_card_numbers("paid with 4111 1111 1111 1111 today") == "paid with [card number removed] today"
    )
    assert (
        redact_card_numbers("order 1234 5678 9012 3456") == "order 1234 5678 9012 3456"
    )  # not a card (Luhn)
    [p] = from_record(
        CFG.source("policies"),
        {"id": "x", "clinic_id": "c", "body": "card 4111-1111-1111-1111"},
        max_chars=500,
        payments=True,
    )
    assert "4111" not in p.text


def test_who_sees_what() -> None:
    p = Passage("notes", "n1", 0, "t", "x", tenant="north-clinic", allow=("team:cardio",))
    assert NORTH_CARDIO.sees(p)
    assert not Viewer(frozenset({"south-clinic"}), frozenset({"team:cardio"})).sees(p)
    assert not Viewer(frozenset({"north-clinic"}), frozenset({"team:ortho"})).sees(p)
    assert Viewer(None, frozenset({"team:cardio"})).sees(p)  # single-tenant
    open_p = Passage("policies", "p1", 0, "t", "x", tenant="north-clinic")
    assert Viewer(frozenset({"north-clinic"}), frozenset()).sees(open_p)
    dated = Passage("notes", "n1", 0, "t", "x", dated="2026-10-02", record_type="note")
    assert Filters(since="2026-10-01", until="2026-10-03", record_type="note").match(dated)
    assert not Filters(until="2026-10-01").match(dated) and not Filters(record_type="message").match(dated)
    assert not Filters(since="2026-01-01").match(Passage("notes", "n1", 0, "t", "x"))


def test_principals_from_a_token() -> None:
    claims = {"realm_access": {"roles": ["nurse"]}, "groups": ["/cardio", "icu"]}
    assert principals(claims, "n-17", "roles", "groups") == [
        "role:nurse",
        "team:cardio",
        "team:icu",
        "user:n-17",
    ]
    assert principals({"roles": "doctor"}, "", "roles", "groups") == ["role:doctor"]
    assert principals({"roles": {"admin": {}}}, "", "roles", "groups") == ["role:admin"]


# ---------------------------------------------------------------------- models
def gateway(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_embeddings_go_through_the_gateway_in_batches() -> None:
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(
            {"auth": req.headers["authorization"], "purpose": req.headers["x-valxel-purpose"], **body}
        )
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": [0.5] * 4} for i in range(len(body["input"]))]}
        )

    emb = GatewayEmbedder(
        base_url="https://gw/", api_key="k", model="m", dimensions=4, component="kn", client=gateway(handler)
    )
    assert len(emb.embed([f"t{i}" for i in range(70)], purpose="document")) == 70
    assert [len(s["input"]) for s in seen] == [64, 6] and seen[0]["auth"] == "Bearer k"
    assert seen[0]["purpose"] == "document" and seen[0]["dimensions"] == 4
    with pytest.raises(ValueError):
        GatewayEmbedder(base_url="https://gw", api_key="", model="m", dimensions=4, component="kn")


@pytest.mark.parametrize(
    "handler, message",
    [
        (lambda r: httpx.Response(403, json={}), "answered 403"),
        (lambda r: httpx.Response(200, json={"data": []}), "returned 0 embeddings"),
        (
            lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]}),
            "returned 1 dimensions",
        ),
    ],
)
def test_gateway_failures_say_what_happened(handler: Any, message: str) -> None:
    emb = GatewayEmbedder(
        base_url="https://gw", api_key="k", model="m", dimensions=4, component="kn", client=gateway(handler)
    )
    with pytest.raises(ModelUnavailable, match=message):
        emb.embed(["x"], purpose="query")


def test_an_unreachable_gateway_is_named() -> None:
    def down(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    emb = GatewayEmbedder(
        base_url="https://gw", api_key="k", model="m", dimensions=4, component="kn", client=gateway(down)
    )
    with pytest.raises(ModelUnavailable, match="can't be reached"):
        emb.embed(["x"], purpose="query")


def test_models_by_environment() -> None:
    local, rr = make_models(Settings())
    assert rr is None and local(type("V", (), {"model": "m", "dimensions": 32})()).dimensions == 32
    with pytest.raises(RuntimeError, match="KNOWLEDGE_GATEWAY_URL"):
        make_models(Settings(environment="production"))
    gw, rr = make_models(Settings(gateway_url="https://gw", gateway_key="k"))
    assert isinstance(rr, GatewayReranker) and isinstance(
        gw(type("V", (), {"model": "m", "dimensions": 8})()), GatewayEmbedder
    )
    assert make_models(Settings(gateway_url="https://gw", gateway_key="k", rerank_model=""))[1] is None


# ---------------------------------------------------------------------- retrieval
class FixedReranker:
    def __init__(self, fail: bool = False) -> None:
        self.fail, self.calls = fail, 0

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        self.calls += 1
        if self.fail:
            raise ModelUnavailable("down")
        return [*[(i, 1.0 - i / 100) for i in reversed(range(len(documents)))][:top_n], (999, 0.0)]


def test_hybrid_search_then_rerank() -> None:
    store, _ = seeded()
    plain = Retriever(store, embedder_for)
    hits, v = plain.search("warfarin dose atrial fibrillation", NORTH_CARDIO, Filters(), 5)
    assert hits[0].passage.source_id == "notes:n-1001" and v.version == 1 and len(hits) == 5
    rr = FixedReranker()
    reranked, _ = Retriever(store, embedder_for, rr).search(
        "warfarin dose atrial fibrillation", NORTH_CARDIO, Filters(), 5
    )
    assert rr.calls == 1 and reranked[0].passage.source_id != "notes:n-1001"  # the reranker's order wins
    # A reranker that's down: the merged order stands.
    fallback, _ = Retriever(store, embedder_for, FixedReranker(fail=True)).search(
        "warfarin dose atrial fibrillation", NORTH_CARDIO, Filters(), 5
    )
    assert [h.passage for h in fallback] == [h.passage for h in hits]


def test_no_active_index_says_what_to_do() -> None:
    r = Retriever(MemoryStore(), embedder_for)
    with pytest.raises(NoIndex, match="knowledge-admin rebuild"):
        r.search("x", NORTH_CARDIO, Filters())


# ---------------------------------------------------------------------- ingestion
def test_events_replace_and_delete_records() -> None:
    store, v = seeded()
    ing = Ingestor(CFG, store, embedder_for, clock=lambda: 1000.0)
    notes = CFG.source("notes")
    ing.handle_event(
        notes,
        {
            "type": "note.amended",
            "time": "2026-10-03T10:00:00Z",
            "data": {
                "note_id": "n-1001",
                "clinic_id": "north-clinic",
                "care_team": "cardio",
                "body": "Warfarin now 3 mg.",
            },
        },
        v,
    )
    [p] = store.document(v.version, "notes", "n-1001", NORTH_CARDIO)
    assert p.text == "Warfarin now 3 mg."
    state = store.source_states()[state_key("notes", v.version)]
    assert state["applied_at"] == 1000.0 and state["last_event_at"] is not None
    ing.handle_event(notes, {"type": "note.deleted", "subject": "n-1001", "data": {}}, v)
    assert store.document(v.version, "notes", "n-1001", NORTH_CARDIO) == []
    with pytest.raises(ValueError, match="without the record's id"):
        ing.handle_event(notes, {"type": "note.deleted", "data": {}}, v)
    with pytest.raises(ValueError, match="must be an object"):
        ing.handle_event(notes, {"type": "x", "data": "text"}, v)
    # A record whose text is now empty leaves the index.
    ing.handle_event(notes, {"type": "note.amended", "data": {"note_id": "n-1002", "body": ""}}, v)
    assert store.document(v.version, "notes", "n-1002", NORTH_CARDIO) == []


def test_a_scan_indexes_what_changed_and_deletes_what_is_gone() -> None:
    store, v = seeded()
    clock = iter([5000.0, 6000.0])
    ing = Ingestor(CFG, store, embedder_for, clock=lambda: next(clock))
    pol = CFG.source("policies")
    rows = [
        {
            "id": "pol-1",
            "clinic_id": "north-clinic",
            "title": "Warfarin monitoring protocol",
            "updated_at": "2026-06-01",
            "body": SEEDED_POL1,
        },
        {
            "id": "pol-9",
            "clinic_id": "north-clinic",
            "title": "New",
            "updated_at": "2026-11-01",
            "body": "New policy.",
        },
        {"title": "no id"},
    ]
    # pol-1 is as seeded (unchanged), pol-9 is new, pol-2 and pol-3 are gone; a row without an id is skipped.
    assert ing.scan(pol, rows, v) == (1, 2)
    assert ing.scan(pol, rows[:2], v) == (0, 0)  # nothing changed
    assert store.source_states()[state_key("policies", v.version)]["last_scan_at"] == 6000.0
    lags = lag_seconds(CFG, store, v.version, now=6000.0 + 20 * 60)
    assert lags["policies"] == 300.0 and lags["notes"] is None and "scanned-forms" not in lags


def test_lag_of_an_events_source() -> None:
    store, v = seeded()
    key = state_key("notes", v.version)
    store.set_source_state(key, applied_at=100.0, last_event_at=50.0, behind=0)
    assert lag_seconds(CFG, store, v.version, 1000.0)["notes"] == 0.0
    store.set_source_state(key, behind=12)
    assert lag_seconds(CFG, store, v.version, 1000.0)["notes"] == 950.0


class Msg:
    def __init__(self, value: Any, offset: int = 7, err: Any = None) -> None:
        self._v, self._o, self._e = value, offset, err

    def value(self) -> bytes:
        return self._v if isinstance(self._v, bytes) else json.dumps(self._v).encode()

    def error(self) -> Any:
        return self._e

    def offset(self) -> int:
        return self._o

    def topic(self) -> str:
        return "emr.notes.v1"

    def partition(self) -> int:
        return 0


class FakeConsumer:
    def __init__(self, conf: dict[str, Any], messages: list[Any]) -> None:
        self.conf, self.messages, self.committed, self.seeks, self.closed = (
            conf,
            list(messages),
            [],
            [],
            False,
        )

    def subscribe(self, topics: list[str]) -> None:
        self.topics = topics

    def poll(self, timeout: float) -> Any:
        return self.messages.pop(0) if self.messages else None

    def commit(self, message: Any, asynchronous: bool) -> None:
        self.committed.append(message.offset())

    def seek(self, tp: Any) -> None:
        self.seeks.append(tp.offset)

    def assignment(self) -> list[str]:
        return ["tp"]

    def get_watermark_offsets(self, tp: Any, timeout: float, cached: bool) -> tuple[int, int]:
        return (0, 10)

    def position(self, tps: list[Any]) -> list[Any]:
        return [type("P", (), {"offset": 8})()]

    def close(self) -> None:
        self.closed = True


def test_an_events_runner_commits_after_indexing_and_retries_failures() -> None:
    store, v = seeded()
    ing = Ingestor(CFG, store, embedder_for)
    good = {
        "type": "note.signed",
        "data": {"note_id": "n-77", "clinic_id": "north-clinic", "care_team": "cardio", "body": "New."},
    }
    consumers: list[FakeConsumer] = []

    def factory(conf: dict[str, Any]) -> FakeConsumer:
        consumers.append(
            FakeConsumer(conf, [Msg(good, 1), Msg(b"not json", 2), None, Msg(None, 3, err="broker")])
        )
        return consumers[-1]

    r = EventsRunner(
        ing,
        CFG.source("notes"),
        v,
        group="kn.notes.v1",
        consumer_factory=factory,
        kafka={"bootstrap.servers": "b"},
    )
    c = consumers[0]
    assert c.conf["group.id"] == "kn.notes.v1" and c.conf["enable.auto.commit"] is False
    for _ in range(4):
        r.step()
    assert c.committed == [1, 2]  # the unreadable one is skipped, not retried forever
    assert [p.record_id for p in store.document(v.version, "notes", "n-77", NORTH_CARDIO)] == ["n-77"]
    assert store.source_states()[state_key("notes", v.version)]["behind"] == 2

    class Broken(MemoryStore):
        def replace_record(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("database down")

    broken = Broken()
    bv = broken.create_version("m", 1536, 1)
    r2 = EventsRunner(
        Ingestor(CFG, broken, embedder_for),
        CFG.source("notes"),
        bv,
        group="g",
        consumer_factory=lambda conf: FakeConsumer(conf, [Msg(good, 5)]),
        kafka={},
    )
    r2.stop.set()  # don't wait out the back-off
    r2.step()
    assert r2.consumer.seeks == [5] and r2.consumer.committed == []
    r2.run()  # stopped: subscribes, then closes
    assert r2.consumer.closed


def test_a_scan_runner_and_the_worker() -> None:
    store, v = seeded()
    ing = Ingestor(CFG, store, embedder_for)
    rows = [{"id": "pol-1", "clinic_id": "north-clinic", "body": "Only this."}]
    runner = ScanRunner(ing, CFG.source("policies"), v, rows=lambda s: rows)
    runner.step()
    assert store.fingerprints(v.version, "policies").keys() == {"pol-1"}

    def broken(s: Any) -> list[dict[str, Any]]:
        raise RuntimeError("store down")

    ScanRunner(ing, CFG.source("policies"), v, rows=broken).step()  # logged, not raised
    made: list[FakeConsumer] = []
    w = Worker(
        ing,
        name="kn",
        kafka={},
        consumer_factory=lambda conf: made.append(FakeConsumer(conf, [])) or made[-1],
        rows=lambda s: [],
    )
    w.refresh()
    assert sorted(k for k in w.runners) == [("messages", 1), ("notes", 1), ("policies", 1)]
    nv = store.create_version("m2", 1536, 1)  # a rebuild: the new version gets its own runners
    w.refresh()
    runner = w.runners[("notes", nv.version)]
    assert isinstance(runner, EventsRunner) and runner.consumer.conf["group.id"] == f"kn.notes.v{nv.version}"
    store.activate(nv.version)
    store.drop(v.version)
    w.refresh()
    assert {ver for _, ver in w.runners} == {nv.version}
    for r in w.runners.values():
        r.stop.set()


# ---------------------------------------------------------------------- versions and admin
def test_versions_switch_and_drop() -> None:
    store, v = seeded()
    v2 = store.create_version("m2", 1536, 1)
    assert active(store).version == v.version and v2.state == "building"  # type: ignore[union-attr]
    store.activate(v2.version)
    assert [x.state for x in store.versions()] == ["retired", "active"]
    with pytest.raises(ValueError):
        store.drop(v2.version)
    with pytest.raises(KeyError):
        store.activate(99)
    store.drop(v.version)
    assert [x.version for x in store.versions()] == [v2.version]


def test_admin_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_CONFIG_PATH", CONFIG)
    store, _ = seeded()
    out = io.StringIO()
    assert admin.main(["rebuild"], store=store, out=out) == 0
    assert "Version 2 is building" in out.getvalue()
    out = io.StringIO()
    assert admin.main(["activate", "2"], store=store, out=out) == 1
    assert "hasn't caught up: messages, notes, policies" in out.getvalue()
    assert admin.main(["activate", "2", "--force"], store=store, out=io.StringIO()) == 0
    out = io.StringIO()
    assert admin.main(["status"], store=store, out=out) == 0
    assert (
        "v1 retired" in out.getvalue()
        and "v2 active" in out.getvalue()
        and "not ingested yet" in out.getvalue()
    )
    assert admin.main(["drop", "1"], store=store, out=io.StringIO()) == 0
    assert [x.version for x in store.versions()] == [2]


def test_a_source_far_behind_is_logged_for_the_alert_rule(caplog: pytest.LogCaptureFixture) -> None:
    store, v = seeded()
    store.set_source_state(state_key("notes", v.version), applied_at=1.0, last_event_at=1.0, behind=40)
    w = Worker(Ingestor(CFG, store, embedder_for), name="kn", kafka={})
    with caplog.at_level("WARNING", logger="knowledge.ingest"):
        assert w.report_lag(now=1.0 + 2 * 3600) == ["notes"]
        assert w.report_lag(now=1.0 + 2 * 3600 + 60) == []  # once every 10 minutes
        assert w.report_lag(now=1.0 + 2 * 3600 + 601) == ["notes"]
    assert "knowledge.source_behind notes is 120 minutes behind" in caplog.text
    assert Worker(Ingestor(CFG, MemoryStore(), embedder_for), name="kn", kafka={}).report_lag() == []


def test_a_topic_that_isnt_compacted_is_reported_and_blocks_a_rebuild(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store, _ = seeded()
    policies = {"emr.notes.v1": "compact", "emr.messages.v1": "delete"}

    def describe(topic: str) -> str:
        return policies[topic]

    with caplog.at_level("WARNING", logger="knowledge.ingest"):
        assert check_topics(CFG, store, describe) == ["messages"]
    assert "knowledge.topic_not_compacted messages reads emr.messages.v1" in caplog.text
    assert not_compacted(CFG, store) == ["messages"]
    monkeypatch.setenv("KNOWLEDGE_CONFIG_PATH", CONFIG)
    out = io.StringIO()
    assert admin.main(["rebuild"], store=store, out=out) == 1
    assert "Not rebuilding: the topics of messages aren't compacted" in out.getvalue()
    assert admin.main(["rebuild", "--force"], store=store, out=io.StringIO()) == 0
    # Fixed: the next check clears it. A topic that can't be described is left as it was.
    policies["emr.messages.v1"] = "compact,delete"
    assert check_topics(CFG, store, describe) == [] and not_compacted(CFG, store) == []

    def down(topic: str) -> str:
        raise RuntimeError("broker down")

    assert check_topics(CFG, store, down) == []
