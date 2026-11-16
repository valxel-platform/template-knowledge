"""The Postgres store: the same answers as the memory store, and access enforced by the database
itself (row-level security), so a bug in the tools' own filters still can't leak a passage.

Needs KNOWLEDGE_TEST_DATABASE_URL: Postgres with pgvector, as a login that may create roles (CI runs
the pgvector image). Each test works in a database of its own."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

from evals.run import passed, run
from knowledge.app import seed
from knowledge.config import load
from knowledge.ingest import Ingestor
from knowledge.models import HashEmbedder
from knowledge.passages import Filters, Viewer
from knowledge.pgstore import PgStore, migrate
from knowledge.retriever import Retriever

from .conftest import CONFIG, CORPUS

URL = os.environ.get("KNOWLEDGE_TEST_DATABASE_URL", "")
pytestmark = [pytest.mark.postgres, pytest.mark.skipif(not URL, reason="set KNOWLEDGE_TEST_DATABASE_URL")]
CFG = load(CONFIG)
NURSE = Viewer(frozenset({"north-clinic"}), frozenset({"user:n-17", "team:cardio"}))
ORTHO = Viewer(frozenset({"north-clinic"}), frozenset({"user:d-4", "team:ortho"}))


def with_db(url: str, name: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=f"/{name}"))


@pytest.fixture
def dsn() -> Iterator[str]:
    name = f"kn_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute(f'create database "{name}"')
    url = with_db(URL, name)
    migrate(url)
    migrate(url)  # idempotent
    yield url
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute(f'drop database "{name}" with (force)')


def embedder_for(v: Any) -> HashEmbedder:
    return HashEmbedder(v.dimensions, v.model)


@pytest.fixture
def store(dsn: str) -> Iterator[PgStore]:
    s = PgStore(dsn)
    seed(s, CFG, embedder_for, CORPUS)
    yield s
    s.close()


def ids(hits: Any) -> list[str]:
    return [h.passage.source_id for h in hits]


def test_hybrid_search_in_postgres(store: PgStore) -> None:
    r = Retriever(store, embedder_for)
    hits, v = r.search("How was the warfarin dose changed for atrial fibrillation?", NURSE, Filters(), 8)
    assert ids(hits)[0] == "notes:n-1001" and v.version == 1
    assert ids(r.search("amlodipine", NURSE, Filters(), 3)[0])[0] == "notes:n-1010"
    only = ids(
        r.search(
            "warfarin",
            NURSE,
            Filters(sources=("policies",), since="2026-01-01", until="2026-12-31", record_type="policy"),
            8,
        )[0]
    )
    assert only and all(i.startswith("policies:") for i in only)
    # Another team never sees cardiology's notes, by keyword or by meaning, and a filtered nearest-
    # neighbour search still returns as many as asked when there are that many to see.
    ortho = ids(r.search("warfarin dose atrial fibrillation INR", ORTHO, Filters(), 5)[0])
    assert "notes:n-1001" not in ortho and "messages:m-201" not in ortho and len(ortho) == 5


def test_documents_counts_and_sources(store: PgStore) -> None:
    v = store.versions()[0]
    [p] = store.document(v.version, "notes", "n-1001", NURSE)
    assert p.title == "Anticoagulation review" and p.dated == "2026-10-02" and "555-0100" not in p.text
    assert store.document(v.version, "notes", "n-1001", ORTHO) == []
    assert store.counts(v.version, NURSE)["notes"] == (6, 6) and store.counts(v.version, ORTHO)["notes"] == (
        2,
        2,
    )
    assert store.counts(v.version, Viewer(None, frozenset({"team:cardio"})))["notes"][0] == 7
    store.set_source_state("notes@v1", applied_at=1.0)
    store.set_source_state("notes@v1", behind=3)
    assert store.source_states()["notes@v1"] == {"applied_at": 1.0, "behind": 3}


def test_writes_versions_and_scans(store: PgStore) -> None:
    v = store.versions()[0]
    ing = Ingestor(CFG, store, embedder_for)
    ing.delete(CFG.source("notes"), "n-1001", v)
    assert store.document(v.version, "notes", "n-1001", NURSE) == []
    assert ing.scan(
        CFG.source("policies"), [{"id": "pol-1", "clinic_id": "north-clinic", "body": "Changed."}], v
    ) == (1, 2)
    assert set(store.fingerprints(v.version, "policies")) == {"pol-1"}
    v2 = store.create_version("other-model", 64, 2)
    assert v2.version == 2 and v2.state == "building"
    store.activate(v2.version)
    assert [x.state for x in store.versions()] == ["retired", "active"]
    with pytest.raises(ValueError):
        store.drop(v2.version)
    with pytest.raises(KeyError):
        store.activate(9)
    store.drop(v.version)
    assert [x.version for x in store.versions()] == [2]


def test_the_database_enforces_access_for_the_tools_login(store: PgStore, dsn: str) -> None:
    """The tools' login is a member of knowledge_reader only: it can't write, and without the
    caller's settings it sees nothing at all, whatever query it runs."""
    login = f"tools_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(f"create role {login} login password 'pw' in role knowledge_reader")
    parts = urlsplit(dsn)
    tools = urlunsplit(
        parts._replace(netloc=f"{login}:pw@{parts.hostname}" + (f":{parts.port}" if parts.port else ""))
    )
    try:
        with psycopg.connect(tools) as conn:
            conn.execute("set role knowledge_reader")
            assert conn.execute("select count(*) from knowledge_passage").fetchone() == (0,)
            conn.execute(
                "select set_config('knowledge.tenants', 'south-clinic', false),"
                " set_config('knowledge.principals', 'team:cardio', false)"
            )
            got = {r[0] for r in conn.execute("select record_id from knowledge_passage")}
            assert got == {"n-1005", "m-203", "pol-3"}  # the south clinic's cardiology records and its policy
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute("delete from knowledge_passage")
        reader = PgStore(tools)
        try:
            hits, _ = Retriever(reader, embedder_for).search("warfarin INR", NURSE, Filters(), 8)
            assert "notes:n-1001" in ids(hits)
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                reader.delete_record(1, "notes", "n-1001")
        finally:
            reader.close()
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(f"drop owned by {login}")
            conn.execute(f"drop role {login}")


def test_the_evals_pass_against_postgres(dsn: str) -> None:
    cases, summary = run(database_url=dsn)
    assert passed(cases, summary), [c for c in cases if not c["ok"]]
