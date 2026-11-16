"""Events checked against the approved data contracts (contracts/), parked on the dead-letter topic
when they break them, and counted per dataset for reconciliation. All data is synthetic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from knowledge.contracts import EVENTS, ContractError, Contracts
from knowledge.ingest import EventsRunner, Ingestor, Worker

from .test_parts import CFG, NORTH_CARDIO, FakeConsumer, Msg, embedder_for, seeded

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "valxel:dataset/emr.notes/1.3.0",
    "type": "object",
    "properties": {
        "note_id": {"type": "string", "x-classification": "none"},
        "clinic_id": {"type": "string", "x-classification": "none"},
        "care_team": {"type": "string", "x-classification": "none"},
        "author_id": {"type": "string", "x-classification": "pii"},
        "body": {"type": "string", "x-classification": "phi"},
        "kind": {"type": "string", "enum": ["progress", "discharge"], "x-classification": "none"},
    },
    "required": ["note_id", "clinic_id"],
    "additionalProperties": False,
}


@pytest.fixture
def contracts(tmp_path: Path) -> Contracts:
    (tmp_path / "emr.notes.json").write_text(json.dumps(SCHEMA))
    (tmp_path / "lock.yaml").write_text("emr.notes: {version: 1.3.0, sha: abc, topic: emr.notes.v1}\n")
    return Contracts.load(tmp_path)


def count(dataset: str, outcome: str) -> float:
    return EVENTS.labels(dataset, outcome)._value.get()


def test_a_repo_without_contracts_checks_nothing(tmp_path: Path) -> None:
    empty = Contracts.load(tmp_path)
    assert empty.for_topic("emr.notes.v1") is None and empty.dataset("emr.notes.v1") == "emr.notes.v1"


def test_data_is_pruned_to_the_contract_and_then_validated(contracts: Contracts) -> None:
    c = contracts.for_topic("emr.notes.v1")
    assert c is not None and c.dataset == "emr.notes" and c.version == "1.3.0"
    newer = {"note_id": "n-1", "clinic_id": "north-clinic", "body": "Synthetic.", "added_later": "x"}
    assert c.check(newer) == {"note_id": "n-1", "clinic_id": "north-clinic", "body": "Synthetic."}
    with pytest.raises(ContractError) as e:
        c.check({"note_id": "n-1"})
    assert (e.value.field, e.value.rule) == ("clinic_id", "required")
    with pytest.raises(ContractError) as e:
        c.check({"note_id": "n-1", "clinic_id": "c", "kind": "memo-with-a-secret"})
    assert (e.value.field, e.value.rule) == ("kind", "enum")
    assert "memo-with-a-secret" not in str(e.value)  # the field, never its value


class Producer:
    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes, dict[str, bytes]]] = []

    def produce(self, topic: str, key: Any, value: bytes, headers: list[tuple[str, bytes]]) -> None:
        self.sent.append((topic, value, dict(headers)))

    def flush(self, timeout: float) -> None:
        pass


class KeyedMsg(Msg):
    def key(self) -> bytes:
        return b"n-1"


def test_an_event_that_breaks_its_contract_is_parked_and_counted(contracts: Contracts) -> None:
    store, v = seeded()
    ing = Ingestor(CFG, store, embedder_for, contracts=contracts)
    good = {
        "type": "note.signed",
        "data": {
            "note_id": "n-77",
            "clinic_id": "north-clinic",
            "care_team": "cardio",
            "body": "New.",
            "added_later": True,
        },
    }
    bad = {"type": "note.signed", "data": {"note_id": "n-78", "care_team": "cardio", "body": "Synthetic."}}
    producer = Producer()
    made: list[FakeConsumer] = []
    r = EventsRunner(
        ing,
        CFG.source("notes"),
        v,
        group="kn.notes.v1",
        consumer_factory=lambda conf: (
            made.append(FakeConsumer(conf, [KeyedMsg(good, 1), KeyedMsg(bad, 2)])) or made[-1]
        ),
        kafka={},
        producer_factory=lambda conf: producer,
    )
    applied, parked = count("emr.notes", "applied"), count("emr.notes", "parked")
    r.step()
    r.step()
    assert made[0].committed == [1, 2]  # parked, then moved past: one bad record never stops the partition
    assert [p.record_id for p in store.document(v.version, "notes", "n-77", NORTH_CARDIO)] == ["n-77"]
    assert store.document(v.version, "notes", "n-78", NORTH_CARDIO) == []
    [(topic, value, headers)] = producer.sent
    assert topic == "emr.notes.v1.dlt" and json.loads(value) == bad
    assert headers["valxel-field"] == b"clinic_id" and headers["valxel-rule"] == b"required"
    assert count("emr.notes", "applied") == applied + 1 and count("emr.notes", "parked") == parked + 1


def test_the_worker_hands_its_producer_to_every_runner(contracts: Contracts) -> None:
    store, _ = seeded()
    producer = Producer()
    w = Worker(
        Ingestor(CFG, store, embedder_for, contracts=contracts),
        name="kn",
        kafka={},
        consumer_factory=lambda conf: FakeConsumer(conf, []),
        producer_factory=lambda conf: producer,
        rows=lambda s: [],
    )
    w.refresh()
    runner = w.runners[("notes", 1)]
    assert (
        isinstance(runner, EventsRunner)
        and runner.producer_factory({}) is producer
        and runner.dataset == "emr.notes"
    )
    for r in w.runners.values():
        r.stop.set()
