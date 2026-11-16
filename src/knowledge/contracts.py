"""Data contracts for the topics this repo reads (the design "Datasets, contracts and reporting").

The SDLC tool writes contracts/<id>.json (the JSON Schema generated from each approved ODCS contract
this repo reads) and contracts/lock.yaml (dataset -> version, hash, topic) on the feature branch.
They ship in the image, so nothing here reads GitHub at run time.

Reading an event: drop the properties this copy doesn't have (a newer, compatible producer may add
optional ones), then validate the rest. An event that fails is not applied: it goes to the topic's
dead-letter topic (<topic>.dlt) with the field that failed, never its value.

Every event is counted by dataset and outcome (knowledge_dataset_events_total): the platform's
dataset-drift rule compares what each consumer applied with what the producer published.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from prometheus_client import Counter

log = logging.getLogger("knowledge.contracts")

EVENTS = Counter(
    "knowledge_dataset_events",
    "Events read per dataset: applied, parked (dead-letter topic) or skipped (unreadable)",
    ["dataset", "outcome"],
)


SUMMARY = "valxel.datasets"
_counts: dict[str, dict[str, int]] = {}
_counts_lock = threading.Lock()


def count(dataset: str, outcome: str) -> None:
    """One event of a dataset: the meter, and the summary line's count."""
    EVENTS.labels(dataset, outcome).inc()
    with _counts_lock:
        by = _counts.setdefault(dataset, {})
        by[outcome] = by.get(outcome, 0) + 1


def summary_line() -> str | None:
    """valxel.datasets {"<dataset>":{"<outcome>":n}} for what was counted since the last line, or
    None. The worker logs it once a minute; in staging the SDLC tool sums these per component and
    compares what each producer published with what each consumer applied, skipped or parked."""
    with _counts_lock:
        taken = {d: dict(sorted(o.items())) for d, o in sorted(_counts.items()) if o}
        _counts.clear()
    return f"{SUMMARY} {json.dumps(taken, separators=(',', ':'))}" if taken else None


class ContractError(ValueError):
    """An event's data breaks its contract. The message names the field, never its value."""

    def __init__(self, dataset: str, field: str, rule: str) -> None:
        super().__init__(f"{dataset}: {field or '(data)'} breaks the data contract ({rule})")
        self.dataset, self.field, self.rule = dataset, field, rule


@dataclass(frozen=True)
class Contract:
    dataset: str
    topic: str
    version: str
    schema: dict[str, Any]

    def check(self, data: dict[str, Any]) -> dict[str, Any]:
        """The event's data as this copy knows it, or ContractError."""
        known = self.schema.get("properties") or {}
        kept = {k: v for k, v in data.items() if k in known}
        for e in sorted(Draft202012Validator(self.schema).iter_errors(kept), key=lambda e: list(e.path)):
            field = ".".join(str(p) for p in e.path) or (
                str(e.message.split("'")[1]) if e.validator == "required" and "'" in e.message else ""
            )
            raise ContractError(self.dataset, field, str(e.validator))
        return kept


class Contracts:
    """The contracts in a directory, by topic. Empty when the repo reads no contracted topic."""

    def __init__(self, by_topic: dict[str, Contract]) -> None:
        self.by_topic = by_topic

    @classmethod
    def load(cls, directory: str | Path = "contracts") -> Contracts:
        root = Path(directory)
        lock = root / "lock.yaml"
        if not lock.exists():
            return cls({})
        out = {}
        for dataset, row in (yaml.safe_load(lock.read_text()) or {}).items():
            schema = json.loads((root / f"{dataset}.json").read_text())
            Draft202012Validator.check_schema(schema)
            out[str(row["topic"])] = Contract(str(dataset), str(row["topic"]), str(row["version"]), schema)
        return cls(out)

    def for_topic(self, topic: str) -> Contract | None:
        return self.by_topic.get(topic)

    def dataset(self, topic: str) -> str:
        c = self.by_topic.get(topic)
        return c.dataset if c else topic
