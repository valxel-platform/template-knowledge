"""Where passages live, in versions.

An index version is one embedding model and one chunking version. Changing either builds a new
version beside the active one; the ingestion worker fills both, agents keep searching the active
one, and `knowledge-admin activate` switches once the new one's evals pass. The old one is dropped
later (`knowledge-admin drop`).

`MemoryStore` is for tests, evals and local runs. `PgStore` (pgstore.py) is the real one: Postgres
with pgvector and full-text search, access enforced by row-level security as well.

Every read takes a Viewer and applies it inside the search, so a top-k list never comes back short
because of access, and never leaks through scores.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, replace
from typing import Protocol

from .passages import Filters, Passage, Viewer

WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset(
    (
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "with",
    )
)


@dataclass(frozen=True)
class IndexVersion:
    version: int
    model: str
    dimensions: int
    chunking: int
    state: str  # building | active | retired


class Store(Protocol):
    def versions(self) -> list[IndexVersion]: ...
    def create_version(self, model: str, dimensions: int, chunking: int) -> IndexVersion: ...
    def activate(self, version: int) -> None: ...
    def drop(self, version: int) -> None: ...
    def replace_record(
        self,
        version: int,
        source: str,
        record_id: str,
        passages: list[Passage],
        vectors: list[list[float]],
        fingerprint: str,
    ) -> None: ...
    def delete_record(self, version: int, source: str, record_id: str) -> None: ...
    def fingerprints(self, version: int, source: str) -> dict[str, str]: ...
    def keyword(
        self, version: int, query: str, viewer: Viewer, filters: Filters, limit: int
    ) -> list[tuple[Passage, float]]: ...
    def nearest(
        self, version: int, vector: list[float], viewer: Viewer, filters: Filters, limit: int
    ) -> list[tuple[Passage, float]]: ...
    def document(self, version: int, source: str, record_id: str, viewer: Viewer) -> list[Passage]: ...
    def counts(self, version: int, viewer: Viewer) -> dict[str, tuple[int, int]]: ...
    def set_source_state(self, source: str, **state: object) -> None: ...
    def source_states(self) -> dict[str, dict[str, object]]: ...


def active(store: Store) -> IndexVersion | None:
    return next((v for v in store.versions() if v.state == "active"), None)


def live_versions(store: Store) -> list[IndexVersion]:
    """The versions the ingestion worker keeps up to date: the active one and any being built."""
    return [v for v in store.versions() if v.state in ("active", "building")]


def words(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOP]


class MemoryStore:
    def __init__(self) -> None:
        self._versions: dict[int, IndexVersion] = {}
        self._passages: dict[int, dict[tuple[str, str], list[tuple[Passage, list[float]]]]] = {}
        self._prints: dict[int, dict[tuple[str, str], str]] = {}
        self._states: dict[str, dict[str, object]] = {}
        self._lock = threading.RLock()

    def versions(self) -> list[IndexVersion]:
        with self._lock:
            return sorted(self._versions.values(), key=lambda v: v.version)

    def create_version(self, model: str, dimensions: int, chunking: int) -> IndexVersion:
        with self._lock:
            n = max(self._versions, default=0) + 1
            v = IndexVersion(n, model, dimensions, chunking, "building")
            self._versions[n], self._passages[n], self._prints[n] = v, {}, {}
            return v

    def activate(self, version: int) -> None:
        with self._lock:
            if version not in self._versions:
                raise KeyError(f"No index version {version}")
            for n, v in list(self._versions.items()):
                if v.state == "active" and n != version:
                    self._versions[n] = replace(v, state="retired")
            self._versions[version] = replace(self._versions[version], state="active")

    def drop(self, version: int) -> None:
        with self._lock:
            if self._versions.get(version) and self._versions[version].state == "active":
                raise ValueError("Activate another version before dropping the active one")
            self._versions.pop(version, None)
            self._passages.pop(version, None)
            self._prints.pop(version, None)

    def replace_record(
        self,
        version: int,
        source: str,
        record_id: str,
        passages: list[Passage],
        vectors: list[list[float]],
        fingerprint: str,
    ) -> None:
        with self._lock:
            self._passages[version][(source, record_id)] = list(zip(passages, vectors, strict=True))
            self._prints[version][(source, record_id)] = fingerprint

    def delete_record(self, version: int, source: str, record_id: str) -> None:
        with self._lock:
            self._passages[version].pop((source, record_id), None)
            self._prints[version].pop((source, record_id), None)

    def fingerprints(self, version: int, source: str) -> dict[str, str]:
        with self._lock:
            return {rid: fp for (src, rid), fp in self._prints[version].items() if src == source}

    def _visible(self, version: int, viewer: Viewer, filters: Filters) -> list[tuple[Passage, list[float]]]:
        with self._lock:
            rows = [r for recs in self._passages.get(version, {}).values() for r in recs]
        return [(p, v) for p, v in rows if viewer.sees(p) and filters.match(p)]

    def keyword(
        self, version: int, query: str, viewer: Viewer, filters: Filters, limit: int
    ) -> list[tuple[Passage, float]]:
        terms = set(words(query))
        rows = self._visible(version, viewer, filters)
        if not terms or not rows:
            return []
        docs = [Counter(words(f"{p.title} {p.text}")) for p, _ in rows]
        df = Counter(t for d in docs for t in terms if t in d)
        n = len(docs)
        scored = []
        for (p, _), d in zip(rows, docs, strict=True):
            s = sum(math.log(1 + n / df[t]) * d[t] / (d[t] + 1.2) for t in terms if t in d)
            if s > 0:
                scored.append((p, s))
        scored.sort(key=lambda x: (-x[1], x[0].source_id, x[0].n))
        return scored[:limit]

    def nearest(
        self, version: int, vector: list[float], viewer: Viewer, filters: Filters, limit: int
    ) -> list[tuple[Passage, float]]:
        scored = [
            (p, sum(a * b for a, b in zip(vector, v, strict=True)))
            for p, v in self._visible(version, viewer, filters)
        ]
        scored.sort(key=lambda x: (-x[1], x[0].source_id, x[0].n))
        return scored[:limit]

    def document(self, version: int, source: str, record_id: str, viewer: Viewer) -> list[Passage]:
        with self._lock:
            rows = self._passages.get(version, {}).get((source, record_id), [])
        return [p for p, _ in sorted(rows, key=lambda r: r[0].n) if viewer.sees(p)]

    def counts(self, version: int, viewer: Viewer) -> dict[str, tuple[int, int]]:
        """Records and passages per source, of those the caller may see."""
        out: dict[str, list[int]] = {}
        with self._lock:
            for (src, _), recs in self._passages.get(version, {}).items():
                seen = [p for p, _ in recs if viewer.sees(p)]
                if not seen:
                    continue
                c = out.setdefault(src, [0, 0])
                c[0] += 1
                c[1] += len(seen)
        return {k: (v[0], v[1]) for k, v in out.items()}

    def set_source_state(self, source: str, **state: object) -> None:
        with self._lock:
            self._states.setdefault(source, {}).update(state)

    def source_states(self) -> dict[str, dict[str, object]]:
        with self._lock:
            return {k: dict(v) for k, v in self._states.items()}
