"""Hybrid search: words and meaning, merged, then reranked.

Each question runs as full-text search (catches record numbers, drug and product codes, names) and
as nearest-neighbour search over embeddings (catches the same thing said in other words). The two
ranked lists are merged by reciprocal rank fusion, and the gateway's reranker orders the best of
them. If the reranker is unavailable, the merged order stands: search degrades, it doesn't fail.

Access and filters are applied inside both searches (store.py), so every candidate is one the
caller may see.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from .models import Embedder, ModelUnavailable, Reranker
from .passages import Filters, Passage, Viewer
from .store import IndexVersion, Store, active

log = logging.getLogger(__name__)

RRF_K = 60  # the usual constant: rank 1 scores 1/61, rank 10 scores 1/70


class NoIndex(RuntimeError):
    pass


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


class Retriever:
    def __init__(
        self,
        store: Store,
        embedder_for: Callable[[IndexVersion], Embedder],
        reranker: Reranker | None = None,
        *,
        candidates: int = 50,
    ) -> None:
        self.store, self.embedder_for, self.reranker, self.candidates = (
            store,
            embedder_for,
            reranker,
            candidates,
        )

    def version(self) -> IndexVersion:
        v = active(self.store)
        if v is None:
            raise NoIndex("No index is active yet: run `knowledge-admin rebuild`, then activate it.")
        return v

    def search(
        self, question: str, viewer: Viewer, filters: Filters, k: int = 8
    ) -> tuple[list[Hit], IndexVersion]:
        v = self.version()
        lexical = self.store.keyword(v.version, question, viewer, filters, self.candidates)
        vector = self.embedder_for(v).embed([question], purpose="query")[0]
        semantic = self.store.nearest(v.version, vector, viewer, filters, self.candidates)
        fused: dict[tuple[str, str, int], tuple[Passage, float]] = {}
        for ranked in (lexical, semantic):
            for rank, (p, _) in enumerate(ranked, start=1):
                key = (p.source, p.record_id, p.n)
                prev = fused.get(key, (p, 0.0))[1]
                fused[key] = (p, prev + 1.0 / (RRF_K + rank))
        merged = sorted(fused.values(), key=lambda x: (-x[1], x[0].source_id, x[0].n))
        pool = merged[: max(k * 3, 20)]
        if self.reranker is not None and len(pool) > 1:
            try:
                order = self.reranker.rerank(
                    question, [f"{p.title}\n{p.text}".strip() for p, _ in pool], top_n=k
                )
                reranked = [Hit(pool[i][0], s) for i, s in order if 0 <= i < len(pool)]
                if reranked:
                    return reranked[:k], v
            except ModelUnavailable as exc:
                log.warning("Reranking skipped: %s", exc)
        return [Hit(p, s) for p, s in pool[:k]], v

    def document(self, source: str, record_id: str, viewer: Viewer) -> list[Passage]:
        return self.store.document(self.version().version, source, record_id, viewer)
