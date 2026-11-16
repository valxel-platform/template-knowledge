"""Retrieval evaluations: the golden set (evals/questions.yaml) against the sample corpus.

  python -m evals.run                   offline: words-only embeddings, no model (CI, every PR)
  python -m evals.run --live            the real embedding model and reranker through the gateway
                                        (post-deploy in dev: KNOWLEDGE_GATEWAY_URL and _KEY)
  python -m evals.run --database-url …  against Postgres with row-level security, not in memory

Every case is a JUnit test case, for the "retrieval / report" status:
  recall: <question>    the expected passages are in the top 8
  leak: <question>      asked as someone who must not see them, the forbidden records stay hidden
  deleted: <record>     a record deleted after indexing never comes back
  latency p95           search at the 95th percentile is within the bar
Leaks and deletions are platform rules: report.py --must-pass fails the status on any of them,
whatever the pass rate. The run exits non-zero when any bar is missed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import yaml

from knowledge.app import make_models, make_store, seed
from knowledge.config import load
from knowledge.ingest import Ingestor
from knowledge.passages import Filters, Viewer
from knowledge.retriever import Retriever
from knowledge.settings import Settings
from knowledge.store import MemoryStore, Store, active

HERE = Path(__file__).parent
ROOT = HERE.parent


def viewer(spec: dict[str, Any]) -> Viewer:
    t = spec.get("tenants")
    return Viewer(None if t is None else frozenset(t), frozenset(spec.get("principals") or ()))


def run(
    *,
    live: bool = False,
    database_url: str = "",
    questions: Path = HERE / "questions.yaml",
    corpus: Path = HERE / "corpus.jsonl",
    config: Path = ROOT / "knowledge.yaml",
    k: int = 8,
    store: Store | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold = yaml.safe_load(questions.read_text())
    cfg = load(config)
    settings = Settings(
        gateway_url=os.environ.get("KNOWLEDGE_GATEWAY_URL", "") if live else "",
        gateway_key=os.environ.get("KNOWLEDGE_GATEWAY_KEY", ""),
        environment="evals",
    )
    if live and not settings.gateway_url:
        raise SystemExit("--live needs KNOWLEDGE_GATEWAY_URL and KNOWLEDGE_GATEWAY_KEY")
    embedder_for, reranker = make_models(settings)
    own = store is None and bool(database_url)
    store = store or (make_store(settings, database_url) if database_url else MemoryStore())
    try:
        return _run(gold, cfg, store, embedder_for, reranker, settings.candidates, corpus, k)
    finally:
        if own:
            store.close()  # type: ignore[attr-defined]


def _run(
    gold: dict[str, Any],
    cfg: Any,
    store: Store,
    embedder_for: Any,
    reranker: Any,
    candidates: int,
    corpus: Path,
    k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v = seed(store, cfg, embedder_for, corpus)
    retriever = Retriever(store, embedder_for, reranker, candidates=candidates)
    people = gold.get("as") or {}
    cases: list[dict[str, Any]] = []
    timings: list[float] = []

    def search(ask: str, question: str) -> list[str]:
        started = time.perf_counter()
        hits, _ = retriever.search(question, viewer(people[ask]), Filters(), k)
        timings.append((time.perf_counter() - started) * 1000)
        return [h.passage.source_id for h in hits]

    for q in gold.get("recall") or []:
        got = search(q["ask"], q["question"])
        missing = [e for e in q["expect"] if e not in got]
        cases.append(
            {
                "name": f"recall: {q['question']}",
                "ok": not missing,
                "detail": f"asked as {q['ask']}; missing {missing}; got {got}",
            }
        )
    for q in gold.get("leaks") or []:
        got = search(q["ask"], q["question"])
        leaked = [f for f in q["forbid"] if f in got]
        cases.append(
            {
                "name": f"leak: {q['question']}",
                "ok": not leaked,
                "detail": f"asked as {q['ask']}; leaked {leaked}" if leaked else f"asked as {q['ask']}",
            }
        )
    ingestor = Ingestor(cfg, store, embedder_for)
    for q in gold.get("deletions") or []:
        source, rid = q["delete"].split(":", 1)
        ingestor.delete(cfg.source(source), rid, v)
        got = search(q["ask"], q["question"])
        cases.append(
            {
                "name": f"deleted: {q['delete']}",
                "ok": q["delete"] not in got,
                "detail": f"asked as {q['ask']}; got {got}",
            }
        )
    bars = gold.get("bars") or {}
    recall = [c for c in cases if c["name"].startswith("recall:")]
    rate = sum(c["ok"] for c in recall) / len(recall) if recall else 1.0
    timings.sort()
    p95 = timings[min(len(timings) - 1, int(0.95 * len(timings)))] if timings else 0.0
    bar_ms = float(bars.get("p95_ms", 800))
    cases.append(
        {"name": "latency p95", "ok": p95 <= bar_ms, "detail": f"{p95:.0f} ms (bar {bar_ms:.0f} ms)"}
    )
    summary = {
        "recall_at_8": round(rate, 4),
        "recall_bar": float(bars.get("recall_at_8", 0.85)),
        "p95_ms": round(p95, 1),
        "index_version": (active(store) or v).version,
    }
    return cases, summary


def passed(cases: list[dict[str, Any]], summary: dict[str, Any]) -> bool:
    rules = all(c["ok"] for c in cases if c["name"].startswith(("leak:", "deleted:", "latency")))
    return rules and summary["recall_at_8"] >= summary["recall_bar"]


def write_junit(cases: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    suite = ET.Element(
        "testsuite", name="retrieval", tests=str(len(cases)), failures=str(sum(not c["ok"] for c in cases))
    )
    props = ET.SubElement(suite, "properties")
    for key, value in summary.items():
        ET.SubElement(props, "property", name=key, value=str(value))
    for c in cases:
        tc = ET.SubElement(suite, "testcase", classname="retrieval", name=c["name"])
        if not c["ok"]:
            ET.SubElement(tc, "failure", message=c["detail"][:500])
        ET.SubElement(tc, "system-out").text = c["detail"]
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--database-url", default="")
    ap.add_argument("--junit", default="reports/retrieval.xml")
    args = ap.parse_args(argv)
    cases, summary = run(live=args.live, database_url=args.database_url)
    write_junit(cases, summary, Path(args.junit))
    for c in cases:
        if not c["ok"]:
            print(f"FAIL {c['name']}: {c['detail']}")
    print(
        f"recall@8 {summary['recall_at_8']:.2f} (bar {summary['recall_bar']:.2f}), "
        f"p95 {summary['p95_ms']:.0f} ms, "
        f"{sum(c['ok'] for c in cases)}/{len(cases)} cases passed"
    )
    return 0 if passed(cases, summary) else 1


if __name__ == "__main__":
    sys.exit(main())
