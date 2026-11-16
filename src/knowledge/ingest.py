"""The ingestion worker: the product's records in, passages out.

Each index version that is live (active, or being built) is kept up to date by its own runners,
one per source:

  events   a Kafka consumer in group <name>.<source>.v<version>, from the topic's start. A new
           version replays the topic in order while the active one carries on, so neither ever
           applies an older record over a newer one. An offset is committed only once its record
           is in the store; a record that can't be read is logged and skipped, a store or gateway
           failure is retried from the same offset.
  scan     the source's query every `every_minutes`: rows whose content changed are re-indexed,
           rows gone are deleted. Needs no events, for a document store that keeps text.

A deleted record's passages are removed with its event, and a deleted record is never answered
again. Each source's lag (how far its index trails the product) is kept for list_sources and the
`knowledge_source_lag_seconds` metric: more than an hour turns the product's health to Needs
attention (the SDLC tool's alert rule for knowledge components).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

from .config import Config, Source
from .contracts import ContractError, Contracts, count, summary_line
from .models import Embedder
from .passages import from_record
from .store import IndexVersion, Store, live_versions

log = logging.getLogger(__name__)

LAG_ALERT_S = 3600


def fingerprint(record: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()


def state_key(source: str, version: int) -> str:
    return f"{source}@v{version}"


def _event_time(value: Any) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


class Ingestor:
    def __init__(
        self,
        cfg: Config,
        store: Store,
        embedder_for: Callable[[IndexVersion], Embedder],
        *,
        payments: bool = False,
        clock: Callable[[], float] = time.time,
        contracts: Contracts | None = None,
    ) -> None:
        self.cfg, self.store, self.embedder_for, self.payments, self.clock = (
            cfg,
            store,
            embedder_for,
            payments,
            clock,
        )
        # The approved data contracts of the topics read (contracts/, written by the SDLC tool).
        self.contracts = contracts or Contracts({})

    def upsert(self, source: Source, record: dict[str, Any], version: IndexVersion) -> None:
        kept = {k: v for k, v in record.items() if k not in source.exclude}
        passages = from_record(
            source, kept, max_chars=self.cfg.index.chunking.max_chars, payments=self.payments
        )
        rid = str(kept[source.id])
        if not passages:
            self.store.delete_record(version.version, source.name, rid)
            return
        vectors = self.embedder_for(version).embed(
            [f"{p.title}\n{p.text}".strip() for p in passages], purpose="document"
        )
        self.store.replace_record(version.version, source.name, rid, passages, vectors, fingerprint(kept))

    def delete(self, source: Source, record_id: str, version: IndexVersion) -> None:
        self.store.delete_record(version.version, source.name, record_id)

    def handle_event(self, source: Source, event: dict[str, Any], version: IndexVersion) -> None:
        """One CloudEvent: `type` says what happened, `data` is the record (`subject` its id when
        a deletion carries no data)."""
        data = event.get("data") or {}
        if not isinstance(data, dict):
            raise ValueError(f"{source.name}: an event's data must be an object")
        etype = str(event.get("type") or "")
        if source.is_delete(etype):
            rid = str(data.get(source.id) or event.get("subject") or "")
            if not rid:
                raise ValueError(f"{source.name}: a deletion without the record's id")
            self.delete(source, rid, version)
        else:
            contract = self.contracts.for_topic(str(source.options.get("topic") or ""))
            self.upsert(source, contract.check(data) if contract else data, version)
        self.store.set_source_state(
            state_key(source.name, version.version),
            kind="events",
            applied_at=self.clock(),
            last_event_at=_event_time(event.get("time")),
        )

    def scan(self, source: Source, rows: Iterable[dict[str, Any]], version: IndexVersion) -> tuple[int, int]:
        """Index what changed since the last scan and delete what's gone. Returns (changed, deleted)."""
        known = self.store.fingerprints(version.version, source.name)
        seen: set[str] = set()
        changed = 0
        for row in rows:
            kept = {k: v for k, v in row.items() if k not in source.exclude}
            rid = str(kept.get(source.id) or "")
            if not rid:
                log.warning("%s: a row without its id field (%s) was skipped", source.name, source.id)
                continue
            seen.add(rid)
            if known.get(rid) != fingerprint(kept):
                self.upsert(source, kept, version)
                changed += 1
        gone = sorted(set(known) - seen)
        for rid in gone:
            self.delete(source, rid, version)
        self.store.set_source_state(
            state_key(source.name, version.version),
            kind="scan",
            last_scan_at=self.clock(),
            every_minutes=source.options.get("every_minutes", 15),
        )
        return changed, len(gone)


def topic_key(source: str) -> str:
    return f"{source}@topic"


def check_topics(cfg: Config, store: Store, describe: Callable[[str], str]) -> list[str]:
    """Each compacted events source's topic really is compacted (`describe` returns the topic's
    cleanup.policy). Records the answer for list_sources and knowledge-admin, and logs
    knowledge.topic_not_compacted for each that isn't. Returns those sources."""
    wrong = []
    for s in cfg.sources:
        if s.kind != "events":
            continue
        topic = str(s.options["topic"])
        try:
            policy = describe(topic)
        except Exception as exc:  # noqa: BLE001  (an unreachable broker mustn't stop ingestion)
            log.warning("%s: couldn't read %s's cleanup.policy (%s)", s.name, topic, type(exc).__name__)
            continue
        ok = "compact" in policy
        store.set_source_state(topic_key(s.name), topic=topic, cleanup_policy=policy, compacted=ok)
        if not ok:
            wrong.append(s.name)
            log.warning(
                "knowledge.topic_not_compacted %s reads %s, whose cleanup.policy is %s: a rebuild would miss "
                "records older than its retention",
                s.name,
                topic,
                policy,
            )
    return wrong


def not_compacted(cfg: Config, store: Store) -> list[str]:
    """Sources whose topic was found not compacted (by the worker's last check)."""
    states = store.source_states()
    return [
        s.name
        for s in cfg.sources
        if s.kind == "events" and states.get(topic_key(s.name), {}).get("compacted") is False
    ]


def lag_seconds(cfg: Config, store: Store, version: int, now: float) -> dict[str, float | None]:
    """How far each source's index trails the product, in seconds (None: never ingested)."""
    states = store.source_states()
    out: dict[str, float | None] = {}
    for s in cfg.sources:
        if not s.indexed:
            continue
        st = states.get(state_key(s.name, version)) or {}
        if s.kind == "scan":
            last = st.get("last_scan_at")
            every = float(str(st.get("every_minutes") or s.options.get("every_minutes", 15))) * 60
            out[s.name] = None if last is None else max(0.0, now - float(str(last)) - every)
        else:
            if st.get("applied_at") is None:
                out[s.name] = None
            elif not st.get("behind"):
                out[s.name] = 0.0
            else:
                since = st.get("last_event_at") or st.get("applied_at")
                out[s.name] = max(0.0, now - float(str(since)))
    return out


# ---------------------------------------------------------------------- runners
class EventsRunner(threading.Thread):
    """One events source into one index version."""

    def __init__(
        self,
        ingestor: Ingestor,
        source: Source,
        version: IndexVersion,
        *,
        group: str,
        consumer_factory: Callable[[dict[str, Any]], Any],
        kafka: dict[str, Any],
        producer_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        super().__init__(daemon=True, name=f"ingest-{source.name}-v{version.version}")
        self.ingestor, self.source, self.version = ingestor, source, version
        self.consumer = consumer_factory(
            {**kafka, "group.id": group, "enable.auto.commit": False, "auto.offset.reset": "earliest"}
        )
        self.kafka, self.producer_factory = kafka, producer_factory or _kafka_producer
        self.producer: Any = None
        self.topic = str(source.options.get("topic") or "")
        self.dataset = ingestor.contracts.dataset(self.topic)
        self.stop = threading.Event()
        self.last_lag_check = 0.0

    def run(self) -> None:
        topic = str(self.source.options["topic"])
        self.consumer.subscribe([topic])
        try:
            while not self.stop.is_set():
                self.step()
        finally:
            self.consumer.close()

    def step(self) -> None:
        msg = self.consumer.poll(1.0)
        if msg is not None and not msg.error():
            try:
                event = json.loads(msg.value())
                self.ingestor.handle_event(self.source, event, self.version)
                count(self.dataset, "applied")
            except ContractError as exc:  # breaks its contract: park it, naming the field, never its value
                self.park(msg, exc)
            except (ValueError, KeyError, TypeError) as exc:  # a record that can't be read: skip it
                count(self.dataset, "skipped")
                log.error(
                    "%s: skipped an unreadable event at offset %s (%s)",
                    self.source.name,
                    msg.offset(),
                    type(exc).__name__,
                )
            except Exception:  # store or gateway: try the same record again shortly
                log.exception(
                    "%s: couldn't index the event at offset %s; retrying", self.source.name, msg.offset()
                )
                self.consumer.seek(_partition(msg))
                self.stop.wait(5)
                return
            self.consumer.commit(message=msg, asynchronous=False)
        elif msg is not None:
            log.warning("%s: Kafka: %s", self.source.name, msg.error())
        if time.monotonic() - self.last_lag_check > 30:
            self.last_lag_check = time.monotonic()
            self.ingestor.store.set_source_state(
                state_key(self.source.name, self.version.version), kind="events", behind=self.behind()
            )

    def park(self, msg: Any, exc: ContractError) -> None:
        """The event as it came, on <topic>.dlt, with why in its headers."""
        if self.producer is None:
            self.producer = self.producer_factory(self.kafka)
        self.producer.produce(
            f"{self.topic}.dlt",
            key=msg.key(),
            value=msg.value(),
            headers=[
                ("valxel-dataset", exc.dataset.encode()),
                ("valxel-field", exc.field.encode()),
                ("valxel-rule", exc.rule.encode()),
                ("valxel-offset", str(msg.offset()).encode()),
            ],
        )
        self.producer.flush(10)
        count(self.dataset, "parked")
        log.error(
            "knowledge.event_parked dataset=%s field=%s breaks the data contract (%s) offset=%s",
            exc.dataset,
            exc.field,
            exc.rule,
            msg.offset(),
        )

    def behind(self) -> int:
        total = 0
        for tp in self.consumer.assignment():
            _low, high = self.consumer.get_watermark_offsets(tp, timeout=5, cached=False)
            [pos] = self.consumer.position([tp])
            total += max(0, high - pos.offset) if pos.offset >= 0 else high
        return total


def _partition(msg: Any) -> Any:
    from confluent_kafka import TopicPartition

    return TopicPartition(msg.topic(), msg.partition(), msg.offset())


class ScanRunner(threading.Thread):
    """One scan source into one index version, every `every_minutes`."""

    def __init__(
        self,
        ingestor: Ingestor,
        source: Source,
        version: IndexVersion,
        *,
        rows: Callable[[Source], Iterable[dict[str, Any]]],
    ) -> None:
        super().__init__(daemon=True, name=f"scan-{source.name}-v{version.version}")
        self.ingestor, self.source, self.version, self.rows = ingestor, source, version, rows
        self.stop = threading.Event()

    def run(self) -> None:
        every = float(self.source.options.get("every_minutes", 15)) * 60
        while not self.stop.is_set():
            self.step()
            self.stop.wait(every)

    def step(self) -> None:
        try:
            changed, deleted = self.ingestor.scan(self.source, self.rows(self.source), self.version)
            log.info("%s: scan indexed %d changed records and deleted %d", self.source.name, changed, deleted)
        except Exception:
            log.exception("%s: scan failed; next try at the next interval", self.source.name)


def scan_rows(source: Source) -> Iterable[dict[str, Any]]:
    """The scan's query against the source's database (read-only: the query is in knowledge.yaml)."""
    import psycopg
    from psycopg.rows import dict_row

    dsn = os.environ.get(str(source.options["dsn_env"]), "")
    if not dsn:
        raise RuntimeError(f"{source.name}: {source.options['dsn_env']} is not set")
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        conn.read_only = True
        yield from conn.execute(str(source.options["query"]))


class Worker:
    """Starts and stops runners as index versions come and go (checked every 30 seconds)."""

    def __init__(
        self,
        ingestor: Ingestor,
        *,
        name: str,
        kafka: dict[str, Any],
        consumer_factory: Callable[[dict[str, Any]], Any] | None = None,
        producer_factory: Callable[[dict[str, Any]], Any] | None = None,
        rows: Callable[[Source], Iterable[dict[str, Any]]] = scan_rows,
    ) -> None:
        self.ingestor, self.name, self.kafka, self.rows = ingestor, name, kafka, rows
        self.consumer_factory = consumer_factory or _kafka_consumer
        self.producer_factory = producer_factory
        self.describe: Callable[[str], str] = lambda topic: _cleanup_policy(kafka, topic)
        self.runners: dict[tuple[str, int], EventsRunner | ScanRunner] = {}
        self.stop = threading.Event()
        self._warned: dict[str, float] = {}

    def report_lag(self, now: float | None = None) -> list[str]:
        """Log knowledge.source_behind for each source of the active version more than an hour
        behind, at most every 10 minutes per source: the platform's knowledge-lag alert rule counts
        these lines. Returns the sources reported."""
        from .store import active

        v = active(self.ingestor.store)
        if v is None:
            return []
        now = time.time() if now is None else now
        out = []
        for name, lag in lag_seconds(self.ingestor.cfg, self.ingestor.store, v.version, now).items():
            if lag is not None and lag > LAG_ALERT_S and now - self._warned.get(name, 0.0) >= 600:
                self._warned[name] = now
                log.warning("knowledge.source_behind %s is %d minutes behind the product", name, lag // 60)
                out.append(name)
        return out

    def refresh(self) -> None:
        live = {v.version: v for v in live_versions(self.ingestor.store)}
        for (src, ver), r in list(self.runners.items()):
            if ver not in live:
                r.stop.set()
                del self.runners[(src, ver)]
        for v in live.values():
            for s in self.ingestor.cfg.sources:
                if not s.indexed or (s.name, v.version) in self.runners:
                    continue
                runner: EventsRunner | ScanRunner
                if s.kind == "events":
                    runner = EventsRunner(
                        self.ingestor,
                        s,
                        v,
                        group=f"{self.name}.{s.name}.v{v.version}",
                        consumer_factory=self.consumer_factory,
                        kafka=self.kafka,
                        producer_factory=self.producer_factory,
                    )
                else:
                    runner = ScanRunner(self.ingestor, s, v, rows=self.rows)
                self.runners[(s.name, v.version)] = runner
                runner.start()

    def start(self) -> threading.Thread:
        def loop() -> None:
            try:
                check_topics(self.ingestor.cfg, self.ingestor.store, self.describe)
            except Exception:
                log.exception("Ingestion: couldn't check the topics' cleanup policies")
            beat = 0
            while not self.stop.is_set():
                try:
                    self.refresh()
                    self.report_lag()
                except Exception:
                    log.exception("Ingestion: couldn't read the index versions; trying again")
                beat += 1
                if beat % 2 == 0:  # once a minute: the line the platform reconciles in staging
                    line = summary_line()
                    if line:
                        log.info(line)
                self.stop.wait(30)

        t = threading.Thread(target=loop, daemon=True, name="ingest-supervisor")
        t.start()
        return t


def _cleanup_policy(kafka: dict[str, Any], topic: str) -> str:
    from confluent_kafka.admin import AdminClient, ConfigResource  # type: ignore[attr-defined]

    admin = AdminClient(kafka)
    res = ConfigResource(ConfigResource.Type.TOPIC, topic)
    configs = admin.describe_configs([res])[res].result(timeout=15)
    return str(configs["cleanup.policy"].value)


def _kafka_producer(conf: dict[str, Any]) -> Any:
    from confluent_kafka import Producer

    return Producer({**conf, "enable.idempotence": True, "acks": "all"})


def _kafka_consumer(conf: dict[str, Any]) -> Any:
    from confluent_kafka import Consumer

    return Consumer(conf)
