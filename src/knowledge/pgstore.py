"""The passages store in Postgres (schema.sql): pgvector for meaning, full-text search for words.

Reads run as knowledge_reader with the caller's tenants and principals set for the transaction, so
row-level security applies on top of the query's own filters; writes run as knowledge_writer. The
login must be a member of the role it uses (in production the tools' login is a reader only).

Filtered vector search uses pgvector's iterative index scan, so a filter that hides most of the
nearest passages still returns `limit` results when there are that many the caller may see.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from importlib import resources
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .passages import Filters, Passage, Viewer
from .store import IndexVersion, words

SEP = "\x1f"


def migrate(dsn: str) -> None:
    """Create or update the schema, and let the migrating login use both roles (local and dev)."""
    ddl = resources.files("knowledge").joinpath("schema.sql").read_text()
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(ddl)
        conn.execute("grant knowledge_reader, knowledge_writer to current_user")


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


def _row(r: dict[str, Any]) -> Passage:
    d = r.get("dated")
    return Passage(
        source=r["source"],
        record_id=r["record_id"],
        n=r["n"],
        title=r["title"],
        text=r["body"],
        dated=d.isoformat() if isinstance(d, date) else (d or ""),
        record_type=r["record_type"],
        link=r["link"],
        tenant=r["tenant"],
        allow=tuple(r["allow"] or ()),
    )


COLS = sql.SQL("source, record_id, n, title, body, dated, record_type, link, tenant, allow")


class PgStore:
    def __init__(self, dsn: str, *, pool_size: int = 5) -> None:
        self.pool = ConnectionPool(
            dsn, min_size=1, max_size=pool_size, kwargs={"row_factory": dict_row}, open=True
        )
        self._iterative_scan: bool | None = None

    def close(self) -> None:
        self.pool.close()

    @contextmanager
    def _tx(self, role: str, viewer: Viewer | None = None) -> Iterator[psycopg.Cursor[Any]]:
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute(sql.SQL("set local role {}").format(sql.Identifier(role)))
            if viewer is not None:
                tenants = "*" if viewer.tenants is None else SEP.join(sorted(viewer.tenants))
                cur.execute(
                    "select set_config('knowledge.tenants', %s, true),"
                    " set_config('knowledge.principals', %s, true)",
                    (tenants, SEP.join(sorted(viewer.principals))),
                )
                # A filtered nearest-neighbour search must not come back short: pgvector 0.8 keeps
                # scanning the index until it has enough visible rows; older versions search exactly.
                if self._iterative():
                    cur.execute("select set_config('hnsw.iterative_scan', 'relaxed_order', true)")
                else:
                    cur.execute("select set_config('enable_indexscan', 'off', true)")
            yield cur

    def _iterative(self) -> bool:
        if self._iterative_scan is None:
            with self.pool.connection() as conn:
                row: Any = conn.execute(
                    "select extversion from pg_extension where extname = 'vector'"
                ).fetchone()
            parts = [int(x) for x in str((row or {}).get("extversion", "0")).split(".")[:2]]
            self._iterative_scan = tuple(parts) >= (0, 8)
        return self._iterative_scan

    # ------------------------------------------------------------------ versions
    def versions(self) -> list[IndexVersion]:
        with self._tx("knowledge_reader") as cur:
            cur.execute(
                "select version, model, dimensions, chunking, state from knowledge_index order by version"
            )
            return [IndexVersion(**r) for r in cur.fetchall()]

    def create_version(self, model: str, dimensions: int, chunking: int) -> IndexVersion:
        with self._tx("knowledge_writer") as cur:
            cur.execute("lock table knowledge_index in exclusive mode")
            cur.execute(
                "insert into knowledge_index (version, model, dimensions, chunking, state)"
                " select coalesce(max(version), 0) + 1, %s, %s, %s, 'building' from knowledge_index"
                " returning version, model, dimensions, chunking, state",
                (model, dimensions, chunking),
            )
            v = IndexVersion(**cur.fetchone())  # type: ignore[arg-type]
        # The version's own nearest-neighbour index (outside the transaction: it's the table owner's job).
        name = sql.Identifier(f"knowledge_passage_hnsw_v{v.version}")
        with self.pool.connection() as conn:
            conn.execute(
                sql.SQL(
                    "create index if not exists {} on knowledge_passage using hnsw"
                    " ((embedding::vector({})) vector_cosine_ops) where version = {}"
                ).format(name, sql.Literal(v.dimensions), sql.Literal(v.version))
            )
        return v

    def activate(self, version: int) -> None:
        with self._tx("knowledge_writer") as cur:
            cur.execute("select 1 from knowledge_index where version = %s", (version,))
            if cur.fetchone() is None:
                raise KeyError(f"No index version {version}")
            cur.execute(
                "update knowledge_index set state = 'retired' where state = 'active' and version <> %s",
                (version,),
            )
            cur.execute("update knowledge_index set state = 'active' where version = %s", (version,))

    def drop(self, version: int) -> None:
        with self._tx("knowledge_writer") as cur:
            cur.execute("select state from knowledge_index where version = %s", (version,))
            row = cur.fetchone()
            if row and row["state"] == "active":
                raise ValueError("Activate another version before dropping the active one")
            cur.execute("delete from knowledge_index where version = %s", (version,))
        with self.pool.connection() as conn:
            conn.execute(
                sql.SQL("drop index if exists {}").format(
                    sql.Identifier(f"knowledge_passage_hnsw_v{version}")
                )
            )

    # ------------------------------------------------------------------ writes
    def replace_record(
        self,
        version: int,
        source: str,
        record_id: str,
        passages: list[Passage],
        vectors: list[list[float]],
        fingerprint: str,
    ) -> None:
        with self._tx("knowledge_writer") as cur:
            cur.execute(
                "delete from knowledge_passage where version = %s and source = %s and record_id = %s",
                (version, source, record_id),
            )
            for p, v in zip(passages, vectors, strict=True):
                cur.execute(
                    "insert into knowledge_passage (version, source, record_id, n, title, body, dated,"
                    " record_type, link, tenant, allow, embedding)"
                    " values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)",
                    (
                        version,
                        source,
                        record_id,
                        p.n,
                        p.title,
                        p.text,
                        p.dated or None,
                        p.record_type,
                        p.link,
                        p.tenant,
                        list(p.allow),
                        _vec(v),
                    ),
                )
            cur.execute(
                "insert into knowledge_record (version, source, record_id, fingerprint)"
                " values (%s, %s, %s, %s)"
                " on conflict (version, source, record_id) do update set fingerprint = excluded.fingerprint",
                (version, source, record_id, fingerprint),
            )

    def delete_record(self, version: int, source: str, record_id: str) -> None:
        with self._tx("knowledge_writer") as cur:
            cur.execute(
                "delete from knowledge_passage where version = %s and source = %s and record_id = %s",
                (version, source, record_id),
            )
            cur.execute(
                "delete from knowledge_record where version = %s and source = %s and record_id = %s",
                (version, source, record_id),
            )

    def fingerprints(self, version: int, source: str) -> dict[str, str]:
        with self._tx("knowledge_writer") as cur:
            cur.execute(
                "select record_id, fingerprint from knowledge_record where version = %s and source = %s",
                (version, source),
            )
            return {r["record_id"]: r["fingerprint"] for r in cur.fetchall()}

    # ------------------------------------------------------------------ reads
    @staticmethod
    def _where(version: int, viewer: Viewer, filters: Filters) -> tuple[sql.Composable, list[Any]]:
        parts: list[sql.Composable] = [sql.SQL("version = %s")]
        args: list[Any] = [version]
        # The query's own access filter; row-level security applies the same rule again.
        if viewer.tenants is not None:
            parts.append(sql.SQL("tenant = any(%s)"))
            args.append(sorted(viewer.tenants))
        parts.append(sql.SQL("(cardinality(allow) = 0 or allow && %s::text[])"))
        args.append(sorted(viewer.principals))
        if filters.sources:
            parts.append(sql.SQL("source = any(%s)"))
            args.append(list(filters.sources))
        if filters.record_type:
            parts.append(sql.SQL("record_type = %s"))
            args.append(filters.record_type)
        if filters.since:
            parts.append(sql.SQL("dated >= %s"))
            args.append(filters.since)
        if filters.until:
            parts.append(sql.SQL("dated <= %s"))
            args.append(filters.until)
        return sql.SQL(" and ").join(parts), args

    def keyword(
        self, version: int, query: str, viewer: Viewer, filters: Filters, limit: int
    ) -> list[tuple[Passage, float]]:
        terms = sorted(set(words(query)))
        if not terms:
            return []
        where, args = self._where(version, viewer, filters)
        q = sql.SQL(
            "select {cols}, ts_rank_cd(tsv, to_tsquery('english', %s)) as score from knowledge_passage"
            " where {where} and tsv @@ to_tsquery('english', %s)"
            " order by score desc, source, record_id, n limit %s"
        ).format(cols=COLS, where=where)
        expr = " | ".join(terms)
        with self._tx("knowledge_reader", viewer) as cur:
            cur.execute(q, [expr, *args, expr, limit])
            return [(_row(r), float(r["score"])) for r in cur.fetchall()]

    def nearest(
        self, version: int, vector: list[float], viewer: Viewer, filters: Filters, limit: int
    ) -> list[tuple[Passage, float]]:
        dims = sql.Literal(len(vector))
        where, args = self._where(version, viewer, filters)
        q = sql.SQL(
            "select {cols}, 1 - (embedding::vector({d}) <=> %s::vector({d})) as score from knowledge_passage"
            " where {where} order by embedding::vector({d}) <=> %s::vector({d}) limit %s"
        ).format(cols=COLS, d=dims, where=where)
        v = _vec(vector)
        with self._tx("knowledge_reader", viewer) as cur:
            cur.execute(q, [v, *args, v, limit])
            return [(_row(r), float(r["score"])) for r in cur.fetchall()]

    def document(self, version: int, source: str, record_id: str, viewer: Viewer) -> list[Passage]:
        where, args = self._where(version, viewer, Filters(sources=(source,)))
        q = sql.SQL(
            "select {cols} from knowledge_passage where {where} and record_id = %s order by n"
        ).format(cols=COLS, where=where)
        with self._tx("knowledge_reader", viewer) as cur:
            cur.execute(q, [*args, record_id])
            return [_row(r) for r in cur.fetchall()]

    def counts(self, version: int, viewer: Viewer) -> dict[str, tuple[int, int]]:
        """Records and passages per source, of those the caller may see."""
        where, args = self._where(version, viewer, Filters())
        q = sql.SQL(
            "select source, count(distinct record_id) as records, count(*) as passages from knowledge_passage"
            " where {where} group by source"
        ).format(where=where)
        with self._tx("knowledge_reader", viewer) as cur:
            cur.execute(q, args)
            return {r["source"]: (int(r["records"]), int(r["passages"])) for r in cur.fetchall()}

    def set_source_state(self, source: str, **state: object) -> None:
        with self._tx("knowledge_writer") as cur:
            cur.execute(
                "insert into knowledge_source_state (source, state) values (%s, %s::jsonb)"
                " on conflict (source) do update set state = knowledge_source_state.state || excluded.state,"
                " updated_at = now()",
                (source, json.dumps(state, default=str)),
            )

    def source_states(self) -> dict[str, dict[str, object]]:
        with self._tx("knowledge_reader") as cur:
            cur.execute("select source, state from knowledge_source_state")
            return {r["source"]: dict(r["state"]) for r in cur.fetchall()}
