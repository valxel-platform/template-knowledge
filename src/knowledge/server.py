"""Retrieval as MCP tools: what agents call to answer from the product's own records.

    search         a question, optional filters, k (default 8, at most 20) -> passages with
                   source ids, titles, dates, links and scores; only what the caller may see
    get_document   a source id from a search result -> the record's text the caller may see
    list_sources   what the index holds and how fresh each source is, so an agent can say what
                   it can't know (a source of files that isn't indexed, a source that is behind)

The conventions of the Valxel MCP server template hold: read-only annotations, a scope on every
tool, one audit record per call (source ids returned; the question only as a hash), usage metrics,
rate limits, tenants from the token. Agents cite the source ids they use.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from mcp.server import MCPServer
from mcp.server.auth.provider import TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from .audit import Auditor, make_sink
from .config import Config
from .ingest import LAG_ALERT_S, lag_seconds
from .models import ModelUnavailable
from .passages import Filters, Viewer
from .retriever import NoIndex, Retriever
from .security import RateLimiter, caller, guarded, principals, tenants
from .settings import Settings

READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
DOCUMENT_CHARS = 20_000  # get_document returns at most this much text


class PassageOut(BaseModel):
    source_id: str = Field(
        description="Cite this: <source>:<record id>. Pass it to get_document for the whole record."
    )
    passage: int
    title: str
    text: str
    date: str | None = None
    record_type: str | None = None
    link: str | None = None
    score: float


class SearchResult(BaseModel):
    passages: list[PassageOut]
    index_version: int
    notes: list[str] = Field(
        default_factory=list, description="What this search couldn't cover, if anything."
    )


class DocumentOut(BaseModel):
    source_id: str
    title: str
    date: str | None = None
    link: str | None = None
    text: str
    truncated: bool


class SourceOut(BaseModel):
    name: str
    title: str
    indexed: bool
    records: int = Field(description="Records in the index that you may see")
    lag_seconds: float | None = Field(
        description="How far the index trails the product; null: not ingested yet"
    )
    note: str = ""


class Sources(BaseModel):
    sources: list[SourceOut]
    index_version: int | None


def make_auditor(settings: Settings) -> Auditor:
    sink = make_sink(
        settings.audit_sink,
        bootstrap=settings.audit_kafka_bootstrap,
        topic=settings.audit_topic,
        config=dict(settings.audit_kafka_config),
        data_class=settings.data_class,
    )
    return Auditor(
        sink,
        service=settings.name,
        product=settings.product,
        environment=settings.environment,
        data_class=settings.data_class,
        required=settings.audit_required,
    )


def _notes(cfg: Config, lags: dict[str, float | None], wanted: tuple[str, ...]) -> list[str]:
    out = []
    for s in cfg.sources:
        if wanted and s.name not in wanted:
            continue
        if not s.indexed:
            out.append(
                f"{s.title} ({s.name}) holds files that aren't searchable; say so rather than assume they "
                "say nothing."
            )
        elif (lag := lags.get(s.name)) is None:
            out.append(f"{s.title} ({s.name}) isn't indexed yet.")
        elif lag > LAG_ALERT_S:
            out.append(
                f"{s.title} ({s.name}) is {round(lag / 60)} minutes behind the product: "
                "recent changes may be "
                "missing."
            )
    return out


def build_server(
    settings: Settings,
    *,
    verifier: TokenVerifier,
    cfg: Config,
    retriever: Retriever,
    auditor: Auditor | None = None,
    clock: Callable[[], float] = time.time,
) -> MCPServer:
    auditor = auditor or make_auditor(settings)
    limiter = RateLimiter(settings.rate_per_minute, settings.rate_burst)
    read = guarded(settings.read_scope, limiter, auditor)
    names = tuple(s.name for s in cfg.sources)
    server = MCPServer(
        settings.name,
        instructions=(
            "Search the product's own records and documents. Cite the source_id of every passage you use. "
            "Call list_sources to see what is covered and how fresh it is."
        ),
        token_verifier=verifier,
        auth=AuthSettings(
            issuer_url=settings.oidc_issuer or "https://issuer.invalid",  # type: ignore[arg-type]
            resource_server_url=settings.resource_url,  # type: ignore[arg-type]
            required_scopes=[settings.read_scope],
            validate_token_resource=True,
        ),
    )

    def viewer() -> Viewer:
        """Who is asking: their tenants and principals, from their token. Searches see only those."""
        claims: dict[str, Any] = getattr(caller(), "claims", None) or {}
        tok = caller()
        mine: frozenset[str] | None = None
        if settings.tenant_claim:
            found = tenants(claims, settings.tenant_claim)
            if not found:
                raise ToolError("Your token names no organization, so no records are available to you.")
            mine = frozenset(found)
        return Viewer(
            mine,
            frozenset(
                principals(claims, tok.subject or tok.client_id, settings.roles_claim, settings.teams_claim)
            ),
        )

    def lags() -> tuple[int | None, dict[str, float | None]]:
        try:
            v = retriever.version()
        except NoIndex:
            return None, {}
        return v.version, lag_seconds(cfg, retriever.store, v.version, clock())

    @server.tool(annotations=READ)
    @read
    def search(
        question: str = Field(
            min_length=3, max_length=500, description="What you want to find, in plain words"
        ),
        sources: list[str] = Field(  # noqa: B008  (pydantic reads it as the parameter's schema)
            default_factory=list,
            max_length=20,
            description="Only these sources; list_sources names them",
        ),
        since: str = Field(
            default="",
            pattern=r"^$|^\d{4}-\d{2}-\d{2}$",
            description="Records dated on or after (YYYY-MM-DD)",
        ),
        until: str = Field(
            default="",
            pattern=r"^$|^\d{4}-\d{2}-\d{2}$",
            description="Records dated on or before (YYYY-MM-DD)",
        ),
        record_type: str = Field(default="", max_length=60),
        k: int = Field(default=8, ge=1, le=20, description="How many passages"),
    ) -> SearchResult:
        """Search the product's records and documents. Returns the passages that best answer the question,
        only from records you may see. Cite each passage's source_id."""
        unknown = sorted(set(sources) - set(names))
        if unknown:
            raise ToolError(f"No source named {', '.join(unknown)}. Call list_sources for the names.")
        who = viewer()
        try:
            hits, v = retriever.search(question, who, Filters(tuple(sources), since, until, record_type), k)
        except NoIndex as exc:
            raise ToolError(str(exc)) from exc
        except ModelUnavailable as exc:
            raise ToolError(f"Search is unavailable right now ({exc}). Try again shortly.") from exc
        _, lag = lags()
        return SearchResult(
            passages=[
                PassageOut(
                    source_id=h.passage.source_id,
                    passage=h.passage.n,
                    title=h.passage.title,
                    text=h.passage.text,
                    date=h.passage.dated or None,
                    record_type=h.passage.record_type or None,
                    link=h.passage.link or None,
                    score=round(h.score, 4),
                )
                for h in hits
            ],
            index_version=v.version,
            notes=_notes(cfg, lag, tuple(sources)),
        )

    @server.tool(annotations=READ)
    @read
    def get_document(
        source_id: str = Field(
            pattern=r"^[a-z][a-z0-9_-]{1,39}:.+$",
            max_length=300,
            description="A source_id from a search result",
        ),
    ) -> DocumentOut:
        """The text of one record, as far as you may see it. Use it when a passage isn't enough."""
        source, record_id = source_id.split(":", 1)
        if source not in names:
            raise ToolError(f"No source named {source}. Call list_sources for the names.")
        try:
            parts = retriever.document(source, record_id, viewer())
        except NoIndex as exc:
            raise ToolError(str(exc)) from exc
        if not parts:  # missing and not yours look the same
            raise ToolError(f"No record {source_id}. Use search to find source ids.")
        text = "\n\n".join(p.text for p in parts)
        return DocumentOut(
            source_id=source_id,
            title=parts[0].title,
            date=parts[0].dated or None,
            link=parts[0].link or None,
            text=text[:DOCUMENT_CHARS],
            truncated=len(text) > DOCUMENT_CHARS,
        )

    @server.tool(annotations=READ)
    @read
    def list_sources() -> Sources:
        """What the index holds, how many records you may see in each source, and how fresh each one is."""
        who = viewer()
        version, lag = lags()
        counts = retriever.store.counts(version, who) if version is not None else {}
        out = []
        for s in cfg.sources:
            note = "Files, not indexed: they can't be searched." if not s.indexed else ""
            if s.indexed and (lg := lag.get(s.name)) is not None and lg > LAG_ALERT_S:
                note = f"{round(lg / 60)} minutes behind the product."
            out.append(
                SourceOut(
                    name=s.name,
                    title=s.title,
                    indexed=s.indexed,
                    records=counts.get(s.name, (0, 0))[0],
                    lag_seconds=lag.get(s.name) if s.indexed else None,
                    note=note,
                )
            )
        return Sources(sources=out, index_version=version)

    return server
