"""Serves the knowledge tools over Streamable HTTP at /mcp, with /healthz and /metrics for the
platform, and (KNOWLEDGE_INGEST=true) runs the ingestion worker in the same process."""

from __future__ import annotations

import json
import logging
from functools import cache
from pathlib import Path
from typing import Any

from mcp.server.transport_security import TransportSecuritySettings
from prometheus_client.core import GaugeMetricFamily
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import cohorts, errors, events, flags, management, telemetry, usage
from .config import Config, load
from .contracts import Contracts
from .ingest import Ingestor, Worker, lag_seconds
from .models import Embedder, GatewayEmbedder, GatewayReranker, HashEmbedder, Reranker
from .retriever import Retriever
from .security import JwtVerifier
from .server import build_server
from .settings import Settings
from .store import IndexVersion, MemoryStore, Store, active

log = logging.getLogger(__name__)
DEPLOYED = ("dev", "staging", "production")


def make_models(settings: Settings) -> tuple[Any, Reranker | None]:
    """Embedders per index version (each version has its own model) and the reranker. Without a
    gateway, words-only hash embeddings: for local runs, never in a deployed environment."""
    if not settings.gateway_url:
        if settings.environment in DEPLOYED:
            raise RuntimeError(
                "Set KNOWLEDGE_GATEWAY_URL: deployed environments embed through the LLM gateway"
            )

        @cache
        def local(model: str, dims: int) -> Embedder:
            return HashEmbedder(dims, model)

        return (lambda v: local(v.model, v.dimensions)), None

    @cache
    def gateway(model: str, dims: int) -> Embedder:
        return GatewayEmbedder(
            base_url=settings.gateway_url,
            api_key=settings.gateway_key,
            model=model,
            dimensions=dims,
            component=settings.name,
        )

    reranker = (
        GatewayReranker(
            base_url=settings.gateway_url,
            api_key=settings.gateway_key,
            model=settings.rerank_model,
            component=settings.name,
        )
        if settings.rerank_model
        else None
    )
    return (lambda v: gateway(v.model, v.dimensions)), reranker


def make_store(settings: Settings, dsn: str = "") -> Store:
    dsn = dsn or settings.database_url
    if dsn:
        from .pgstore import PgStore

        return PgStore(dsn)
    if settings.environment in DEPLOYED:
        raise RuntimeError("Set KNOWLEDGE_DATABASE_URL: deployed environments keep passages in Postgres")
    return MemoryStore()


def seed(
    store: Store, cfg: Config, embedder_for: Any, path: str | Path, *, payments: bool = False
) -> IndexVersion:
    """A fresh active version filled from a JSON-lines file of {"source", "record"} (local runs and evals)."""
    v = store.create_version(cfg.index.embedding_model, cfg.index.dimensions, cfg.index.chunking.version)
    ingestor = Ingestor(cfg, store, embedder_for, payments=payments)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            ingestor.upsert(cfg.source(row["source"]), row["record"], v)
    store.activate(v.version)
    return v


class LagCollector:
    """knowledge_source_lag_seconds{source}: how far each source's index trails the product."""

    def __init__(self) -> None:
        self.cfg: Config | None = None
        self.store: Store | None = None

    def collect(self) -> Any:
        import time

        g = GaugeMetricFamily(
            "knowledge_source_lag_seconds", "How far the active index trails the product", labels=["source"]
        )
        try:
            v = active(self.store) if self.store is not None else None
            if v is not None and self.cfg is not None and self.store is not None:
                for name, lag in lag_seconds(self.cfg, self.store, v.version, time.time()).items():
                    if lag is not None:
                        g.add_metric([name], lag)
        except Exception:
            log.exception("Couldn't read the sources' lag")
        yield g


LAG = LagCollector()
usage.registry.register(LAG)


def create_app(
    settings: Settings | None = None,
    verifier: Any = None,
    store: Store | None = None,
    auditor: Any = None,
    cfg: Config | None = None,
    retriever: Retriever | None = None,
    seed_path: str | Path | None = None,
) -> Any:
    settings = settings or Settings()
    telemetry.setup(settings.name)
    errors.start_errors()
    flags.configure(settings)
    cfg = cfg or load(
        settings.config_path, data_class=settings.data_class, multi_tenant=bool(settings.tenant_claim)
    )
    embedder_for, reranker = make_models(settings)
    store = store or make_store(settings)
    if seed_path:
        seed(store, cfg, embedder_for, seed_path, payments=settings.data_class == "payments")
    elif isinstance(store, MemoryStore) and active(store) is None:  # a local run: an empty index to start
        v = store.create_version(cfg.index.embedding_model, cfg.index.dimensions, cfg.index.chunking.version)
        store.activate(v.version)
    retriever = retriever or Retriever(store, embedder_for, reranker, candidates=settings.candidates)
    verifier = verifier or JwtVerifier(
        issuer=settings.oidc_issuer, jwks_url=settings.oidc_jwks_url, audience=settings.resource_url
    )
    server = build_server(settings, verifier=verifier, cfg=cfg, retriever=retriever, auditor=auditor)

    if settings.ingest:
        writer = make_store(settings, settings.ingest_database_url) if settings.ingest_database_url else store
        ingestor = Ingestor(
            cfg,
            writer,
            embedder_for,
            payments=settings.data_class == "payments",
            contracts=Contracts.load(settings.contracts_dir),
        )
        kafka = {"bootstrap.servers": settings.ingest_kafka_bootstrap, **settings.ingest_kafka_config}
        Worker(ingestor, name=settings.name, kafka=kafka).start()

    @server.custom_route("/healthz", methods=["GET"], include_in_schema=False)  # type: ignore[untyped-decorator]
    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    events.register(usage.registry)
    LAG.cfg, LAG.store = cfg, store

    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=bool(settings.allowed_hosts), allowed_hosts=settings.allowed_hosts
    )
    app = server.streamable_http_app(
        stateless_http=True, json_response=True, transport_security=security, host=settings.host
    )
    from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

    cohorts.start_reporting()
    events.start_reporting()
    return OpenTelemetryMiddleware(
        errors.TraceResponseMiddleware(cohorts.CohortMiddleware(usage.UsageMiddleware(app))),
        excluded_urls="healthz,metrics",
    )


def main() -> None:
    import uvicorn

    settings = Settings()
    telemetry.setup_logging(settings.name)
    app = create_app(settings)
    # /metrics on the management port (MANAGEMENT_PORT, 8081), apart from the API, as in the Java
    # templates: the mesh lets the data plane's metrics collector reach that port only.
    management.serve(usage.metrics_body, host=settings.host)
    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None)
