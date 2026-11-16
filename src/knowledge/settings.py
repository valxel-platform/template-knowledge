"""Configuration from the environment. Auth has no defaults: a server without an issuer refuses to
start rather than serve product data to anyone. What is indexed, and who may see it, is in
knowledge.yaml (config.py), reviewed like code."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KNOWLEDGE_")

    name: str = "knowledge"
    host: str = "0.0.0.0"  # noqa: S104  (inside the container)
    port: int = 8080
    # This server's public URL (the token audience / RFC 8707 resource).
    resource_url: str = "http://localhost:8080/mcp"
    oidc_issuer: str = ""
    oidc_jwks_url: str = ""
    # The token claim naming the caller's tenants (Keycloak organizations). Every search sees only
    # those tenants' passages. Empty: a single-tenant product, no limit.
    tenant_claim: str = "organization"
    # Claims that name the caller's roles and teams. A passage's `allow` list (knowledge.yaml) is
    # matched against user:<sub>, role:<role> and team:<team>.
    roles_claim: str = "roles"
    teams_claim: str = "groups"
    read_scope: str = "knowledge:read"
    # Per caller: calls per minute, and burst.
    rate_per_minute: int = 120
    rate_burst: int = 30
    # What to index and who may see it.
    config_path: str = "knowledge.yaml"
    # The passages store. Empty: in memory, filled with the sample corpus (local runs only).
    # The tools connect as a member of knowledge_reader (SELECT only; row-level security applies),
    # the ingestion worker as a member of knowledge_writer (schema.sql).
    database_url: str = ""
    ingest_database_url: str = ""
    # The product's LLM gateway: embeddings (/v1/embeddings) and reranking (/v1/rerank). Embedding
    # model and dimensions come from knowledge.yaml; the gateway applies the data-class policy.
    gateway_url: str = ""
    gateway_key: str = ""
    rerank_model: str = "rerank"  # the gateway's name for the reranker; empty turns reranking off
    # How many keyword and vector candidates each search takes before merging and reranking.
    candidates: int = 50
    # The ingestion worker: runs in the same process, consuming each events source.
    ingest: bool = False
    ingest_kafka_bootstrap: str = ""
    ingest_kafka_config: dict[str, str] = {}  # security.protocol, sasl.*: as for the audit topic
    # Audit trail (audit.py): "kafka" in deployed environments, "log" locally.
    product: str = ""  # e.g. emr
    environment: str = ""  # e.g. production
    data_class: str = "general"  # general | phi | payments
    audit_sink: str = "log"
    audit_kafka_bootstrap: str = ""
    audit_topic: str = "audit.v1"
    audit_kafka_config: dict[str, str] = {}  # extra librdkafka settings: security.protocol, sasl.*
    audit_required: bool = True  # a call whose record can't be published fails
    # Feature flags: the product's Flipt over OFREP. Unset, every flag reads as its default (off).
    flags_url: str = ""
    flags_environment: str = ""
    flags_namespace: str = ""
    flags_token: str = ""
    flags_timeout_s: float = 0.25
    flags_cache_s: float = 30.0
    # Hostnames clients may use (DNS rebinding protection); empty allows any, for local use.
    allowed_hosts: list[str] = []
