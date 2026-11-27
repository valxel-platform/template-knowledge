# template-knowledge

The Valxel template for a **knowledge** repo: retrieval over a product's own records, so its agents
answer from them and cite what they used. It keeps an index of passages from the product's events
(or a scan of a document store) in the product's Postgres, and serves it as MCP tools over
Streamable HTTP at `/mcp`:

| Tool | Takes | Returns |
| --- | --- | --- |
| `search` | a question; optional `sources`, `since`, `until`, `record_type`; `k` (default 8, at most 20) | passages with a `source_id` to cite, title, date, link and score; only from records the caller may see; `notes` on what the search couldn't cover |
| `get_document` | a `source_id` from a search | the record's text, as far as the caller may see it |
| `list_sources` | nothing | each source, whether it's searchable, how many records the caller may see in it, and how far it trails the product |

The example indexes an EMR: clinical notes and care-team messages from events, clinic policies
from a document store, and scanned referral forms named but not indexed. Valxel writes your
product's `knowledge.yaml` and golden set during the build; people approve them at G5.

## How it fits together

```
product events ──> ingestion worker ──> passages (Postgres, pgvector + full text, row-level security)
document store ──> (scan, every 15 min)       ▲ embeddings through the LLM gateway
                                              │
agent ──(person's token)──> search tools ─────┘ filtered by the caller's clinic and teams
```

- **What is indexed** is `knowledge.yaml` (`src/knowledge/config.py` has the reference). Each
  source names where records come from (`events`, `scan` or `files`), the id and text fields, the
  tenant field, an `allow` list built from the record's own access fields
  (`team:{care_team}`, `user:{author}`), and fields that are never indexed (`exclude`).
- **Who sees what** is decided per passage, inside the search query: the caller's tenants (the
  token's organization claim) and principals (`user:<sub>`, `role:<role>`, `team:<group>`) must
  match the passage's tenant and `allow` list. Postgres enforces the same rule with row-level
  security (`schema.sql`), so a bug in the tools still can't return another clinic's notes. A
  record whose access fields are missing is visible to nobody, not to everyone.
- **Search is hybrid.** Full-text search catches record numbers, drug and product codes and names.
  Vector search catches the same thing said in other words. The two are merged by reciprocal rank,
  and the gateway's reranker orders the best of them; if the reranker is down, the merged order
  stands. Filtered vector search never comes back short (pgvector 0.8's iterative index scan; older
  versions search exactly).
- **Freshness.** Each live index version has its own consumer per events source
  (`<name>.<source>.v<version>`), from the topic's start, committing only once a record is stored.
  A deletion removes the record's passages, and it's never answered again. A scan re-indexes only
  rows whose content changed and deletes rows that are gone. A source more than an hour behind logs
  `knowledge.source_behind`, which the platform's `knowledge-lag` alert rule turns into Needs
  attention, and `list_sources` and search results say so.
- **Data classes.** Embeddings and reranking go through the product's LLM gateway, which applies
  the data class: PHI only to a provider under the BAA or the self-hosted model (decided at G4). A
  payments product never indexes card fields (`knowledge.yaml` is refused) and card numbers are
  removed from any text.
- **The gateway** must serve `POST /v1/embeddings` (OpenAI-style) and `POST /v1/rerank`
  (Cohere-style). Without a gateway, local runs use words-only hash embeddings; deployed
  environments refuse to start without one.

## Index versions

An index version is one embedding model and one chunking version. To change either, edit
`knowledge.yaml` and run `knowledge-admin rebuild`: the ingestion worker fills the new version
beside the active one, from every source. Then run the retrieval evals against it, and
`knowledge-admin activate <n>` once it has caught up and the evals pass. `knowledge-admin drop <n>`
removes a retired version. `knowledge-admin status` shows each version and each source's lag. Run
these as the database owner; `knowledge-admin migrate` creates the schema.

A rebuild replays each events source's topic from its start, so the topic must keep every record.
At G4 the architecture agent decides, per source and with people approving, between a compacted
topic and a scan. A compacted topic means each event carries the whole record keyed by its id, a
deletion is a `.deleted` event, and Kafka keeps each record's latest event. The provider that
publishes the topic is told to keep it that way during the build. `knowledge.yaml` marks it
`compacted: true` and refuses an events source without it.

At start, the worker reads each topic's `cleanup.policy`. A topic that isn't compacted logs
`knowledge.topic_not_compacted`, `list_sources` says a rebuild would miss older records, and
`knowledge-admin rebuild` refuses until it's fixed (or `--force`). A source that can't meet this
should be a scan.

**Data contracts.** Every events source's topic has an approved data contract (ODCS v3, in the
product's governance repo). The SDLC tool writes the schemas this repo reads into `contracts/`
(`<dataset>.json` and `lock.yaml`), and they ship in the image. The worker drops properties its
copy doesn't know (a newer, compatible producer), validates the rest, and parks an event that fails
on `<topic>.dlt`, with the field and rule in its headers and never the value. It counts every event
by dataset and outcome (`knowledge_dataset_events_total{outcome="applied|parked|skipped"}`) for the
platform's reconciliation. Kafka allows only consumer groups that start with the component's name,
so `name` in `knowledge.yaml` is the component's name.

## Evals

`evals/questions.yaml` is the golden set:
- recall questions with the passages that must come back;
- leak cases, asked as someone who must not see the answer;
- deletion cases.

`python -m evals.run` runs it against the sample corpus (`evals/corpus.jsonl`, invented records)
on every PR. That is the `retrieval` check, against Postgres in CI. Recall must reach the bar in
`questions.yaml`; any leak, any deleted record or a slow p95 fails it whatever the recall
(`report.py --must-pass`).

After the dev deploy, `--live` runs the same set with the real embedding model and reranker
through the gateway, as part of the integration suite. G6 shows both beside the last release's
numbers. Bars are starting values: tighten them, never loosen.

## Conventions every tool follows

The Valxel MCP server template's conventions hold here unchanged: every tool is read-only
(`readOnlyHint`), declares its scope (`knowledge:read`), is rate-limited per caller, traced
(OpenTelemetry), counted (`usage.py`) and audited (`audit.py`). The audit record names the caller
and the `source_id`s returned. The question itself is recorded only as `question_sha256`, so a
search can be matched to a later complaint without the audit trail holding PHI. Results say what to
do next and never echo the question.

## Auth

The server is an OAuth 2.0 resource server (MCP authorization spec):

- Tokens are JWTs from the product's IdP, checked for signature (JWKS), issuer, expiry and
  audience = this server's URL (`KNOWLEDGE_RESOURCE_URL`, RFC 8707). Tokens for another resource are refused.
- Calls without a token get 401 with a pointer to `/.well-known/oauth-protected-resource/mcp`,
  so MCP clients can discover the IdP. Tokens without the read scope get 403.
- Agents call with the person's delegated token (token exchange), so searches see what that person
  may see. A client-credentials token sees only passages open to its tenant.

## Ports

The API is on 8080, with `/healthz` and `/readyz` for Kubernetes' probes. `/metrics` is on the
management port, `MANAGEMENT_PORT` (8081), as Spring's actuator is in the Java templates. The
platform's metrics collector scrapes it there, and the product's mesh lets the collector reach that
port and no other. The API port serves no metrics.

## Run it

```
uv sync
uv run pytest                        # starts the server on a free port and calls it over HTTP
KNOWLEDGE_TEST_DATABASE_URL=postgresql://postgres@localhost/postgres uv run pytest   # and the Postgres store
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run python -m evals.run           # the golden set, offline
KNOWLEDGE_OIDC_ISSUER=... KNOWLEDGE_OIDC_JWKS_URL=... uv run knowledge-server
```

| Variable | Default | |
| --- | --- | --- |
| `KNOWLEDGE_NAME` | `knowledge` | Server name; agents see tools as `<name>__<tool>` |
| `KNOWLEDGE_RESOURCE_URL` | `http://localhost:8080/mcp` | This server's public URL, the token audience |
| `KNOWLEDGE_OIDC_ISSUER`, `KNOWLEDGE_OIDC_JWKS_URL` | | Required |
| `KNOWLEDGE_READ_SCOPE` | `knowledge:read` | |
| `KNOWLEDGE_TENANT_CLAIM` | `organization` | The token claim naming the caller's tenants. Empty for a single-tenant product |
| `KNOWLEDGE_ROLES_CLAIM`, `KNOWLEDGE_TEAMS_CLAIM` | `roles`, `groups` | Claims matched against `allow` (`role:`, `team:`); Keycloak's `realm_access.roles` too |
| `KNOWLEDGE_CONFIG_PATH` | `knowledge.yaml` | What is indexed and who may see it |
| `KNOWLEDGE_DATABASE_URL` | | Postgres with pgvector, as a login in `knowledge_reader` only (the tools). Unset: in memory, local runs only |
| `KNOWLEDGE_INGEST_DATABASE_URL` | | A login in `knowledge_writer`, for the ingestion worker |
| `KNOWLEDGE_GATEWAY_URL`, `KNOWLEDGE_GATEWAY_KEY` | | The product's LLM gateway: embeddings and reranking |
| `KNOWLEDGE_RERANK_MODEL` | `rerank` | The gateway's name for the reranker; empty turns reranking off |
| `KNOWLEDGE_CANDIDATES` | `50` | Keyword and vector candidates per search, before merging |
| `KNOWLEDGE_INGEST` | `false` | Run the ingestion worker in this process |
| `KNOWLEDGE_INGEST_KAFKA_BOOTSTRAP`, `KNOWLEDGE_INGEST_KAFKA_CONFIG` | | The product's Kafka, and extra client settings (JSON) |
| `KNOWLEDGE_RATE_PER_MINUTE`, `KNOWLEDGE_RATE_BURST` | `120`, `30` | Per caller, per replica |
| `KNOWLEDGE_ALLOWED_HOSTS` | `[]` | Host headers to accept (DNS rebinding protection); set in production |
| `KNOWLEDGE_PRODUCT`, `KNOWLEDGE_ENVIRONMENT`, `KNOWLEDGE_DATA_CLASS` | | Stamped on every audit record; `payments` removes card numbers; a deployed environment needs a database and a gateway |
| `KNOWLEDGE_AUDIT_SINK`, `KNOWLEDGE_AUDIT_KAFKA_BOOTSTRAP`, `KNOWLEDGE_AUDIT_TOPIC`, `KNOWLEDGE_AUDIT_KAFKA_CONFIG`, `KNOWLEDGE_AUDIT_REQUIRED` | `log`, , `audit.v1`, `{}`, `true` | As in the MCP server template |
| `KNOWLEDGE_FLAGS_URL`, `KNOWLEDGE_FLAGS_ENVIRONMENT`, `KNOWLEDGE_FLAGS_NAMESPACE`, `KNOWLEDGE_FLAGS_TOKEN` | | The product's Flipt, read through OpenFeature |
| `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES` | | Standard OpenTelemetry settings |

`/metrics` adds `knowledge_source_lag_seconds{source}` to the usage metrics.

## Who calls this server

`usage.py` records every tool call with two names for the caller, neither of them a person:

| Name | From | Example |
| --- | --- | --- |
| `client` | The OAuth client the token was issued to (`azp` or `client_id`) | `intake-agent`, `claude-ai` |
| `app` | The software, as it names itself: `_meta["io.modelcontextprotocol/clientInfo"]` on every request (protocol 2026-07-28 on), an older client's `initialize`, or else its User-Agent | `claude-code`, `cursor` |

Per call there is one log line, for example:

```json
{"message": "tools/call search", "mcp.tool": "search", "mcp.client.id": "intake-agent",
 "mcp.client.app": "claude-code", "mcp.client.version": "2.1.4", "event.outcome": "success",
 "event.duration_ms": 41.7, "trace.id": "..."}
```

The span carries `gen_ai.tool.name`, `mcp.client.id` and `mcp.client.app`, and `/metrics` (scraped
inside the cluster; the gateway routes only `/mcp` outside) has:

```
mcp_tool_calls_total{tool, client, app, outcome}          # outcome: success, denied, failure
mcp_tool_call_duration_seconds{tool, client, app}         # histogram
```

so "which clients call which tools, and how often do they fail?" is one PromQL query:
`sum by (client, app, tool, outcome) (increase(mcp_tool_calls_total[1d]))`. Each
`server/discover` or `initialize` is logged on `mcp.clients` with the client's name, version and
protocol version, which shows who still connects with an old client. At most 100 OAuth clients and
50 apps are named in the metrics; later ones count as `other`. Who did what to which records stays
in the audit trail, which names the person (`sub`) and is read under stricter rules.

## Errors and traces

Every response carries a W3C `traceresponse` header with its trace id (`errors.py`), which the web
and mobile apps put on the errors they report, so the SDLC tool sees a failure people hit and an
error here as one incident. With `SENTRY_DSN` set, unhandled exceptions and `ERROR` logs go to
Sentry, grouped into issues, with the exception type, stack, endpoint path, a `trace_id` tag and a
`component` tag (`component.py`, which the SDLC tool sets to the component's name when it creates a
repo from this template, via `.valxel/identity`)
only: never exception messages, formatted messages, local variables, queries, headers, bodies,
users or breadcrumbs. Also `ENVIRONMENT` and `SENTRY_RELEASE`. Unset, Sentry is off; logs, traces
and metrics stay in OpenChoreo. For PHI products use a Sentry plan under a BAA, or self-hosted Sentry.

## Errors and traces

Every response carries a W3C `traceresponse` header with its trace id (`errors.py`), which the web
and mobile apps put on the errors they report, so the SDLC tool sees a failure people hit and an
error here as one incident. With `SENTRY_DSN` set, unhandled exceptions and `ERROR` logs go to
Sentry, grouped into issues, with the exception type, stack, endpoint path, a `trace_id` tag and a
`component` tag (`component.py`, which the SDLC tool sets to the component's name when it creates a
repo from this template, via `.valxel/identity`)
only: never exception messages, formatted messages, local variables, queries, headers, bodies,
users or breadcrumbs. Also `ENVIRONMENT` and `SENTRY_RELEASE`. Unset, Sentry is off; logs, traces
and metrics stay in OpenChoreo. For PHI products use a Sentry plan under a BAA, or self-hosted Sentry.

## Delivery

CI (`ci.yml`) runs `unit` (lint, format, types, tests with coverage, the Postgres store against
pgvector), `retrieval` (the golden set), dependency review and an image build; CodeQL comes from
the repo's default setup. These are the required checks the Valxel SDLC sets on repos with the
`knowledge` role. After each deployment the SDLC sends
`valxel-deployed`, which runs `post-deploy.yml` (`deployed/smoke.py`) against the new release.

## Performance tests

When the Valxel SDLC labels a pull request `valxel:perf-smoke` (a tool, the store, security or the
runtime changes; or the reviewer saw a performance risk), `perf.yml` loads the tools in-process
with `perf/smoke.py` on main's code and the PR's, on one runner, and `perf/compare.py` posts
`perf-smoke / report`: failed when the PR is more than 20% slower than main at the 95th or 99th
percentile, its error rate rises, or it misses the budget (500 ms, 1 s, under 1% errors). `CALLS` in `perf/smoke.py` are a search and the
record behind its best passage. A knowledge server has no staging load
test (it needs delegated tokens); production timings and the rollout's comparison watch it.
