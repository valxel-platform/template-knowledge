"""knowledge.yaml: what the product's agents may search, where it comes from, and who may see it.

    index:
      embedding_model: text-embedding-3-small   # through the gateway; the product decides at G4
      dimensions: 1536
      chunking: { max_chars: 1200, version: 1 }
    sources:
      - name: notes                     # the source's name; source ids are <name>:<record id>
        title: Clinical notes
        from: { events: { topic: emr.notes.v1 } }        # or { scan: {...} }, or { files: {...} }
        id: note_id                     # the record's id field
        text: [body]                    # fields whose text is indexed, in order
        heading: title                  # the passage title (optional)
        date: signed_at                 # the record's date (optional): search filters on it
        record_type: note
        link: "https://emr.example/notes/{note_id}"     # where a person opens it (optional)
        tenant: clinic_id               # the tenant (organization) field; required when multi-tenant
        allow: ["team:{care_team_id}", "user:{author_id}"]   # who may see it; empty: the whole tenant
        exclude: [phone, ssn]           # fields never indexed, even if named in text

Source kinds:
  events   a Kafka topic of CloudEvents; `delete_types` (default: types ending in .deleted) remove
           the record, every other type replaces it.
  scan     a query against a read-only database (`dsn_env` names the variable holding its URL),
           run every `every_minutes`: new and changed rows are indexed, rows gone are deleted.
           For a document store that keeps text but publishes no events.
  files    a store of files (PDFs, scans) that isn't indexed: list_sources names it, so an agent
           can say it can't search those rather than answer as if they don't exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

NAME = re.compile(r"^[a-z][a-z0-9_-]{1,39}$")
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
PRINCIPAL = re.compile(r"^(user|role|team):")
# Field names that hold card data: a payments product never indexes them (VX: card data stays put).
CARD_FIELD = re.compile(r"(card|pan|cvv|cvc|expiry)", re.IGNORECASE)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Chunking:
    max_chars: int = 1200
    version: int = 1


@dataclass(frozen=True)
class IndexSpec:
    embedding_model: str
    dimensions: int
    chunking: Chunking = Chunking()


@dataclass(frozen=True)
class Source:
    name: str
    title: str
    kind: str  # events | scan | files
    options: dict[str, Any]
    id: str = "id"
    text: tuple[str, ...] = ()
    heading: str = ""
    date: str = ""
    record_type: str = ""
    link: str = ""
    tenant: str = ""
    allow: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    @property
    def indexed(self) -> bool:
        return self.kind != "files"

    def is_delete(self, event_type: str) -> bool:
        types = self.options.get("delete_types")
        return event_type in types if types else event_type.endswith(".deleted")


@dataclass(frozen=True)
class Config:
    index: IndexSpec
    sources: tuple[Source, ...] = field(default_factory=tuple)

    def source(self, name: str) -> Source:
        for s in self.sources:
            if s.name == name:
                return s
        raise KeyError(name)


def load(path: str | Path, *, data_class: str = "general", multi_tenant: bool = True) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return parse(raw, data_class=data_class, multi_tenant=multi_tenant)


def parse(raw: dict[str, Any], *, data_class: str = "general", multi_tenant: bool = True) -> Config:
    problems: list[str] = []
    idx = raw.get("index") or {}
    model = str(idx.get("embedding_model") or "")
    if not model:
        problems.append("index.embedding_model is required")
    dims = idx.get("dimensions")
    if not isinstance(dims, int) or not 8 <= dims <= 4096:
        problems.append("index.dimensions must be a whole number between 8 and 4096")
        dims = 0
    ch = idx.get("chunking") or {}
    chunking = Chunking(int(ch.get("max_chars", 1200)), int(ch.get("version", 1)))
    if not 200 <= chunking.max_chars <= 8000:
        problems.append("index.chunking.max_chars must be between 200 and 8000")
    sources: list[Source] = []
    seen: set[str] = set()
    for i, s in enumerate(raw.get("sources") or []):
        where = f"sources[{i}]"
        name = str(s.get("name") or "")
        if not NAME.match(name):
            problems.append(f"{where}.name must be lowercase letters, digits, - or _ (got {name!r})")
        elif name in seen:
            problems.append(f"{where}: two sources are named {name}")
        seen.add(name)
        frm = s.get("from") or {}
        kinds = [k for k in ("events", "scan", "files") if k in frm]
        if len(kinds) != 1:
            problems.append(f"{name}: `from` needs exactly one of events, scan or files")
            continue
        kind = kinds[0]
        opts = dict(frm[kind] or {})
        if kind == "events" and not opts.get("topic"):
            problems.append(f"{name}: events need a topic")
        if kind == "scan":
            if not (opts.get("dsn_env") and opts.get("query")):
                problems.append(f"{name}: a scan needs dsn_env and query")
            opts.setdefault("every_minutes", 15)
        text = tuple(str(t) for t in (s.get("text") or []))
        if kind != "files" and not text:
            problems.append(f"{name}: name the fields whose text is indexed (text)")
        exclude = tuple(str(t) for t in (s.get("exclude") or []))
        if overlap := sorted(set(text) & set(exclude)):
            problems.append(f"{name}: {', '.join(overlap)} is both indexed and excluded")
        if data_class == "payments" and (cards := [t for t in text if CARD_FIELD.search(t)]):
            problems.append(f"{name}: a payments product never indexes card data ({', '.join(cards)})")
        tenant = str(s.get("tenant") or "")
        if multi_tenant and kind != "files" and not tenant:
            problems.append(f"{name}: name the tenant field (or set KNOWLEDGE_TENANT_CLAIM empty)")
        allow = tuple(str(a) for a in (s.get("allow") or []))
        for a in allow:
            if not PRINCIPAL.match(a):
                problems.append(f"{name}: allow entries start with user:, role: or team: (got {a!r})")
        sources.append(
            Source(
                name=name,
                title=str(s.get("title") or name),
                kind=kind,
                options=opts,
                id=str(s.get("id") or "id"),
                text=text,
                heading=str(s.get("heading") or ""),
                date=str(s.get("date") or ""),
                record_type=str(s.get("record_type") or ""),
                link=str(s.get("link") or ""),
                tenant=tenant,
                allow=allow,
                exclude=exclude,
            )
        )
    if not sources:
        problems.append("name at least one source")
    if problems:
        raise ConfigError("knowledge.yaml: " + "; ".join(problems))
    return Config(IndexSpec(model, int(dims), chunking), tuple(sources))


def fill(template: str, record: dict[str, Any]) -> str:
    """A link or principal with {field} placeholders filled from the record; "" if one is missing."""
    missing = False

    def sub(m: re.Match[str]) -> str:
        nonlocal missing
        v = record.get(m.group(1))
        if v in (None, ""):
            missing = True
            return ""
        return str(v)

    out = PLACEHOLDER.sub(sub, template)
    return "" if missing else out
