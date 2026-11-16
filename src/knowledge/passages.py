"""Passages: the pieces of a record that search returns, who may see each one, and how a record
becomes passages.

A record is split at its headings and paragraphs, packed up to `max_chars` each, so a passage is a
readable piece an agent can cite. Every passage keeps the record's tenant and `allow` list, so
access is decided per passage, inside the search query (store.py), never on results afterwards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from .config import Source, fill

HEADING = re.compile(r"^#{1,6}\s")
SENTENCE = re.compile(r"(?<=[.!?])\s+")
# 13 to 19 digits, optionally grouped by spaces or dashes: a card number if it passes Luhn.
CARD_NUMBER = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")


@dataclass(frozen=True)
class Passage:
    source: str
    record_id: str
    n: int
    title: str
    text: str
    dated: str = ""  # YYYY-MM-DD, or ""
    record_type: str = ""
    link: str = ""
    tenant: str = ""
    allow: tuple[str, ...] = ()

    @property
    def source_id(self) -> str:
        return f"{self.source}:{self.record_id}"


@dataclass(frozen=True)
class Viewer:
    """Who is asking: their tenants (None: a single-tenant product) and principals
    (user:<sub>, role:<role>, team:<team>)."""

    tenants: frozenset[str] | None
    principals: frozenset[str]

    def sees(self, p: Passage) -> bool:
        if self.tenants is not None and p.tenant not in self.tenants:
            return False
        return not p.allow or bool(self.principals.intersection(p.allow))


@dataclass(frozen=True)
class Filters:
    sources: tuple[str, ...] = ()
    since: str = ""  # YYYY-MM-DD, inclusive
    until: str = ""
    record_type: str = ""

    def match(self, p: Passage) -> bool:
        if self.sources and p.source not in self.sources:
            return False
        if self.record_type and p.record_type != self.record_type:
            return False
        if (self.since or self.until) and not p.dated:
            return False
        if self.since and p.dated < self.since:
            return False
        return not (self.until and p.dated > self.until)


def split(text: str, max_chars: int) -> list[str]:
    """Pieces of at most max_chars, broken at headings, then paragraphs, then sentences."""
    blocks: list[str] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = para.strip()
        if not para:
            continue
        # A heading starts a new block, so a section's title travels with its first paragraph.
        lines = para.splitlines()
        cur: list[str] = []
        for line in lines:
            if HEADING.match(line) and cur:
                blocks.append("\n".join(cur))
                cur = []
            cur.append(line)
        if cur:
            blocks.append("\n".join(cur))
    pieces: list[str] = []
    buf = ""
    for block in blocks:
        for part in _fit(block, max_chars):
            starts_section = bool(HEADING.match(part))
            if buf and (starts_section or len(buf) + 2 + len(part) > max_chars):
                pieces.append(buf)
                buf = ""
            buf = f"{buf}\n\n{part}" if buf else part
    if buf:
        pieces.append(buf)
    return pieces


def _fit(block: str, max_chars: int) -> list[str]:
    if len(block) <= max_chars:
        return [block]
    out: list[str] = []
    buf = ""
    for sentence in SENTENCE.split(block):
        while len(sentence) > max_chars:  # one very long sentence: cut it
            if buf:
                out.append(buf)
                buf = ""
            out.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if buf and len(buf) + 1 + len(sentence) > max_chars:
            out.append(buf)
            buf = ""
        buf = f"{buf} {sentence}" if buf else sentence
    if buf:
        out.append(buf)
    return out


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def redact_card_numbers(text: str) -> str:
    """Card numbers removed: a payments product never indexes them, even inside free text."""

    def sub(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return "[card number removed]" if _luhn(digits) else m.group(0)

    return CARD_NUMBER.sub(sub, text)


def _day(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, date):
        return value.isoformat()[:10]
    s = str(value)
    return s[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", s) else ""


def from_record(
    source: Source, record: dict[str, Any], *, max_chars: int, payments: bool = False
) -> list[Passage]:
    """The record's passages. Excluded fields never reach the text; for a payments product, card
    numbers are removed wherever they appear."""
    record_id = str(record.get(source.id) or "")
    if not record_id:
        raise ValueError(f"{source.name}: a record without its id field ({source.id})")
    kept = {k: v for k, v in record.items() if k not in source.exclude}
    parts = [str(kept[f]).strip() for f in source.text if kept.get(f) not in (None, "")]
    text = "\n\n".join(parts)
    if payments:
        text = redact_card_numbers(text)
    title = str(kept.get(source.heading) or "") if source.heading else ""
    allow = tuple(sorted({p for p in (fill(a, kept) for a in source.allow) if p}))
    if source.allow and not allow:
        # Access fields are missing: nobody may see it, rather than everyone in the tenant.
        allow = ("user:nobody",)
    tenant = str(kept.get(source.tenant) or "") if source.tenant else ""
    dated = _day(kept.get(source.date)) if source.date else ""
    link = fill(source.link, kept) if source.link else ""
    return [
        Passage(
            source=source.name,
            record_id=record_id,
            n=i,
            title=title,
            text=piece,
            dated=dated,
            record_type=source.record_type,
            link=link,
            tenant=tenant,
            allow=allow,
        )
        for i, piece in enumerate(split(text, max_chars))
    ]
