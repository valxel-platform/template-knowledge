"""Embeddings and reranking, always through the product's LLM gateway: the gateway decides which
providers the product's data class allows (PHI only under a BAA or to the self-hosted model;
payments never leave), meters spend and records usage.

The gateway is expected to serve the OpenAI-style embeddings route (POST /v1/embeddings) and the
Cohere-style rerank route (POST /v1/rerank). `HashEmbedder` needs no model at all: tests, evals on
every PR, and local runs use it.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Protocol

import httpx

WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: list[str], *, purpose: str) -> list[list[float]]: ...


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]: ...


class ModelUnavailable(RuntimeError):
    """The gateway refused or failed: the message says which, never what was sent."""


def _normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class HashEmbedder:
    """Words hashed into a fixed-size vector (the hashing trick). Not a language model: two texts
    are close when they share words. Deterministic and free, for tests and offline evals."""

    def __init__(self, dimensions: int = 256, model: str = "hash-v1") -> None:
        self.dimensions, self.model = dimensions, model

    def embed(self, texts: list[str], *, purpose: str) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * self.dimensions
            for word in WORD.findall(text.lower()):
                h = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big")
                v[h % self.dimensions] += 1.0 if (h >> 32) & 1 else -1.0
            out.append(_normalize(v))
        return out


class GatewayEmbedder:
    BATCH = 64

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int,
        component: str,
        client: httpx.Client | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("KNOWLEDGE_GATEWAY_KEY is required with KNOWLEDGE_GATEWAY_URL")
        self.model, self.dimensions, self.component = model, dimensions, component
        self.url = base_url.rstrip("/") + "/v1/embeddings"
        self.client = client or httpx.Client(timeout=timeout_s)
        self.headers = {"Authorization": f"Bearer {api_key}", "x-valxel-component": component}

    def embed(self, texts: list[str], *, purpose: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            batch = texts[i : i + self.BATCH]
            body = _post(
                self.client,
                self.url,
                {"model": self.model, "input": batch, "dimensions": self.dimensions},
                {**self.headers, "x-valxel-purpose": purpose},
            )
            rows = sorted(body.get("data") or [], key=lambda r: int(r.get("index", 0)))
            if len(rows) != len(batch):
                raise ModelUnavailable(f"The gateway returned {len(rows)} embeddings for {len(batch)} texts")
            for r in rows:
                vec = [float(x) for x in r["embedding"]]
                if len(vec) != self.dimensions:
                    raise ModelUnavailable(
                        f"{self.model} returned {len(vec)} dimensions; knowledge.yaml says {self.dimensions}"
                    )
                out.append(vec)
        return out


class GatewayReranker:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        component: str,
        client: httpx.Client | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        self.model = model
        self.url = base_url.rstrip("/") + "/v1/rerank"
        self.client = client or httpx.Client(timeout=timeout_s)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "x-valxel-component": component,
            "x-valxel-purpose": "rerank",
        }

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        body = _post(
            self.client,
            self.url,
            {"model": self.model, "query": query, "documents": documents, "top_n": top_n},
            self.headers,
        )
        return [(int(r["index"]), float(r.get("relevance_score", 0.0))) for r in body.get("results") or []]


def _post(client: httpx.Client, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    try:
        resp = client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise ModelUnavailable(f"The LLM gateway can't be reached ({type(exc).__name__})") from exc
    if resp.status_code >= 400:
        raise ModelUnavailable(f"The LLM gateway answered {resp.status_code} on {url.rsplit('/', 1)[-1]}")
    data: dict[str, Any] = resp.json()
    return data
