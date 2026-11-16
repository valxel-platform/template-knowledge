"""Index versions, from the command line (run as the database's owner, with KNOWLEDGE_DATABASE_URL).

knowledge-admin migrate          create or update the schema (schema.sql)
knowledge-admin status           the versions, and each source's records and lag
knowledge-admin rebuild          a new version from knowledge.yaml's embedding model and chunking;
                                 the ingestion worker fills it from every source
knowledge-admin activate N       switch searches to version N, once it has caught up and its
                                 retrieval evals pass (--force skips the catch-up check)
knowledge-admin drop N           delete a retired version and its passages
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import TextIO

from .config import load
from .ingest import lag_seconds
from .settings import Settings
from .store import Store


def main(argv: list[str] | None = None, *, store: Store | None = None, out: TextIO | None = None) -> int:
    ap = argparse.ArgumentParser(prog="knowledge-admin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate")
    sub.add_parser("status")
    sub.add_parser("rebuild")
    a = sub.add_parser("activate")
    a.add_argument("version", type=int)
    a.add_argument("--force", action="store_true")
    d = sub.add_parser("drop")
    d.add_argument("version", type=int)
    args = ap.parse_args(argv)
    write = (out or sys.stdout).write
    settings = Settings()
    if args.cmd == "migrate":
        from .pgstore import migrate

        migrate(settings.database_url)
        write("Schema is up to date.\n")
        return 0
    from .app import make_store

    st = store or make_store(settings)
    cfg = load(settings.config_path, data_class=settings.data_class, multi_tenant=bool(settings.tenant_claim))
    if args.cmd == "status":
        for v in st.versions():
            lags = lag_seconds(cfg, st, v.version, time.time())
            write(f"v{v.version} {v.state}: {v.model} ({v.dimensions} dims), chunking {v.chunking}\n")
            for name, lag in lags.items():
                write(f"  {name}: " + ("not ingested yet" if lag is None else f"{lag:.0f}s behind") + "\n")
        return 0
    if args.cmd == "rebuild":
        v = st.create_version(cfg.index.embedding_model, cfg.index.dimensions, cfg.index.chunking.version)
        write(
            f"Version {v.version} is building: the ingestion worker fills it from every source. "
            f"Activate it once it has caught up and its retrieval evals pass.\n"
        )
        return 0
    if args.cmd == "activate":
        if not args.force:
            behind = {
                k: v for k, v in lag_seconds(cfg, st, args.version, time.time()).items() if v is None or v > 0
            }
            if behind:
                write(
                    f"Version {args.version} hasn't caught up: {', '.join(sorted(behind))}. "
                    "Wait, or use --force.\n"
                )
                return 1
        st.activate(args.version)
        write(f"Searches now use version {args.version}.\n")
        return 0
    st.drop(args.version)
    write(f"Version {args.version} is dropped.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
