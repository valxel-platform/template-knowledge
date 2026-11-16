-- The passages store: Postgres with pgvector and full-text search. Run by `knowledge-admin migrate`
-- (idempotent). Access is enforced twice: by the queries (store.py) and by row-level security here,
-- so a bug in the tools still can't return another tenant's or another team's passages.
--
-- Two roles, granted to logins by whoever runs the database:
--   knowledge_reader  the MCP tools: SELECT only, and only the rows the caller may see
--   knowledge_writer  the ingestion worker: reads and writes everything
-- In production give the tools' login knowledge_reader only.

create extension if not exists vector;

do $$ begin
  if not exists (select 1 from pg_roles where rolname = 'knowledge_reader') then
    create role knowledge_reader nologin;
  end if;
  if not exists (select 1 from pg_roles where rolname = 'knowledge_writer') then
    create role knowledge_writer nologin;
  end if;
end $$;

create table if not exists knowledge_index (
  version     integer primary key,
  model       text not null,
  dimensions  integer not null,
  chunking    integer not null,
  state       text not null check (state in ('building', 'active', 'retired')),
  created_at  timestamptz not null default now()
);
create unique index if not exists knowledge_index_one_active on knowledge_index (state) where state = 'active';

create table if not exists knowledge_passage (
  version     integer not null references knowledge_index (version) on delete cascade,
  source      text not null,
  record_id   text not null,
  n           integer not null,
  title       text not null default '',
  body        text not null,
  dated       date,
  record_type text not null default '',
  link        text not null default '',
  tenant      text not null default '',
  allow       text[] not null default '{}',
  embedding   vector not null,  -- dimensions vary by version; each version has its own HNSW index
  tsv         tsvector generated always as (to_tsvector('english', title || ' ' || body)) stored,
  primary key (version, source, record_id, n)
);
create index if not exists knowledge_passage_tsv on knowledge_passage using gin (tsv);
create index if not exists knowledge_passage_tenant on knowledge_passage (version, tenant);

-- What each record looked like when indexed, so a scan re-indexes only what changed.
create table if not exists knowledge_record (
  version     integer not null references knowledge_index (version) on delete cascade,
  source      text not null,
  record_id   text not null,
  fingerprint text not null,
  primary key (version, source, record_id)
);

create table if not exists knowledge_source_state (
  source      text primary key,
  state       jsonb not null default '{}',
  updated_at  timestamptz not null default now()
);

alter table knowledge_passage enable row level security;
alter table knowledge_passage force row level security;

-- Settings are per transaction (set_config(..., true)); a missing setting matches nothing.
-- Lists are joined with the unit separator, which never appears in a name.
drop policy if exists knowledge_read on knowledge_passage;
create policy knowledge_read on knowledge_passage for select to knowledge_reader using (
  (current_setting('knowledge.tenants', true) = '*'
   or tenant = any (string_to_array(current_setting('knowledge.tenants', true), E'\x1f')))
  and (cardinality(allow) = 0
   or allow && string_to_array(current_setting('knowledge.principals', true), E'\x1f'))
);
drop policy if exists knowledge_write on knowledge_passage;
create policy knowledge_write on knowledge_passage for all to knowledge_writer using (true) with check (true);

grant select on knowledge_index, knowledge_passage, knowledge_source_state to knowledge_reader;
grant select, insert, update, delete on knowledge_index, knowledge_passage, knowledge_record,
  knowledge_source_state to knowledge_writer;
