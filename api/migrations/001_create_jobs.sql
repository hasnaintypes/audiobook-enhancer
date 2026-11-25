-- Job-lifecycle schema (docs/07-Decisions.md D004/D007's deferred Postgres
-- schema, built here at Phase 6 as D007 always intended). One job per row;
-- the polling-worker model (gap.md Step 9) means a Colab-side worker reads
-- and updates these rows directly -- this API never holds a live link to
-- that worker.
--
-- Apply with a DIRECT (unpooled) connection, not the pooled app connection
-- string:
--   psql "$DATABASE_URL_UNPOOLED" -f api/migrations/001_create_jobs.sql

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    progress SMALLINT NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    source_url TEXT NOT NULL,
    preset TEXT NOT NULL CHECK (preset IN ('gentle', 'balanced', 'strong', 'custom')),
    add_music BOOLEAN NOT NULL DEFAULT FALSE,
    idempotency_key TEXT NOT NULL UNIQUE,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
