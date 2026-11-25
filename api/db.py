"""Real Postgres-backed job store (docs/07-Decisions.md D004/D007's deferred
job-lifecycle schema). Mirrors the pure-logic/impure-I/O split used
throughout audiobook_engine (e.g. jobstore/storage.py's boto3 calls,
runner/job_runner.py's run_job): the JobStore protocol below is the
testable interface api/routes.py depends on; PostgresJobStore is the real
implementation, excluded from the unit suite (real network/DB) like
storage.py's upload_file/download_file already are.

psycopg is imported lazily inside each method, not at module level, so this
module (and anything that merely imports it, like api/main.py) doesn't
require psycopg to be installed just to be imported -- same lazy-import
convention as stitch_job/run_job/mix_job's soundfile/boto3 imports.
"""

from __future__ import annotations

import os
from typing import Protocol
from uuid import UUID

from api.models import EnhancementPreset, Job, JobStatus


class JobStore(Protocol):
    def insert_job(
        self, source_url: str, preset: EnhancementPreset, add_music: bool, idempotency_key: str
    ) -> Job: ...

    def get_job(self, job_id: UUID) -> Job | None: ...

    def update_status(self, job_id: UUID, status: JobStatus) -> Job | None: ...


def _get_connection():
    import psycopg
    import psycopg.rows

    database_url = os.environ["DATABASE_URL"]
    return psycopg.connect(database_url, row_factory=psycopg.rows.dict_row)


class PostgresJobStore:
    """Real implementation, backed by Neon Postgres (D007). Requires
    DATABASE_URL in the environment -- not covered by tests/unit, verified
    live against the real dev database instead (gap.md evidence).
    """

    def insert_job(
        self, source_url: str, preset: EnhancementPreset, add_music: bool, idempotency_key: str
    ) -> Job:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO jobs (source_url, preset, add_music, idempotency_key)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (idempotency_key) DO NOTHING
                    RETURNING *
                    """,
                    (source_url, preset.value, add_music, idempotency_key),
                )
                row = cur.fetchone()
                if row is None:
                    # Same idempotency key as an existing job -- return that
                    # job, not a duplicate (idempotent submit).
                    cur.execute(
                        "SELECT * FROM jobs WHERE idempotency_key = %s", (idempotency_key,)
                    )
                    row = cur.fetchone()
                conn.commit()
                return Job.from_row(row)

    def get_job(self, job_id: UUID) -> Job | None:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM jobs WHERE id = %s", (str(job_id),))
                row = cur.fetchone()
                return Job.from_row(row) if row is not None else None

    def update_status(self, job_id: UUID, status: JobStatus) -> Job | None:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs SET status = %s, updated_at = now()
                    WHERE id = %s
                    RETURNING *
                    """,
                    (status.value, str(job_id)),
                )
                row = cur.fetchone()
                conn.commit()
                return Job.from_row(row) if row is not None else None
