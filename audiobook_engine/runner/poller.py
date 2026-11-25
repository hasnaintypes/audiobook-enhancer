"""Colab-side job poller (gap.md Step 9 follow-up): the actual worker for
the polling-worker execution model api/ was built against (resolves PRD
Open Assumption #6 -- the API never holds a live link to this process, so
an MCP/Colab disconnect here has zero effect on job state in Postgres).

Claims a queued row from the real jobs table (api/db.py's
PostgresJobStore), downloads+processes it with the existing engine
(ingestion/youtube.py, runner/job_runner.py, stitching/stitch.py -- none
reimplemented here), and reports the real outcome back. Uses the job's own
Postgres UUID as the object-storage job_id, which is the missing link
gap.md flagged between the two id spaces.

process_one_job/run_poller_loop are impure (network/ffmpeg/DB) and
excluded from the unit suite, same convention as run_job/stitch_job/
mix_job. Only the pure validate_job_before_processing is unit-tested here.
"""

from __future__ import annotations

import time
from pathlib import Path

from api.db import JobStore
from api.models import Job


def validate_job_before_processing(job: Job) -> str | None:
    """Return an error message if this job can't be processed yet, else
    None. add_music=True has no music-source mechanism anywhere in the
    system yet (no upload endpoint, no fixed track) -- fail fast with a
    specific reason rather than silently ignoring the flag or guessing at
    a source that isn't this poller's decision to invent.
    """
    if job.add_music:
        return (
            "Music mixing requested (addMusic=true) but no music-source mechanism "
            "exists yet (no upload endpoint or fixed track) -- not yet supported."
        )
    return None


def process_one_job(
    job_store: JobStore,
    work_dir: Path,
    preset_tier_caps: dict | None = None,
    tier2_venv_python: Path | None = None,
) -> bool:
    """Claim and process at most one queued job. Returns False if nothing
    was queued, True if a job was claimed (whether it succeeded, failed
    validation, or failed processing -- all three count as "handled").
    """
    from audiobook_engine.config.presets import resolve_max_tier
    from audiobook_engine.ingestion.youtube import extract_audio
    from audiobook_engine.runner.job_runner import run_job
    from audiobook_engine.stitching.stitch import stitch_job

    job = job_store.claim_next_queued_job()
    if job is None:
        return False

    validation_error = validate_job_before_processing(job)
    if validation_error is not None:
        job_store.fail_job(job.id, validation_error)
        return True

    job_id = str(job.id)
    try:
        extracted = extract_audio(job.source_url, work_dir)
        max_tier = resolve_max_tier(job.preset.value, preset_tier_caps)
        run_job(
            job_id,
            extracted.audio_path,
            work_dir,
            max_tier=max_tier,
            tier2_venv_python=tier2_venv_python,
        )
        stitch_job(job_id, work_dir)
        job_store.succeed_job(job.id)
    except Exception as exc:  # noqa: BLE001 -- any processing failure must reach Postgres, not crash the poller
        job_store.fail_job(job.id, str(exc))

    return True


def run_poller_loop(
    job_store: JobStore,
    work_dir: Path,
    poll_interval_s: float = 10.0,
    max_iterations: int | None = None,
) -> None:
    """The actual Colab-cell entrypoint -- runs until max_iterations (for a
    bounded smoke-test cell) or forever (a long-running cell) if None.
    """
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        handled = process_one_job(job_store, work_dir)
        iterations += 1
        if not handled:
            time.sleep(poll_interval_s)
