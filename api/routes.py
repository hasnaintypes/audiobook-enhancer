"""The five job-lifecycle endpoints (docs/06-Roadmap.md Phase 6). Route
handlers depend on the JobStore protocol (api/db.py) and the artifact
lister (api/artifacts.py) via FastAPI Depends, so tests can override both
with fakes and exercise this whole layer without a real database or
network call -- the one abstraction this increment introduces, and it
exists specifically to keep this file unit-testable, not speculatively.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from api.artifacts import list_job_artifacts as _list_job_artifacts
from api.auth import require_api_key
from api.db import JobStore, PostgresJobStore
from api.models import (
    Job,
    JobArtifact,
    JobStatus,
    JobSubmitRequest,
    PreviewRequest,
    PreviewResult,
    can_cancel,
)

router = APIRouter(dependencies=[Depends(require_api_key)])

_default_job_store = PostgresJobStore()


def get_job_store() -> JobStore:
    return _default_job_store


def get_artifact_lister():
    return _list_job_artifacts


def _get_job_or_404(job_store: JobStore, job_id: UUID) -> Job:
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@router.post("/jobs", response_model=Job)
def submit_job(payload: JobSubmitRequest, job_store: JobStore = Depends(get_job_store)) -> Job:
    return job_store.insert_job(
        source_url=payload.source_url,
        preset=payload.preset,
        add_music=payload.add_music,
        idempotency_key=payload.idempotency_key,
    )


@router.get("/jobs/{job_id}", response_model=Job)
def get_job(job_id: UUID, job_store: JobStore = Depends(get_job_store)) -> Job:
    return _get_job_or_404(job_store, job_id)


@router.post("/jobs/{job_id}/cancel", response_model=Job)
def cancel_job(job_id: UUID, job_store: JobStore = Depends(get_job_store)) -> Job:
    job = _get_job_or_404(job_store, job_id)
    if not can_cancel(job.status):
        raise HTTPException(
            status_code=409, detail=f"Job {job_id} is already {job.status.value}, cannot cancel"
        )
    cancelled = job_store.update_status(job_id, status=JobStatus.CANCELLED)
    assert cancelled is not None  # job existed one line above; not deleted between calls
    return cancelled


@router.get("/jobs/{job_id}/artifacts", response_model=list[JobArtifact])
def get_job_artifacts(
    job_id: UUID,
    job_store: JobStore = Depends(get_job_store),
    artifact_lister=Depends(get_artifact_lister),
) -> list[JobArtifact]:
    _get_job_or_404(job_store, job_id)
    return artifact_lister(str(job_id))


@router.post("/previews")
def create_preview(payload: PreviewRequest) -> PreviewResult:
    raise HTTPException(
        status_code=501,
        detail=(
            "Preview generation needs real audio processing (yt-dlp/ffmpeg), which this "
            "API cannot do under the polling-worker execution model (gap.md Step 9) -- "
            "not yet implemented, pending a preview-worker design."
        ),
    )
