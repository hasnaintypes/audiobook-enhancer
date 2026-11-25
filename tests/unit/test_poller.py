from datetime import datetime, timezone
from uuid import uuid4

from api.models import EnhancementPreset, Job, JobStatus
from audiobook_engine.runner.poller import validate_job_before_processing


def _job(add_music: bool) -> Job:
    now = datetime.now(timezone.utc)
    return Job(
        id=uuid4(),
        status=JobStatus.QUEUED,
        progress=0,
        sourceUrl="https://youtu.be/example",
        preset=EnhancementPreset.BALANCED,
        addMusic=add_music,
        error=None,
        createdAt=now,
        updatedAt=now,
    )


def test_validate_job_before_processing_allows_job_without_music():
    job = _job(add_music=False)

    assert validate_job_before_processing(job) is None


def test_validate_job_before_processing_rejects_add_music_job():
    job = _job(add_music=True)

    error = validate_job_before_processing(job)

    assert error is not None
    assert "music" in error.lower()
