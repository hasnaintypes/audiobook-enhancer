"""Lists a job's real artifacts in Neon Object Storage, for GET
/jobs/{id}/artifacts. Reuses audiobook_engine/jobstore/storage.py's existing
get_s3_client/BUCKET_NAME/object_key/presigned_url rather than
reimplementing storage access -- this module only adds "which of these
known kinds actually exist for this job" and the JobArtifact shape.

Excluded from the unit suite (real network), like storage.py's own
boto3-touching functions.
"""

from __future__ import annotations

from api.models import JobArtifact

# Kinds that represent a real deliverable a user would want to download --
# "original"/"chunks"/"manifest" are internal bookkeeping, not artifacts.
_ARTIFACT_KINDS = ("enhanced", "music_mixed")
_ARTIFACT_FILENAME = "final.wav"


def list_job_artifacts(job_id: str) -> list[JobArtifact]:
    from botocore.exceptions import ClientError

    from audiobook_engine.jobstore.storage import (
        BUCKET_NAME,
        get_s3_client,
        object_key,
        presigned_url,
    )

    client = get_s3_client()
    artifacts: list[JobArtifact] = []
    for kind in _ARTIFACT_KINDS:
        key = object_key(job_id, kind, _ARTIFACT_FILENAME)
        try:
            head = client.head_object(Bucket=BUCKET_NAME, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                continue
            raise
        artifacts.append(
            JobArtifact(
                name=f"{kind}/{_ARTIFACT_FILENAME}",
                url=presigned_url(key),
                sizeBytes=head["ContentLength"],
            )
        )
    return artifacts
