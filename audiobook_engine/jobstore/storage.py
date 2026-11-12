"""Object storage client — thin wrapper around boto3 pointed at Neon's
S3-compatible Object Storage (see neon.ts, bucket "audiobooks").

Reads AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_ENDPOINT_URL_S3 /
AWS_REGION from the environment (populated by `neon deploy`/`neon env pull`
into .env.local, or injected directly in Colab — see
notebooks/colab_bootstrap.ipynb). Never hardcode credentials here.

This module does real network I/O and is not covered by tests/unit;
key-naming logic is kept in pure functions below so that part IS testable.
"""

from __future__ import annotations

from pathlib import Path

BUCKET_NAME = "audiobooks"


def object_key(job_id: str, kind: str, filename: str) -> str:
    """Build a consistent storage key for an artifact belonging to a job.

    kind is one of: "original", "chunks", "enhanced", "music_mixed".
    Keeping this as a pure function makes the naming convention testable
    without any network access.
    """
    if not job_id or "/" in job_id:
        raise ValueError(f"Invalid job_id: {job_id!r}")
    allowed_kinds = {"original", "chunks", "enhanced", "music_mixed"}
    if kind not in allowed_kinds:
        raise ValueError(f"Unknown artifact kind: {kind!r}, expected one of {allowed_kinds}")
    return f"jobs/{job_id}/{kind}/{filename}"


def get_s3_client():
    """Return a boto3 S3 client configured for Neon Object Storage.

    Path-style addressing is required by Neon (see neon-object-storage
    skill) — the AWS SDK default (virtual-hosted-style) will NOT work.
    """
    import boto3
    from botocore.config import Config

    return boto3.client("s3", config=Config(s3={"addressing_style": "path"}))


def upload_file(local_path: Path, key: str) -> None:
    client = get_s3_client()
    client.upload_file(str(local_path), BUCKET_NAME, key)


def download_file(key: str, local_path: Path) -> None:
    client = get_s3_client()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(BUCKET_NAME, key, str(local_path))


def presigned_url(key: str, expires_in: int = 3600) -> str:
    client = get_s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET_NAME, "Key": key},
        ExpiresIn=expires_in,
    )
