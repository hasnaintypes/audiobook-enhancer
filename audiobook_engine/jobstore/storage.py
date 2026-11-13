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

import json
import tempfile
from pathlib import Path

from audiobook_engine.chunking.manifest import ChunkManifest, manifest_from_dict, manifest_to_dict

BUCKET_NAME = "audiobooks"


def object_key(job_id: str, kind: str, filename: str) -> str:
    """Build a consistent storage key for an artifact belonging to a job.

    kind is one of: "original", "chunks", "enhanced", "music_mixed",
    "manifest". Keeping this as a pure function makes the naming convention
    testable without any network access.
    """
    if not job_id or "/" in job_id:
        raise ValueError(f"Invalid job_id: {job_id!r}")
    allowed_kinds = {"original", "chunks", "enhanced", "music_mixed", "manifest"}
    if kind not in allowed_kinds:
        raise ValueError(f"Unknown artifact kind: {kind!r}, expected one of {allowed_kinds}")
    return f"jobs/{job_id}/{kind}/{filename}"


def get_s3_client():
    """Return a boto3 S3 client configured for Neon Object Storage.

    Path-style addressing AND SigV4 are both required by Neon (see
    neon-object-storage skill: "Path-style addressing and SigV4 only").
    Without pinning signature_version explicitly, boto3 generated SigV2-
    style presigned URLs (AWSAccessKeyId/Signature/Expires params) that
    Neon rejected with AccessDenied — confirmed by an actual failed
    download, not just a theoretical mismatch.
    """
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
    )


def upload_file(local_path: Path, key: str) -> None:
    client = get_s3_client()
    client.upload_file(str(local_path), BUCKET_NAME, key)


def download_file(key: str, local_path: Path) -> None:
    client = get_s3_client()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(BUCKET_NAME, key, str(local_path))


def delete_file(key: str) -> None:
    client = get_s3_client()
    client.delete_object(Bucket=BUCKET_NAME, Key=key)


def presigned_url(key: str, expires_in: int = 3600) -> str:
    client = get_s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET_NAME, "Key": key},
        ExpiresIn=expires_in,
    )


def upload_manifest(manifest: ChunkManifest) -> None:
    """Serialize and upload a job's chunk-completion manifest as
    jobs/<job_id>/manifest/chunks.json — the interim per-chunk bookkeeping
    docs/07-Decisions.md D007 defers a real Postgres schema for until
    Phase 6 (see chunking/manifest.py's module docstring).
    """
    key = object_key(manifest.job_id, "manifest", "chunks.json")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(manifest_to_dict(manifest), f)
        temp_path = Path(f.name)
    try:
        upload_file(temp_path, key)
    finally:
        temp_path.unlink(missing_ok=True)


def download_manifest(job_id: str) -> ChunkManifest:
    """Download and parse a job's chunk-completion manifest. Raises
    ValueError (via manifest_from_dict) if the manifest is missing expected
    keys or malformed — e.g. from a mid-write interruption.
    """
    key = object_key(job_id, "manifest", "chunks.json")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = Path(f.name)
    try:
        download_file(key, temp_path)
        with open(temp_path) as fh:
            data = json.load(fh)
    finally:
        temp_path.unlink(missing_ok=True)
    return manifest_from_dict(data)
