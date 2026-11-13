"""Per-job chunk-completion manifest — the interim resumability bookkeeping.

docs/07-Decisions.md D007 defers the real Postgres job-lifecycle schema to
Phase 6; docs/02-Architecture.md anticipates "chunk-level already-processed
markers" as the interim mechanism in the meantime. This module is that
interim mechanism: one small JSON manifest per job, stored as an object in
Neon Object Storage (kind="manifest") rather than a database row.

The pure logic here (ChunkManifest, manifest_to_dict/from_dict,
pending_chunk_indices, mark_chunk_complete) is fully unit-tested. The
boto3-touching upload_manifest/download_manifest helpers live in
jobstore/storage.py alongside the other artifact I/O, and are excluded
from unit tests like the rest of that module's real network calls.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkManifest:
    job_id: str
    total_chunks: int
    completed_indices: frozenset[int]


def manifest_to_dict(manifest: ChunkManifest) -> dict:
    return {
        "job_id": manifest.job_id,
        "total_chunks": manifest.total_chunks,
        "completed_indices": sorted(manifest.completed_indices),
    }


def manifest_from_dict(data: dict) -> ChunkManifest:
    """Raises ValueError on malformed/missing keys — this boundary must
    tolerate a half-written or corrupt manifest after an interruption.
    """
    try:
        job_id = data["job_id"]
        total_chunks = data["total_chunks"]
        completed_indices = data["completed_indices"]
    except KeyError as exc:
        raise ValueError(f"Malformed manifest, missing key: {exc}") from exc

    if not isinstance(job_id, str) or not job_id:
        raise ValueError(f"Invalid job_id in manifest: {job_id!r}")
    if not isinstance(total_chunks, int) or total_chunks < 0:
        raise ValueError(f"Invalid total_chunks in manifest: {total_chunks!r}")
    if not isinstance(completed_indices, list) or not all(
        isinstance(i, int) for i in completed_indices
    ):
        raise ValueError(f"Invalid completed_indices in manifest: {completed_indices!r}")

    return ChunkManifest(
        job_id=job_id,
        total_chunks=total_chunks,
        completed_indices=frozenset(completed_indices),
    )


def pending_chunk_indices(manifest: ChunkManifest) -> list[int]:
    """Indices in range(total_chunks) not yet completed, in order."""
    return [i for i in range(manifest.total_chunks) if i not in manifest.completed_indices]


def mark_chunk_complete(manifest: ChunkManifest, index: int) -> ChunkManifest:
    """Return a NEW ChunkManifest with index added — pure, immutable,
    idempotent (marking an already-completed index again is a no-op, since
    a crash-then-retry could re-report the same chunk done).
    """
    if not 0 <= index < manifest.total_chunks:
        raise ValueError(
            f"Chunk index {index} out of range for total_chunks={manifest.total_chunks}"
        )
    return ChunkManifest(
        job_id=manifest.job_id,
        total_chunks=manifest.total_chunks,
        completed_indices=manifest.completed_indices | {index},
    )
