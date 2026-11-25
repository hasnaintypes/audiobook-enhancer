"""Pydantic request/response models for the job-lifecycle API.

Field names and JSON shapes match frontend/src/lib/api.ts EXACTLY -- that
file was written first, against docs/02-Architecture.md's planned contract,
before this backend existed, so it is the real contract to implement
against, not something to redesign. camelCase aliases are explicit per
field rather than an alias-generator, since there are only a handful of
fields and this project prefers explicit over clever (see CLAUDE.md).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EnhancementPreset(str, Enum):
    GENTLE = "gentle"
    BALANCED = "balanced"
    STRONG = "strong"
    CUSTOM = "custom"


_TERMINAL_STATUSES = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED})


def can_cancel(status: JobStatus) -> bool:
    """Only a job that hasn't already reached a terminal state can be
    cancelled -- same spirit as chunking/manifest.py's idempotent
    mark_chunk_complete: a no-op/rejected transition is not an error in the
    data, just a transition that was never valid.
    """
    return status not in _TERMINAL_STATUSES


class JobSubmitRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source_url: str = Field(alias="sourceUrl")
    preset: EnhancementPreset
    add_music: bool = Field(default=False, alias="addMusic")
    idempotency_key: str = Field(alias="idempotencyKey")


class Job(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: UUID
    status: JobStatus
    progress: int
    source_url: str = Field(alias="sourceUrl")
    preset: EnhancementPreset
    error: str | None = None
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    @classmethod
    def from_row(cls, row: dict) -> "Job":
        return cls.model_validate(
            {
                "id": row["id"],
                "status": row["status"],
                "progress": row["progress"],
                "sourceUrl": row["source_url"],
                "preset": row["preset"],
                "error": row["error"],
                "createdAt": row["created_at"],
                "updatedAt": row["updated_at"],
            }
        )


class JobArtifact(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    url: str
    size_bytes: int = Field(alias="sizeBytes")


class PreviewRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source_url: str = Field(alias="sourceUrl")
    preset: EnhancementPreset


class PreviewResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    original_url: str = Field(alias="originalUrl")
    enhanced_url: str = Field(alias="enhancedUrl")
