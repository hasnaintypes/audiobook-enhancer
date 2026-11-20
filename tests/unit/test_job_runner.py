import numpy as np
import pytest

from audiobook_engine.chunking.boundaries import ChunkBoundary
from audiobook_engine.chunking.manifest import ChunkManifest
from audiobook_engine.runner.job_runner import (
    chunk_filename,
    resolve_manifest,
    slice_chunk_samples,
)

SAMPLE_RATE = 16_000


def test_chunk_filename_is_zero_padded():
    assert chunk_filename(0) == "chunk_0000.wav"
    assert chunk_filename(7) == "chunk_0007.wav"
    assert chunk_filename(123) == "chunk_0123.wav"


def test_chunk_filename_sorts_lexicographically_in_numeric_order():
    names = [chunk_filename(i) for i in [10, 2, 1, 20]]
    assert sorted(names) == [chunk_filename(i) for i in [1, 2, 10, 20]]


def test_slice_chunk_samples_extracts_expected_range():
    samples = np.arange(SAMPLE_RATE * 10, dtype=np.float32)
    boundary = ChunkBoundary(start_s=2.0, end_s=5.0, cut_reason="silence_gap")

    chunk = slice_chunk_samples(samples, SAMPLE_RATE, boundary)

    assert len(chunk) == 3 * SAMPLE_RATE
    assert chunk[0] == samples[2 * SAMPLE_RATE]
    assert chunk[-1] == samples[5 * SAMPLE_RATE - 1]


def test_slice_chunk_samples_covers_full_audio_with_no_gaps_or_overlaps():
    samples = np.arange(SAMPLE_RATE * 9, dtype=np.float32)
    boundaries = [
        ChunkBoundary(start_s=0.0, end_s=3.0, cut_reason="max_duration_fallback"),
        ChunkBoundary(start_s=3.0, end_s=6.0, cut_reason="max_duration_fallback"),
        ChunkBoundary(start_s=6.0, end_s=9.0, cut_reason="end_of_audio"),
    ]

    chunks = [slice_chunk_samples(samples, SAMPLE_RATE, b) for b in boundaries]

    assert np.array_equal(np.concatenate(chunks), samples)


def test_resolve_manifest_starts_fresh_when_none_exists():
    manifest = resolve_manifest("job-1", total_chunks=4, existing=None)

    assert manifest == ChunkManifest(job_id="job-1", total_chunks=4, completed_indices=frozenset())


def test_resolve_manifest_resumes_matching_existing_manifest():
    existing = ChunkManifest(job_id="job-1", total_chunks=4, completed_indices=frozenset({0, 1}))

    manifest = resolve_manifest("job-1", total_chunks=4, existing=existing)

    assert manifest == existing


def test_resolve_manifest_rejects_total_chunks_mismatch():
    existing = ChunkManifest(job_id="job-1", total_chunks=4, completed_indices=frozenset({0}))

    with pytest.raises(ValueError):
        resolve_manifest("job-1", total_chunks=5, existing=existing)


def test_resolve_manifest_rejects_job_id_mismatch():
    existing = ChunkManifest(job_id="job-1", total_chunks=4, completed_indices=frozenset())

    with pytest.raises(ValueError):
        resolve_manifest("job-2", total_chunks=4, existing=existing)
