import pytest

from audiobook_engine.chunking.manifest import (
    ChunkManifest,
    manifest_from_dict,
    manifest_to_dict,
    mark_chunk_complete,
    pending_chunk_indices,
)


def _manifest(total_chunks: int = 5, completed: frozenset[int] = frozenset()) -> ChunkManifest:
    return ChunkManifest(job_id="sample-001", total_chunks=total_chunks, completed_indices=completed)


def test_manifest_round_trips_through_dict():
    manifest = _manifest(total_chunks=4, completed=frozenset({0, 2}))
    data = manifest_to_dict(manifest)
    restored = manifest_from_dict(data)
    assert restored == manifest


def test_pending_indices_excludes_completed():
    manifest = _manifest(total_chunks=5, completed=frozenset({1, 3}))
    assert pending_chunk_indices(manifest) == [0, 2, 4]


def test_pending_indices_all_pending_when_none_completed():
    manifest = _manifest(total_chunks=3, completed=frozenset())
    assert pending_chunk_indices(manifest) == [0, 1, 2]


def test_pending_indices_empty_when_all_completed():
    manifest = _manifest(total_chunks=3, completed=frozenset({0, 1, 2}))
    assert pending_chunk_indices(manifest) == []


def test_mark_chunk_complete_is_idempotent():
    manifest = _manifest(total_chunks=3, completed=frozenset({0}))
    once = mark_chunk_complete(manifest, 1)
    twice = mark_chunk_complete(once, 1)
    assert once == twice
    assert once.completed_indices == frozenset({0, 1})


def test_mark_chunk_complete_rejects_out_of_range_index():
    manifest = _manifest(total_chunks=3)
    with pytest.raises(ValueError):
        mark_chunk_complete(manifest, 3)
    with pytest.raises(ValueError):
        mark_chunk_complete(manifest, -1)


def test_manifest_from_dict_rejects_malformed_data():
    with pytest.raises(ValueError):
        manifest_from_dict({"job_id": "sample-001", "total_chunks": 3})  # missing key

    with pytest.raises(ValueError):
        manifest_from_dict(
            {"job_id": "sample-001", "total_chunks": "not-an-int", "completed_indices": []}
        )

    with pytest.raises(ValueError):
        manifest_from_dict(
            {"job_id": "", "total_chunks": 3, "completed_indices": []}
        )
