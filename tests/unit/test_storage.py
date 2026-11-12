import pytest

from audiobook_engine.jobstore.storage import object_key


def test_object_key_builds_expected_path():
    key = object_key("job-123", "original", "source.wav")
    assert key == "jobs/job-123/original/source.wav"


def test_object_key_rejects_invalid_kind():
    with pytest.raises(ValueError):
        object_key("job-123", "bogus", "source.wav")


def test_object_key_rejects_job_id_with_slash():
    with pytest.raises(ValueError):
        object_key("job/123", "original", "source.wav")


def test_object_key_rejects_empty_job_id():
    with pytest.raises(ValueError):
        object_key("", "original", "source.wav")
