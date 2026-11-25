from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.models import EnhancementPreset, Job, JobArtifact, JobStatus
from api.routes import get_artifact_lister, get_job_store


class FakeJobStore:
    """In-memory JobStore double -- lets api/routes.py be fully exercised
    without a real Postgres connection, same idea as storage.py's real
    network calls being excluded from unit tests while the pure logic
    around them is tested directly.
    """

    def __init__(self):
        self._jobs: dict[UUID, Job] = {}
        self._by_idempotency_key: dict[str, UUID] = {}

    def insert_job(
        self, source_url: str, preset: EnhancementPreset, add_music: bool, idempotency_key: str
    ) -> Job:
        existing_id = self._by_idempotency_key.get(idempotency_key)
        if existing_id is not None:
            return self._jobs[existing_id]

        now = datetime.now(timezone.utc)
        job = Job(
            id=uuid4(),
            status=JobStatus.QUEUED,
            progress=0,
            sourceUrl=source_url,
            preset=preset,
            error=None,
            createdAt=now,
            updatedAt=now,
        )
        self._jobs[job.id] = job
        self._by_idempotency_key[idempotency_key] = job.id
        return job

    def get_job(self, job_id: UUID) -> Job | None:
        return self._jobs.get(job_id)

    def update_status(self, job_id: UUID, status: JobStatus) -> Job | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        updated = job.model_copy(update={"status": status, "updated_at": datetime.now(timezone.utc)})
        self._jobs[job_id] = updated
        return updated


@pytest.fixture
def fake_job_store():
    return FakeJobStore()


@pytest.fixture
def client(fake_job_store):
    app.dependency_overrides[get_job_store] = lambda: fake_job_store
    app.dependency_overrides[get_artifact_lister] = lambda: (lambda job_id: [])
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


AUTH_HEADERS = {"Authorization": "Bearer test-key"}


@pytest.fixture(autouse=True)
def api_keys_env(monkeypatch):
    monkeypatch.setenv("API_KEYS", "test-key,other-key")


def _submit_payload(idempotency_key: str = "idem-1") -> dict:
    return {
        "sourceUrl": "https://youtu.be/example",
        "preset": "balanced",
        "addMusic": False,
        "idempotencyKey": idempotency_key,
    }


def test_submit_job_returns_queued_job(client):
    response = client.post("/jobs", json=_submit_payload(), headers=AUTH_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["progress"] == 0
    assert body["sourceUrl"] == "https://youtu.be/example"
    assert body["preset"] == "balanced"


def test_submit_job_is_idempotent_on_same_key(client):
    first = client.post("/jobs", json=_submit_payload("same-key"), headers=AUTH_HEADERS).json()
    second = client.post("/jobs", json=_submit_payload("same-key"), headers=AUTH_HEADERS).json()

    assert first["id"] == second["id"]


def test_submit_job_with_different_keys_creates_different_jobs(client):
    first = client.post("/jobs", json=_submit_payload("key-a"), headers=AUTH_HEADERS).json()
    second = client.post("/jobs", json=_submit_payload("key-b"), headers=AUTH_HEADERS).json()

    assert first["id"] != second["id"]


def test_get_job_returns_submitted_job(client):
    submitted = client.post("/jobs", json=_submit_payload(), headers=AUTH_HEADERS).json()

    response = client.get(f"/jobs/{submitted['id']}", headers=AUTH_HEADERS)

    assert response.status_code == 200
    assert response.json()["id"] == submitted["id"]


def test_get_unknown_job_returns_404(client):
    response = client.get(f"/jobs/{uuid4()}", headers=AUTH_HEADERS)

    assert response.status_code == 404


def test_cancel_queued_job_succeeds(client):
    submitted = client.post("/jobs", json=_submit_payload(), headers=AUTH_HEADERS).json()

    response = client.post(f"/jobs/{submitted['id']}/cancel", headers=AUTH_HEADERS)

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_cancel_already_cancelled_job_returns_409(client):
    submitted = client.post("/jobs", json=_submit_payload(), headers=AUTH_HEADERS).json()
    client.post(f"/jobs/{submitted['id']}/cancel", headers=AUTH_HEADERS)

    response = client.post(f"/jobs/{submitted['id']}/cancel", headers=AUTH_HEADERS)

    assert response.status_code == 409


def test_cancel_unknown_job_returns_404(client):
    response = client.post(f"/jobs/{uuid4()}/cancel", headers=AUTH_HEADERS)

    assert response.status_code == 404


def test_get_artifacts_for_unknown_job_returns_404(client):
    response = client.get(f"/jobs/{uuid4()}/artifacts", headers=AUTH_HEADERS)

    assert response.status_code == 404


def test_get_artifacts_for_known_job_returns_list(fake_job_store):
    app.dependency_overrides[get_job_store] = lambda: fake_job_store
    app.dependency_overrides[get_artifact_lister] = lambda: (
        lambda job_id: [JobArtifact(name="enhanced/final.wav", url="https://example/x", sizeBytes=123)]
    )
    with TestClient(app) as client:
        submitted = client.post("/jobs", json=_submit_payload(), headers=AUTH_HEADERS).json()

        response = client.get(f"/jobs/{submitted['id']}/artifacts", headers=AUTH_HEADERS)

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["name"] == "enhanced/final.wav"
        assert body[0]["sizeBytes"] == 123
    app.dependency_overrides.clear()


def test_previews_returns_501(client):
    response = client.post(
        "/previews",
        json={"sourceUrl": "https://youtu.be/example", "preset": "balanced"},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 501


def test_missing_auth_header_returns_401(client):
    response = client.get(f"/jobs/{uuid4()}")

    assert response.status_code == 401


def test_invalid_api_key_returns_401(client):
    response = client.get(f"/jobs/{uuid4()}", headers={"Authorization": "Bearer wrong-key"})

    assert response.status_code == 401
