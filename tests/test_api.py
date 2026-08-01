from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from universal_memory.api import ApiKeyAuthorizationPolicy, Capability, create_app
from universal_memory.vector_store_manager import (
    DeterministicHashEmbedder,
    InMemoryVectorBackend,
    SimpleEntityExtractor,
    VectorStoreManager,
)

API_KEY = "local-test-api-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}


def manager_and_backend() -> tuple[VectorStoreManager, InMemoryVectorBackend]:
    backend = InMemoryVectorBackend()
    return (
        VectorStoreManager(
            embedder=DeterministicHashEmbedder(),
            backend=backend,
            entity_extractor=SimpleEntityExtractor(),
            encryption_key=b"e" * 32,
            entity_index_key=b"i" * 32,
        ),
        backend,
    )


def client_for(*capabilities: Capability) -> TestClient:
    manager, _ = manager_and_backend()
    granted = capabilities or tuple(Capability)
    return TestClient(create_app(manager, ApiKeyAuthorizationPolicy(API_KEY, granted)))


def test_http_lifecycle_round_trips_rich_metadata_without_network() -> None:
    client = client_for()
    stored = client.post(
        "/store",
        headers=AUTH,
        json={
            "content": "Python notes for @Alice",
            "metadata": {"kind": "note", "user_field": 7},
            "source": "notebook://team/private",
            "event_timestamp": "2026-07-31T21:26:05.753+05:30",
        },
    )
    assert stored.status_code == 201
    store_body = stored.json()
    memory_id = store_body["memory_id"]
    assert store_body["source"] == "notebook://team/private"
    assert "@Alice" in store_body["entities"]
    assert store_body["event_timestamp"] == "2026-07-31T15:56:05.753000Z"
    assert store_body["created_at"] != store_body["event_timestamp"]

    retrieved = client.post("/retrieve", headers=AUTH, json={"memory_id": memory_id})
    assert retrieved.status_code == 200
    memory = retrieved.json()["memories"][0]
    assert memory["content"] == "Python notes for @Alice"
    assert memory["metadata"] == {"kind": "note", "user_field": 7}
    assert memory["source"] == "notebook://team/private"

    searched = client.post("/search", headers=AUTH, json={"query": "Python", "limit": 5})
    assert searched.status_code == 200
    assert searched.json()["memories"][0]["memory_id"] == memory_id

    forgotten = client.post("/forget", headers=AUTH, json={"memory_id": memory_id})
    assert forgotten.status_code == 200
    assert forgotten.json() == {"deleted": 1}


def test_authentication_is_fail_closed_and_does_not_echo_credentials() -> None:
    manager, _ = manager_and_backend()
    no_policy = TestClient(create_app(manager))
    missing = no_policy.post("/retrieve", json={})
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"

    invalid = TestClient(create_app(manager, ApiKeyAuthorizationPolicy(API_KEY))).post(
        "/retrieve",
        headers={"Authorization": "Bearer do-not-echo-this"},
        json={},
    )
    assert invalid.status_code == 401
    assert "do-not-echo-this" not in invalid.text


def test_capabilities_are_enforced_per_operation() -> None:
    client = client_for(Capability.READ)
    assert client.post("/retrieve", headers=AUTH, json={}).status_code == 200
    assert client.post("/search", headers=AUTH, json={"query": "x"}).status_code == 200
    assert client.post("/store", headers=AUTH, json={"content": "x"}).status_code == 403
    assert client.post("/forget", headers=AUTH, json={"memory_id": "x"}).status_code == 403


def test_valid_auth_still_fails_closed_when_manager_unconfigured() -> None:
    client = TestClient(create_app(None, ApiKeyAuthorizationPolicy(API_KEY)))
    response = client.post("/retrieve", headers=AUTH, json={})
    assert response.status_code == 503
    assert "key" not in response.text.casefold()


def test_invalid_payloads_and_ambiguous_selectors_are_stable_422() -> None:
    client = client_for()
    naive = client.post(
        "/store",
        headers=AUTH,
        json={"content": "naive", "event_timestamp": "2026-07-31T21:26:05"},
    )
    assert naive.status_code == 422

    nan = client.post(
        "/store",
        headers={**AUTH, "Content-Type": "application/json"},
        content='{"content":"bad","metadata":{"value":NaN}}',
    )
    assert nan.status_code == 422
    assert nan.json() == {"detail": "invalid memory request"}

    ambiguous_retrieve = client.post(
        "/retrieve", headers=AUTH, json={"memory_id": "one", "entities": ["@Alice"]}
    )
    ambiguous_forget = client.post(
        "/forget", headers=AUTH, json={"memory_id": "one", "entities": ["@Alice"]}
    )
    assert ambiguous_retrieve.status_code == ambiguous_forget.status_code == 422
    assert client.post("/forget", headers=AUTH, json={}).status_code == 422
    assert client.post("/retrieve", headers=AUTH, json={"entities": [""]}).status_code == 422


def test_corrupt_record_maps_to_sanitized_stable_error() -> None:
    manager, backend = manager_and_backend()
    manager.store("private", memory_id="one")
    record = backend.get("one")
    assert record is not None
    backend._records["one"] = replace(record, vector=(record.vector[0] + 1.0, *record.vector[1:]))
    client = TestClient(create_app(manager, ApiKeyAuthorizationPolicy(API_KEY)))

    response = client.post("/retrieve", headers=AUTH, json={"memory_id": "one"})
    assert response.status_code == 500
    assert response.json() == {"detail": "stored memory record is corrupt"}
    assert "private" not in response.text


def test_authentication_precedes_json_validation_and_body_limits() -> None:
    client = client_for()
    malformed = client.post(
        "/store",
        headers={"Content-Type": "application/json"},
        content="{not-json",
    )
    assert malformed.status_code == 401

    oversized = client.post(
        "/store",
        headers={**AUTH, "Content-Type": "application/json"},
        content=b"x" * (1_048_576 + 1),
    )
    assert oversized.status_code == 413
    assert oversized.json() == {"detail": "request body exceeds the configured limit"}


def test_backend_failures_are_sanitized_as_json_503() -> None:
    class FailingBackend(InMemoryVectorBackend):
        def get(self, memory_id: str):
            raise RuntimeError("backend-secret")

    service = VectorStoreManager(
        embedder=DeterministicHashEmbedder(),
        backend=FailingBackend(),
        entity_extractor=SimpleEntityExtractor(),
        encryption_key=b"e" * 32,
        entity_index_key=b"i" * 32,
    )
    client = TestClient(
        create_app(service, ApiKeyAuthorizationPolicy(API_KEY)),
        raise_server_exceptions=False,
    )
    response = client.post("/retrieve", headers=AUTH, json={"memory_id": "one"})
    assert response.status_code == 503
    assert response.json() == {"detail": "memory backend unavailable"}
    assert "backend-secret" not in response.text
