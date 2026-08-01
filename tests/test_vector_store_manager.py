from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from universal_memory.vector_store_manager import (
    CorruptRecordError,
    InMemoryVectorBackend,
    MemoryLimitError,
    MemoryLimits,
    MemoryValidationError,
    VectorStoreManager,
)


class KeywordEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        lowered = text.casefold()
        return [float("python" in lowered), float("garden" in lowered), 1.0]


class MentionExtractor:
    def extract(self, text: str) -> list[str]:
        return [word for word in text.split() if word.startswith("@")]


@pytest.fixture
def components() -> tuple[VectorStoreManager, InMemoryVectorBackend]:
    backend = InMemoryVectorBackend()
    manager = VectorStoreManager(
        embedder=KeywordEmbedder(),
        backend=backend,
        entity_extractor=MentionExtractor(),
        encryption_key=b"e" * 32,
        entity_index_key=b"i" * 32,
    )
    return manager, backend


def make_manager(
    *, limits: MemoryLimits | None = None, embedder: object | None = None
) -> tuple[VectorStoreManager, InMemoryVectorBackend]:
    backend = InMemoryVectorBackend()
    return (
        VectorStoreManager(
            embedder=embedder or KeywordEmbedder(),
            backend=backend,
            entity_extractor=MentionExtractor(),
            encryption_key=b"e" * 32,
            entity_index_key=b"i" * 32,
            limits=limits,
        ),
        backend,
    )


def test_rich_metadata_round_trip_and_backend_receives_no_plaintext_labels(components) -> None:
    manager, backend = components
    event_timestamp = datetime(2026, 7, 31, 21, 26, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    stored = manager.store(
        "secret for @Alice",
        {"private": "hidden", "nested": {"kept": True}},
        source="email://private-alice-inbox",
        event_timestamp=event_timestamp,
        memory_id="memory-1",
    )

    retrieved = manager.retrieve(memory_id="memory-1")[0]
    expected_event_time = event_timestamp.astimezone(timezone.utc)
    assert stored.source == retrieved.source == "email://private-alice-inbox"
    assert stored.entities == retrieved.entities == ("@Alice",)
    assert stored.event_timestamp == retrieved.event_timestamp == expected_event_time
    assert retrieved.metadata == {"private": "hidden", "nested": {"kept": True}}
    assert retrieved.created_at == stored.created_at
    assert stored.created_at != expected_event_time

    record = backend.get("memory-1")
    assert record is not None
    persisted = record.ciphertext + record.nonce + "".join(record.entity_indexes).encode()
    for plaintext in (b"secret", b"hidden", b"alice", b"email", b"inbox"):
        assert plaintext not in persisted.lower()
    assert not hasattr(record, "source")
    assert not hasattr(record, "entities")
    assert record.event_timestamp == expected_event_time
    assert len(record.nonce) == 12
    assert len(next(iter(record.entity_indexes))) == 64


def test_chronological_retrieval_and_semantic_search(components) -> None:
    manager, _ = components
    now = datetime.now(timezone.utc)
    manager.store("new Python event for @Alice", memory_id="new", event_timestamp=now)
    manager.store("old garden event for @Alice", memory_id="old", event_timestamp=now - timedelta(days=1))
    manager.store("other event for @Bob", memory_id="other", event_timestamp=now + timedelta(days=1))

    assert [item.memory_id for item in manager.retrieve(entities=["@ALICE"])] == ["new", "old"]
    results = manager.search("advanced Python")
    assert results[0].memory_id == "new"
    assert results[0].score is not None and results[0].score > results[1].score


def test_strict_json_rejected_before_embedding() -> None:
    embedder = KeywordEmbedder()
    manager, _ = make_manager(embedder=embedder)
    cycle: dict[str, object] = {}
    cycle["self"] = cycle

    bad_values = [
        {1: "non-string key"},
        {"nan": float("nan")},
        {"inf": float("inf")},
        {"tuple": (1, 2)},
        cycle,
    ]
    for value in bad_values:
        with pytest.raises(MemoryValidationError):
            manager.store("never embedded", value)  # type: ignore[arg-type]
    assert embedder.calls == 0


def test_manager_enforces_depth_byte_entity_vector_and_record_caps() -> None:
    limits = MemoryLimits(
        max_content_bytes=10,
        max_source_bytes=4,
        max_id_bytes=4,
        max_metadata_bytes=8,
        max_json_depth=2,
        max_entities=1,
        max_entity_bytes=3,
        max_vector_dimensions=3,
        max_records=1,
    )
    manager, _ = make_manager(limits=limits)

    with pytest.raises(MemoryLimitError):
        manager.store("éééééé")  # twelve UTF-8 bytes
    with pytest.raises(MemoryLimitError):
        manager.store("ok", source="12345")
    with pytest.raises(MemoryLimitError):
        manager.store("ok", memory_id="12345")
    with pytest.raises(MemoryLimitError):
        manager.store("ok", {"a": "123456"})
    with pytest.raises(MemoryLimitError):
        manager.store("ok", {"a": {"b": {"c": 1}}})
    with pytest.raises(MemoryLimitError):
        manager.store("@aa @bb")

    manager.store("ok", memory_id="one")
    with pytest.raises(MemoryLimitError):
        manager.store("ok", memory_id="two")

    class TooWide:
        def embed(self, text: str) -> list[float]:
            return [0.0, 0.0, 0.0, 0.0]

    wide, _ = make_manager(limits=limits, embedder=TooWide())
    with pytest.raises(MemoryLimitError):
        wide.store("ok")


def test_nonfinite_vector_and_invalid_entity_items_are_rejected() -> None:
    class NanEmbedder:
        def embed(self, text: str) -> list[float]:
            return [float("nan")]

    manager, _ = make_manager(embedder=NanEmbedder())
    with pytest.raises(MemoryValidationError, match="finite vector"):
        manager.store("content")

    class BadExtractor:
        def extract(self, text: str) -> list[object]:
            return [7]

    backend = InMemoryVectorBackend()
    manager = VectorStoreManager(
        embedder=KeywordEmbedder(),
        backend=backend,
        entity_extractor=BadExtractor(),  # type: ignore[arg-type]
        encryption_key=b"e" * 32,
        entity_index_key=b"i" * 32,
    )
    with pytest.raises(MemoryValidationError):
        manager.store("content")


def test_simultaneous_selectors_are_rejected(components) -> None:
    manager, _ = components
    with pytest.raises(MemoryValidationError, match="cannot be combined"):
        manager.retrieve(memory_id="one", entities=["@Alice"])
    with pytest.raises(MemoryValidationError, match="cannot be combined"):
        manager.forget(memory_id="one", entities=["@Alice"])
    with pytest.raises(MemoryValidationError, match="required"):
        manager.forget()


def test_all_clear_backend_fields_are_authenticated(components) -> None:
    manager, backend = components
    manager.store("secret @Alice", memory_id="one")
    original = backend.get("one")
    assert original is not None

    tampered_records = [
        replace(original, created_at=original.created_at + timedelta(seconds=1)),
        replace(original, event_timestamp=original.event_timestamp + timedelta(seconds=1)),
        replace(original, vector=(original.vector[0] + 1.0, *original.vector[1:])),
        replace(original, entity_indexes=frozenset({"0" * 64})),
    ]
    for tampered in tampered_records:
        backend._records["one"] = tampered
        with pytest.raises(CorruptRecordError, match="stored memory record is corrupt"):
            manager.retrieve(memory_id="one")
    backend._records["one"] = original


def test_entity_delete_authenticates_indexes_before_deleting(components) -> None:
    manager, backend = components
    manager.store("secret @Alice", memory_id="one")
    record = backend.get("one")
    assert record is not None
    alice_index = next(iter(record.entity_indexes))
    backend._records["one"] = replace(record, entity_indexes=frozenset({alice_index, "0" * 64}))

    with pytest.raises(CorruptRecordError):
        manager.forget(entities=["@Alice"])
    assert backend.get("one") is not None


def test_malformed_backend_record_has_sanitized_domain_error(components) -> None:
    manager, backend = components
    backend._records["bad"] = object()  # type: ignore[assignment]
    with pytest.raises(CorruptRecordError) as raised:
        manager.retrieve()
    assert str(raised.value) == "stored memory record is corrupt"
    assert raised.value.detail == "stored memory record is corrupt"


def test_keys_are_validated() -> None:
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        VectorStoreManager(
            embedder=KeywordEmbedder(),
            backend=InMemoryVectorBackend(),
            entity_extractor=MentionExtractor(),
            encryption_key=b"short",
            entity_index_key=b"i" * 32,
        )


def test_large_finite_vectors_produce_finite_similarity() -> None:
    class HugeEmbedder:
        def embed(self, text: str) -> list[float]:
            return [1e308, -1e308]

    manager, _ = make_manager(embedder=HugeEmbedder())
    manager.store("huge", memory_id="huge")
    score = manager.search("huge")[0].score
    assert score is not None
    assert score == pytest.approx(1.0)


def test_backend_get_and_delete_failures_are_sanitized() -> None:
    class FailingBackend(InMemoryVectorBackend):
        fail_get = False
        fail_delete = False

        def get(self, memory_id: str):
            if self.fail_get:
                raise RuntimeError("get-secret")
            return super().get(memory_id)

        def delete(self, memory_ids):
            if self.fail_delete:
                raise RuntimeError("delete-secret")
            return super().delete(memory_ids)

    from universal_memory.vector_store_manager import StorageBackendError

    backend = FailingBackend()
    manager = VectorStoreManager(
        embedder=KeywordEmbedder(),
        backend=backend,
        entity_extractor=MentionExtractor(),
        encryption_key=b"e" * 32,
        entity_index_key=b"i" * 32,
    )
    manager.store("one", memory_id="one")
    backend.fail_get = True
    with pytest.raises(StorageBackendError) as get_error:
        manager.retrieve(memory_id="one")
    assert "get-secret" not in str(get_error.value)
    backend.fail_get = False
    backend.fail_delete = True
    with pytest.raises(StorageBackendError) as delete_error:
        manager.forget(memory_id="one")
    assert "delete-secret" not in str(delete_error.value)
