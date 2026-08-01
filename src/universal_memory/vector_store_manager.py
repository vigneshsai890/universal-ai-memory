"""Encrypted vector memory management without plaintext persistence or logging.

Vectors, identifiers, timestamps, and blind indexes remain clear for backend operations.
They are *authenticated*, not encrypted, by binding canonical digests to AES-GCM AAD.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import struct
import threading
import unicodedata
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class MemoryServiceError(Exception):
    """Base class for errors safe for transport adapters to classify."""

    detail = "memory operation failed"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.detail)


class MemoryValidationError(MemoryServiceError, ValueError):
    """Caller input violates the manager contract."""

    detail = "invalid memory request"


class MemoryConflictError(MemoryServiceError):
    """A caller-selected identifier already exists."""

    detail = "memory identifier already exists"


class MemoryLimitError(MemoryValidationError):
    """A configured resource cap was exceeded."""

    detail = "memory request exceeds a configured limit"


class StorageBackendError(MemoryServiceError):
    """The persistence adapter failed without exposing implementation details."""

    detail = "memory backend unavailable"


class CorruptRecordError(MemoryServiceError):
    """A persisted record is malformed or fails authentication."""

    detail = "stored memory record is corrupt"


class Embedder(Protocol):
    """Maps text to a stable numeric vector."""

    def embed(self, text: str) -> Sequence[float]: ...


class EntityExtractor(Protocol):
    """Extracts entity labels while plaintext exists in process memory."""

    def extract(self, text: str) -> Iterable[str]: ...


@dataclass(frozen=True, slots=True)
class MemoryLimits:
    """Manager-enforced limits; byte limits are measured as UTF-8."""

    max_content_bytes: int = 65_536
    max_source_bytes: int = 2_048
    max_id_bytes: int = 256
    max_metadata_bytes: int = 65_536
    max_json_depth: int = 12
    max_entities: int = 64
    max_entity_bytes: int = 256
    max_vector_dimensions: int = 4_096
    max_records: int = 10_000

    def __post_init__(self) -> None:
        for name in (
            "max_content_bytes",
            "max_source_bytes",
            "max_id_bytes",
            "max_metadata_bytes",
            "max_json_depth",
            "max_entities",
            "max_entity_bytes",
            "max_vector_dimensions",
            "max_records",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class EncryptedVectorRecord:
    """Backend-safe record containing ciphertext plus authenticated clear indexes."""

    memory_id: str
    created_at: datetime
    event_timestamp: datetime
    vector: tuple[float, ...]
    entity_indexes: frozenset[str]
    nonce: bytes
    ciphertext: bytes


class VectorBackend(Protocol):
    """Persistence adapter. Implementations receive ciphertext and blind indexes only."""

    def add(self, record: EncryptedVectorRecord) -> None: ...

    def get(self, memory_id: str) -> EncryptedVectorRecord | None: ...

    def list(self) -> list[EncryptedVectorRecord]: ...

    def delete(self, memory_ids: Iterable[str]) -> int: ...


@dataclass(frozen=True, slots=True)
class Memory:
    memory_id: str
    content: str
    metadata: Mapping[str, Any]
    source: str | None
    event_timestamp: datetime
    entities: tuple[str, ...]
    created_at: datetime
    score: float | None = None


class InMemoryVectorBackend:
    """Thread-safe backend intended for tests and single-process development."""

    def __init__(self) -> None:
        self._records: dict[str, EncryptedVectorRecord] = {}
        self._lock = threading.RLock()

    def add(self, record: EncryptedVectorRecord) -> None:
        with self._lock:
            if record.memory_id in self._records:
                raise MemoryConflictError
            self._records[record.memory_id] = record

    def get(self, memory_id: str) -> EncryptedVectorRecord | None:
        with self._lock:
            return self._records.get(memory_id)

    def list(self) -> list[EncryptedVectorRecord]:
        with self._lock:
            return list(self._records.values())

    def delete(self, memory_ids: Iterable[str]) -> int:
        with self._lock:
            deleted = 0
            for memory_id in set(memory_ids):
                if self._records.pop(memory_id, None) is not None:
                    deleted += 1
            return deleted


class DeterministicHashEmbedder:
    """Small offline embedder for development; replace with a production model."""

    _TOKEN = re.compile(r"\w+", re.UNICODE)

    def __init__(self, dimensions: int = 64) -> None:
        if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions < 1:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    def embed(self, text: str) -> Sequence[float]:
        vector = [0.0] * self._dimensions
        for token in self._TOKEN.findall(text.casefold()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self._dimensions
            vector[index] += 1.0 if digest[8] & 1 else -1.0
        return vector


class SimpleEntityExtractor:
    """Offline example extractor for @mentions and title-cased phrases."""

    _ENTITY = re.compile(r"(?:@[\w.-]+|\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*\b)")

    def extract(self, text: str) -> Iterable[str]:
        return (match.group(0) for match in self._ENTITY.finditer(text))


class VectorStoreManager:
    """Coordinates validation, encryption, blind indexing, and retrieval.

    Private payload fields are encrypted. Backend-visible vectors, IDs, timestamps,
    and blind indexes are not private; their canonical values are authenticated as
    AES-GCM associated data and checked before any ordering, scoring, or deletion.
    """

    _AAD_PREFIX = b"universal-memory:v2:"
    _INDEX_PATTERN = re.compile(r"[0-9a-f]{64}\Z")

    def __init__(
        self,
        *,
        embedder: Embedder,
        backend: VectorBackend,
        entity_extractor: EntityExtractor,
        encryption_key: bytes,
        entity_index_key: bytes,
        limits: MemoryLimits | None = None,
    ) -> None:
        if len(encryption_key) != 32:
            raise ValueError("encryption_key must be exactly 32 bytes")
        if len(entity_index_key) < 32:
            raise ValueError("entity_index_key must be at least 32 bytes")
        if hmac.compare_digest(encryption_key, entity_index_key):
            raise ValueError("encryption and entity index keys must be distinct")
        self._embedder = embedder
        self._backend = backend
        self._entity_extractor = entity_extractor
        self._cipher = AESGCM(encryption_key)
        self._entity_index_key = bytes(entity_index_key)
        self._limits = limits or MemoryLimits()

    @property
    def limits(self) -> MemoryLimits:
        return self._limits

    def store(
        self,
        content: str,
        metadata: Mapping[str, Any] | None = None,
        *,
        source: str | None = None,
        event_timestamp: datetime | None = None,
        memory_id: str | None = None,
    ) -> Memory:
        # All caller-controlled payload data is validated and canonicalized before
        # invoking an embedder, which may be a remote or otherwise untrusted adapter.
        content = self._text(content, "content", self._limits.max_content_bytes)
        if source is not None:
            source = self._text(source, "source", self._limits.max_source_bytes)
        identifier = str(uuid.uuid4()) if memory_id is None else self._text(
            memory_id, "memory_id", self._limits.max_id_bytes
        )
        user_metadata, _ = self._canonical_metadata({} if metadata is None else metadata)
        created_at = datetime.now(timezone.utc)
        event_time = created_at if event_timestamp is None else self._utc(event_timestamp, "event_timestamp")

        records = self._listed_records()
        if len(records) >= self._limits.max_records:
            raise MemoryLimitError("record count limit exceeded")
        if self._backend_get(identifier) is not None:
            raise MemoryConflictError

        vector = self._validated_vector(self._embedder.embed(content))
        extracted_entities = self._extract_entities(content)
        entity_indexes = frozenset(self._entity_index(entity) for entity in extracted_entities)
        payload = self._canonical_json_bytes(
            {
                "content": content,
                "entities": list(extracted_entities),
                "event_timestamp": self._canonical_timestamp(event_time),
                "metadata": user_metadata,
                "source": source,
            }
        )
        nonce = os.urandom(12)
        record_fields = EncryptedVectorRecord(
            memory_id=identifier,
            created_at=created_at,
            event_timestamp=event_time,
            vector=vector,
            entity_indexes=entity_indexes,
            nonce=nonce,
            ciphertext=b"",
        )
        ciphertext = self._cipher.encrypt(nonce, payload, self._aad(record_fields))
        try:
            self._backend.add(
                EncryptedVectorRecord(
                    memory_id=identifier,
                    created_at=created_at,
                    event_timestamp=event_time,
                    vector=vector,
                    entity_indexes=entity_indexes,
                    nonce=nonce,
                    ciphertext=ciphertext,
                )
            )
        except MemoryConflictError:
            raise
        except ValueError as exc:
            raise MemoryConflictError from exc
        except Exception as exc:
            raise StorageBackendError from exc
        return Memory(
            memory_id=identifier,
            content=content,
            metadata=user_metadata,
            source=source,
            event_timestamp=event_time,
            entities=extracted_entities,
            created_at=created_at,
        )

    def retrieve(
        self,
        *,
        memory_id: str | None = None,
        entities: Sequence[str] = (),
        limit: int = 20,
        newest_first: bool = True,
    ) -> list[Memory]:
        """Retrieve by ID, or by authenticated event chronology and entities."""
        self._validate_limit(limit)
        entity_values = self._selector_entities(entities)
        if memory_id is not None and entity_values:
            raise MemoryValidationError("memory_id and entities cannot be combined")
        if memory_id is not None:
            identifier = self._text(memory_id, "memory_id", self._limits.max_id_bytes)
            record = self._backend_get(identifier)
            if record is None:
                return []
            memory = self._decrypt(record)
            if memory.memory_id != identifier:
                raise CorruptRecordError
            return [memory]

        authenticated = self._authenticated_records(self._listed_records())
        required = frozenset(self._entity_index(entity) for entity in entity_values)
        selected = [(record, memory) for record, memory in authenticated if required.issubset(record.entity_indexes)]
        selected.sort(
            key=lambda item: (item[1].event_timestamp, item[1].memory_id),
            reverse=newest_first,
        )
        return [memory for _, memory in selected[:limit]]

    def search(
        self,
        query: str,
        *,
        entities: Sequence[str] = (),
        limit: int = 10,
    ) -> list[Memory]:
        """Return entity-filtered memories ordered by authenticated vectors."""
        query = self._text(query, "query", self._limits.max_content_bytes)
        self._validate_limit(limit)
        entity_values = self._selector_entities(entities)
        query_vector = self._validated_vector(self._embedder.embed(query))
        required = frozenset(self._entity_index(entity) for entity in entity_values)
        scored: list[tuple[float, Memory]] = []
        for record, memory in self._authenticated_records(self._listed_records()):
            if not required.issubset(record.entity_indexes):
                continue
            if len(record.vector) != len(query_vector):
                continue
            scored.append((self._cosine(query_vector, record.vector), memory))
        scored.sort(
            key=lambda item: (item[0], item[1].event_timestamp, item[1].memory_id),
            reverse=True,
        )
        return [
            Memory(
                memory_id=memory.memory_id,
                content=memory.content,
                metadata=memory.metadata,
                source=memory.source,
                event_timestamp=memory.event_timestamp,
                entities=memory.entities,
                created_at=memory.created_at,
                score=score,
            )
            for score, memory in scored[:limit]
        ]

    def forget(self, *, memory_id: str | None = None, entities: Sequence[str] = ()) -> int:
        """Delete only records whose complete clear envelope authenticates."""
        entity_values = self._selector_entities(entities)
        if memory_id is not None and entity_values:
            raise MemoryValidationError("memory_id and entities cannot be combined")
        if memory_id is not None:
            identifier = self._text(memory_id, "memory_id", self._limits.max_id_bytes)
            record = self._backend_get(identifier)
            if record is None:
                return 0
            memory = self._decrypt(record)
            if memory.memory_id != identifier:
                raise CorruptRecordError
            return self._backend_delete([identifier])
        if not entity_values:
            raise MemoryValidationError("memory_id or at least one entity is required")

        # Authenticate every record before trusting blind indexes or deleting IDs.
        required = frozenset(self._entity_index(entity) for entity in entity_values)
        authenticated = self._authenticated_records(self._listed_records())
        identifiers = [
            memory.memory_id
            for record, memory in authenticated
            if required.issubset(record.entity_indexes)
        ]
        return self._backend_delete(identifiers)

    def _authenticated_records(
        self, records: Iterable[EncryptedVectorRecord]
    ) -> list[tuple[EncryptedVectorRecord, Memory]]:
        return [(record, self._decrypt(record)) for record in records]

    def _decrypt(self, record: EncryptedVectorRecord, score: float | None = None) -> Memory:
        try:
            memory_id = self._text(record.memory_id, "memory_id", self._limits.max_id_bytes)
            created_at = self._utc(record.created_at, "created_at")
            event_time_clear = self._utc(record.event_timestamp, "event_timestamp")
            vector = self._validated_vector(record.vector)
            indexes = self._validated_indexes(record.entity_indexes)
            if not isinstance(record.nonce, bytes) or len(record.nonce) != 12:
                raise CorruptRecordError
            if not isinstance(record.ciphertext, bytes) or len(record.ciphertext) < 16:
                raise CorruptRecordError
            validated_record = EncryptedVectorRecord(
                memory_id=memory_id,
                created_at=created_at,
                event_timestamp=event_time_clear,
                vector=vector,
                entity_indexes=indexes,
                nonce=record.nonce,
                ciphertext=record.ciphertext,
            )
            plaintext = self._cipher.decrypt(
                record.nonce,
                record.ciphertext,
                self._aad(validated_record),
            )
            payload = json.loads(
                plaintext,
                parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("non-finite number")),
            )
            if not isinstance(payload, dict) or set(payload) != {
                "content", "entities", "event_timestamp", "metadata", "source"
            }:
                raise CorruptRecordError
            content = self._text(payload["content"], "content", self._limits.max_content_bytes)
            source_value = payload["source"]
            source = None if source_value is None else self._text(
                source_value, "source", self._limits.max_source_bytes
            )
            metadata, _ = self._canonical_metadata(payload["metadata"])
            entities_value = payload["entities"]
            if not isinstance(entities_value, list):
                raise CorruptRecordError
            entities = self._validated_entity_items(entities_value)
            timestamp_value = payload["event_timestamp"]
            if not isinstance(timestamp_value, str):
                raise CorruptRecordError
            event_time = self._utc(datetime.fromisoformat(timestamp_value), "event_timestamp")
            if self._canonical_timestamp(event_time) != timestamp_value:
                raise CorruptRecordError
            if event_time != event_time_clear:
                raise CorruptRecordError
            if frozenset(self._entity_index(entity) for entity in entities) != indexes:
                raise CorruptRecordError
            return Memory(
                memory_id=memory_id,
                content=content,
                metadata=metadata,
                source=source,
                event_timestamp=event_time,
                entities=entities,
                created_at=created_at,
                score=score,
            )
        except CorruptRecordError:
            raise
        except (InvalidTag, AttributeError, KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise CorruptRecordError from exc
        except Exception as exc:
            # Backend data and crypto/parser failures never escape with sensitive detail.
            raise CorruptRecordError from exc

    def _extract_entities(self, content: str) -> tuple[str, ...]:
        try:
            extracted = self._entity_extractor.extract(content)
            return self._validated_entity_items(extracted, deduplicate=True)
        except MemoryServiceError:
            raise
        except Exception as exc:
            raise MemoryValidationError("entity extractor returned invalid entities") from exc

    def _validated_entity_items(
        self, entities: Iterable[Any], *, deduplicate: bool = False
    ) -> tuple[str, ...]:
        if isinstance(entities, (str, bytes)):
            raise MemoryValidationError("entities must be a sequence of strings")
        result: list[str] = []
        seen: set[str] = set()
        try:
            iterator = iter(entities)
        except TypeError as exc:
            raise MemoryValidationError("entities must be a sequence of strings") from exc
        for entity in iterator:
            value = self._text(entity, "entity", self._limits.max_entity_bytes)
            normalized = self._normalize_entity(value)
            if not normalized:
                raise MemoryValidationError("entities must not be blank")
            if deduplicate and normalized in seen:
                continue
            if len(result) >= self._limits.max_entities:
                raise MemoryLimitError("entity count limit exceeded")
            seen.add(normalized)
            result.append(value)
        return tuple(result)

    def _selector_entities(self, entities: Sequence[str]) -> tuple[str, ...]:
        return self._validated_entity_items(entities)

    def _entity_index(self, entity: str) -> str:
        normalized = self._normalize_entity(entity)
        if not normalized:
            raise MemoryValidationError("entities must not be blank")
        return hmac.new(
            self._entity_index_key,
            normalized.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _validated_indexes(self, indexes: Any) -> frozenset[str]:
        if not isinstance(indexes, (set, frozenset)) or len(indexes) > self._limits.max_entities:
            raise CorruptRecordError
        if any(not isinstance(value, str) or not self._INDEX_PATTERN.fullmatch(value) for value in indexes):
            raise CorruptRecordError
        return frozenset(indexes)

    def _backend_get(self, memory_id: str) -> EncryptedVectorRecord | None:
        try:
            return self._backend.get(memory_id)
        except Exception as exc:
            raise StorageBackendError from exc

    def _backend_delete(self, memory_ids: Iterable[str]) -> int:
        try:
            deleted = self._backend.delete(memory_ids)
        except Exception as exc:
            raise StorageBackendError from exc
        if not isinstance(deleted, int) or isinstance(deleted, bool) or deleted < 0:
            raise StorageBackendError
        return deleted

    def _listed_records(self) -> list[EncryptedVectorRecord]:
        try:
            records = self._backend.list()
        except Exception as exc:
            raise StorageBackendError from exc
        if not isinstance(records, list) or len(records) > self._limits.max_records:
            raise CorruptRecordError
        return records

    def _canonical_metadata(self, metadata: Any) -> tuple[dict[str, Any], bytes]:
        if not isinstance(metadata, Mapping):
            raise MemoryValidationError("metadata must be a JSON object")
        normalized = self._strict_json(metadata, depth=1, active=set())
        if not isinstance(normalized, dict):
            raise MemoryValidationError("metadata must be a JSON object")
        encoded = self._canonical_json_bytes(normalized)
        if len(encoded) > self._limits.max_metadata_bytes:
            raise MemoryLimitError("metadata byte limit exceeded")
        return normalized, encoded

    def _strict_json(self, value: Any, *, depth: int, active: set[int]) -> Any:
        if value is None or isinstance(value, (str, bool)):
            if isinstance(value, str):
                try:
                    value.encode("utf-8")
                except UnicodeError as exc:
                    raise MemoryValidationError("metadata contains invalid Unicode") from exc
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise MemoryValidationError("metadata numbers must be finite")
            return value
        if depth > self._limits.max_json_depth:
            raise MemoryLimitError("metadata JSON depth limit exceeded")
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in active:
                raise MemoryValidationError("metadata must not contain cycles")
            active.add(identity)
            try:
                result: dict[str, Any] = {}
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise MemoryValidationError("metadata object keys must be strings")
                    result[key] = self._strict_json(item, depth=depth + 1, active=active)
                return result
            finally:
                active.remove(identity)
        if isinstance(value, list):
            identity = id(value)
            if identity in active:
                raise MemoryValidationError("metadata must not contain cycles")
            active.add(identity)
            try:
                return [self._strict_json(item, depth=depth + 1, active=active) for item in value]
            finally:
                active.remove(identity)
        raise MemoryValidationError("metadata must contain only strict JSON values")

    @staticmethod
    def _canonical_json_bytes(value: Any) -> bytes:
        try:
            return json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError) as exc:
            raise MemoryValidationError("value is not canonical JSON") from exc

    def _validated_vector(self, vector: Sequence[float]) -> tuple[float, ...]:
        if isinstance(vector, (str, bytes)):
            raise MemoryValidationError("embedder must return a numeric vector")
        try:
            if len(vector) > self._limits.max_vector_dimensions:
                raise MemoryLimitError("vector dimension limit exceeded")
            values = tuple(float(value) for value in vector)
        except MemoryServiceError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise MemoryValidationError("embedder must return a numeric vector") from exc
        if not values or any(not math.isfinite(value) for value in values):
            raise MemoryValidationError("embedder must return a non-empty finite vector")
        return values

    @staticmethod
    def _normalize_entity(entity: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", entity).casefold().split())

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise MemoryValidationError("limit must be between 1 and 100")

    def _text(self, value: Any, field_name: str, max_bytes: int) -> str:
        if not isinstance(value, str):
            raise MemoryValidationError(f"{field_name} must be a string")
        if not value:
            raise MemoryValidationError(f"{field_name} must not be empty")
        try:
            size = len(value.encode("utf-8"))
        except UnicodeError as exc:
            raise MemoryValidationError(f"{field_name} contains invalid Unicode") from exc
        if size > max_bytes:
            raise MemoryLimitError(f"{field_name} byte limit exceeded")
        return value

    @classmethod
    def _canonical_timestamp(cls, value: datetime) -> str:
        utc_value = cls._utc(value, "timestamp")
        return utc_value.isoformat(timespec="microseconds").replace("+00:00", "Z")

    @classmethod
    def _aad(cls, record: EncryptedVectorRecord) -> bytes:
        vector_bytes = b"".join(struct.pack("!d", value) for value in record.vector)
        indexes_bytes = json.dumps(
            sorted(record.entity_indexes), separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        envelope = {
            "created_at": cls._canonical_timestamp(record.created_at),
            "entity_indexes_sha256": hashlib.sha256(indexes_bytes).hexdigest(),
            "event_timestamp": cls._canonical_timestamp(record.event_timestamp),
            "memory_id": record.memory_id,
            "vector_sha256": hashlib.sha256(vector_bytes).hexdigest(),
        }
        return cls._AAD_PREFIX + json.dumps(
            envelope,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @staticmethod
    def _utc(value: datetime, field_name: str) -> datetime:
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise MemoryValidationError(f"{field_name} must be timezone-aware")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        scale = max(
            max((abs(value) for value in left), default=0.0),
            max((abs(value) for value in right), default=0.0),
        )
        if scale == 0.0:
            return 0.0
        scaled_left = tuple(value / scale for value in left)
        scaled_right = tuple(value / scale for value in right)
        left_norm = math.sqrt(math.fsum(value * value for value in scaled_left))
        right_norm = math.sqrt(math.fsum(value * value for value in scaled_right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        score = math.fsum(
            a * b for a, b in zip(scaled_left, scaled_right, strict=True)
        ) / (left_norm * right_norm)
        if not math.isfinite(score):
            raise CorruptRecordError
        return max(-1.0, min(1.0, score))
