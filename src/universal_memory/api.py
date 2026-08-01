"""Authenticated FastAPI adapters for :mod:`universal_memory.vector_store_manager`."""

from __future__ import annotations

import base64
import hmac
import os
from collections.abc import Collection
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Protocol

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from .vector_store_manager import (
    CorruptRecordError,
    DeterministicHashEmbedder,
    InMemoryVectorBackend,
    Memory,
    MemoryConflictError,
    MemoryLimitError,
    MemoryServiceError,
    MemoryValidationError,
    SimpleEntityExtractor,
    StorageBackendError,
    VectorStoreManager,
)
MAX_REQUEST_BODY_BYTES = 1_048_576

MAX_TEXT_CHARS = 65_536
MAX_SOURCE_CHARS = 2_048
MAX_ID_CHARS = 256
MAX_ENTITY_CHARS = 256
MAX_ENTITIES = 64
Entity = Annotated[str, Field(min_length=1, max_length=MAX_ENTITY_CHARS)]


class Capability(StrEnum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"


class AuthorizationPolicy(Protocol):
    """Resolve a bearer credential to capabilities, or ``None`` if invalid."""

    def capabilities_for(self, bearer_token: str) -> Collection[Capability | str] | None: ...


class ApiKeyAuthorizationPolicy:
    """Constant-time single API-key policy for local runnable mode."""

    def __init__(
        self,
        api_key: str,
        capabilities: Collection[Capability | str] = tuple(Capability),
    ) -> None:
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must not be empty")
        self._api_key = api_key
        self._capabilities = frozenset(Capability(value) for value in capabilities)

    def capabilities_for(self, bearer_token: str) -> Collection[Capability] | None:
        if not isinstance(bearer_token, str) or not hmac.compare_digest(bearer_token, self._api_key):
            return None
        return self._capabilities


class DenyAllAuthorizationPolicy:
    """Default policy used when an application forgot to inject authorization."""

    def capabilities_for(self, bearer_token: str) -> None:
        return None


class StoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str | None = Field(default=None, min_length=1, max_length=MAX_SOURCE_CHARS)
    event_timestamp: AwareDatetime | None = None
    memory_id: str | None = Field(default=None, min_length=1, max_length=MAX_ID_CHARS)


class StoreResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    memory_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    source: str | None = Field(min_length=1, max_length=MAX_SOURCE_CHARS)
    event_timestamp: datetime
    entities: list[Entity] = Field(max_length=MAX_ENTITIES)
    created_at: datetime


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "allOf": [
                {
                    "not": {
                        "required": ["memory_id", "entities"],
                        "properties": {"entities": {"minItems": 1}},
                    }
                }
            ]
        },
    )
    memory_id: str | None = Field(default=None, min_length=1, max_length=MAX_ID_CHARS)
    entities: list[Entity] = Field(default_factory=list, max_length=MAX_ENTITIES)
    limit: int = Field(default=20, ge=1, le=100)
    newest_first: bool = True

    @model_validator(mode="after")
    def selectors_are_unambiguous(self) -> "RetrieveRequest":
        if self.memory_id is not None and self.entities:
            raise ValueError("memory_id and entities cannot be combined")
        return self


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    entities: list[Entity] = Field(default_factory=list, max_length=MAX_ENTITIES)
    limit: int = Field(default=10, ge=1, le=100)


class ForgetRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "oneOf": [
                {
                    "required": ["memory_id"],
                    "properties": {
                        "memory_id": {"type": "string", "minLength": 1, "maxLength": MAX_ID_CHARS},
                        "entities": {"type": "array", "maxItems": 0},
                    },
                },
                {
                    "required": ["entities"],
                    "properties": {"entities": {"type": "array", "minItems": 1, "maxItems": MAX_ENTITIES}},
                },
            ]
        },
    )
    memory_id: str | None = Field(default=None, min_length=1, max_length=MAX_ID_CHARS)
    entities: list[Entity] = Field(default_factory=list, max_length=MAX_ENTITIES)

    @model_validator(mode="after")
    def require_one_selector(self) -> "ForgetRequest":
        if self.memory_id is not None and self.entities:
            raise ValueError("memory_id and entities cannot be combined")
        if self.memory_id is None and not self.entities:
            raise ValueError("memory_id or at least one entity is required")
        return self


class MemoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    memory_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    content: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    metadata: dict[str, Any]
    source: str | None = Field(min_length=1, max_length=MAX_SOURCE_CHARS)
    event_timestamp: datetime
    entities: list[Entity] = Field(max_length=MAX_ENTITIES)
    created_at: datetime
    score: float | None = None


class MemoriesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    memories: list[MemoryResponse]


class ForgetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deleted: int = Field(ge=0)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detail: str


_COMMON_ERRORS = {
    401: {"model": ErrorResponse, "description": "Missing or invalid bearer credential"},
    403: {"model": ErrorResponse, "description": "Credential lacks the required capability"},
    413: {"model": ErrorResponse, "description": "A configured manager limit was exceeded"},
    422: {"model": ErrorResponse, "description": "Request validation failed"},
    500: {"model": ErrorResponse, "description": "A stored record is corrupt"},
    503: {"model": ErrorResponse, "description": "Memory manager is not configured"},
}
_STORE_ERRORS = {
    **_COMMON_ERRORS,
    409: {"model": ErrorResponse, "description": "Memory identifier already exists"},
    413: {"model": ErrorResponse, "description": "A configured manager limit was exceeded"},
}


def _response(memory: Memory) -> MemoryResponse:
    return MemoryResponse(
        memory_id=memory.memory_id,
        content=memory.content,
        metadata=dict(memory.metadata),
        source=memory.source,
        event_timestamp=memory.event_timestamp,
        entities=list(memory.entities),
        created_at=memory.created_at,
        score=memory.score,
    )


def _store_response(memory: Memory) -> StoreResponse:
    return StoreResponse(
        memory_id=memory.memory_id,
        source=memory.source,
        event_timestamp=memory.event_timestamp,
        entities=list(memory.entities),
        created_at=memory.created_at,
    )


def create_app(
    manager: VectorStoreManager | None = None,
    authorization_policy: AuthorizationPolicy | None = None,
) -> FastAPI:
    """Create an API with injected storage and authorization dependencies.

    Omitting ``authorization_policy`` denies every request. This deliberate default
    prevents an embedded or accidentally exposed app from becoming unauthenticated.
    """

    app = FastAPI(
        title="Universal AI Memory API",
        version="0.2.0",
        description=(
            "Local-first encrypted payload storage. Backend-visible vectors, IDs, "
            "timestamps, and blind indexes are authenticated but are not encrypted."
        ),
    )
    app.state.manager = manager
    app.state.authorization_policy = authorization_policy or DenyAllAuthorizationPolicy()
    bearer = HTTPBearer(auto_error=False, scheme_name="BearerAuth")

    def get_manager() -> VectorStoreManager:
        configured = app.state.manager
        if configured is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="memory manager is not configured",
            )
        return configured

    @app.middleware("http")
    async def request_boundary(request: Request, call_next):
        if request.url.path in {"/store", "/retrieve", "/search", "/forget"}:
            authorization = request.headers.get("authorization", "")
            scheme, separator, token = authorization.partition(" ")
            if separator == "" or scheme.casefold() != "bearer" or not token:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "bearer authentication required"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            try:
                granted = app.state.authorization_policy.capabilities_for(token)
            except Exception:
                granted = None
            if granted is None:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "invalid bearer credential"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > MAX_REQUEST_BODY_BYTES:
                        raise ValueError
                except ValueError:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={"detail": "request body exceeds the configured limit"},
                    )
            body = await request.body()
            if len(body) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={"detail": "request body exceeds the configured limit"},
                )
        return await call_next(request)

    def require_capability(capability: Capability):
        def authorize(
            credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
        ) -> None:
            if credentials is None or credentials.scheme.casefold() != "bearer":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="bearer authentication required",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            try:
                granted = app.state.authorization_policy.capabilities_for(credentials.credentials)
            except Exception:
                granted = None
            if granted is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="invalid bearer credential",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            try:
                capabilities = frozenset(Capability(value) for value in granted)
            except (TypeError, ValueError):
                capabilities = frozenset()
            if capability not in capabilities:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="bearer credential lacks required capability",
                )

        return authorize

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "request validation failed"},
        )

    @app.exception_handler(MemoryLimitError)
    async def memory_limit_handler(_request: Request, exc: MemoryLimitError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, content={"detail": exc.detail})

    @app.exception_handler(MemoryValidationError)
    async def memory_validation_handler(_request: Request, exc: MemoryValidationError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": exc.detail})

    @app.exception_handler(MemoryConflictError)
    async def memory_conflict_handler(_request: Request, exc: MemoryConflictError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": exc.detail})

    @app.exception_handler(CorruptRecordError)
    async def corrupt_record_handler(_request: Request, exc: CorruptRecordError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": exc.detail})

    @app.exception_handler(StorageBackendError)
    async def storage_backend_handler(_request: Request, exc: StorageBackendError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": exc.detail})

    @app.exception_handler(MemoryServiceError)
    async def memory_error_handler(_request: Request, _exc: MemoryServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "memory operation failed"},
        )

    @app.post(
        "/store",
        response_model=StoreResponse,
        status_code=status.HTTP_201_CREATED,
        operation_id="storeMemory",
        responses=_STORE_ERRORS,
        dependencies=[Depends(require_capability(Capability.WRITE))],
    )
    def store(request: StoreRequest, service: VectorStoreManager = Depends(get_manager)) -> StoreResponse:
        memory = service.store(
            request.content,
            request.metadata,
            source=request.source,
            event_timestamp=request.event_timestamp,
            memory_id=request.memory_id,
        )
        return _store_response(memory)

    @app.post(
        "/retrieve",
        response_model=MemoriesResponse,
        operation_id="retrieveMemories",
        responses=_COMMON_ERRORS,
        dependencies=[Depends(require_capability(Capability.READ))],
    )
    def retrieve(
        request: RetrieveRequest,
        service: VectorStoreManager = Depends(get_manager),
    ) -> MemoriesResponse:
        memories = service.retrieve(
            memory_id=request.memory_id,
            entities=request.entities,
            limit=request.limit,
            newest_first=request.newest_first,
        )
        return MemoriesResponse(memories=[_response(memory) for memory in memories])

    @app.post(
        "/search",
        response_model=MemoriesResponse,
        operation_id="searchMemories",
        responses=_COMMON_ERRORS,
        dependencies=[Depends(require_capability(Capability.READ))],
    )
    def search(request: SearchRequest, service: VectorStoreManager = Depends(get_manager)) -> MemoriesResponse:
        memories = service.search(request.query, entities=request.entities, limit=request.limit)
        return MemoriesResponse(memories=[_response(memory) for memory in memories])

    @app.post(
        "/forget",
        response_model=ForgetResponse,
        operation_id="forgetMemories",
        responses=_COMMON_ERRORS,
        dependencies=[Depends(require_capability(Capability.DELETE))],
    )
    def forget(request: ForgetRequest, service: VectorStoreManager = Depends(get_manager)) -> ForgetResponse:
        return ForgetResponse(deleted=service.forget(memory_id=request.memory_id, entities=request.entities))

    return app


def manager_from_environment() -> VectorStoreManager:
    """Build the local starter manager from base64 keys without logging them."""
    encryption_key = _required_key("UNIVERSAL_MEMORY_ENCRYPTION_KEY", exact_length=32)
    index_key = _required_key("UNIVERSAL_MEMORY_INDEX_KEY", minimum_length=32)
    return VectorStoreManager(
        embedder=DeterministicHashEmbedder(),
        backend=InMemoryVectorBackend(),
        entity_extractor=SimpleEntityExtractor(),
        encryption_key=encryption_key,
        entity_index_key=index_key,
    )


def authorization_policy_from_environment() -> ApiKeyAuthorizationPolicy:
    """Build the runnable API's fail-closed bearer policy from an environment key."""
    api_key = os.environ.get("UNIVERSAL_MEMORY_API_KEY")
    if api_key is None or not api_key:
        raise RuntimeError("required environment variable UNIVERSAL_MEMORY_API_KEY is not set")
    return ApiKeyAuthorizationPolicy(api_key)


def _required_key(name: str, *, exact_length: int | None = None, minimum_length: int | None = None) -> bytes:
    encoded = os.environ.get(name)
    if encoded is None:
        raise RuntimeError(f"required environment variable {name} is not set")
    try:
        value = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"{name} must be valid base64") from exc
    if exact_length is not None and len(value) != exact_length:
        raise RuntimeError(f"{name} must decode to exactly {exact_length} bytes")
    if minimum_length is not None and len(value) < minimum_length:
        raise RuntimeError(f"{name} must decode to at least {minimum_length} bytes")
    return value


def configured_app() -> FastAPI:
    """Uvicorn factory requiring encryption keys and a bearer API key."""
    return create_app(manager_from_environment(), authorization_policy_from_environment())


def main() -> None:
    uvicorn.run("universal_memory.api:configured_app", factory=True, host="127.0.0.1", port=8000)


# Import-safe and fail-closed: inject both dependencies with create_app(...).
app = create_app()
