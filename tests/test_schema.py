from __future__ import annotations

from pathlib import Path

from universal_memory.api import ApiKeyAuthorizationPolicy, create_app
from universal_memory.vector_store_manager import (
    DeterministicHashEmbedder,
    InMemoryVectorBackend,
    SimpleEntityExtractor,
    VectorStoreManager,
)

ROOT = Path(__file__).resolve().parents[1]


def manager() -> VectorStoreManager:
    return VectorStoreManager(
        embedder=DeterministicHashEmbedder(),
        backend=InMemoryVectorBackend(),
        entity_extractor=SimpleEntityExtractor(),
        encryption_key=b"e" * 32,
        entity_index_key=b"i" * 32,
    )


def test_generated_openapi_has_security_capability_errors_and_limits() -> None:
    schema = create_app(manager(), ApiKeyAuthorizationPolicy("test-key")).openapi()
    security_scheme = schema["components"]["securitySchemes"]["BearerAuth"]
    assert security_scheme["type"] == "http"
    assert security_scheme["scheme"] == "bearer"

    expected_operations = {
        "/store": "storeMemory",
        "/retrieve": "retrieveMemories",
        "/search": "searchMemories",
        "/forget": "forgetMemories",
    }
    for path, operation_id in expected_operations.items():
        operation = schema["paths"][path]["post"]
        assert operation["operationId"] == operation_id
        assert operation["security"] == [{"BearerAuth": []}]
        assert {"401", "403", "413", "422", "500", "503"} <= set(operation["responses"])
    assert "409" in schema["paths"]["/store"]["post"]["responses"]

    retrieve_schema = schema["components"]["schemas"]["RetrieveRequest"]
    retrieve = retrieve_schema["properties"]
    assert retrieve["entities"]["maxItems"] == 64
    assert retrieve["entities"]["items"]["maxLength"] == 256
    assert retrieve["memory_id"]["anyOf"][0]["maxLength"] == 256
    assert retrieve_schema["allOf"][0]["not"]["required"] == ["memory_id", "entities"]

    forget_schema = schema["components"]["schemas"]["ForgetRequest"]
    assert len(forget_schema["oneOf"]) == 2
    id_branch = forget_schema["oneOf"][0]
    assert id_branch["properties"]["memory_id"]["type"] == "string"
    assert id_branch["properties"]["entities"]["maxItems"] == 0
    assert forget_schema["oneOf"][1]["properties"]["entities"]["minItems"] == 1


def test_checked_openapi_tracks_generated_contract_and_is_wheel_included() -> None:
    artifact = (ROOT / "openapi.yaml").read_text(encoding="utf-8")
    for marker in (
        "version: 0.2.0",
        "BearerAuth:",
        "scheme: bearer",
        "operationId: storeMemory",
        "operationId: retrieveMemories",
        "operationId: searchMemories",
        "operationId: forgetMemories",
        "'401'",
        "'403'",
        "'413'",
        "'422'",
        "'500'",
        "maxItems: 64",
        "maxLength: 256",
    ):
        assert marker in artifact

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"openapi.yaml" = "universal_memory/openapi.yaml"' in pyproject
