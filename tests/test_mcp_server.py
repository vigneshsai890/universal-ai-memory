from __future__ import annotations

import json

import pytest

from universal_memory.api import Capability
from universal_memory.mcp_server import create_mcp_server
from universal_memory.vector_store_manager import (
    DeterministicHashEmbedder,
    InMemoryVectorBackend,
    SimpleEntityExtractor,
    VectorStoreManager,
)


def manager() -> VectorStoreManager:
    return VectorStoreManager(
        embedder=DeterministicHashEmbedder(),
        backend=InMemoryVectorBackend(),
        entity_extractor=SimpleEntityExtractor(),
        encryption_key=b"e" * 32,
        entity_index_key=b"i" * 32,
    )


def tool(server, name: str):
    registered = server._tool_manager.get_tool(name)
    assert registered is not None
    return registered


def test_mcp_defaults_to_no_capabilities() -> None:
    server = create_mcp_server(manager())
    with pytest.raises(PermissionError, match="write"):
        tool(server, "store").fn(content="private")
    with pytest.raises(PermissionError, match="read"):
        tool(server, "retrieve").fn()
    with pytest.raises(PermissionError, match="read"):
        tool(server, "search").fn(query="private")
    with pytest.raises(PermissionError, match="delete"):
        tool(server, "forget").fn(memory_id="one")


def test_mcp_capabilities_are_independent() -> None:
    service = manager()
    write_server = create_mcp_server(service, [Capability.WRITE])
    stored = tool(write_server, "store").fn(content="private", memory_id="one")
    assert stored["memory_id"] == "one"
    with pytest.raises(PermissionError):
        tool(write_server, "retrieve").fn(memory_id="one")

    read_server = create_mcp_server(service, [Capability.READ])
    assert tool(read_server, "retrieve").fn(memory_id="one")["memories"][0]["content"] == "private"
    with pytest.raises(PermissionError):
        tool(read_server, "forget").fn(memory_id="one")

    delete_server = create_mcp_server(service, [Capability.DELETE])
    assert tool(delete_server, "forget").fn(memory_id="one") == {"deleted": 1}


def test_mcp_schema_has_entity_item_and_collection_limits() -> None:
    schema_text = json.dumps(tool(create_mcp_server(manager()), "retrieve").parameters)
    assert '"maxItems": 64' in schema_text
    assert '"maxLength": 256' in schema_text
    assert '"minimum": 1' in schema_text
    assert '"maximum": 100' in schema_text
