"""Capability-gated Python MCP FastMCP adapter for universal memory."""

from __future__ import annotations

import os
from collections.abc import Collection
from datetime import datetime
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .api import Capability, manager_from_environment
from .vector_store_manager import Memory, VectorStoreManager

Content = Annotated[str, Field(min_length=1, max_length=65_536)]
Source = Annotated[str, Field(min_length=1, max_length=2_048)]
MemoryId = Annotated[str, Field(min_length=1, max_length=256)]
Entity = Annotated[str, Field(min_length=1, max_length=256)]
Entities = Annotated[list[Entity], Field(max_length=64)]


def _memory_dict(memory: Memory) -> dict[str, Any]:
    return {
        "memory_id": memory.memory_id,
        "content": memory.content,
        "metadata": dict(memory.metadata),
        "source": memory.source,
        "event_timestamp": memory.event_timestamp.isoformat(),
        "entities": list(memory.entities),
        "created_at": memory.created_at.isoformat(),
        "score": memory.score,
    }


def _store_dict(memory: Memory) -> dict[str, Any]:
    return {
        "memory_id": memory.memory_id,
        "source": memory.source,
        "event_timestamp": memory.event_timestamp.isoformat(),
        "entities": list(memory.entities),
        "created_at": memory.created_at.isoformat(),
    }


def create_mcp_server(
    manager: VectorStoreManager,
    capabilities: Collection[Capability | str] = (),
) -> FastMCP:
    """Create a server with fixed capabilities supplied by its trusted launcher.

    The default grants nothing. MCP transports that authenticate clients should
    construct a separately scoped server/session rather than exposing these tools
    with ambient authority.
    """

    granted = frozenset(Capability(value) for value in capabilities)
    server = FastMCP("universal-ai-memory")

    def require(capability: Capability) -> None:
        if capability not in granted:
            raise PermissionError(f"MCP capability required: {capability.value}")

    @server.tool(name="store")
    def store(
        content: Content,
        metadata: dict[str, Any] | None = None,
        source: Source | None = None,
        event_timestamp: datetime | None = None,
        memory_id: MemoryId | None = None,
    ) -> dict[str, Any]:
        """Encrypt and store a memory; requires the write capability."""
        require(Capability.WRITE)
        memory = manager.store(
            content,
            metadata,
            source=source,
            event_timestamp=event_timestamp,
            memory_id=memory_id,
        )
        return _store_dict(memory)

    @server.tool(name="retrieve")
    def retrieve(
        memory_id: MemoryId | None = None,
        entities: Entities | None = None,
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
        newest_first: bool = True,
    ) -> dict[str, Any]:
        """Retrieve memories; requires read and rejects ambiguous selectors."""
        require(Capability.READ)
        memories = manager.retrieve(
            memory_id=memory_id,
            entities=entities or (),
            limit=limit,
            newest_first=newest_first,
        )
        return {"memories": [_memory_dict(memory) for memory in memories]}

    @server.tool(name="search")
    def search(
        query: Content,
        entities: Entities | None = None,
        limit: Annotated[int, Field(ge=1, le=100)] = 10,
    ) -> dict[str, Any]:
        """Search memories; requires the read capability."""
        require(Capability.READ)
        memories = manager.search(query, entities=entities or (), limit=limit)
        return {"memories": [_memory_dict(memory) for memory in memories]}

    @server.tool(name="forget")
    def forget(
        memory_id: MemoryId | None = None,
        entities: Entities | None = None,
    ) -> dict[str, int]:
        """Delete selected memories; requires the delete capability."""
        require(Capability.DELETE)
        return {"deleted": manager.forget(memory_id=memory_id, entities=entities or ())}

    return server


def capabilities_from_environment() -> frozenset[Capability]:
    """Parse explicit comma-separated MCP capabilities; absence grants none."""
    raw = os.environ.get("UNIVERSAL_MEMORY_MCP_CAPABILITIES", "")
    values = [value.strip() for value in raw.split(",") if value.strip()]
    try:
        return frozenset(Capability(value) for value in values)
    except ValueError as exc:
        raise RuntimeError("UNIVERSAL_MEMORY_MCP_CAPABILITIES contains an unknown capability") from exc


def main() -> None:
    create_mcp_server(manager_from_environment(), capabilities_from_environment()).run()


if __name__ == "__main__":
    main()
