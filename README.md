# Universal AI Memory starter

Python 3.12 local-first reference service with injected embedding, entity extraction, authorization, and vector persistence adapters. Private payload fields use AES-256-GCM; normalized entity labels are represented in the backend as HMAC-SHA-256 blind indexes. The included backend and deterministic embedder are development implementations.

## Install and test

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest
```

Tests are in-process and make no network requests. `openapi.yaml` is included in the wheel as `universal_memory/openapi.yaml`.

## Run the HTTP API

Provide two distinct base64 keys and a non-empty bearer API key. Key-generation commands print secrets to the terminal; store real values in a secret manager, not source or logs.

```bash
export UNIVERSAL_MEMORY_ENCRYPTION_KEY="$(openssl rand -base64 32)"
export UNIVERSAL_MEMORY_INDEX_KEY="$(openssl rand -base64 32)"
export UNIVERSAL_MEMORY_API_KEY="$(openssl rand -hex 32)"
universal-memory-api
```

The API binds to `127.0.0.1:8000`. Every operation requires `Authorization: Bearer $UNIVERSAL_MEMORY_API_KEY`. Runnable mode grants that key read, write, and delete capabilities. Embedded applications must inject both a manager and an `AuthorizationPolicy` into `create_app`; omitting a policy denies all requests. A recognized credential without the operation's capability receives 403, while a missing or invalid credential receives 401.

```bash
curl -H "Authorization: Bearer $UNIVERSAL_MEMORY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"content":"Met Alice at launch","metadata":{"project":"demo"}}' \
  http://127.0.0.1:8000/store
```

## Run MCP

MCP tools are independently gated as `read`, `write`, and `delete`. No capability is granted by default. The local stdio launcher accepts an explicit comma-separated capability set:

```bash
export UNIVERSAL_MEMORY_MCP_CAPABILITIES=read,write,delete
universal-memory-mcp
```

Use `create_mcp_server(manager, capabilities={Capability.READ})` for a least-privilege embedded server. An MCP transport with per-client authentication should construct separately scoped sessions rather than sharing ambient capabilities.

## Data and validation model

A store request may provide `content`, strict-JSON `metadata`, an encrypted `source`, a timezone-aware `event_timestamp`, and an optional ID. `created_at` is always service generated. Event chronology uses `event_timestamp`. Entity labels extracted during storage are encrypted in the payload and exposed only to authorized responses; filters use keyed blind indexes.

`VectorStoreManager` is the enforcement boundary even when HTTP/MCP schemas are bypassed. Default caps are: content 65,536 UTF-8 bytes, source 2,048 bytes, ID 256 bytes, canonical metadata 65,536 bytes and depth 12, 64 entities of 256 bytes each, 4,096 vector dimensions, and 10,000 records. Metadata must be acyclic strict JSON with string object keys and finite numbers. Validation and canonicalization occur before embedding. Limits can be reduced through `MemoryLimits`.

ID and entity selectors cannot be combined. Delete-by-entity authenticates complete records before trusting blind indexes or IDs. Malformed, unauthenticated, or corrupt records produce a sanitized domain error and are not used for sorting, scoring, or deletion.

## Security boundaries

- Content, metadata, source, event timestamp, and raw entity labels are encrypted in the payload.
- **Vectors are not encrypted and may leak semantic information.** IDs, ingestion/event timestamps, nonce, ciphertext length, vectors, and HMAC entity indexes are visible to the backend.
- Clear created/event timestamps plus digests of the vector and entity-index set are encoded canonically into AES-GCM associated data. This authenticates those fields; it does not make them private.
- Blind indexes hide raw labels but permit equality correlation. Use an independent high-entropy index key.
- The API uses bearer authentication, not transport encryption. Keep the default loopback binding or add trusted TLS before crossing a network boundary.
- The in-memory backend loses records on restart and is not suitable for multi-process or production deployment. Production adapters also need durable atomic capacity enforcement, key rotation, auditing, and operational controls.
- The service does not intentionally log plaintext or secrets. Decrypted API/MCP responses are returned only after capability checks.
