# ADR-0003: ClickHouse MCP In Local And Client-Server Modes

## Status

Accepted

## Date

2026-06-28

## Context

The original MCP wrapper in `scripts/narrative_explainer_mcp.py` is tied to the
DuckDB/parquet path and performs narrative-selection and explanation assembly
inside the server process. V2 is moving the graph itself into ClickHouse, and
the next serving shape needs to support:

- one shared ClickHouse graph across many users
- local direct-to-ClickHouse MCP usage for development
- remote client/server usage where many clients consume one shared API
- compatibility with the existing MCP capability surface
- removal of server-side narrative reasoning so the client LLM can use raw
  graph capabilities directly

## Decision

V2 will add a new Rust MCP surface inside `v2/` with three modes:

- `mcp-stdio`
  Direct local MCP stdio mode against ClickHouse.
- `serve-api`
  HTTP API mode exposing the same MCP request surface for shared remote use.
- `mcp-proxy`
  Thin local stdio proxy that forwards MCP requests to the remote API.

The v2 MCP keeps the existing high-level tool names for compatibility, but the
tool payloads become raw structured graph bundles instead of server-written
natural-language reasoning.

The server now does:

- ClickHouse reads
- deterministic ranking and retrieval helpers already embodied in the graph and
  supporting-doc candidate selection
- raw bundle assembly
- MCP transport and HTTP transport

The server no longer does:

- final narrative text synthesis
- contradiction prose
- regime-choice prose
- day-frame prose

That LLM-side reasoning is delegated to the client consuming the MCP.

## Consequences

Positive:

- One shared ClickHouse graph can serve many users without copying graph state
  per client.
- MCP local mode and client/server mode use the same Rust implementation.
- Consumers keep a compatible MCP tool surface while gaining direct access to
  lower-level graph data.
- The expensive DuckDB adapter path is avoided for v2 serving.

Tradeoffs:

- The compatibility tool names now return raw structured payloads rather than
  finished prose, so client prompts must take on the final reasoning step.
- Existing consumers that relied on the old `db` filesystem semantics need to
  treat `db` as a legacy alias for ClickHouse database selection, or omit it.
- The v2 MCP depends on the ClickHouse graph being queryable and sufficiently
  materialized before use.
