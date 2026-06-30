# News Narrative Explainer V3

`v3/` is the performance-focused Rust rebuild of the current Python graph
builder, with DuckDB kept on the client side and parquet kept as the canonical
storage format.

## Architecture

The new target shape is:

- Rust builds the graph once per day from local source parquet.
- Hot graph artifacts stay in partitioned parquet for cheap remote scans.
- Cold document payloads are split into separate partitioned parquet so large
  content fields do not bloat the hot query path.
- A lightweight Rust HTTP server exposes those parquet files with byte-range
  support so DuckDB clients can read them remotely with `httpfs`.

This keeps the expensive matching and materialization path under direct Rust
control while still letting DuckDB do selective remote reads, projection
pruning, predicate pruning, and local caching on the consumer side.

## Artifact Layout

Default output root:

```text
/Users/jamiepearcey/projects/research/news-narrative-explainer/data/narrative_graph_parquet_v3
```

Key daily artifacts now include:

- `graph_doc_nodes_daily/partition_date=YYYY-MM-DD/part-000.parquet`
  Hot document-node slice for graph traversal and joins.
- `doc_payload_daily/partition_date=YYYY-MM-DD/part-000.parquet`
  Cold payload slice for title, summary, body, and other heavy review fields.
- `doc_review_daily/partition_date=YYYY-MM-DD/part-*.parquet`
  Chunked lighter review payload slices plus `index.json` with sorted `doc_id`
  ranges.
- `doc_detail_daily/partition_date=YYYY-MM-DD/part-*.parquet`
  Chunked heavier detail payload slices plus `index.json` with sorted `doc_id`
  ranges.
- `silver_*` daily parquet outputs
- `gold_*` daily parquet outputs
- `manifests/materialized_days.json`

Default benchmark artifact location:

```text
/Users/jamiepearcey/projects/research/news-narrative-explainer/v3/results/
```

## Commands

Build one local day from already-downloaded parquet:

```bash
cargo run --release --bin news-narrative-v3 -- build-local-day \
  --date 2026-06-05 \
  --input-glob "gdelt_candidates_20d_full/dt=*/part-*.parquet"
```

Build one local day and persist a benchmark artifact:

```bash
cargo run --release --bin news-narrative-v3 -- benchmark-local-day \
  --date 2026-06-05 \
  --input-glob "gdelt_candidates_20d_full/dt=*/part-*.parquet"
```

Serve the materialized parquet tree over HTTP with range support:

```bash
cargo run --release --bin news-narrative-v3 -- serve-artifacts \
  --output-root /Users/jamiepearcey/projects/research/news-narrative-explainer/data/narrative_graph_parquet_v3 \
  --bind 127.0.0.1:8789
```

Print example DuckDB client SQL for remote reads:

```bash
cargo run --release --bin news-narrative-v3 -- print-duckdb-client-sql \
  --base-url http://127.0.0.1:8789 \
  --date 2026-06-05
```

Run the hosted-parquet MCP with the same explanation-oriented tool surface as
the older Python MCP:

```bash
uv run --with 'duckdb>=1.0' python v3/v3_narrative_explainer_mcp.py
```

Run the self-contained `v3` validation client against the hosted MCP and the
hybrid retrieval server:

```bash
uv run --with 'duckdb>=1.0' python v3/v3_validate_mcp_narrative.py \
  --date 2026-05-31 \
  --hybrid-search-url http://127.0.0.1:8788/search
```

Run the hybrid retrieval server with Swagger docs:

```bash
cargo run --release --bin qdrant_day -- serve --bind 127.0.0.1:8788
```

Swagger UI is then available at [http://127.0.0.1:8788/docs](http://127.0.0.1:8788/docs).

Batch rebuild indexed collections over HTTP:

```bash
curl -X POST http://127.0.0.1:8788/index/rebuild-days \
  -H 'content-type: application/json' \
  -d '{
    "dates": ["2026-06-05", "2026-06-06"],
    "recreate": true
  }'
```

## DuckDB Client Pattern

DuckDB remains the query client, not the graph builder.

Typical flow:

1. Read hot remote parquet directly over HTTP.
2. Filter down to the small set of `doc_id` values worth inspecting.
3. Hydrate broader light review rows from `doc_review_daily`.
4. Use the `doc_id` sidecar index to target only the relevant
   `doc_detail_daily` chunk files for the final review stage.
4. Optionally persist local DuckDB cache tables if the same slice will be
   reused repeatedly.

A ready-to-run example lives at `v3/sql/duckdb_remote_client.sql`.

## V3 MCP

`v3/v3_narrative_explainer_mcp.py` is a Python MCP server that keeps the
existing MCP tool names and output shape, but resolves its query workspace from
the hosted parquet tree instead of the older local parquet adapter.

`v3/` now also carries its own supporting Python query/matching modules, so the
MCP no longer imports core query logic from the legacy `scripts/` directory.

The default `db` argument is:

```text
http://127.0.0.1:8789
```

That value should point at the root served by `serve-artifacts`. The MCP then:

1. fetches `manifest.json`
2. creates a temporary DuckDB catalog that points at hosted parquet over
   `httpfs`
3. exposes the hot graph and silver/gold layers as parquet-backed views rather
   than copied local tables
4. loads `doc_review_daily_file_index` and `doc_detail_daily_file_index`
   sidecar metadata when available
5. hydrates `doc_review_daily` for reranking and only the targeted
   `doc_detail_daily` chunk files for shortlisted documents

Repeated hosted queries now also reuse a persistent local DuckDB catalog cache
under:

```text
/Users/jamiepearcey/projects/research/news-narrative-explainer/data/.cache/v3_duckdb_query
```

That cache is keyed by store root, manifest fingerprint, requested date scope,
gold scope, and load profile.

In addition to the day/factor explanation tools, the MCP now also exposes
`assess_event_impact`, which is an event-first path for questions such as:

- "Would a Hormuz disruption likely impact WTI, Gold, or DXY?"
- "Does this refinery outage look relevant for crack spreads or crude?"

That tool uses the hybrid retrieval layer first, then runs the existing
deterministic asset/factor transmission rules over the retrieved corpus hits.
It expects either:

- a `date`, which maps to the default day collection name
  `news_narrative_v3_YYYYMMDD_allminilm`
- or an explicit `collection`

If the HTTP hybrid endpoint is unavailable, the MCP falls back to a local
`cargo run --release --bin qdrant_day -- search ...` call against the same
Qdrant collection.

`qdrant_day` now indexes from the `v3` artifact root rather than the legacy
`bronze_candidates` parquet. For each document it joins:

- `graph_doc_nodes_daily` for source metadata plus `market_context_text`
- `doc_review_daily` for `relevant_text`, `metadata_json`, and `quotations`
- `doc_payload_daily` for `body_text` when present

Because many current source rows still do not have first-class article
`title`, `summary_text`, or `body_text`, the indexer also derives fallback
search/display text from normalized keyword spans and URL slugs so retrieval is
not operating on null payloads.

## Benchmark Workflow

For each performance pass:

1. Run `benchmark-local-day` on the same day slice.
2. Save the emitted JSON artifact in `v3/results/`.
3. Compare input load time, bronze build time, silver build time, write time,
   and row counts per materialized layer.

Benchmarks should be kept as durable artifacts rather than transient terminal
output.

## Current Scope

The current `v3` implementation:

- reads local source parquet
- rebuilds bronze candidates in Rust
- builds silver and gold parquet outputs in Rust
- splits hot document-node output from cold payload output
- serves generated parquet over HTTP with single-range byte requests
- exposes artifact-server admin endpoints for fetch/load/fetch-and-load day
  operations plus Swagger docs
- serves hybrid Qdrant retrieval over HTTP with OpenAPI and Swagger UI
- resolves the hosted-parquet MCP path through direct DuckDB parquet views
  instead of eager temp-table copies
- keeps the MCP/query/validator Python path self-contained within `v3/`
- emits benchmark JSON artifacts for repeated comparisons

## Notes

- The artifact server still keeps a small surface: static file serving,
  `GET`/`HEAD` with single `Range` header support, plus explicit admin `POST`
  endpoints for fetch/load orchestration.
- The first optimization target remains the Rust materialization path, not SQL
  engine rewrites.
- End-to-end runtime against the full local corpus still needs another live
  benchmark pass after the current bronze-row runtime failure is resolved.

## Artifact Admin API

Run the parquet artifact server with docs:

```bash
cargo run --release --bin news-narrative-v3 -- serve-artifacts --bind 127.0.0.1:8789
```

Swagger UI is then available at [http://127.0.0.1:8789/docs](http://127.0.0.1:8789/docs).

Fetch one or more exported GCS days into the existing local day layout:

```bash
curl -X POST http://127.0.0.1:8789/admin/fetch-days \
  -H 'content-type: application/json' \
  -d '{
    "dates": ["2026-06-05"],
    "gcs_uri_template": "gs://market-news-datwa/gdelt/day-{date}/*.parquet",
    "overwrite": true
  }'
```

Load one or more locally fetched days into the graph parquet output:

```bash
curl -X POST http://127.0.0.1:8789/admin/load-days \
  -H 'content-type: application/json' \
  -d '{
    "dates": ["2026-06-05"],
    "overwrite_day": true
  }'
```

Fetch and then load days in one request:

```bash
curl -X POST http://127.0.0.1:8789/admin/fetch-and-load-days \
  -H 'content-type: application/json' \
  -d '{
    "dates": ["2026-06-05", "2026-06-06"],
    "overwrite_fetch": true,
    "overwrite_day": true
  }'
```
