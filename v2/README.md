# News Narrative Explainer V2

ClickHouse-first reimplementation of the narrative graph pipeline.

This version keeps the existing deterministic taxonomy and output concepts, but
changes the execution model:

- BigQuery exports directly to Google Cloud Storage with native `EXPORT DATA`
- ClickHouse loads the exported parquet from GCS with a `gcs(...)` query
- Rust handles the small set of row-level enrichments that are awkward in
  ClickHouse:
  - HTML/entity normalization
  - page-title fallback extraction from `gkg_extras`
  - source classification
  - market-context sentence selection
- ClickHouse performs the factor/asset matching and the silver/gold graph
  materialization

## Layout

- `src/main.rs`
  Rust CLI for orchestration, enrichment, and query helpers.
- `sql/schema.sql`
  ClickHouse schema and analytical views.
- `sql/load_gcs_into_clickhouse.sql`
  ClickHouse `INSERT ... SELECT FROM gcs(...)` load template.
- `sql/load_local_parquet_into_clickhouse.sql`
  ClickHouse `INSERT ... SELECT FROM file(...)` load template for already
  downloaded parquet.

## Environment

Local ClickHouse via Docker Compose:

```bash
docker compose -f /Users/jamiepearcey/projects/research/news-narrative-explainer/v2/docker-compose.yml up -d
```

ClickHouse HTTP connection:

```bash
export CLICKHOUSE_URL="http://localhost:8123"
export CLICKHOUSE_DATABASE="default"
export CLICKHOUSE_USER="default"
export CLICKHOUSE_PASSWORD=""
```

GCS HMAC credentials for ClickHouse object-storage reads:

```bash
export GCS_HMAC_ACCESS_KEY="..."
export GCS_HMAC_SECRET_KEY="..."
```

BigQuery export requires the `bq` CLI to be installed and authenticated.

## Bootstrap

Apply schema and load taxonomy/config tables:

```bash
cargo run -- bootstrap
```

## Export BigQuery To GCS

This uses native BigQuery `EXPORT DATA` to write partitioned parquet directly
to Google Cloud Storage.

```bash
cargo run -- export-bigquery-to-gcs \
  --project your-gcp-project \
  --location US \
  --start 2026-06-15T00:00:00Z \
  --end 2026-06-25T00:00:00Z \
  --bucket your-bucket \
  --prefix news-narrative-v2/gdelt \
  --rows-per-day 2000
```

Dry-run SQL generation only:

```bash
cargo run -- export-bigquery-to-gcs \
  --project your-gcp-project \
  --location US \
  --start 2026-06-15T00:00:00Z \
  --end 2026-06-25T00:00:00Z \
  --bucket your-bucket \
  --dry-run \
  --include-queries
```

## Load GCS Into ClickHouse

This step is intentionally a ClickHouse query, not a Rust-side parquet copy.

```bash
cargo run -- load-gcs-into-clickhouse \
  --gcs-url "https://storage.googleapis.com/your-bucket/news-narrative-v2/gdelt/dt=*/part-*.parquet" \
  --start-date 2026-06-15 \
  --end-date 2026-06-25
```

## Load Local Parquet Into ClickHouse

This path is for already-downloaded parquet such as the existing 20-day corpus.
It uses ClickHouse `file(...)` directly.

```bash
cargo run -- load-local-parquet-into-clickhouse \
  --input-glob "gdelt_candidates_20d_full/dt=*/part-*.parquet" \
  --start-date 2026-06-05 \
  --end-date 2026-06-25
```

For the corpus currently present in the repo, the first discovered file is:

```text
/Users/jamiepearcey/projects/research/news-narrative-explainer/data/gdelt_candidates_20d_full/dt=2026-06-05/part-20260626T211008Z-bigquery-window.parquet
```

Operational caveat:

- `file(...)` reads from ClickHouse `user_files_path`, not arbitrary host
  paths. The provided Compose stack mounts the existing local corpora into
  `/var/lib/clickhouse/user_files/`, so the command uses a relative path such
  as `gdelt_candidates_20d_full/dt=*/part-*.parquet`.
- Local file loads are now cataloged in ClickHouse `ingest_file_catalog` with
  per-file path, SHA-256, size, partition date, and loaded row count so reruns
  can skip already-loaded parquet files.

## Enrich Bronze

Transform raw exported rows into enriched `bronze_candidates`:

```bash
cargo run -- enrich-bronze \
  --start-date 2026-06-15 \
  --end-date 2026-06-25 \
  --batch-size 2000
```

This step now runs in bounded memory:

- it processes one `partition_date` at a time
- it pages through raw bronze rows in `document_identifier, ingested_at` order
- it inserts each transformed batch immediately instead of retaining the full
  window in memory

## Build The ClickHouse Narrative Graph

Materialize the silver relationship layers and expose gold layers through
ClickHouse views:

```bash
cargo run -- build-clickhouse-graph \
  --start-date 2026-06-15 \
  --end-date 2026-06-25 \
  --source-uri "gs://your-bucket/news-narrative-v2/gdelt"
```

The build step only processes partitions that are not already recorded in
`graph_build_partitions`.

## Query

Summary:

```bash
cargo run -- query --view summary
```

Top factors:

```bash
cargo run -- query --view top-factors --limit 20
```

Asset narratives:

```bash
cargo run -- query \
  --view asset-narratives \
  --asset-label WTI \
  --start-date 2026-06-18 \
  --end-date 2026-06-23 \
  --limit 10
```

Explain move:

```bash
cargo run -- query \
  --view explain-move \
  --asset-label WTI \
  --start-date 2026-06-18 \
  --end-date 2026-06-23 \
  --limit 10
```

## Benchmark Rust Work

This measures the non-ClickHouse Rust-side transform/scoring work only.

```bash
cargo run -- benchmark-rust-work --iterations 200000
```

Current local synthetic benchmark baseline:

- `transform_bronze_row`: about `99.9k ops/sec` and about `10.0us` per row
- `supporting_doc_relevance`: about `98.3k ops/sec` and about `10.2us` per row

Live ClickHouse-backed checks already performed:

- one-day local parquet load for `2026-06-05` loaded `178845` rows into
  `bronze_raw_gdelt`
- ClickHouse reported about `6.34s` load time, about `28.2k rows/sec`, about
  `104.4 MiB/sec`, and about `1.05 GiB` query peak memory for that day-level
  insert
- one-day `enrich-bronze` for `2026-06-05` inserted `178845` rows in about
  `46.66s`
- the current one-day `build-clickhouse-graph` attempt is still the dominant
  bottleneck and currently fails with ClickHouse `MEMORY_LIMIT_EXCEEDED`
  before asset-factor materialization completes

Durable benchmark record:

- `/Users/jamiepearcey/projects/research/news-narrative-explainer/docs/benchmarks/v2-clickhouse-baselines.md`
- `/Users/jamiepearcey/projects/research/news-narrative-explainer/v2/results/2026-06-28-v2-benchmark-baseline.json`

Recommended live benchmark flow:

```bash
cargo run -- load-local-parquet-into-clickhouse \
  --input-glob "gdelt_candidates_20d_full/dt=2026-06-05/part-*.parquet" \
  --start-date 2026-06-05 \
  --end-date 2026-06-06

cargo run -- enrich-bronze \
  --start-date 2026-06-05 \
  --end-date 2026-06-06 \
  --batch-size 2000
```

Then inspect:

```sql
SELECT *
FROM ingest_file_catalog
ORDER BY loaded_at DESC
LIMIT 20;
```

## ClickHouse MCP

V2 now includes a ClickHouse-backed MCP surface in Rust. It does not use
DuckDB, and it does not perform final narrative reasoning on the server.
Instead it exposes raw structured graph bundles so the client LLM can reason
over them.

Modes:

- `cargo run -- mcp-stdio`
  Direct stdio MCP against the configured ClickHouse database.
- `cargo run -- serve-api --bind 127.0.0.1:8788`
  Shared HTTP API for many clients against one ClickHouse graph.
- `cargo run -- mcp-proxy --api-base-url http://127.0.0.1:8788`
  Thin local stdio proxy that forwards MCP requests to the shared API.

Supported compatibility tools include:

- `explain_move`
- `summarize_narrative`
- `supporting_docs`
- `explain_day`
- `explain_cross_asset_move`
- `build_narrative_frame`
- `find_contradictory_assets`
- `explain_asset_via_day_context`
- `similar_days`
- `intraday_evolution`

V2 also exposes lower-level raw tools directly:

- `summary`
- `top_factors`
- `top_assets`
- `factor_daily`
- `tone_tails`
- `asset_narratives`
- `asset_timeline`
- `factor_crossovers`
- `asset_crossovers`
- `query_clickhouse`

Transport notes:

- `mcp-stdio` and `mcp-proxy` both speak stdio MCP with `Content-Length`
  framing.
- `serve-api` exposes `GET /health`, `GET /tools`, and `POST /mcp`.
- The compatibility tool names now return raw JSON bundles rather than final
  prose summaries. Final reasoning should happen in the consuming client.

## Notes

- V2 uses stable hashed ids for source, asset, geo, cluster, and doc identity
  instead of the mutable DuckDB append-only dictionary tables.
- Gold layers are exposed as ClickHouse views over the silver tables instead of
  being rewritten day-by-day.
- The Rust enrichment path is now bounded by `--batch-size`, but the local
  benchmark command still runs against an in-process synthetic sample rather
  than live ClickHouse data.
- V2 now includes a direct ClickHouse MCP in both local and client/server
  modes, so shared multi-user serving no longer needs the legacy DuckDB-based
  MCP path.
- The ClickHouse graph build SQL is now valid on stock ClickHouse semantics,
  but a one-day asset-factor build still OOMs on the current query shape and
  needs another reduction in join fanout before the full end-to-end path is
  benchmark-complete.
- The current local validation covers Rust compile/unit checks only. Live
  BigQuery, GCS, and ClickHouse integration still needs environment-backed
  validation.
