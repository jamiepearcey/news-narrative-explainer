# ADR-0002: ClickHouse-First V2 Narrative Pipeline

## Status

Accepted

## Date

2026-06-28

## Context

The existing narrative pipeline has already moved canonical storage toward
partitioned parquet and uses DuckDB as the local analytical adapter. That path
works for local research, but the next version needs a warehouse-oriented build
shape that can:

- export directly from BigQuery into Google Cloud Storage without row-by-row
  local materialization
- load parquet from object storage directly into the analytical engine
- keep the bulk of deterministic factor matching and aggregation inside the
  analytical engine
- avoid growing a Python-heavy orchestration surface for work that is either
  warehouse-native or better expressed in a compiled CLI

## Decision

The project will add a parallel `v2/` implementation that is ClickHouse-first.

The v2 execution model is:

- BigQuery exports directly to GCS with native `EXPORT DATA`
- ClickHouse loads exported parquet from GCS with a `gcs(...)` query
- Rust performs the narrow row-level enrichment steps that are awkward to
  express or maintain in ClickHouse SQL:
  - HTML/entity normalization
  - fallback title extraction from embedded GKG extras
  - source classification
  - market-context sentence extraction
- ClickHouse performs deterministic factor matching, asset matching, silver
  relationship materialization, and gold analytical rollups
- Gold daily layers are exposed as ClickHouse views over silver tables instead
  of being rewritten as mutable daily tables

V2 also replaces the mutable append-only source/asset/geo dictionary-id
assignment with stable hashed ids.

## Consequences

Positive:

- BigQuery to GCS export is native and avoids the slower local Arrow-to-parquet
  handoff path.
- ClickHouse can ingest directly from GCS and handle the bulk of the graph
  build inside the warehouse.
- Rust replaces Python for the non-SQL orchestration and enrichment edge cases.
- The v1 DuckDB path remains available while v2 is validated.

Tradeoffs:

- V2 is currently a parallel implementation rather than a drop-in replacement.
- Stable hashed ids are simpler operationally but differ from the v1 mutable
  dictionary-id model.
- Gold views trade rewrite simplicity for dependence on ClickHouse query-time
  computation.
- Live integration validation still depends on an environment with BigQuery,
  GCS HMAC access, and ClickHouse.
