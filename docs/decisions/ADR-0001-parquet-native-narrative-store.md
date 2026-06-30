# ADR-0001: Parquet-Native Narrative Store

## Status

Accepted

## Date

2026-06-27

## Context

The original narrative graph builder accumulated all retained history into one
DuckDB database file. Raw inputs were day-partitioned on disk, but the derived
graph layers were stored in one mutable artifact. That made long-horizon
materialization cost grow with retained history and tied the MCP directly to a
monolithic DuckDB store.

The project needs near-constant incremental cost per new day, durable storage
that stays query-engine-neutral, and the ability to load only the requested day
plus a small amount of surrounding context into the MCP.

## Decision

The canonical narrative graph store will be partitioned parquet, not a single
DuckDB database file.

The new storage shape is:

- daily parquet partitions for `bronze_candidates`
- daily parquet partitions for the intraday relationship layers
  `silver_event_graph`, `silver_factor_mentions`,
  `silver_asset_factor_mentions`, and `silver_market_context_mentions`
- daily parquet partitions for `gold_factor_buckets_daily` and
  `gold_asset_factor_panel_daily`
- daily parquet partitions for
  `gold_factor_crossover_links_daily` and
  `gold_asset_factor_crossover_links_daily`

DuckDB remains the local analytical adapter. The MCP now resolves a parquet
root into a temporary DuckDB workspace scoped to the requested day window for
detailed layers while loading the compact daily gold layers across the retained
horizon.

## Consequences

Positive:

- Incremental materialization is day-bounded.
- Storage is engine-neutral and portable to DuckDB, ClickHouse, Spark, Polars,
  or other analytical systems later.
- The MCP no longer requires a full-history mutable DuckDB file.
- Detailed intraday data can remain scoped to the requested day while daily
  summary history stays available for tools such as `similar_days`.

Tradeoffs:

- There is now an explicit adapter layer between canonical storage and the MCP.
- Querying raw parquet directly is less convenient than using one persistent
  DuckDB file unless an adapter is present.
- Cross-day links must be materialized deliberately rather than assumed from one
  consolidated database.
