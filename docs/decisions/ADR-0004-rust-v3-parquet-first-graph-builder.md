# ADR-0004: Rust V3 Parquet-First Graph Builder

## Status

Accepted

## Context

The original Python pipeline is semantically correct but slow. The parallel
ClickHouse-first `v2/` experiment improved ingest but exposed a worse failure
mode in the graph build: the expensive factor/asset matching work was pushed
into SQL joins that created too much intermediate fanout and ultimately hit
memory limits on realistic day partitions.

The core requirement is not “use a different storage engine.” The requirement
is to own the expensive transformation path in a runtime where memory layout,
parallelism, batching, and matching strategy are directly controllable and
benchmarkable.

## Decision

Create `v3/` as a Rust parquet-first builder that mirrors the current Python
parquet-native output layout and rebuilds the graph in Rust rather than in
Python or ClickHouse SQL.

The initial `v3/` scope is intentionally partition-local:

- read one day of local source parquet
- build bronze candidates in Rust
- build silver factor, asset-factor, market-context, and event layers in Rust
- materialize daily gold rollups and crossover tables back to partitioned
  parquet
- split hot graph document-node parquet from cold document-payload parquet
- serve the resulting parquet tree over HTTP with byte-range support for
  DuckDB clients
- emit explicit benchmark artifacts for each run

## Consequences

Positive:

- performance work becomes direct and measurable at the real hot path
- avoids the current ClickHouse join-fanout bottleneck for graph construction
- keeps the parquet-native output layout already used by the project
- lets DuckDB remain a selective remote-read client instead of the build engine
- keeps heavy content fields out of the hot graph scan path
- allows iterative optimization without changing user-facing data products

Negative:

- introduces another implementation path that must be kept semantically aligned
  with the Python reference
- initial parity work will precede deeper optimization
- some query conveniences previously delegated to DuckDB or ClickHouse must be
  re-expressed in Rust materialization code

## Follow-Up

- benchmark `v3` against the existing 20-day local corpus day by day
- identify dominant fanout and allocation hotspots
- optimize matching and silver-row construction before expanding scope beyond
  local parquet input
