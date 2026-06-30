# V2 ClickHouse Benchmark Baselines

This document records the current benchmark and validation baseline for the
ClickHouse-first `v2/` pipeline as of 2026-06-28.

## Scope

- Rust-side synthetic microbenchmarks for the work that still runs outside
  ClickHouse
- Live one-day local parquet ingest into ClickHouse
- Live one-day Rust enrichment into ClickHouse
- Live graph-build attempt and its current failure mode
- Reproducible commands for rerunning the same checks

These numbers are local baselines, not throughput guarantees.

Saved benchmark artifact:

- `/Users/jamiepearcey/projects/research/news-narrative-explainer/v2/results/2026-06-28-v2-benchmark-baseline.json`

## Environment

- Project: `/Users/jamiepearcey/projects/research/news-narrative-explainer/v2`
- Benchmark date: `2026-06-28`
- Local source day used for the live ingest check: `2026-06-05`
- Local parquet file:
  `/Users/jamiepearcey/projects/research/news-narrative-explainer/data/gdelt_candidates_20d_full/dt=2026-06-05/part-20260626T211008Z-bigquery-window.parquet`
- File size: `695287449` bytes
- SHA-256:
  `1d1b6c1f28fa9bfe188990129d1b4abceb82bf57e23e6ba9403a082e6d0862a2`

## Rust Synthetic Benchmarks

Command:

```bash
cargo run -- benchmark-rust-work --iterations 200000
```

Observed baseline in `release` mode:

- `transform_bronze_row`: about `99.9k ops/sec`, about `10.0us` per row
- `supporting_doc_relevance`: about `98.3k ops/sec`, about `10.2us` per row

Interpretation:

- These numbers cover only the Rust-side transform/scoring helpers.
- They do not include ClickHouse read or write cost.
- They are useful for detecting regressions in the remaining non-SQL path.

## Live One-Day ClickHouse Ingest

### Setup

Schema bootstrap:

```bash
cargo run -- bootstrap
```

One-day local parquet load:

```bash
cargo run -- load-local-parquet-into-clickhouse \
  --input-glob "gdelt_candidates_20d_full/dt=2026-06-05/part-*.parquet" \
  --start-date 2026-06-05 \
  --end-date 2026-06-06
```

Validation query:

```sql
SELECT count()
FROM default.bronze_raw_gdelt
WHERE partition_date = toDate('2026-06-05');
```

Catalog inspection:

```sql
SELECT source_path, content_sha256, file_size_bytes, partition_date, row_count, status
FROM default.ingest_file_catalog
WHERE partition_date = toDate('2026-06-05')
ORDER BY loaded_at DESC;
```

### Observed Baseline

- Loaded rows: `178845`
- ClickHouse reported processed volume: about `661.70 MiB`
- ClickHouse reported insert time: about `6.34s`
- ClickHouse reported throughput: about `28.2k rows/sec`
- ClickHouse reported bandwidth: about `104.36 MiB/sec`
- ClickHouse reported peak query memory: about `1.05 GiB`
- End-to-end CLI wall time: about `7.89s`
- Rust process max RSS from `/usr/bin/time -lp`: about `9.1 MiB`

### Reload Skip Validation

The same load command was rerun without changing the source parquet.

Observed result:

- `loaded_files`: empty
- `skipped_already_loaded_files`: the same `2026-06-05` parquet file

This confirms `ingest_file_catalog` is preventing duplicate local parquet
reloads when path, checksum, and partition date match a prior successful load.

Observed no-op reload baseline:

- wall time: about `1.35s`
- Rust process max RSS: about `8.9 MiB`

## Live One-Day Enrichment

Command:

```bash
cargo run --release -- enrich-bronze \
  --start-date 2026-06-05 \
  --end-date 2026-06-06 \
  --batch-size 2000
```

Observed baseline:

- Inserted rows: `178845`
- Wall time: about `46.66s`
- Rust process max RSS from `/usr/bin/time -lp`: about `1.51 GiB`
- Rust process peak memory footprint from `/usr/bin/time -lp`: about `3.66 GiB`

Interpretation:

- The bounded-memory rewrite fixed the previous whole-window behavior, but the
  current one-day run is still materially memory-heavy because the process
  keeps a large in-memory `HashSet<u64>` of existing doc ids for the partition
  and performs many HTTP round trips plus JSON transforms.

## Live One-Day Graph Build

Command:

```bash
cargo run --release -- build-clickhouse-graph \
  --start-date 2026-06-05 \
  --end-date 2026-06-06 \
  --source-uri "file://local-gdelt-corpus"
```

Current result:

- The one-day graph build does not complete successfully on the current design.
- After fixing ClickHouse alias-shadowing bugs, rewriting unsupported
  non-equi joins into `CROSS JOIN + WHERE`, and trying
  `SETTINGS join_algorithm = 'grace_hash'`, the asset-factor materialization
  still fails with ClickHouse `MEMORY_LIMIT_EXCEEDED`.

Observed failure baselines:

- first failing run: about `100.59s` wall time before memory-limit failure
- second failing run with `grace_hash`: about `123.83s` wall time before
  memory-limit failure
- live `system.processes` sample during the `grace_hash` run:
  about `81.14s` elapsed, about `251,510` rows read, about `1.14 GB` read, and
  about `9.92 GB` ClickHouse query memory

Interpretation:

- The dominant unresolved bottleneck is the ClickHouse-side graph build,
  specifically the joins around `matched_assets`, `asset_pattern_hits`,
  `factor_pattern_hits`, and `asset_factor_scores`.
- Query benchmarks that depend on a complete asset-factor graph are not yet
  representative until this stage is redesigned or materially reduced in
  fanout.

## Partial Query Baselines

These were measured against the partially materialized benchmark database where
`silver_event_graph` and `silver_factor_mentions` existed, but
`silver_asset_factor_mentions` and `silver_market_context_mentions` remained
empty because the graph build failed before completion.

- `query --view summary`: about `0.28s` wall time
- `query --view top-factors --limit 10`: about `0.26s` wall time
- `query --view top-assets --limit 10`: about `0.05s` wall time and returned
  an empty result because asset-factor rows were not materialized

## Benchmark-Driven Fixes Applied During This Pass

- Fixed ClickHouse alias-shadowing in the bronze batch fetch by qualifying
  `partition_date` and `ingested_at`.
- Fixed Rust `event_time` serialization for ClickHouse `DateTime64(3)` inserts.
- Added explicit CTE column aliases in graph SQL so downstream CTEs resolve
  stable names instead of qualified names such as `a.asset_label`.
- Rewrote unsupported non-equi joins in the graph SQL into
  ClickHouse-supported `CROSS JOIN + WHERE` forms.
- Added `CLICKHOUSE_TIMEOUT_SECONDS` support to the HTTP client so long-running
  stages can be benchmarked instead of failing at the default request timeout.

## Recommended Follow-On Benchmarks

One-day enrichment on the same partition:

```bash
cargo run -- enrich-bronze \
  --start-date 2026-06-05 \
  --end-date 2026-06-06 \
  --batch-size 2000
```

Suggested metrics to capture on the next run:

- elapsed wall time
- rows transformed per second
- peak RSS for the Rust process
- ClickHouse peak query memory during reads and writes

One-day graph build on the same partition:

```bash
cargo run -- build-clickhouse-graph \
  --start-date 2026-06-05 \
  --end-date 2026-06-06 \
  --source-uri "file://local-gdelt-corpus"
```

## Caveats

- Docker Compose configuration exists in `v2/docker-compose.yml`, but the live
  check on 2026-06-28 used a directly started local ClickHouse server because
  the Docker daemon was unavailable at that time.
- BigQuery export and GCS-backed ClickHouse ingest have not yet been benchmarked
  live in this workspace.
- The current cataloging and duplicate-skip validation has been confirmed only
  for local parquet loads.
- A full one-day ClickHouse graph build is not yet viable on this machine with
  the current SQL shape, so end-to-end asset-factor and supporting-doc query
  baselines remain blocked on that redesign.
