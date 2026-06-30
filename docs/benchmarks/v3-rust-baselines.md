# V3 Rust Baselines

## 2026-06-28 one-day local parquet baseline

Dataset:

- input day: `2026-06-05`
- input file:
  `/Users/jamiepearcey/projects/research/news-narrative-explainer/data/gdelt_candidates_20d_full/dt=2026-06-05/part-20260626T211008Z-bigquery-window.parquet`
- row count: `178,845`

Command:

```bash
cargo run --release --bin news-narrative-v3 -- benchmark-local-day \
  --date 2026-06-05 \
  --input-glob '/Users/jamiepearcey/projects/research/news-narrative-explainer/data/gdelt_candidates_20d_full/dt=2026-06-05/part-*.parquet' \
  --output-root /Users/jamiepearcey/projects/research/news-narrative-explainer/data/narrative_graph_parquet_v3 \
  --overwrite-day
```

Saved artifact:

- `/Users/jamiepearcey/projects/research/news-narrative-explainer/v3/results/2026-06-05-v3-benchmark.json`

Observed stage timings after the current chunked review/detail split:

- `load_input`: `401.22ms`
- `build_bronze`: `1653.61ms`
- `build_silver`: `6036.82ms`
- `build_gold`: `61394.91ms`
- `write_outputs`: `34900.85ms`

Observed output counts:

- `graph_doc_nodes_daily`: `178,845`
- `doc_payload_daily`: `178,845`
- `doc_review_daily`: `178,845`
- `doc_detail_daily`: `178,845`
- `silver_event_graph`: `171,000`
- `silver_factor_mentions`: `1,557,804`
- `silver_asset_factor_mentions`: `2,112,877`
- `silver_market_context_mentions`: `651,690`
- `gold_factor_buckets_daily`: `4,965`
- `gold_asset_factor_panel_daily`: `12,683`
- `gold_factor_crossover_links_daily`: `4,746`
- `gold_asset_factor_crossover_links_daily`: `11,653`

Notes:

- This run passed after hardening the bronze-row string and timestamp parsing
  path so dirty-but-recoverable values no longer fail the entire day.
- `doc_review_daily` and `doc_detail_daily` now also emit `index.json` sidecar
  manifests with sorted `doc_id` chunk ranges so hosted readers can target a
  small subset of payload files.
- The new write path increases write cost materially because each cold payload
  family now emits many chunk files in addition to the compatibility
  `part-000.parquet` file.
- The `build_gold` jump in this overwrite run needs separate profiling; it is
  not explained by the payload chunk/index work alone.

## 2026-06-28 hosted parquet query baseline

Dataset:

- hosted root:
  `http://127.0.0.1:8789`
- served artifact root:
  `/Users/jamiepearcey/projects/research/news-narrative-explainer/data/narrative_graph_parquet_v3`
- representative query:
  `explain-move`
- asset: `WTI`
- day: `2026-06-05`

Command shape:

```bash
cargo run --release --bin news-narrative-v3 -- serve-artifacts \
  --output-root /Users/jamiepearcey/projects/research/news-narrative-explainer/data/narrative_graph_parquet_v3 \
  --bind 127.0.0.1:8789

uv run --with 'duckdb>=1.0' python3 - <<'PY'
import importlib.util, json, statistics, subprocess, sys, time, tracemalloc
from pathlib import Path

base = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
sys.path.insert(0, str(base / "v3"))
store_spec = importlib.util.spec_from_file_location(
    "v3_remote_parquet_store_mod",
    base / "v3" / "v3_remote_parquet_store.py",
)
store_mod = importlib.util.module_from_spec(store_spec)
store_spec.loader.exec_module(store_mod)

store = "http://127.0.0.1:8789"
asset = "WTI"
date = "2026-06-05"
runs = []
for _ in range(3):
    tracemalloc.start()
    t0 = time.perf_counter()
    with store_mod.resolve_query_db(store, date, date) as db:
        proc = subprocess.run(
            [
                "uv",
                "run",
                "--with",
                "duckdb>=1.0",
                "python3",
                str(base / "scripts" / "query_narrative_graph.py"),
                "--db",
                str(db),
                "--view",
                "explain-move",
                "--asset-label",
                asset,
                "--start-date",
                date,
                "--end-date",
                date,
                "--limit",
                "10",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    payload = json.loads(proc.stdout)
    runs.append(
        {
            "seconds": elapsed,
            "peak_python_bytes": peak,
            "docs": len(payload["supporting_docs"]),
            "narratives": len(payload["top_narratives"]),
        }
    )
print(json.dumps(runs, indent=2))
PY
```

Saved artifact:

- `/Users/jamiepearcey/projects/research/news-narrative-explainer/v3/results/2026-06-28-v3-hosted-query-benchmark.json`

Observed results:

- Old adapter, materialized temp tables:
  `10.276s` average over 3 runs
- New adapter, direct parquet views plus deferred payload hydrate:
  `6.409s` average over 3 runs
- Chunked review/detail files plus `doc_id`-range sidecar indexes:
  `2.506s` average over 5 runs
- `supporting_docs` inside one resolved DuckDB catalog dropped to about
  `1.188s` average, while `asset_narratives` and `asset_timeline` remained
  about `40ms` each

Notes:

- The main win came from removing the eager `INSERT INTO ... SELECT FROM
  read_parquet(...)` stage in the hosted-parquet adapter.
- The new path still creates a small temporary DuckDB catalog file, but the
  large remote partitions remain parquet-backed views instead of copied local
  tables.
- `supporting_docs` now filters on hot graph/silver relations first, hydrates
  lighter review payloads for reranking, and then uses `doc_id`-range sidecar
  indexes to read only the relevant `doc_detail_daily` chunk files for the
  final shortlist.

## 2026-06-28 MCP smoke timings

Saved artifact:

- `/Users/jamiepearcey/projects/research/news-narrative-explainer/v3/results/2026-06-28-v3-mcp-smoke.json`

Representative one-shot handler timings against `http://127.0.0.1:8789`
for `2026-06-05`:

- `explain_move`: `1.75s`
- `summarize_narrative`: `3.57s`
- `supporting_docs`: `1.79s`
- `explain_day`: `2.35s`
- `explain_cross_asset_move`: `2.27s`
- `build_narrative_frame`: `2.31s`
- `find_contradictory_assets`: `2.31s`
- `explain_asset_via_day_context`: `2.35s`
- `query_duckdb`: `0.03s`
- `similar_days`: `0.04s`
- `intraday_evolution`: `0.10s`

Notes:

- All copied v3 MCP tool handlers completed successfully on this smoke pass.
- `similar_days` is now explicitly on a `gold_only` full-history path.
- The day-context tools were then reduced from multi-second per-asset
  orchestration to bulk universe queries plus persistent hosted-catalog reuse,
  which brought them back down to roughly the same order of magnitude as
  `explain_move`.

## 2026-06-28 persistent query-cache timings

Saved artifact:

- `/Users/jamiepearcey/projects/research/news-narrative-explainer/v3/results/2026-06-28-v3-query-cache-benchmark.json`

Cold vs warm handler timings after adding the persistent local DuckDB cache at
`data/.cache/v3_duckdb_query/`:

- `explain_move`: `2.77s` cold -> `1.33s` warm
- `similar_days`: `0.10s` cold -> `0.03s` warm
- `explain_day`: `8.78s` cold -> `8.94s` warm

Notes:

- The persistent cache eliminates repeated hosted-catalog rebuild work for the
  same store/date/load-profile scope.
- Warm-call improvement is real for the setup-sensitive tools.
- `explain_day` barely changes because its remaining wall time is dominated by
  single-day cross-asset reasoning and supporting-doc orchestration, not by
  hosted-catalog setup.
