# News Narrative Explainer

Standalone scripts for identifying the news narratives that were present around
an asset move after it happened.

This folder is intentionally narrow:

- build a local narrative graph from GDELT-style candidate parquet
- classify rows into deterministic macro, geopolitical, commodity, corporate,
  and crypto factors
- look up which factors were most active for a chosen asset and window
- pull supporting documents for those factors
- render a plain markdown brief for review

It does not rank price-prediction candidates and does not make trading claims.

## Layout

- `config/news_narrative_taxonomy.json`
  Deterministic factor taxonomy.
- `scripts/build_narrative_graph.py`
  Builds a local DuckDB narrative graph from parquet input.
- `scripts/fetch_gdelt_bigquery_candidates.py`
  Fetches one-day GDELT BigQuery GKG candidate rows directly to local parquet.
- `scripts/export_gdelt_bigquery_window.py`
  Exports a multi-day capped GDELT candidate window to partitioned parquet
  using the faster Arrow result path.
- `scripts/query_narrative_graph.py`
  Query helper for post-hoc narrative identification.
- `scripts/parquet_narrative_store.py`
  Writes the graph as daily parquet partitions and can hydrate a day-scoped
  DuckDB workspace for MCP queries.
- `scripts/render_narrative_brief.py`
  Turns an `explain-move` JSON payload into a markdown note.
- `scripts/narrative_explainer_mcp.py`
  Minimal stdio MCP wrapper exposing explanation and summary tools.
- `scripts/validate_mcp_narrative.py`
  Two-stage wrapper: local MCP explanation first, then constrained web
  validation against the MCP claims.

## Expected Input Columns

Input parquet should contain:

- `record_datetime`
- `partition_date`
- `source_common_name`
- `document_identifier`
- `v2_themes`
- `v2_tone`
- `v2_locations`
- `v2_persons`
- `v2_organizations`
- `all_names`

Optional text-bearing columns are consumed when present:

- `title` or `article_title` or `headline`
- `summary` or `snippet` or `description`
- `text` or `article_text` or `body_text` or `content`

Optional metadata columns are also preserved in `bronze_candidates` and returned
with supporting documents:

- `metadata_json`
- `gkg_extras` or `extras`
- `sharing_image`
- `related_images`
- `social_image_embeds`
- `social_video_embeds`
- `quotations`
- `amounts`
- `dates`
- `gcam`
- `translation_info`

When those fields are present, the local graph stores them directly and also
builds a `relevant_text` field that blends article text with names, themes,
organizations, and locations for post-hoc narrative review.

The public GDELT BigQuery GKG table does not expose full article body text.
The standalone fetcher therefore emits nullable `title`, `summary`, and `text`
columns and fills richer GKG metadata where available.

## Fetch

Default one-day capped fetch:

```bash
python3 scripts/fetch_gdelt_bigquery_candidates.py \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --max-results 50000
```

The fetcher also honors the same credential environment used by the old
`quant-algos` BigQuery connector:

```bash
SOURCE_BIGQUERY_IPC__PROJECT_ID=...
SOURCE_BIGQUERY_IPC__SERVICE_ACCOUNT_JSON='{...}'
```

If `SOURCE_BIGQUERY_IPC__SERVICE_ACCOUNT_JSON` is set and no project is passed,
the script derives the project from the JSON `project_id` field.

Dry-run BigQuery cost estimate:

```bash
python3 scripts/fetch_gdelt_bigquery_candidates.py \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --dry-run \
  --max-results 50000
```

Configurable window:

```bash
python3 scripts/fetch_gdelt_bigquery_candidates.py \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --start 2026-06-24T00:00:00Z \
  --end 2026-06-25T00:00:00Z \
  --max-results 50000
```

Output is written to:

```text
data/gdelt_candidates/dt=YYYY-MM-DD/part-YYYYMMDDTHHMMSSZ-bigquery.parquet
```

Fast multi-day capped export for local testing:

```bash
python3 scripts/export_gdelt_bigquery_window.py \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --start 2026-06-15T00:00:00Z \
  --end 2026-06-25T00:00:00Z \
  --rows-per-day 2000 \
  --output-root data/gdelt_candidates_10d
```

Dry-run estimate for the same 10-day window:

```bash
python3 scripts/export_gdelt_bigquery_window.py \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --start 2026-06-15T00:00:00Z \
  --end 2026-06-25T00:00:00Z \
  --rows-per-day 2000 \
  --output-root data/gdelt_candidates_10d \
  --dry-run
```

## Build

Pull a fresh standalone parquet from BigQuery:

```bash
python3 scripts/fetch_gdelt_bigquery_candidates.py \
  --project your-gcp-project \
  --lookback-hours 6 \
  --max-results 5000
```

Pull a fresh parquet and enrich a bounded number of URLs with title/summary/text:

```bash
python3 scripts/fetch_gdelt_bigquery_candidates.py \
  --project your-gcp-project \
  --lookback-hours 6 \
  --max-results 1000 \
  --enrich-text \
  --enrich-max-docs 100 \
  --estimate-total-rows
```

With `--estimate-total-rows`, the fetch report includes:

- actual parquet size for the fetched sample
- observed bytes per row
- projected full-window parquet size based on the sampled rows
- total matching row count for the requested window

Then build the local narrative graph from the standalone parquet:

```bash
cd news-narrative-explainer
python3 scripts/build_narrative_graph.py \
  --input-glob "data/gdelt_candidates/dt=*/part-*.parquet" \
  --output-db data/narrative_graph.duckdb \
  --overwrite
```

Subsequent runs without `--overwrite` append only unseen `partition_date` values
into the existing DuckDB graph, preserve the daily materialized tables, and
refresh explicit day-to-day crossover link tables for the affected dates.

Parquet-native materialization:

```bash
python3 scripts/parquet_narrative_store.py \
  --input-glob "/absolute/path/to/gdelt_candidates_etl_day_*/*.parquet" \
  --input-glob "/absolute/path/to/gdelt_candidates/dt=*/part-*.parquet" \
  --output-root data/narrative_graph_parquet \
  --overwrite
```

This writes daily parquet partitions for the bronze, silver, gold, and
crossover layers. The parquet store becomes the canonical graph substrate; the
MCP can point its `db` argument at that root and it will build a temporary
day-scoped DuckDB workspace behind the scenes.

## Query

Top factor activity:

```bash
python3 scripts/query_narrative_graph.py --db data/narrative_graph.duckdb --view top-factors --limit 20
```

Narratives for an asset in a window:

```bash
python3 scripts/query_narrative_graph.py \
  --db data/narrative_graph.duckdb \
  --view asset-narratives \
  --asset-label WTI \
  --start-date 2026-06-18 \
  --end-date 2026-06-23 \
  --limit 10
```

Crossovers for an asset between consecutive days:

```bash
python3 scripts/query_narrative_graph.py \
  --db data/narrative_graph.duckdb \
  --view asset-crossovers \
  --asset-label WTI \
  --factor-label oil \
  --start-date 2026-06-18 \
  --end-date 2026-06-23 \
  --limit 10
```

Explain a move with factors, timeline, and documents:

```bash
python3 scripts/query_narrative_graph.py \
  --db data/narrative_graph.duckdb \
  --view explain-move \
  --asset-label WTI \
  --start-date 2026-06-18 \
  --end-date 2026-06-23 \
  --limit 10 > results/wti_explain_move.json
```

Render a markdown brief:

```bash
python3 scripts/render_narrative_brief.py \
  --input results/wti_explain_move.json \
  --output results/wti_explain_move.md
```

Two-stage validation for a day-level narrative:

```bash
python3 scripts/validate_mcp_narrative.py \
  --db data/narrative_graph.duckdb \
  --date 2026-06-24 \
  --json > results/validated_2026-06-24.json
```

Direct desk-question mode:

```bash
python3 scripts/validate_mcp_narrative.py \
  --db data/narrative_graph.duckdb \
  --date 2026-06-24 \
  --question "What is the single best explanation for today’s cross-asset market behaviour?" \
  --question "Which asset most contradicts the dominant narrative?"
```

The current direct-question taxonomy covers:

- best / dominant explanation
- contradictory asset
- contradiction evidence
- unexplained / unknown areas
- overreaction vs underreaction
- narrative assumptions

The MCP now also exposes a higher-level structured reasoning tool:

- `build_narrative_frame`
  Returns a narrative frame with:
  - `primary_regime`
  - `dominant_narrative`
  - `best_explanation`
  - `top_competing_hypotheses`
    Each hypothesis now includes explicit `confidence`, `explains`, and
    `weakness` fields in addition to model-composition details.
  - `rankings.evidence_strength`
  - `rankings.market_impact`
  - `rankings.transmission_plausibility`
  - `first_link`
  - `transmission_chain`
  - `blocking_overlay`
  - `weakest_asset`
  - `unresolved_assets`
  - `consistency_warnings`
  - `confidence_summary`
  - diagnostics and supporting references

This wrapper:

- calls the local MCP first
- consumes the MCP-native `build_narrative_frame` output as the primary
  structured evidence object
- preserves the MCP trust fields (`fit_confidence`, `contradiction_score`,
  `unsupported_assets`, and `Unsupported / cannot answer`)
- uses the frame's regime, transmission, weakest-asset, unresolved-asset, and
  diagnostics fields instead of reconstructing those outside the MCP
- performs a constrained web search against those MCP claims
- labels each claim as `confirmed`, `refined`, `unsupported`, or a trust-gated
  MCP status such as `mcp_unresolved`, `mcp_low_confidence`, or
  `mcp_high_contradiction`

The intent is to validate or refine the MCP answer, not to let the web replace
the local evidence model.

In plain-text mode, the wrapper now starts with a compact desk-style summary:

- `Desk answers`
  Direct answers for question prompts passed with `--question`.
- `MCP trust`
  Stage-one fit, contradiction, and unsupported-asset signals.
- `Best-fit explanation`
  The top combined explanation plus its validation bucket.
- `Validated claims` / `Refined claims`
  Claims with the strongest external support.
- `MCP unresolved`
  Claims the wrapper intentionally does not try to over-validate because the
  MCP itself marked them unresolved or too weak.

For true edge cases where the built-in explanation helpers are still too narrow,
the MCP also exposes a guarded read-only DuckDB query path. It accepts only
single-statement `SELECT` or `WITH` queries, blocks mutation/DDL/PRAGMA
statements, and applies a limit when one is not supplied.

Two more trust-oriented MCP tools are available:

- `similar_days`
  Finds prior local days with the closest daily factor mix.
- `intraday_evolution`
  Summarizes hourly factor leadership when there are enough intraday buckets,
  and explicitly says it cannot answer reliably when the local data is too
  sparse.

The main day-level MCP outputs now also emit explicit trust scaffolding:

- `fit_confidence`
  A normalized combination-fit score for how well the selected narrative set
  explains the requested assets.
- `contradiction_score`
  A normalized contradiction penalty combining direct contradictions,
  unresolved assets, and lack of direct support.
- `unsupported_assets`
  Assets that remain unsupported or unresolved in the current local slice.
- `Unsupported / cannot answer`
  A compact block that makes missing or unresolved cases explicit instead of
  hiding them inside the narrative prose.

## Experimental Semantic Retrieval

An additive experimental helper is available for a dense-retrieval path over
`bronze_candidates`:

- pluggable embedding backends with a local-first default
- Qdrant TurboQuant for compressed ANN search
- Qdrant oversampling plus rescoring on stored higher-precision vectors

This does not replace the deterministic graph or current MCP tools. It is a
separate retrieval layer intended for RAG-style evidence lookup.

Embedding backend options:

- `ollama`
  Local default. No remote API dependency. Default model: `embeddinggemma`.
- `openai`
  Optional hosted baseline. Supports native embedding shortening through the
  `dimensions` parameter.
- `sentence-transformers`
  Optional in-process local backend.

Index a date window from the parquet-native or DuckDB-backed graph:

```bash
python3 scripts/qdrant_turboquant_rag.py index \
  --db data/narrative_graph_parquet \
  --collection news_narrative_bronze \
  --qdrant-url http://localhost:6333 \
  --embedding-provider ollama \
  --embedding-model embeddinggemma \
  --start-date 2026-06-18 \
  --end-date 2026-06-25 \
  --turbo-bits bits2 \
  --batch-size 64 \
  --recreate
```

Query the compressed index with rescoring enabled:

```bash
python3 scripts/qdrant_turboquant_rag.py search \
  --collection news_narrative_bronze \
  --qdrant-url http://localhost:6333 \
  --embedding-provider ollama \
  --embedding-model embeddinggemma \
  --query "hawkish Fed repricing lifting the dollar and pressuring gold" \
  --start-date 2026-06-18 \
  --end-date 2026-06-25 \
  --oversampling 2.0 \
  --limit 10
```

Optional shortened-dimension run:

```bash
python3 scripts/qdrant_turboquant_rag.py index \
  --db data/narrative_graph_parquet \
  --collection news_narrative_bronze_384 \
  --qdrant-url http://localhost:6333 \
  --embedding-provider openai \
  --embedding-model text-embedding-3-large \
  --dimensions 384 \
  --start-date 2026-06-18 \
  --end-date 2026-06-25 \
  --recreate
```

For local backends, `--dimensions` is optional and applies prefix truncation,
not native Matryoshka shortening.

Runtime requirements:

- a reachable Qdrant server, preferably `1.18+` for TurboQuant
- for the local default, a reachable Ollama server and an installed embedding
  model such as `embeddinggemma`
- `OPENAI_API_KEY` or `--openai-api-key` only when `--embedding-provider openai`
- `uv` if `openai` or `qdrant-client` are not already installed for the active
  Python, or if you use the optional `sentence-transformers` backend

The helper stores bronze-document payloads including `partition_date`,
`source_domain`, `source_type`, `theme_tags`, title/summary, and derived market
context so semantic hits can still be constrained by date and source filters.

## Tests

```bash
python3 scripts/test_build_narrative_graph.py
python3 scripts/test_fetch_gdelt_bigquery_candidates.py
python3 scripts/test_export_gdelt_bigquery_window.py
python3 scripts/test_query_narrative_graph.py
python3 scripts/test_narrative_explainer_mcp.py
uv run --with duckdb>=1.0 --with qdrant-client>=1.15 python3 scripts/test_qdrant_turboquant_rag.py
```

## MCP Wrapper

The standalone folder also includes a minimal MCP-style stdio server for
explanation workflows:

```bash
python3 scripts/narrative_explainer_mcp.py
```

Exposed tools:

- `explain_move`
  Returns the local factor explanation payload for an asset and date window.
- `summarize_narrative`
  Returns a short deterministic text summary built from the local explanation,
  using stored title, summary, body excerpt, and derived `relevant_text` when
  available.
- `supporting_docs`
  Returns the document list for an asset and optional factor.

The MCP wrapper is intentionally narrow and explanation-oriented. It does not
expose prediction, lead/lag, or market-ranking tools.
