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
- `scripts/query_narrative_graph.py`
  Query helper for post-hoc narrative identification.
- `scripts/render_narrative_brief.py`
  Turns an `explain-move` JSON payload into a markdown note.
- `scripts/narrative_explainer_mcp.py`
  Minimal stdio MCP wrapper exposing explanation and summary tools.

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
  --enrich-max-docs 100
```

Then build the local narrative graph from the standalone parquet:

```bash
cd news-narrative-explainer
python3 scripts/build_narrative_graph.py \
  --input-glob "data/gdelt_candidates/dt=*/part-*.parquet" \
  --output-db data/narrative_graph.duckdb \
  --overwrite
```

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

## Tests

```bash
python3 scripts/test_build_narrative_graph.py
python3 scripts/test_fetch_gdelt_bigquery_candidates.py
python3 scripts/test_query_narrative_graph.py
python3 scripts/test_narrative_explainer_mcp.py
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
