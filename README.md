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

When those fields are present, the local graph stores them directly and also
builds a `relevant_text` field that blends article text with names, themes,
organizations, and locations for post-hoc narrative review.

## Build

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
