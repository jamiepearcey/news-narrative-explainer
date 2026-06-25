# Results Notes

This directory is for generated narrative outputs. Generated result payloads and
briefs are ignored by git; this README records the validated workflow.

## Capped Daily Fetch

The standalone BigQuery fetcher defaults to a 24 hour window and writes parquet
to `data/gdelt_candidates/dt=YYYY-MM-DD/part-*.parquet`.

Validation command for a capped run:

```bash
python3 scripts/fetch_gdelt_bigquery_candidates.py \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --max-results 50000
```

Dry-run cost check:

```bash
python3 scripts/fetch_gdelt_bigquery_candidates.py \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --dry-run \
  --max-results 50000
```

Build after fetch:

```bash
python3 scripts/build_narrative_graph.py \
  --input-glob "data/gdelt_candidates/dt=*/part-*.parquet" \
  --output-db data/narrative_graph.duckdb \
  --overwrite
```

## Daily Size Estimate

The existing local quant-algos candidate parquet sample is 14 files, 2,462,641
rows, and about 2.5 GiB with the older narrower schema. That is about 1.1 KiB
per row, or roughly 51 MiB per 50,000 rows.

For the richer standalone fetch, the script reports a cautious compressed
parquet estimate of roughly 2-10 KiB per row because `Quotations`, `GCAM`, and
`Extras` can dominate row width. At the test cap of 50,000 rows, expect roughly
95-477 MiB of parquet before downstream DuckDB indexing. Real size should be
measured from the written part file after the first credentialed run.

## Text Availability

`gdelt-bq.gdeltv2.gkg_partitioned` provides GKG metadata, source URLs, themes,
entities, locations, tone, quotations, media fields, and `Extras`; it does not
provide full article body text. The fetcher emits nullable `title`, `summary`,
and `text` columns so any future text-bearing source can use the same graph
builder without schema changes.
