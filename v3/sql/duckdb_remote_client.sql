INSTALL httpfs;
LOAD httpfs;

-- Optional: keep a local cache if you reuse the same remote slice repeatedly.
-- SET enable_http_metadata_cache = true;

-- Read hot graph parquet remotely. DuckDB can prune projected columns and
-- skip parquet row groups when predicates can be satisfied from metadata.
WITH factor_window AS (
    SELECT
        bucket_time,
        factor_id,
        factor_label,
        doc_count,
        avg_tone
    FROM read_parquet(
        'http://127.0.0.1:8789/gold_factor_buckets_daily/bucket_time=2026-06-05/part-000.parquet'
    )
),
top_factors AS (
    SELECT
        factor_id,
        factor_label,
        sum(doc_count) AS total_docs
    FROM factor_window
    GROUP BY 1, 2
    ORDER BY total_docs DESC
    LIMIT 25
),
candidate_docs AS (
    SELECT
        doc_id,
        factor_ids,
        factor_labels,
        source_domain,
        market_context_score
    FROM read_parquet(
        'http://127.0.0.1:8789/graph_doc_nodes_daily/partition_date=2026-06-05/part-000.parquet'
    )
    WHERE market_context_score >= 0.2
)
SELECT *
FROM top_factors;

-- Hydrate cold payloads only after narrowing the document set.
SELECT
    payload.doc_id,
    payload.title,
    payload.summary_text,
    payload.body_text
FROM read_parquet(
    'http://127.0.0.1:8789/doc_payload_daily/partition_date=2026-06-05/part-000.parquet'
) AS payload
WHERE payload.doc_id IN (
    SELECT doc_id
    FROM candidate_docs
    LIMIT 50
);
