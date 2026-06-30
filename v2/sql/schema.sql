CREATE TABLE IF NOT EXISTS factor_dictionary (
    factor_id UInt32,
    factor_label String,
    factor_group String
)
ENGINE = MergeTree
ORDER BY factor_id;

CREATE TABLE IF NOT EXISTS factor_rule_patterns (
    factor_id UInt32,
    factor_label String,
    factor_group String,
    pattern String
)
ENGINE = MergeTree
ORDER BY (factor_id, pattern);

CREATE TABLE IF NOT EXISTS factor_rule_assets (
    factor_id UInt32,
    asset_label String
)
ENGINE = MergeTree
ORDER BY (factor_id, asset_label);

CREATE TABLE IF NOT EXISTS asset_rule_patterns (
    asset_label String,
    pattern String
)
ENGINE = MergeTree
ORDER BY (asset_label, pattern);

CREATE TABLE IF NOT EXISTS asset_context_required (
    asset_label String
)
ENGINE = MergeTree
ORDER BY asset_label;

CREATE TABLE IF NOT EXISTS bronze_raw_gdelt (
    record_datetime String,
    partition_date Date,
    source_common_name Nullable(String),
    document_identifier String,
    title Nullable(String),
    summary Nullable(String),
    text Nullable(String),
    v2_themes Nullable(String),
    v2_tone Nullable(String),
    v2_locations Nullable(String),
    v2_persons Nullable(String),
    v2_organizations Nullable(String),
    all_names Nullable(String),
    metadata_json Nullable(String),
    ingested_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree
PARTITION BY partition_date
ORDER BY (partition_date, document_identifier, ingested_at);

CREATE TABLE IF NOT EXISTS bronze_candidates (
    doc_id UInt64,
    record_datetime String,
    event_time DateTime64(3),
    partition_date Date,
    source_domain String,
    document_identifier String,
    v2_themes Nullable(String),
    v2_tone Nullable(String),
    v2_locations Nullable(String),
    v2_persons Nullable(String),
    v2_organizations Nullable(String),
    all_names Nullable(String),
    title Nullable(String),
    summary_text Nullable(String),
    body_text Nullable(String),
    relevant_text Nullable(String),
    metadata_json Nullable(String),
    gkg_extras Nullable(String),
    sharing_image Nullable(String),
    related_images Nullable(String),
    social_image_embeds Nullable(String),
    social_video_embeds Nullable(String),
    quotations Nullable(String),
    amounts Nullable(String),
    dates Nullable(String),
    gcam Nullable(String),
    translation_info Nullable(String),
    source_type String,
    source_priority UInt8,
    market_context_text Nullable(String),
    market_context_score Float64,
    tone Nullable(Float64),
    geo_labels Array(String),
    match_text String,
    asset_match_text String,
    ingested_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree
PARTITION BY partition_date
ORDER BY (partition_date, doc_id);

CREATE TABLE IF NOT EXISTS silver_event_graph (
    event_time DateTime64(3),
    bucket_time Date,
    partition_date Date,
    cluster_id UInt64,
    doc_id UInt64,
    factor_ids Array(UInt32),
    factor_labels Array(String),
    asset_ids Array(UInt64),
    asset_labels Array(String),
    geo_ids Array(UInt64),
    geo_labels Array(String),
    source_id UInt64,
    source_domain String,
    tone Nullable(Float64),
    novelty Float64,
    source_weight Float64,
    classification_confidence Float64,
    model_version String,
    prompt_version String,
    created_at DateTime64(3)
)
ENGINE = MergeTree
PARTITION BY bucket_time
ORDER BY (bucket_time, doc_id, cluster_id);

CREATE TABLE IF NOT EXISTS silver_factor_mentions (
    bucket_time Date,
    event_time DateTime64(3),
    partition_date Date,
    doc_id UInt64,
    cluster_id UInt64,
    factor_id UInt32,
    factor_label String,
    geo_id UInt64,
    geo_label String,
    source_id UInt64,
    source_domain String,
    tone Nullable(Float64),
    novelty Float64,
    source_weight Float64,
    classification_confidence Float64
)
ENGINE = MergeTree
PARTITION BY bucket_time
ORDER BY (bucket_time, factor_id, geo_id, doc_id);

CREATE TABLE IF NOT EXISTS silver_asset_factor_mentions (
    bucket_time Date,
    event_time DateTime64(3),
    partition_date Date,
    doc_id UInt64,
    cluster_id UInt64,
    factor_id UInt32,
    factor_label String,
    asset_id UInt64,
    asset_label String,
    geo_id UInt64,
    geo_label String,
    source_id UInt64,
    source_domain String,
    tone Nullable(Float64),
    novelty Float64,
    source_weight Float64,
    classification_confidence Float64,
    asset_factor_relevance Float64
)
ENGINE = MergeTree
PARTITION BY bucket_time
ORDER BY (bucket_time, asset_id, factor_id, geo_id, doc_id);

CREATE TABLE IF NOT EXISTS silver_market_context_mentions (
    bucket_time Date,
    event_time DateTime64(3),
    partition_date Date,
    doc_id UInt64,
    cluster_id UInt64,
    factor_label String,
    asset_label String,
    source_domain String,
    source_type String,
    source_priority UInt8,
    market_context_text Nullable(String),
    market_context_score Float64,
    classification_confidence Float64
)
ENGINE = MergeTree
PARTITION BY bucket_time
ORDER BY (bucket_time, asset_label, factor_label, doc_id);

CREATE TABLE IF NOT EXISTS graph_build_partitions (
    partition_date Date,
    source_uri String,
    processed_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree
ORDER BY (partition_date, source_uri, processed_at);

CREATE TABLE IF NOT EXISTS ingest_file_catalog (
    source_kind String,
    source_path String,
    content_sha256 Nullable(String),
    file_size_bytes Nullable(UInt64),
    partition_date Date,
    row_count Nullable(UInt64),
    status String,
    loaded_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree
ORDER BY (source_kind, source_path, partition_date, loaded_at);

CREATE VIEW IF NOT EXISTS gold_factor_buckets_daily_base AS
SELECT
    bucket_time,
    factor_id,
    factor_label,
    geo_id,
    geo_label,
    uniqExact(doc_id) AS doc_count,
    count() AS mention_count,
    uniqExact(source_id) AS unique_sources,
    uniqExact(geo_id) AS geo_count,
    avg(tone) AS tone_mean,
    avg(abs(ifNull(tone, 0.0))) AS avg_abs_tone,
    avg(novelty) AS novelty_mean,
    uniqExactIf(doc_id, tone <= -5) AS negative_tail_count,
    uniqExactIf(doc_id, tone >= 5) AS positive_tail_count,
    avg(classification_confidence) AS confidence_mean,
    min(event_time) AS first_seen,
    max(event_time) AS last_seen,
    if(doc_count = 0, CAST(NULL, 'Nullable(Float64)'), unique_sources / doc_count) AS source_dispersion
FROM silver_factor_mentions
GROUP BY bucket_time, factor_id, factor_label, geo_id, geo_label;

CREATE VIEW IF NOT EXISTS gold_factor_buckets_daily AS
WITH roll AS (
    SELECT
        *,
        avg(tone_mean) OVER (
            PARTITION BY factor_id
            ORDER BY bucket_time
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS tone_mean_30d_avg,
        stddevSampStable(tone_mean) OVER (
            PARTITION BY factor_id
            ORDER BY bucket_time
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS tone_mean_30d_std
    FROM gold_factor_buckets_daily_base
)
SELECT
    bucket_time,
    factor_id,
    factor_label,
    geo_id,
    geo_label,
    doc_count,
    mention_count,
    unique_sources,
    geo_count,
    tone_mean,
    if(tone_mean_30d_std = 0 OR isNull(tone_mean_30d_std), CAST(NULL, 'Nullable(Float64)'), (tone_mean - tone_mean_30d_avg) / tone_mean_30d_std) AS tone_zscore_30d,
    avg_abs_tone,
    novelty_mean,
    negative_tail_count,
    positive_tail_count,
    source_dispersion,
    confidence_mean,
    first_seen,
    last_seen,
    doc_count * (0.5 + ifNull(source_dispersion, 0.0)) * (1.0 + (ifNull(avg_abs_tone, 0.0) / 5.0)) AS narrative_score
FROM roll;

CREATE VIEW IF NOT EXISTS gold_asset_factor_panel_daily_base AS
SELECT
    bucket_time,
    factor_id,
    factor_label,
    asset_id,
    asset_label,
    geo_id,
    geo_label,
    uniqExact(doc_id) AS doc_count,
    count() AS mention_count,
    uniqExact(source_id) AS unique_sources,
    uniqExact(geo_id) AS geo_count,
    avg(tone) AS tone_mean,
    avg(abs(ifNull(tone, 0.0))) AS avg_abs_tone,
    avg(novelty) AS novelty_mean,
    avg(classification_confidence) AS confidence,
    avg(asset_factor_relevance) AS asset_factor_relevance_mean,
    if(doc_count = 0, CAST(NULL, 'Nullable(Float64)'), unique_sources / doc_count) AS source_dispersion
FROM silver_asset_factor_mentions
GROUP BY bucket_time, factor_id, factor_label, asset_id, asset_label, geo_id, geo_label;

CREATE VIEW IF NOT EXISTS gold_asset_factor_panel_daily AS
WITH roll AS (
    SELECT
        *,
        avg(tone_mean) OVER (
            PARTITION BY asset_id, factor_id
            ORDER BY bucket_time
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS tone_mean_30d_avg,
        stddevSampStable(tone_mean) OVER (
            PARTITION BY asset_id, factor_id
            ORDER BY bucket_time
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS tone_mean_30d_std
    FROM gold_asset_factor_panel_daily_base
)
SELECT
    bucket_time,
    asset_id,
    asset_label,
    factor_id,
    factor_label,
    geo_id,
    geo_label,
    doc_count,
    mention_count,
    unique_sources,
    geo_count,
    tone_mean,
    if(tone_mean_30d_std = 0 OR isNull(tone_mean_30d_std), CAST(NULL, 'Nullable(Float64)'), (tone_mean - tone_mean_30d_avg) / tone_mean_30d_std) AS tone_zscore_30d,
    avg_abs_tone,
    novelty_mean,
    doc_count * ifNull(source_dispersion, 0.0) * (0.5 + ifNull(asset_factor_relevance_mean, 0.0) / 8.0) AS event_intensity,
    source_dispersion,
    confidence,
    doc_count
        * (0.5 + ifNull(source_dispersion, 0.0))
        * (0.5 + ifNull(asset_factor_relevance_mean, 0.0) / 8.0)
        * (1.0 + (ifNull(avg_abs_tone, 0.0) / 5.0)) AS narrative_score
FROM roll;

CREATE VIEW IF NOT EXISTS gold_factor_crossover_links_daily AS
SELECT
    prev.bucket_time AS prior_bucket_time,
    curr.bucket_time AS bucket_time,
    curr.factor_id,
    curr.factor_label,
    curr.geo_id,
    curr.geo_label,
    prev.doc_count AS prior_doc_count,
    curr.doc_count,
    prev.narrative_score AS prior_narrative_score,
    curr.narrative_score,
    curr.doc_count - prev.doc_count AS doc_count_delta,
    curr.narrative_score - prev.narrative_score AS narrative_score_delta
FROM gold_factor_buckets_daily curr
INNER JOIN gold_factor_buckets_daily prev
    ON prev.factor_id = curr.factor_id
   AND prev.geo_id = curr.geo_id
   AND prev.bucket_time = curr.bucket_time - toIntervalDay(1);

CREATE VIEW IF NOT EXISTS gold_asset_factor_crossover_links_daily AS
SELECT
    prev.bucket_time AS prior_bucket_time,
    curr.bucket_time AS bucket_time,
    curr.asset_id,
    curr.asset_label,
    curr.factor_id,
    curr.factor_label,
    curr.geo_id,
    curr.geo_label,
    prev.doc_count AS prior_doc_count,
    curr.doc_count,
    prev.narrative_score AS prior_narrative_score,
    curr.narrative_score,
    curr.doc_count - prev.doc_count AS doc_count_delta,
    curr.narrative_score - prev.narrative_score AS narrative_score_delta
FROM gold_asset_factor_panel_daily curr
INNER JOIN gold_asset_factor_panel_daily prev
    ON prev.asset_id = curr.asset_id
   AND prev.factor_id = curr.factor_id
   AND prev.geo_id = curr.geo_id
   AND prev.bucket_time = curr.bucket_time - toIntervalDay(1);
