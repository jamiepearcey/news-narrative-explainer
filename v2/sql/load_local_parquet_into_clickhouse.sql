INSERT INTO bronze_raw_gdelt (
    record_datetime,
    partition_date,
    source_common_name,
    document_identifier,
    title,
    summary,
    text,
    v2_themes,
    v2_tone,
    v2_locations,
    v2_persons,
    v2_organizations,
    all_names,
    metadata_json
)
SELECT
    toString(record_datetime),
    toDate(partition_date),
    nullIf(toString(source_common_name), ''),
    toString(document_identifier),
    nullIf(toString(title), ''),
    nullIf(toString(summary), ''),
    nullIf(toString(text), ''),
    nullIf(toString(v2_themes), ''),
    nullIf(toString(v2_tone), ''),
    nullIf(toString(v2_locations), ''),
    nullIf(toString(v2_persons), ''),
    nullIf(toString(v2_organizations), ''),
    nullIf(toString(all_names), ''),
    nullIf(toString(metadata_json), '')
FROM file(
    '__LOCAL_GLOB__',
    'Parquet'
)
WHERE toDate(partition_date) >= toDate('__START_DATE__')
  AND toDate(partition_date) < toDate('__END_DATE__');
