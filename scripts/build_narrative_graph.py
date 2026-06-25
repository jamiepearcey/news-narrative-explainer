#!/usr/bin/env python3
"""Build a local deterministic news narrative graph."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_SCRIPTS_DIR = SCRIPT_DIR.parent
if str(PARENT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_SCRIPTS_DIR))

from _duckdb_bootstrap import ensure_duckdb

ensure_duckdb(__file__)

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable
from urllib.parse import urlparse

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_GLOB = ROOT / "data" / "gdelt_candidates" / "dt=*" / "part-*.parquet"
DEFAULT_DB = ROOT / "data" / "narrative_graph.duckdb"
DEFAULT_TAXONOMY = ROOT / "config" / "news_narrative_taxonomy.json"


@dataclass(frozen=True)
class FactorRule:
    factor_id: int
    label: str
    group: str
    patterns: tuple[str, ...]
    asset_hints: tuple[str, ...]


def stable_u64(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def parse_delimited_field(raw: str | None) -> list[str]:
    if not raw:
        return []
    values: list[str] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        item = re.sub(r",[0-9]+$", "", item).strip()
        if item:
            values.append(item)
    return values


def parse_tone(raw: str | None) -> float | None:
    if not raw:
        return None
    head = raw.split(",", 1)[0].strip()
    if not head:
        return None
    try:
        return float(head)
    except ValueError:
        return None


def parse_record_datetime(raw: str | None, partition_date: str | None) -> datetime:
    if raw:
        text = raw.strip()
        for fmt in ("%Y%m%d%H%M%S", "%Y%m%d"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                pass
    if partition_date:
        return datetime.strptime(partition_date, "%Y-%m-%d")
    raise ValueError("record_datetime or partition_date is required")


def extract_source_domain(source_common_name: str | None, document_identifier: str) -> str:
    if source_common_name and source_common_name.strip():
        return source_common_name.strip().lower()
    parsed = urlparse(document_identifier)
    if parsed.netloc:
        return parsed.netloc.lower()
    return "unknown"


def extract_geo_labels(raw: str | None) -> list[str]:
    if not raw:
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = [part.strip() for part in entry.split("#") if part.strip()]
        for part in parts:
            upper = part.upper()
            if len(upper) == 2 and upper.isalpha() and upper not in seen:
                seen.add(upper)
                labels.append(upper)
    return labels


def load_taxonomy(path: Path) -> list[FactorRule]:
    data = json.loads(path.read_text())
    rules = []
    for item in data["factors"]:
        rules.append(
            FactorRule(
                factor_id=int(item["id"]),
                label=item["label"],
                group=item["group"],
                patterns=tuple(pattern.upper() for pattern in item["patterns"]),
                asset_hints=tuple(item.get("asset_hints", [])),
            )
        )
    return rules


def normalized_match_text(
    themes: Iterable[str],
    persons: Iterable[str],
    organizations: Iterable[str],
    names: Iterable[str],
    locations: Iterable[str],
) -> str:
    return " | ".join([*themes, *persons, *organizations, *names, *locations]).upper()


def classify_factors(match_text: str, rules: list[FactorRule]) -> list[FactorRule]:
    matched: list[FactorRule] = []
    for rule in rules:
        if any(pattern in match_text for pattern in rule.patterns):
            matched.append(rule)
    return matched


def classification_confidence(match_count: int) -> float:
    return min(0.95, 0.55 + 0.08 * match_count)


def initialize_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TABLE bronze_candidates (
            doc_id UBIGINT,
            record_datetime VARCHAR,
            event_time TIMESTAMP,
            partition_date DATE,
            source_domain VARCHAR,
            document_identifier VARCHAR,
            v2_themes VARCHAR,
            v2_tone VARCHAR,
            v2_locations VARCHAR,
            v2_persons VARCHAR,
            v2_organizations VARCHAR,
            all_names VARCHAR,
            tone DOUBLE
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE factor_dictionary (
            factor_id INTEGER,
            factor_label VARCHAR,
            factor_group VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE source_dictionary (
            source_id INTEGER,
            source_domain VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE asset_dictionary (
            asset_id INTEGER,
            asset_label VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE geo_dictionary (
            geo_id INTEGER,
            geo_label VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE silver_event_graph (
            event_time TIMESTAMP,
            cluster_id UBIGINT,
            doc_id UBIGINT,
            factor_ids INTEGER[],
            factor_labels VARCHAR[],
            asset_ids INTEGER[],
            asset_labels VARCHAR[],
            geo_ids INTEGER[],
            geo_labels VARCHAR[],
            source_id INTEGER,
            source_domain VARCHAR,
            tone DOUBLE,
            novelty DOUBLE,
            source_weight DOUBLE,
            classification_confidence DOUBLE,
            model_version VARCHAR,
            prompt_version VARCHAR,
            created_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE silver_factor_mentions (
            bucket_time DATE,
            event_time TIMESTAMP,
            doc_id UBIGINT,
            cluster_id UBIGINT,
            factor_id INTEGER,
            factor_label VARCHAR,
            geo_id INTEGER,
            geo_label VARCHAR,
            source_id INTEGER,
            source_domain VARCHAR,
            tone DOUBLE,
            novelty DOUBLE,
            source_weight DOUBLE,
            classification_confidence DOUBLE
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE silver_asset_factor_mentions (
            bucket_time DATE,
            event_time TIMESTAMP,
            doc_id UBIGINT,
            cluster_id UBIGINT,
            factor_id INTEGER,
            factor_label VARCHAR,
            asset_id INTEGER,
            asset_label VARCHAR,
            geo_id INTEGER,
            geo_label VARCHAR,
            source_id INTEGER,
            source_domain VARCHAR,
            tone DOUBLE,
            novelty DOUBLE,
            source_weight DOUBLE,
            classification_confidence DOUBLE
        )
        """
    )


def flush_pending_rows(
    con: duckdb.DuckDBPyConnection,
    bronze_rows: list[tuple[object, ...]],
    silver_rows: list[tuple[object, ...]],
    factor_mentions: list[tuple[object, ...]],
    asset_factor_mentions: list[tuple[object, ...]],
) -> None:
    if bronze_rows:
        con.executemany(
            "INSERT INTO bronze_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            bronze_rows,
        )
        bronze_rows.clear()
    if silver_rows:
        con.executemany(
            "INSERT INTO silver_event_graph VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            silver_rows,
        )
        silver_rows.clear()
    if factor_mentions:
        con.executemany(
            "INSERT INTO silver_factor_mentions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            factor_mentions,
        )
        factor_mentions.clear()
    if asset_factor_mentions:
        con.executemany(
            "INSERT INTO silver_asset_factor_mentions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            asset_factor_mentions,
        )
        asset_factor_mentions.clear()


def build_narrative_graph(
    input_glob: str,
    output_db: Path,
    taxonomy_path: Path,
    overwrite: bool,
    input_batch_size: int = 5_000,
    flush_every_rows: int = 25_000,
) -> None:
    del input_batch_size, flush_every_rows
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_db.exists():
        output_db.unlink()

    rules = load_taxonomy(taxonomy_path)
    con = duckdb.connect(str(output_db))
    initialize_schema(con)
    con.executemany(
        "INSERT INTO factor_dictionary VALUES (?, ?, ?)",
        [(rule.factor_id, rule.label, rule.group) for rule in rules],
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE factor_rule_patterns (
            factor_id INTEGER,
            factor_label VARCHAR,
            factor_group VARCHAR,
            pattern VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE factor_rule_assets (
            factor_id INTEGER,
            asset_label VARCHAR
        )
        """
    )
    con.executemany(
        "INSERT INTO factor_rule_patterns VALUES (?, ?, ?, ?)",
        [
            (rule.factor_id, rule.label, rule.group, pattern)
            for rule in rules
            for pattern in rule.patterns
        ],
    )
    con.executemany(
        "INSERT INTO factor_rule_assets VALUES (?, ?)",
        [
            (rule.factor_id, asset_label)
            for rule in rules
            for asset_label in rule.asset_hints
        ],
    )

    con.execute(
        f"""
        INSERT INTO bronze_candidates
        SELECT
            CAST(hash(document_identifier) AS UBIGINT) AS doc_id,
            record_datetime,
            COALESCE(
                try_strptime(record_datetime, '%Y%m%d%H%M%S'),
                try_strptime(record_datetime, '%Y%m%d'),
                CAST(try_cast(partition_date AS DATE) AS TIMESTAMP)
            ) AS event_time,
            try_cast(partition_date AS DATE) AS partition_date,
            lower(
                COALESCE(
                    NULLIF(trim(source_common_name), ''),
                    NULLIF(regexp_extract(document_identifier, '^[a-z]+://([^/]+)', 1), ''),
                    'unknown'
                )
            ) AS source_domain,
            document_identifier,
            v2_themes,
            v2_tone,
            v2_locations,
            v2_persons,
            v2_organizations,
            all_names,
            try_cast(regexp_extract(COALESCE(v2_tone, ''), '^\\s*(-?[0-9]+(?:\\.[0-9]+)?)', 1) AS DOUBLE) AS tone
        FROM read_parquet({json.dumps(input_glob)})
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE bronze_enriched AS
        SELECT
            *,
            CASE
                WHEN array_length(list_distinct(regexp_extract_all(COALESCE(v2_locations, ''), '#([A-Z]{2})#', 1))) = 0
                    THEN ['GLOBAL']
                ELSE array_sort(list_distinct(regexp_extract_all(COALESCE(v2_locations, ''), '#([A-Z]{2})#', 1)))
            END AS geo_labels,
            upper(
                concat_ws(
                    ' | ',
                    COALESCE(v2_themes, ''),
                    COALESCE(v2_persons, ''),
                    COALESCE(v2_organizations, ''),
                    COALESCE(all_names, ''),
                    COALESCE(v2_locations, '')
                )
            ) AS match_text
        FROM bronze_candidates
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE matched_factors AS
        SELECT DISTINCT
            b.doc_id,
            p.factor_id,
            p.factor_label
        FROM bronze_enriched b
        JOIN factor_rule_patterns p
          ON contains(b.match_text, p.pattern)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE matched_assets AS
        SELECT DISTINCT
            m.doc_id,
            m.factor_id,
            m.factor_label,
            a.asset_label
        FROM matched_factors m
        JOIN factor_rule_assets a
          ON a.factor_id = m.factor_id
        """
    )
    con.execute(
        """
        INSERT INTO source_dictionary
        SELECT
            row_number() OVER (ORDER BY source_domain)::INTEGER AS source_id,
            source_domain
        FROM (
            SELECT DISTINCT b.source_domain
            FROM bronze_enriched b
            JOIN matched_factors m USING (doc_id)
        )
        """
    )
    con.execute(
        """
        INSERT INTO geo_dictionary
        SELECT
            row_number() OVER (ORDER BY geo_label)::INTEGER AS geo_id,
            geo_label
        FROM (
            SELECT DISTINCT unnest(b.geo_labels) AS geo_label
            FROM bronze_enriched b
            JOIN matched_factors m USING (doc_id)
        )
        """
    )
    con.execute(
        """
        INSERT INTO asset_dictionary
        SELECT
            row_number() OVER (ORDER BY asset_label)::INTEGER AS asset_id,
            asset_label
        FROM (
            SELECT DISTINCT asset_label
            FROM matched_assets
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE factor_rollup AS
        SELECT
            doc_id,
            list(factor_id ORDER BY factor_id) AS factor_ids,
            list(factor_label ORDER BY factor_id) AS factor_labels,
            count(*) AS factor_count
        FROM matched_factors
        GROUP BY doc_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE asset_rollup AS
        SELECT
            m.doc_id,
            list(a.asset_id ORDER BY a.asset_id) AS asset_ids,
            list(m.asset_label ORDER BY a.asset_id) AS asset_labels
        FROM matched_assets m
        JOIN asset_dictionary a USING (asset_label)
        GROUP BY m.doc_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE geo_rollup AS
        SELECT
            g.doc_id,
            list(d.geo_id ORDER BY d.geo_id) AS geo_ids,
            list(g.geo_label ORDER BY d.geo_id) AS geo_labels
        FROM (
            SELECT DISTINCT
                b.doc_id,
                unnest(b.geo_labels) AS geo_label
            FROM bronze_enriched b
            JOIN matched_factors m USING (doc_id)
        ) g
        JOIN geo_dictionary d USING (geo_label)
        GROUP BY g.doc_id
        """
    )
    con.execute(
        """
        INSERT INTO silver_event_graph
        SELECT
            b.event_time,
            CAST(hash(b.source_domain || '|' || b.document_identifier) AS UBIGINT) AS cluster_id,
            b.doc_id,
            f.factor_ids,
            f.factor_labels,
            COALESCE(a.asset_ids, []::INTEGER[]) AS asset_ids,
            COALESCE(a.asset_labels, []::VARCHAR[]) AS asset_labels,
            g.geo_ids,
            g.geo_labels,
            s.source_id,
            b.source_domain,
            b.tone,
            1.0 AS novelty,
            1.0 AS source_weight,
            least(0.95, 0.55 + (0.08 * f.factor_count)) AS classification_confidence,
            'narrative_graph.phase1.v1' AS model_version,
            'deterministic-narrative-taxonomy' AS prompt_version,
            current_timestamp::TIMESTAMP AS created_at
        FROM bronze_enriched b
        JOIN factor_rollup f USING (doc_id)
        JOIN geo_rollup g USING (doc_id)
        LEFT JOIN asset_rollup a USING (doc_id)
        JOIN source_dictionary s USING (source_domain)
        """
    )
    con.execute(
        """
        INSERT INTO silver_factor_mentions
        SELECT
            CAST(b.event_time AS DATE) AS bucket_time,
            b.event_time,
            b.doc_id,
            CAST(hash(b.source_domain || '|' || b.document_identifier) AS UBIGINT) AS cluster_id,
            m.factor_id,
            m.factor_label,
            d.geo_id,
            g.geo_label,
            s.source_id,
            b.source_domain,
            b.tone,
            1.0 AS novelty,
            1.0 AS source_weight,
            least(0.95, 0.55 + (0.08 * f.factor_count)) AS classification_confidence
        FROM bronze_enriched b
        JOIN factor_rollup f USING (doc_id)
        JOIN matched_factors m USING (doc_id)
        JOIN source_dictionary s USING (source_domain)
        CROSS JOIN UNNEST(b.geo_labels) AS g(geo_label)
        JOIN geo_dictionary d USING (geo_label)
        """
    )
    con.execute(
        """
        INSERT INTO silver_asset_factor_mentions
        SELECT
            CAST(b.event_time AS DATE) AS bucket_time,
            b.event_time,
            b.doc_id,
            CAST(hash(b.source_domain || '|' || b.document_identifier) AS UBIGINT) AS cluster_id,
            m.factor_id,
            m.factor_label,
            a.asset_id,
            a.asset_label,
            d.geo_id,
            g.geo_label,
            s.source_id,
            b.source_domain,
            b.tone,
            1.0 AS novelty,
            1.0 AS source_weight,
            least(0.95, 0.55 + (0.08 * f.factor_count)) AS classification_confidence
        FROM bronze_enriched b
        JOIN factor_rollup f USING (doc_id)
        JOIN matched_assets m USING (doc_id)
        JOIN asset_dictionary a USING (asset_label)
        JOIN source_dictionary s USING (source_domain)
        CROSS JOIN UNNEST(b.geo_labels) AS g(geo_label)
        JOIN geo_dictionary d USING (geo_label)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE gold_factor_buckets_daily_base AS
        SELECT
            bucket_time,
            factor_id,
            factor_label,
            geo_id,
            geo_label,
            COUNT(*)::INTEGER AS news_count,
            COUNT(DISTINCT source_id)::INTEGER AS unique_sources,
            AVG(tone) AS tone_mean,
            AVG(novelty) AS novelty_mean,
            SUM(CASE WHEN tone <= -5 THEN 1 ELSE 0 END)::INTEGER AS negative_tail_count,
            SUM(CASE WHEN tone >= 5 THEN 1 ELSE 0 END)::INTEGER AS positive_tail_count,
            AVG(classification_confidence) AS confidence_mean,
            MIN(event_time) AS first_seen,
            MAX(event_time) AS last_seen,
            CASE
                WHEN COUNT(*) = 0 THEN NULL
                ELSE CAST(COUNT(DISTINCT source_id) AS DOUBLE) / CAST(COUNT(*) AS DOUBLE)
            END AS source_dispersion
        FROM silver_factor_mentions
        GROUP BY 1, 2, 3, 4, 5
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE gold_factor_buckets_daily AS
        WITH roll AS (
            SELECT
                *,
                AVG(tone_mean) OVER (
                    PARTITION BY factor_id
                    ORDER BY bucket_time
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) AS tone_mean_30d_avg,
                STDDEV_SAMP(tone_mean) OVER (
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
            news_count,
            unique_sources,
            tone_mean,
            CASE
                WHEN tone_mean_30d_std IS NULL OR tone_mean_30d_std = 0 THEN NULL
                ELSE (tone_mean - tone_mean_30d_avg) / tone_mean_30d_std
            END AS tone_zscore_30d,
            novelty_mean,
            negative_tail_count,
            positive_tail_count,
            source_dispersion,
            confidence_mean,
            first_seen,
            last_seen
        FROM roll
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE gold_asset_factor_panel_daily_base AS
        SELECT
            bucket_time,
            factor_id,
            factor_label,
            asset_id,
            asset_label,
            geo_id,
            geo_label,
            COUNT(*)::INTEGER AS news_count,
            COUNT(DISTINCT source_id)::INTEGER AS unique_sources,
            AVG(tone) AS tone_mean,
            AVG(novelty) AS novelty_mean,
            AVG(classification_confidence) AS confidence,
            CASE
                WHEN COUNT(*) = 0 THEN NULL
                ELSE CAST(COUNT(DISTINCT source_id) AS DOUBLE) / CAST(COUNT(*) AS DOUBLE)
            END AS source_dispersion
        FROM silver_asset_factor_mentions
        GROUP BY 1, 2, 3, 4, 5, 6, 7
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE gold_asset_factor_panel_daily AS
        WITH roll AS (
            SELECT
                *,
                AVG(tone_mean) OVER (
                    PARTITION BY asset_id, factor_id
                    ORDER BY bucket_time
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) AS tone_mean_30d_avg,
                STDDEV_SAMP(tone_mean) OVER (
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
            news_count,
            unique_sources,
            tone_mean,
            CASE
                WHEN tone_mean_30d_std IS NULL OR tone_mean_30d_std = 0 THEN NULL
                ELSE (tone_mean - tone_mean_30d_avg) / tone_mean_30d_std
            END AS tone_zscore_30d,
            novelty_mean,
            CAST(news_count AS DOUBLE) * COALESCE(source_dispersion, 0.0) AS event_intensity,
            source_dispersion,
            confidence
        FROM roll
        """
    )
    con.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", default=str(DEFAULT_INPUT_GLOB))
    parser.add_argument("--output-db", default=str(DEFAULT_DB))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--input-batch-size", type=int, default=1000)
    parser.add_argument("--flush-every-rows", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_narrative_graph(
        input_glob=args.input_glob,
        output_db=Path(args.output_db),
        taxonomy_path=Path(args.taxonomy),
        overwrite=args.overwrite,
        input_batch_size=args.input_batch_size,
        flush_every_rows=args.flush_every_rows,
    )
    return 0


build_factor_graph = build_narrative_graph


if __name__ == "__main__":
    raise SystemExit(main())
