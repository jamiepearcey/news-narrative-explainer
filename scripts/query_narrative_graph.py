#!/usr/bin/env python3
"""Query helper for post-hoc news narrative identification."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _duckdb_bootstrap import ensure_duckdb

ensure_duckdb(__file__)

import argparse
import json
from typing import Any

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "narrative_graph.duckdb"
PRIORITY_CONFIG_PATH = ROOT / "config" / "asset_narrative_priorities.json"
from narrative_text_matching import asset_cues, factor_cues, match_count

ASSET_FACTOR_PRIORITY: dict[str, dict[str, float]] = json.loads(PRIORITY_CONFIG_PATH.read_text())
INDEX_ASSETS = {"NDX", "SPX"}


def rows_to_dicts(columns: list[str], rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    return [dict(zip(columns, row, strict=True)) for row in rows]


def _asset_factor_multiplier(asset_label: str | None, factor_label: str | None) -> float:
    if not asset_label or not factor_label:
        return 1.0
    return ASSET_FACTOR_PRIORITY.get(asset_label, {}).get(factor_label, 1.0)


def _adjust_asset_narrative_rows(rows: list[dict[str, Any]], asset_label: str) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        base_score = float(updated.get("avg_narrative_score") or updated.get("narrative_score") or 0.0)
        multiplier = _asset_factor_multiplier(asset_label, updated.get("factor_label"))
        updated["asset_factor_priority_multiplier"] = multiplier
        updated["adjusted_narrative_score"] = round(base_score * multiplier, 6)
        adjusted.append(updated)
    adjusted.sort(
        key=lambda row: (
            float(row.get("adjusted_narrative_score") or 0.0),
            float(row.get("avg_narrative_score") or row.get("narrative_score") or 0.0),
            float(row.get("doc_count") or 0.0),
            row.get("factor_label") or "",
        ),
        reverse=True,
    )
    return adjusted


def _adjust_asset_narrative_rows_by_asset(
    rows: list[dict[str, Any]],
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        asset_label = str(row.get("asset_label") or "")
        grouped.setdefault(asset_label, []).append(row)
    return {
        asset_label: _adjust_asset_narrative_rows(asset_rows, asset_label)[:limit]
        for asset_label, asset_rows in grouped.items()
    }


def run_query(db_path: Path, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        cursor = con.execute(sql, params or [])
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
        return rows_to_dicts(columns, rows)
    finally:
        con.close()


def table_columns(db_path: Path, table_name: str) -> set[str]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        return {str(row[1]).lower() for row in rows}
    finally:
        con.close()


def relation_exists(db_path: Path, relation_name: str) -> bool:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_name = ?
            LIMIT 1
            """,
            [relation_name],
        ).fetchone()
        return row is not None
    finally:
        con.close()


def _read_parquet_expr(paths: list[str]) -> str:
    return f"read_parquet({json.dumps(paths)}, union_by_name=true)"


def _payload_paths_for_doc_rows(
    db_path: Path,
    index_table: str,
    doc_rows: list[dict[str, Any]],
) -> list[str]:
    if not doc_rows or not relation_exists(db_path, index_table):
        return []
    doc_ids_by_date: dict[str, list[int]] = {}
    for row in doc_rows:
        partition_date = row.get("partition_date")
        doc_id = row.get("doc_id")
        if partition_date is None or doc_id is None:
            continue
        doc_ids_by_date.setdefault(str(partition_date), []).append(int(doc_id))
    if not doc_ids_by_date:
        return []
    date_values = sorted(doc_ids_by_date)
    placeholders = ",".join("?" for _ in date_values)
    index_rows = run_query(
        db_path,
        f"""
        SELECT partition_date, file_path, min_doc_id, max_doc_id
        FROM {index_table}
        WHERE partition_date IN ({placeholders})
        ORDER BY partition_date, min_doc_id, max_doc_id
        """,
        date_values,
    )
    selected: list[str] = []
    seen: set[str] = set()
    for row in index_rows:
        partition_date = str(row["partition_date"])
        min_doc_id = int(row.get("min_doc_id") or 0)
        max_doc_id = int(row.get("max_doc_id") or 0)
        if any(min_doc_id <= doc_id <= max_doc_id for doc_id in doc_ids_by_date.get(partition_date, [])):
            file_path = str(row["file_path"])
            if file_path not in seen:
                seen.add(file_path)
                selected.append(file_path)
    return selected


def _supporting_doc_relevance(doc: dict[str, Any], requested_asset_label: str) -> float:
    factor_terms = factor_cues(doc.get("factor_label"))
    asset_terms = asset_cues(requested_asset_label)
    title = doc.get("title")
    summary = doc.get("summary_text")
    body = doc.get("body_excerpt")
    market_context = doc.get("market_context_text")
    evidence_text = " ".join(filter(None, [market_context, summary, body]))
    title_factor_hits = match_count(title, factor_terms)
    text_factor_hits = match_count(evidence_text, factor_terms)
    title_asset_hits = match_count(title, asset_terms)
    text_asset_hits = match_count(evidence_text, asset_terms)
    source_type = str(doc.get("source_type") or "")
    source_bonus = 3.0 if source_type == "market_wrap" else 2.0 if source_type == "commodity_specialist" else 0.0
    index_penalty = 0.0
    if requested_asset_label in INDEX_ASSETS:
        title = str(doc.get("title") or "")
        market_context_score = float(doc.get("market_context_score") or 0.0)
        if source_type == "company_specific":
            index_penalty -= 8.0
        if source_type == "general_news":
            index_penalty -= 5.0
        if any(token in title.upper() for token in ("(NASDAQ:", "(NYSE:", "FINANCIAL COMPARISON", " VS. ", "QUARTER ENDED")):
            index_penalty -= 12.0
        if market_context_score < 2.0:
            index_penalty -= 4.0
    score = (
        (title_factor_hits * 8.0)
        + (text_factor_hits * 4.0)
        + (title_asset_hits * 6.0)
        + (text_asset_hits * 3.0)
        + source_bonus
        + index_penalty
        + (float(doc.get("classification_confidence") or 0.0) * 2.0)
        + (1.0 if title else 0.0)
        + (0.5 if summary else 0.0)
        + (0.5 if body else 0.0)
    )
    if title_factor_hits + text_factor_hits + title_asset_hits + text_asset_hits == 0:
        score -= 5.0
    return round(score, 3)


def _rerank_supporting_docs(
    rows: list[dict[str, Any]],
    requested_asset_label: str,
    requested_factor_label: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    rescored = []
    for row in rows:
        rescored_row = dict(row)
        rescored_row["relevance_score"] = _supporting_doc_relevance(rescored_row, requested_asset_label)
        rescored.append(rescored_row)
    rescored.sort(
        key=lambda row: (
            float(row.get("relevance_score") or 0.0),
            float(row.get("classification_confidence") or 0.0),
            row.get("event_time"),
            row.get("document_identifier") or "",
        ),
        reverse=True,
    )
    deduped: list[dict[str, Any]] = []
    seen_keys: set[tuple[str | None, str | None, str | None]] = set()
    for row in rescored:
        key = (
            row.get("document_identifier"),
            row.get("asset_label"),
            row.get("factor_label"),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(row)
        if len(deduped) >= limit:
            break
    if requested_factor_label:
        positive = [row for row in deduped if float(row.get("relevance_score") or 0.0) > 0.0]
        if positive:
            return positive[:limit]
    return deduped


def _dedupe_supporting_doc_keys(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_keys: set[tuple[str | None, str | None, str | None]] = set()
    for row in rows:
        key = (
            row.get("document_identifier"),
            row.get("asset_label"),
            row.get("factor_label"),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(row)
    return deduped


def _rerank_supporting_docs_by_asset(
    rows: list[dict[str, Any]],
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        asset_label = str(row.get("asset_label") or "")
        grouped.setdefault(asset_label, []).append(row)
    out: dict[str, list[dict[str, Any]]] = {}
    for asset_label, asset_rows in grouped.items():
        out[asset_label] = _rerank_supporting_docs(asset_rows, asset_label, None, limit)
    return out


def query_summary(db_path: Path) -> dict[str, Any]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = [
            "bronze_candidates",
            "silver_event_graph",
            "silver_factor_mentions",
            "silver_asset_factor_mentions",
            "gold_factor_buckets_daily",
            "gold_asset_factor_panel_daily",
            "gold_factor_crossover_links_daily",
            "gold_asset_factor_crossover_links_daily",
        ]
        counts = {table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables}
        event_span = con.execute(
            "SELECT min(event_time), max(event_time) FROM silver_event_graph"
        ).fetchone()
        bucket_span = con.execute(
            "SELECT min(bucket_time), max(bucket_time), count(DISTINCT bucket_time) FROM gold_factor_buckets_daily"
        ).fetchone()
        build_state = con.execute(
            """
            SELECT
                COUNT(DISTINCT partition_date),
                MIN(partition_date),
                MAX(partition_date)
            FROM graph_build_partitions
            """
        ).fetchone()
        return {
            "database": str(db_path),
            "table_counts": counts,
            "event_span": {"min": event_span[0], "max": event_span[1]},
            "bucket_span": {
                "min": bucket_span[0],
                "max": bucket_span[1],
                "bucket_dates": bucket_span[2],
            },
            "build_partitions": {
                "count": build_state[0],
                "min": build_state[1],
                "max": build_state[2],
            },
        }
    finally:
        con.close()


def query_top_factors(db_path: Path, limit: int) -> list[dict[str, Any]]:
    return run_query(
        db_path,
        """
        SELECT
            factor_label,
            SUM(doc_count) AS doc_count,
            SUM(mention_count) AS mention_count,
            AVG(source_dispersion) AS avg_source_dispersion,
            AVG(tone_mean) AS avg_tone_mean,
            AVG(narrative_score) AS avg_narrative_score
        FROM gold_factor_buckets_daily
        GROUP BY factor_label
        ORDER BY doc_count DESC, avg_narrative_score DESC, factor_label ASC
        LIMIT ?
        """,
        [limit],
    )


def query_top_assets(db_path: Path, limit: int) -> list[dict[str, Any]]:
    return run_query(
        db_path,
        """
        SELECT
            asset_label,
            SUM(doc_count) AS doc_count,
            SUM(mention_count) AS mention_count,
            AVG(source_dispersion) AS avg_source_dispersion,
            AVG(event_intensity) AS avg_event_intensity,
            AVG(narrative_score) AS avg_narrative_score
        FROM gold_asset_factor_panel_daily
        GROUP BY asset_label
        ORDER BY doc_count DESC, avg_narrative_score DESC, asset_label ASC
        LIMIT ?
        """,
        [limit],
    )


def query_factor_daily(db_path: Path, factor_label: str, limit: int) -> list[dict[str, Any]]:
    return run_query(
        db_path,
        """
        SELECT
            bucket_time,
            geo_label,
            doc_count,
            mention_count,
            unique_sources,
            tone_mean,
            tone_zscore_30d,
            avg_abs_tone,
            novelty_mean,
            source_dispersion,
            confidence_mean,
            narrative_score
        FROM gold_factor_buckets_daily
        WHERE factor_label = ?
        ORDER BY bucket_time DESC, narrative_score DESC, geo_label ASC
        LIMIT ?
        """,
        [factor_label, limit],
    )


def query_tone_tails(db_path: Path, limit: int) -> list[dict[str, Any]]:
    return run_query(
        db_path,
        """
        SELECT
            factor_label,
            geo_label,
            bucket_time,
            doc_count,
            mention_count,
            negative_tail_count,
            positive_tail_count,
            tone_mean,
            source_dispersion,
            narrative_score
        FROM gold_factor_buckets_daily
        ORDER BY
            negative_tail_count DESC,
            positive_tail_count DESC,
            narrative_score DESC,
            factor_label ASC
        LIMIT ?
        """,
        [limit],
    )


def query_asset_narratives(
    db_path: Path,
    asset_label: str,
    start_date: str | None,
    end_date: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    rows = run_query(
        db_path,
        """
        WITH filtered AS (
            SELECT *
            FROM silver_asset_factor_mentions
            WHERE asset_label = ?
              AND (? IS NULL OR CAST(event_time AS DATE) >= CAST(? AS DATE))
              AND (? IS NULL OR CAST(event_time AS DATE) <= CAST(? AS DATE))
        ),
        dedup_docs AS (
            SELECT DISTINCT
                doc_id,
                asset_label,
                factor_label,
                source_id,
                CAST(event_time AS DATE) AS bucket_time,
                tone,
                classification_confidence,
                asset_factor_relevance
            FROM filtered
        ),
        geo_stats AS (
            SELECT
                asset_label,
                factor_label,
                COUNT(DISTINCT geo_id)::INTEGER AS geo_count,
                COUNT(*)::INTEGER AS mention_count
            FROM filtered
            GROUP BY asset_label, factor_label
        )
        SELECT
            d.asset_label,
            d.factor_label,
            COUNT(*)::INTEGER AS doc_count,
            g.mention_count,
            COUNT(DISTINCT d.source_id)::INTEGER AS avg_unique_sources,
            g.geo_count AS avg_geo_count,
            AVG(d.tone) AS avg_tone_mean,
            NULL::DOUBLE AS avg_tone_zscore_30d,
            AVG(abs(COALESCE(d.tone, 0.0))) AS avg_abs_tone,
            AVG(d.asset_factor_relevance) AS avg_asset_factor_relevance,
            CASE
                WHEN COUNT(*) = 0 THEN NULL
                ELSE CAST(COUNT(DISTINCT d.source_id) AS DOUBLE) / CAST(COUNT(*) AS DOUBLE)
            END AS avg_source_dispersion,
            CAST(COUNT(DISTINCT d.source_id) AS DOUBLE)
                * (0.5 + (AVG(d.asset_factor_relevance) / 8.0)) AS avg_event_intensity,
            CAST(COUNT(*) AS DOUBLE)
                * (
                    0.5 + CASE
                        WHEN COUNT(*) = 0 THEN 0.0
                        ELSE CAST(COUNT(DISTINCT d.source_id) AS DOUBLE) / CAST(COUNT(*) AS DOUBLE)
                    END
                )
                * (0.5 + (AVG(d.asset_factor_relevance) / 8.0))
                * (1.0 + (AVG(abs(COALESCE(d.tone, 0.0))) / 5.0)) AS avg_narrative_score,
            CAST(COUNT(*) AS DOUBLE)
                * (
                    0.5 + CASE
                        WHEN COUNT(*) = 0 THEN 0.0
                        ELSE CAST(COUNT(DISTINCT d.source_id) AS DOUBLE) / CAST(COUNT(*) AS DOUBLE)
                    END
                )
                * (0.5 + (AVG(d.asset_factor_relevance) / 8.0))
                * (1.0 + (AVG(abs(COALESCE(d.tone, 0.0))) / 5.0)) AS max_narrative_score,
            MIN(d.bucket_time) AS first_bucket,
            MAX(d.bucket_time) AS last_bucket
        FROM dedup_docs d
        JOIN geo_stats g
          ON g.asset_label = d.asset_label
         AND g.factor_label = d.factor_label
        GROUP BY d.asset_label, d.factor_label, g.mention_count, g.geo_count
        ORDER BY avg_narrative_score DESC, doc_count DESC, d.factor_label ASC
        LIMIT ?
        """,
        [asset_label, start_date, start_date, end_date, end_date, max(limit * 5, 25)],
    )
    return _adjust_asset_narrative_rows(rows, asset_label)[:limit]


def query_asset_narratives_bulk(
    db_path: Path,
    asset_labels: list[str],
    start_date: str | None,
    end_date: str | None,
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    if not asset_labels:
        return {}
    placeholders = ",".join("?" for _ in asset_labels)
    rows = run_query(
        db_path,
        f"""
        WITH filtered AS (
            SELECT *
            FROM silver_asset_factor_mentions
            WHERE asset_label IN ({placeholders})
              AND (? IS NULL OR CAST(event_time AS DATE) >= CAST(? AS DATE))
              AND (? IS NULL OR CAST(event_time AS DATE) <= CAST(? AS DATE))
        ),
        dedup_docs AS (
            SELECT DISTINCT
                doc_id,
                asset_label,
                factor_label,
                source_id,
                CAST(event_time AS DATE) AS bucket_time,
                tone,
                classification_confidence,
                asset_factor_relevance
            FROM filtered
        ),
        geo_stats AS (
            SELECT
                asset_label,
                factor_label,
                COUNT(DISTINCT geo_id)::INTEGER AS geo_count,
                COUNT(*)::INTEGER AS mention_count
            FROM filtered
            GROUP BY asset_label, factor_label
        )
        SELECT
            d.asset_label,
            d.factor_label,
            COUNT(*)::INTEGER AS doc_count,
            g.mention_count,
            COUNT(DISTINCT d.source_id)::INTEGER AS avg_unique_sources,
            g.geo_count AS avg_geo_count,
            AVG(d.tone) AS avg_tone_mean,
            NULL::DOUBLE AS avg_tone_zscore_30d,
            AVG(abs(COALESCE(d.tone, 0.0))) AS avg_abs_tone,
            AVG(d.asset_factor_relevance) AS avg_asset_factor_relevance,
            CASE
                WHEN COUNT(*) = 0 THEN NULL
                ELSE CAST(COUNT(DISTINCT d.source_id) AS DOUBLE) / CAST(COUNT(*) AS DOUBLE)
            END AS avg_source_dispersion,
            CAST(COUNT(DISTINCT d.source_id) AS DOUBLE)
                * (0.5 + (AVG(d.asset_factor_relevance) / 8.0)) AS avg_event_intensity,
            CAST(COUNT(*) AS DOUBLE)
                * (
                    0.5 + CASE
                        WHEN COUNT(*) = 0 THEN 0.0
                        ELSE CAST(COUNT(DISTINCT d.source_id) AS DOUBLE) / CAST(COUNT(*) AS DOUBLE)
                    END
                )
                * (0.5 + (AVG(d.asset_factor_relevance) / 8.0))
                * (1.0 + (AVG(abs(COALESCE(d.tone, 0.0))) / 5.0)) AS avg_narrative_score,
            CAST(COUNT(*) AS DOUBLE)
                * (
                    0.5 + CASE
                        WHEN COUNT(*) = 0 THEN 0.0
                        ELSE CAST(COUNT(DISTINCT d.source_id) AS DOUBLE) / CAST(COUNT(*) AS DOUBLE)
                    END
                )
                * (0.5 + (AVG(d.asset_factor_relevance) / 8.0))
                * (1.0 + (AVG(abs(COALESCE(d.tone, 0.0))) / 5.0)) AS max_narrative_score,
            MIN(d.bucket_time) AS first_bucket,
            MAX(d.bucket_time) AS last_bucket
        FROM dedup_docs d
        JOIN geo_stats g
          ON g.asset_label = d.asset_label
         AND g.factor_label = d.factor_label
        GROUP BY d.asset_label, d.factor_label, g.mention_count, g.geo_count
        """,
        asset_labels + [start_date, start_date, end_date, end_date],
    )
    return _adjust_asset_narrative_rows_by_asset(rows, limit)


def query_asset_timeline(
    db_path: Path,
    asset_label: str,
    factor_label: str | None,
    start_date: str | None,
    end_date: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    return run_query(
        db_path,
        """
        WITH filtered AS (
            SELECT *
            FROM silver_asset_factor_mentions
            WHERE asset_label = ?
              AND (? IS NULL OR factor_label = ?)
              AND (? IS NULL OR CAST(event_time AS DATE) >= CAST(? AS DATE))
              AND (? IS NULL OR CAST(event_time AS DATE) <= CAST(? AS DATE))
        ),
        dedup_docs AS (
            SELECT DISTINCT
                bucket_time,
                doc_id,
                asset_label,
                factor_label,
                source_id,
                tone,
                classification_confidence
            FROM filtered
        ),
        geo_stats AS (
            SELECT
                bucket_time,
                asset_label,
                factor_label,
                COUNT(DISTINCT geo_id)::INTEGER AS geo_count,
                COUNT(*)::INTEGER AS mention_count
            FROM filtered
            GROUP BY bucket_time, asset_label, factor_label
        )
        SELECT
            d.bucket_time,
            d.asset_label,
            d.factor_label,
            COUNT(*)::INTEGER AS doc_count,
            g.mention_count,
            COUNT(DISTINCT d.source_id)::INTEGER AS unique_sources,
            g.geo_count,
            AVG(d.tone) AS tone_mean,
            NULL::DOUBLE AS tone_zscore_30d,
            AVG(abs(COALESCE(d.tone, 0.0))) AS avg_abs_tone,
            CASE
                WHEN COUNT(*) = 0 THEN NULL
                ELSE CAST(COUNT(DISTINCT d.source_id) AS DOUBLE) / CAST(COUNT(*) AS DOUBLE)
            END AS source_dispersion,
            CAST(COUNT(DISTINCT d.source_id) AS DOUBLE) AS event_intensity,
            AVG(d.classification_confidence) AS confidence,
            CAST(COUNT(*) AS DOUBLE)
                * (
                    0.5 + CASE
                        WHEN COUNT(*) = 0 THEN 0.0
                        ELSE CAST(COUNT(DISTINCT d.source_id) AS DOUBLE) / CAST(COUNT(*) AS DOUBLE)
                    END
                )
                * (1.0 + (AVG(abs(COALESCE(d.tone, 0.0))) / 5.0)) AS narrative_score
        FROM dedup_docs d
        JOIN geo_stats g
          ON g.bucket_time = d.bucket_time
         AND g.asset_label = d.asset_label
         AND g.factor_label = d.factor_label
        GROUP BY d.bucket_time, d.asset_label, d.factor_label, g.mention_count, g.geo_count
        ORDER BY d.bucket_time DESC, narrative_score DESC, d.factor_label ASC
        LIMIT ?
        """,
        [asset_label, factor_label, factor_label, start_date, start_date, end_date, end_date, limit],
    )


def query_factor_crossovers(
    db_path: Path,
    factor_label: str | None,
    start_date: str | None,
    end_date: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    return run_query(
        db_path,
        """
        SELECT
            prior_bucket_time,
            bucket_time,
            factor_label,
            geo_label,
            prior_doc_count,
            doc_count,
            prior_narrative_score,
            narrative_score,
            doc_count_delta,
            narrative_score_delta
        FROM gold_factor_crossover_links_daily
        WHERE (? IS NULL OR factor_label = ?)
          AND (? IS NULL OR bucket_time >= CAST(? AS DATE))
          AND (? IS NULL OR bucket_time <= CAST(? AS DATE))
        ORDER BY
            bucket_time DESC,
            abs(narrative_score_delta) DESC,
            abs(doc_count_delta) DESC,
            factor_label ASC,
            geo_label ASC
        LIMIT ?
        """,
        [factor_label, factor_label, start_date, start_date, end_date, end_date, limit],
    )


def query_asset_crossovers(
    db_path: Path,
    asset_label: str,
    factor_label: str | None,
    start_date: str | None,
    end_date: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    return run_query(
        db_path,
        """
        SELECT
            prior_bucket_time,
            bucket_time,
            asset_label,
            factor_label,
            geo_label,
            prior_doc_count,
            doc_count,
            prior_narrative_score,
            narrative_score,
            doc_count_delta,
            narrative_score_delta
        FROM gold_asset_factor_crossover_links_daily
        WHERE asset_label = ?
          AND (? IS NULL OR factor_label = ?)
          AND (? IS NULL OR bucket_time >= CAST(? AS DATE))
          AND (? IS NULL OR bucket_time <= CAST(? AS DATE))
        ORDER BY
            bucket_time DESC,
            abs(narrative_score_delta) DESC,
            abs(doc_count_delta) DESC,
            factor_label ASC,
            geo_label ASC
        LIMIT ?
        """,
        [asset_label, factor_label, factor_label, start_date, start_date, end_date, end_date, limit],
    )


def query_supporting_docs(
    db_path: Path,
    asset_label: str,
    factor_label: str | None,
    start_date: str | None,
    end_date: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    candidate_limit = max(limit * 40, 200) if factor_label else max(limit * 20, 100)
    bronze_columns = table_columns(db_path, "bronze_candidates")
    has_doc_payload = relation_exists(db_path, "doc_payload_daily")
    source_type_expr = "b.source_type" if "source_type" in bronze_columns else "CAST(NULL AS VARCHAR) AS source_type"
    source_priority_expr = (
        "b.source_priority" if "source_priority" in bronze_columns else "CAST(NULL AS INTEGER) AS source_priority"
    )
    market_context_expr = (
        "b.market_context_text"
        if "market_context_text" in bronze_columns
        else "CAST(NULL AS VARCHAR) AS market_context_text"
    )
    market_context_score_expr = (
        "b.market_context_score"
        if "market_context_score" in bronze_columns
        else "CAST(NULL AS DOUBLE) AS market_context_score"
    )
    evidence_preference_expr = "b.market_context_text," if "market_context_text" in bronze_columns else ""
    ordering_pref_expr = (
        "WHEN COALESCE(NULLIF(b.market_context_text, ''), NULLIF(b.title, ''), NULLIF(b.summary_text, ''), NULLIF(b.body_text, '')) IS NULL THEN 0"
        if "market_context_text" in bronze_columns
        else "WHEN COALESCE(NULLIF(b.title, ''), NULLIF(b.summary_text, ''), NULLIF(b.body_text, '')) IS NULL THEN 0"
    )
    hot_ordering_pref_expr = (
        "WHEN COALESCE(NULLIF(b.market_context_text, ''), NULLIF(b.title, '')) IS NULL THEN 0"
        if "market_context_text" in bronze_columns
        else "WHEN NULLIF(b.title, '') IS NULL THEN 0"
    )
    source_priority_order_expr = "COALESCE(b.source_priority, 0) DESC," if "source_priority" in bronze_columns else ""
    if has_doc_payload:
        has_doc_review = relation_exists(db_path, "doc_review_daily")
        has_doc_detail = relation_exists(db_path, "doc_detail_daily")
        candidate_rows = run_query(
            db_path,
            f"""
            SELECT DISTINCT
                m.doc_id,
                CAST(m.event_time AS DATE) AS partition_date,
                m.event_time,
                m.asset_label,
                m.factor_label,
                m.geo_label,
                m.source_domain,
                {source_type_expr},
                {source_priority_expr},
                b.title,
                {market_context_expr},
                {market_context_score_expr},
                b.document_identifier,
                b.tone,
                m.classification_confidence
            FROM silver_asset_factor_mentions m
            JOIN bronze_candidates b USING (doc_id)
            WHERE m.asset_label = ?
              AND (? IS NULL OR m.factor_label = ?)
              AND (? IS NULL OR CAST(m.event_time AS DATE) >= CAST(? AS DATE))
              AND (? IS NULL OR CAST(m.event_time AS DATE) <= CAST(? AS DATE))
            ORDER BY
                CASE
                    {hot_ordering_pref_expr}
                    ELSE 1
                END DESC,
                {source_priority_order_expr}
                m.event_time DESC,
                m.classification_confidence DESC,
                m.factor_label ASC
            LIMIT ?
            """,
            [
                asset_label,
                factor_label,
                factor_label,
                start_date,
                start_date,
                end_date,
                end_date,
                candidate_limit,
            ],
        )
        if not candidate_rows:
            return []
        doc_ids = [row["doc_id"] for row in candidate_rows]
        review_paths = _payload_paths_for_doc_rows(db_path, "doc_review_daily_file_index", candidate_rows)
        placeholders = ",".join("?" for _ in doc_ids)
        review_source = "doc_payload_daily"
        if has_doc_review:
            review_source = _read_parquet_expr(review_paths) if review_paths else "doc_review_daily"
            light_payload_rows = run_query(
                db_path,
                f"""
                SELECT
                    doc_id,
                    partition_date,
                    summary_text,
                    relevant_text,
                    metadata_json,
                    gkg_extras,
                    quotations,
                    title AS payload_title,
                    document_identifier AS payload_document_identifier
                FROM {review_source}
                WHERE doc_id IN ({placeholders})
                """,
                doc_ids,
            )
        else:
            light_payload_rows = run_query(
                db_path,
                f"""
                SELECT
                    doc_id,
                    partition_date,
                    summary_text,
                    relevant_text,
                    metadata_json,
                    gkg_extras,
                    quotations,
                    title AS payload_title,
                    document_identifier AS payload_document_identifier
                FROM doc_payload_daily
                WHERE doc_id IN ({placeholders})
                """,
                doc_ids,
            )
        light_payload_by_doc_id = {row["doc_id"]: row for row in light_payload_rows}
        hydrated_rows: list[dict[str, Any]] = []
        for row in candidate_rows:
            payload = light_payload_by_doc_id.get(row["doc_id"], {})
            merged = dict(row)
            merged["title"] = merged.get("title") or payload.get("payload_title")
            merged["summary_text"] = payload.get("summary_text")
            merged["body_excerpt"] = None
            merged["relevant_text"] = payload.get("relevant_text")
            merged["metadata_json"] = payload.get("metadata_json")
            merged["gkg_extras"] = payload.get("gkg_extras")
            merged["quotations"] = payload.get("quotations")
            merged["document_identifier"] = (
                merged.get("document_identifier") or payload.get("payload_document_identifier")
            )
            merged["evidence_text"] = next(
                (
                    value
                    for value in [
                        merged.get("market_context_text"),
                        merged.get("summary_text"),
                        merged.get("title"),
                        merged.get("relevant_text"),
                    ]
                    if value
                ),
                None,
            )
            hydrated_rows.append(merged)
        shortlisted_rows = _rerank_supporting_docs(hydrated_rows, asset_label, factor_label, max(limit * 5, 25))
        shortlisted_rows = _dedupe_supporting_doc_keys(shortlisted_rows)
        shortlisted_doc_ids = [row["doc_id"] for row in shortlisted_rows]
        if not shortlisted_doc_ids:
            return []
        detail_paths = _payload_paths_for_doc_rows(db_path, "doc_detail_daily_file_index", shortlisted_rows)
        full_placeholders = ",".join("?" for _ in shortlisted_doc_ids)
        if has_doc_detail:
            detail_source = _read_parquet_expr(detail_paths) if detail_paths else "doc_detail_daily"
            detail_rows = run_query(
                db_path,
                f"""
                SELECT
                    doc_id,
                    partition_date,
                    body_text,
                    sharing_image,
                    related_images,
                    social_image_embeds,
                    social_video_embeds,
                    amounts,
                    dates,
                    gcam,
                    translation_info
                FROM {detail_source}
                WHERE doc_id IN ({full_placeholders})
                """,
                shortlisted_doc_ids,
            )
            review_rows = run_query(
                db_path,
                f"""
                SELECT
                    doc_id,
                    partition_date,
                    summary_text,
                    relevant_text,
                    metadata_json,
                    gkg_extras,
                    quotations,
                    title AS payload_title,
                    document_identifier AS payload_document_identifier
                FROM {review_source}
                WHERE doc_id IN ({full_placeholders})
                """,
                shortlisted_doc_ids,
            )
            full_payload_by_doc_id = {row["doc_id"]: row for row in review_rows}
            for row in detail_rows:
                full_payload_by_doc_id.setdefault(row["doc_id"], {}).update(row)
        else:
            full_payload_rows = run_query(
                db_path,
                f"""
                SELECT
                    doc_id,
                    partition_date,
                    summary_text,
                    body_text,
                    relevant_text,
                    metadata_json,
                    gkg_extras,
                    sharing_image,
                    related_images,
                    social_image_embeds,
                    social_video_embeds,
                    quotations,
                    amounts,
                    dates,
                    gcam,
                    translation_info,
                    title AS payload_title,
                    document_identifier AS payload_document_identifier
                FROM doc_payload_daily
                WHERE doc_id IN ({full_placeholders})
                """,
                shortlisted_doc_ids,
            )
            full_payload_by_doc_id = {row["doc_id"]: row for row in full_payload_rows}
        final_rows: list[dict[str, Any]] = []
        for row in shortlisted_rows:
            payload = full_payload_by_doc_id.get(row["doc_id"], {})
            merged = dict(row)
            body_text = payload.get("body_text")
            merged["title"] = merged.get("title") or payload.get("payload_title")
            merged["summary_text"] = payload.get("summary_text")
            merged["body_excerpt"] = None if body_text is None else str(body_text)[:1200]
            merged["relevant_text"] = payload.get("relevant_text")
            merged["metadata_json"] = payload.get("metadata_json")
            merged["gkg_extras"] = payload.get("gkg_extras")
            merged["sharing_image"] = payload.get("sharing_image")
            merged["related_images"] = payload.get("related_images")
            merged["social_image_embeds"] = payload.get("social_image_embeds")
            merged["social_video_embeds"] = payload.get("social_video_embeds")
            merged["quotations"] = payload.get("quotations")
            merged["amounts"] = payload.get("amounts")
            merged["dates"] = payload.get("dates")
            merged["gcam"] = payload.get("gcam")
            merged["translation_info"] = payload.get("translation_info")
            merged["document_identifier"] = (
                merged.get("document_identifier") or payload.get("payload_document_identifier")
            )
            merged["evidence_text"] = next(
                (
                    value
                    for value in [
                        merged.get("market_context_text"),
                        merged.get("summary_text"),
                        merged.get("title"),
                        None if body_text is None else str(body_text)[:400],
                        merged.get("relevant_text"),
                    ]
                    if value
                ),
                None,
            )
            final_rows.append(merged)
        return _rerank_supporting_docs(final_rows, asset_label, factor_label, limit)
    rows = run_query(
        db_path,
        f"""
        SELECT DISTINCT
            m.event_time,
            m.asset_label,
            m.factor_label,
            m.geo_label,
            m.source_domain,
            {source_type_expr},
            {source_priority_expr},
            b.title,
            b.summary_text,
            CASE
                WHEN b.body_text IS NULL THEN NULL
                ELSE substr(b.body_text, 1, 1200)
            END AS body_excerpt,
            {market_context_expr},
            {market_context_score_expr},
            b.relevant_text,
            b.metadata_json,
            b.gkg_extras,
            b.sharing_image,
            b.related_images,
            b.social_image_embeds,
            b.social_video_embeds,
            b.quotations,
            b.amounts,
            b.dates,
            b.gcam,
            b.translation_info,
            COALESCE(
                {evidence_preference_expr}
                b.summary_text,
                b.title,
                CASE
                    WHEN b.body_text IS NULL THEN NULL
                    ELSE substr(b.body_text, 1, 400)
                END,
                b.relevant_text
            ) AS evidence_text,
            b.document_identifier,
            b.tone,
            m.classification_confidence
        FROM silver_asset_factor_mentions m
        JOIN bronze_candidates b USING (doc_id)
        WHERE m.asset_label = ?
          AND (? IS NULL OR m.factor_label = ?)
          AND (? IS NULL OR CAST(m.event_time AS DATE) >= CAST(? AS DATE))
          AND (? IS NULL OR CAST(m.event_time AS DATE) <= CAST(? AS DATE))
        ORDER BY
            CASE
                {ordering_pref_expr}
                ELSE 1
            END DESC,
            {source_priority_order_expr}
            m.event_time DESC,
            m.classification_confidence DESC,
            m.factor_label ASC
        LIMIT ?
        """,
        [
            asset_label,
            factor_label,
            factor_label,
            start_date,
            start_date,
            end_date,
            end_date,
            candidate_limit,
        ],
    )
    return _rerank_supporting_docs(rows, asset_label, factor_label, limit)


def query_supporting_docs_bulk(
    db_path: Path,
    asset_labels: list[str],
    start_date: str | None,
    end_date: str | None,
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    if not asset_labels:
        return {}
    bronze_columns = table_columns(db_path, "bronze_candidates")
    has_doc_payload = relation_exists(db_path, "doc_payload_daily")
    source_type_expr = "b.source_type" if "source_type" in bronze_columns else "CAST(NULL AS VARCHAR) AS source_type"
    source_priority_expr = (
        "b.source_priority" if "source_priority" in bronze_columns else "CAST(NULL AS INTEGER) AS source_priority"
    )
    market_context_expr = (
        "b.market_context_text"
        if "market_context_text" in bronze_columns
        else "CAST(NULL AS VARCHAR) AS market_context_text"
    )
    market_context_score_expr = (
        "b.market_context_score"
        if "market_context_score" in bronze_columns
        else "CAST(NULL AS DOUBLE) AS market_context_score"
    )
    hot_ordering_pref_expr = (
        "WHEN COALESCE(NULLIF(b.market_context_text, ''), NULLIF(b.title, '')) IS NULL THEN 0"
        if "market_context_text" in bronze_columns
        else "WHEN NULLIF(b.title, '') IS NULL THEN 0"
    )
    source_priority_order_expr = "COALESCE(b.source_priority, 0) DESC," if "source_priority" in bronze_columns else ""
    candidate_limit = max(limit * 20, 100)
    placeholders = ",".join("?" for _ in asset_labels)
    candidate_rows = run_query(
        db_path,
        f"""
        SELECT DISTINCT
            m.doc_id,
            CAST(m.event_time AS DATE) AS partition_date,
            m.event_time,
            m.asset_label,
            m.factor_label,
            m.geo_label,
            m.source_domain,
            {source_type_expr},
            {source_priority_expr},
            b.title,
            {market_context_expr},
            {market_context_score_expr},
            b.document_identifier,
            b.tone,
            m.classification_confidence
        FROM silver_asset_factor_mentions m
        JOIN bronze_candidates b USING (doc_id)
        WHERE m.asset_label IN ({placeholders})
          AND (? IS NULL OR CAST(m.event_time AS DATE) >= CAST(? AS DATE))
          AND (? IS NULL OR CAST(m.event_time AS DATE) <= CAST(? AS DATE))
        ORDER BY
            m.asset_label ASC,
            CASE
                {hot_ordering_pref_expr}
                ELSE 1
            END DESC,
            {source_priority_order_expr}
            m.event_time DESC,
            m.classification_confidence DESC,
            m.factor_label ASC
        LIMIT ?
        """,
        asset_labels + [start_date, start_date, end_date, end_date, candidate_limit * max(len(asset_labels), 1)],
    )
    if not candidate_rows:
        return {asset_label: [] for asset_label in asset_labels}
    if not has_doc_payload:
        return _rerank_supporting_docs_by_asset(candidate_rows, limit)
    has_doc_review = relation_exists(db_path, "doc_review_daily")
    has_doc_detail = relation_exists(db_path, "doc_detail_daily")
    doc_ids = [row["doc_id"] for row in candidate_rows]
    doc_placeholders = ",".join("?" for _ in doc_ids)
    review_paths = _payload_paths_for_doc_rows(db_path, "doc_review_daily_file_index", candidate_rows)
    review_source = "doc_payload_daily"
    if has_doc_review:
        review_source = _read_parquet_expr(review_paths) if review_paths else "doc_review_daily"
    review_rows = run_query(
        db_path,
        f"""
        SELECT
            doc_id,
            partition_date,
            summary_text,
            relevant_text,
            metadata_json,
            gkg_extras,
            quotations,
            title AS payload_title,
            document_identifier AS payload_document_identifier
        FROM {review_source}
        WHERE doc_id IN ({doc_placeholders})
        """,
        doc_ids,
    )
    review_by_doc_id = {row["doc_id"]: row for row in review_rows}
    hydrated: list[dict[str, Any]] = []
    for row in candidate_rows:
        payload = review_by_doc_id.get(row["doc_id"], {})
        merged = dict(row)
        merged["title"] = merged.get("title") or payload.get("payload_title")
        merged["summary_text"] = payload.get("summary_text")
        merged["body_excerpt"] = None
        merged["relevant_text"] = payload.get("relevant_text")
        merged["metadata_json"] = payload.get("metadata_json")
        merged["gkg_extras"] = payload.get("gkg_extras")
        merged["quotations"] = payload.get("quotations")
        merged["document_identifier"] = merged.get("document_identifier") or payload.get("payload_document_identifier")
        merged["evidence_text"] = next(
            (value for value in [merged.get("market_context_text"), merged.get("summary_text"), merged.get("title"), merged.get("relevant_text")] if value),
            None,
        )
        hydrated.append(merged)
    shortlisted_by_asset = _rerank_supporting_docs_by_asset(hydrated, max(limit * 5, 25))
    shortlisted_rows = _dedupe_supporting_doc_keys(
        [row for asset in asset_labels for row in shortlisted_by_asset.get(asset, [])]
    )
    shortlisted_doc_ids = [row["doc_id"] for row in shortlisted_rows]
    if not shortlisted_doc_ids:
        return {asset_label: [] for asset_label in asset_labels}
    full_placeholders = ",".join("?" for _ in shortlisted_doc_ids)
    detail_paths = _payload_paths_for_doc_rows(db_path, "doc_detail_daily_file_index", shortlisted_rows)
    if has_doc_detail:
        detail_source = _read_parquet_expr(detail_paths) if detail_paths else "doc_detail_daily"
        detail_rows = run_query(
            db_path,
            f"""
            SELECT
                doc_id,
                partition_date,
                body_text,
                sharing_image,
                related_images,
                social_image_embeds,
                social_video_embeds,
                amounts,
                dates,
                gcam,
                translation_info
            FROM {detail_source}
            WHERE doc_id IN ({full_placeholders})
            """,
            shortlisted_doc_ids,
        )
        full_payload_by_doc_id = {row["doc_id"]: row for row in review_rows if row["doc_id"] in shortlisted_doc_ids}
        for row in detail_rows:
            full_payload_by_doc_id.setdefault(row["doc_id"], {}).update(row)
    else:
        full_payload_rows = run_query(
            db_path,
            f"""
            SELECT
                doc_id,
                partition_date,
                summary_text,
                body_text,
                relevant_text,
                metadata_json,
                gkg_extras,
                sharing_image,
                related_images,
                social_image_embeds,
                social_video_embeds,
                quotations,
                amounts,
                dates,
                gcam,
                translation_info,
                title AS payload_title,
                document_identifier AS payload_document_identifier
            FROM doc_payload_daily
            WHERE doc_id IN ({full_placeholders})
            """,
            shortlisted_doc_ids,
        )
        full_payload_by_doc_id = {row["doc_id"]: row for row in full_payload_rows}
    final_rows: list[dict[str, Any]] = []
    for row in shortlisted_rows:
        payload = full_payload_by_doc_id.get(row["doc_id"], {})
        merged = dict(row)
        body_text = payload.get("body_text")
        merged["title"] = merged.get("title") or payload.get("payload_title")
        merged["summary_text"] = payload.get("summary_text")
        merged["body_excerpt"] = None if body_text is None else str(body_text)[:1200]
        merged["relevant_text"] = payload.get("relevant_text")
        merged["metadata_json"] = payload.get("metadata_json")
        merged["gkg_extras"] = payload.get("gkg_extras")
        merged["sharing_image"] = payload.get("sharing_image")
        merged["related_images"] = payload.get("related_images")
        merged["social_image_embeds"] = payload.get("social_image_embeds")
        merged["social_video_embeds"] = payload.get("social_video_embeds")
        merged["quotations"] = payload.get("quotations")
        merged["amounts"] = payload.get("amounts")
        merged["dates"] = payload.get("dates")
        merged["gcam"] = payload.get("gcam")
        merged["translation_info"] = payload.get("translation_info")
        merged["document_identifier"] = merged.get("document_identifier") or payload.get("payload_document_identifier")
        merged["evidence_text"] = next(
            (
                value
                for value in [
                    merged.get("market_context_text"),
                    merged.get("summary_text"),
                    merged.get("title"),
                    None if body_text is None else str(body_text)[:400],
                    merged.get("relevant_text"),
                ]
                if value
            ),
            None,
        )
        final_rows.append(merged)
    return _rerank_supporting_docs_by_asset(final_rows, limit)


def query_explain_move(
    db_path: Path,
    asset_label: str,
    start_date: str | None,
    end_date: str | None,
    limit: int,
) -> dict[str, Any]:
    return {
        "asset_label": asset_label,
        "window": {"start_date": start_date, "end_date": end_date},
        "top_narratives": query_asset_narratives(db_path, asset_label, start_date, end_date, limit),
        "timeline": query_asset_timeline(db_path, asset_label, None, start_date, end_date, limit),
        "supporting_docs": query_supporting_docs(db_path, asset_label, None, start_date, end_date, limit),
    }


def json_default(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument(
        "--view",
        required=True,
        choices=[
            "summary",
            "top-factors",
            "top-assets",
            "factor-daily",
            "factor-crossovers",
            "tone-tails",
            "asset-narratives",
            "asset-timeline",
            "asset-crossovers",
            "supporting-docs",
            "explain-move",
        ],
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--factor-label")
    parser.add_argument("--asset-label")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"database not found: {db_path}")

    if args.view == "summary":
        payload: Any = query_summary(db_path)
    elif args.view == "top-factors":
        payload = query_top_factors(db_path, args.limit)
    elif args.view == "top-assets":
        payload = query_top_assets(db_path, args.limit)
    elif args.view == "factor-daily":
        if not args.factor_label:
            raise SystemExit("--factor-label is required for --view factor-daily")
        payload = query_factor_daily(db_path, args.factor_label, args.limit)
    elif args.view == "factor-crossovers":
        payload = query_factor_crossovers(
            db_path, args.factor_label, args.start_date, args.end_date, args.limit
        )
    elif args.view == "tone-tails":
        payload = query_tone_tails(db_path, args.limit)
    elif args.view == "asset-narratives":
        if not args.asset_label:
            raise SystemExit("--asset-label is required for --view asset-narratives")
        payload = query_asset_narratives(
            db_path, args.asset_label, args.start_date, args.end_date, args.limit
        )
    elif args.view == "asset-timeline":
        if not args.asset_label:
            raise SystemExit("--asset-label is required for --view asset-timeline")
        payload = query_asset_timeline(
            db_path,
            args.asset_label,
            args.factor_label,
            args.start_date,
            args.end_date,
            args.limit,
        )
    elif args.view == "asset-crossovers":
        if not args.asset_label:
            raise SystemExit("--asset-label is required for --view asset-crossovers")
        payload = query_asset_crossovers(
            db_path,
            args.asset_label,
            args.factor_label,
            args.start_date,
            args.end_date,
            args.limit,
        )
    elif args.view == "supporting-docs":
        if not args.asset_label:
            raise SystemExit("--asset-label is required for --view supporting-docs")
        payload = query_supporting_docs(
            db_path,
            args.asset_label,
            args.factor_label,
            args.start_date,
            args.end_date,
            args.limit,
        )
    else:
        if not args.asset_label:
            raise SystemExit("--asset-label is required for --view explain-move")
        payload = query_explain_move(
            db_path, args.asset_label, args.start_date, args.end_date, args.limit
        )

    print(json.dumps(payload, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
