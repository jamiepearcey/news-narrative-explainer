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


def rows_to_dicts(columns: list[str], rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    return [dict(zip(columns, row, strict=True)) for row in rows]


def run_query(db_path: Path, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        cursor = con.execute(sql, params or [])
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
        return rows_to_dicts(columns, rows)
    finally:
        con.close()


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
        ]
        counts = {table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables}
        event_span = con.execute(
            "SELECT min(event_time), max(event_time) FROM silver_event_graph"
        ).fetchone()
        bucket_span = con.execute(
            "SELECT min(bucket_time), max(bucket_time), count(DISTINCT bucket_time) FROM gold_factor_buckets_daily"
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
    return run_query(
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
                classification_confidence
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
            CASE
                WHEN COUNT(*) = 0 THEN NULL
                ELSE CAST(COUNT(DISTINCT d.source_id) AS DOUBLE) / CAST(COUNT(*) AS DOUBLE)
            END AS avg_source_dispersion,
            CAST(COUNT(DISTINCT d.source_id) AS DOUBLE) AS avg_event_intensity,
            CAST(COUNT(*) AS DOUBLE)
                * (
                    0.5 + CASE
                        WHEN COUNT(*) = 0 THEN 0.0
                        ELSE CAST(COUNT(DISTINCT d.source_id) AS DOUBLE) / CAST(COUNT(*) AS DOUBLE)
                    END
                )
                * (1.0 + (AVG(abs(COALESCE(d.tone, 0.0))) / 5.0)) AS avg_narrative_score,
            CAST(COUNT(*) AS DOUBLE)
                * (
                    0.5 + CASE
                        WHEN COUNT(*) = 0 THEN 0.0
                        ELSE CAST(COUNT(DISTINCT d.source_id) AS DOUBLE) / CAST(COUNT(*) AS DOUBLE)
                    END
                )
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
        [asset_label, start_date, start_date, end_date, end_date, limit],
    )


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


def query_supporting_docs(
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
        SELECT DISTINCT
            m.event_time,
            m.asset_label,
            m.factor_label,
            m.geo_label,
            m.source_domain,
            b.title,
            b.summary_text,
            CASE
                WHEN b.body_text IS NULL THEN NULL
                ELSE substr(b.body_text, 1, 1200)
            END AS body_excerpt,
            b.relevant_text,
            b.document_identifier,
            b.tone,
            m.classification_confidence
        FROM silver_asset_factor_mentions m
        JOIN bronze_candidates b USING (doc_id)
        WHERE m.asset_label = ?
          AND (? IS NULL OR m.factor_label = ?)
          AND (? IS NULL OR CAST(m.event_time AS DATE) >= CAST(? AS DATE))
          AND (? IS NULL OR CAST(m.event_time AS DATE) <= CAST(? AS DATE))
        ORDER BY m.event_time DESC, m.classification_confidence DESC, m.factor_label ASC
        LIMIT ?
        """,
        [asset_label, factor_label, factor_label, start_date, start_date, end_date, end_date, limit],
    )


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
            "tone-tails",
            "asset-narratives",
            "asset-timeline",
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
