#!/usr/bin/env python3
"""Hosted-parquet DuckDB adapter for the v3 narrative graph."""

from __future__ import annotations

import json
import hashlib
import os
import urllib.parse
import urllib.request
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

from _duckdb_bootstrap import ensure_duckdb

ensure_duckdb(__file__)

import duckdb

DATE_FORMAT = "%Y-%m-%d"
DEFAULT_HTTP_ROOT = "http://127.0.0.1:8789"
CACHE_SCHEMA_VERSION = "v2"
DEFAULT_QUERY_CACHE_ROOT = (
    Path("/Users/jamiepearcey/projects/research/news-narrative-explainer/data/.cache/v3_duckdb_query")
)

WINDOW_TABLE_LAYOUTS = {
    "graph_doc_nodes_daily": ("partition_date", "copy"),
    "doc_review_daily": ("partition_date", "copy"),
    "doc_detail_daily": ("partition_date", "copy"),
    "doc_payload_daily": ("partition_date", "copy"),
    "silver_event_graph": ("event_date", "copy"),
    "silver_factor_mentions": ("bucket_time", "copy"),
    "silver_asset_factor_mentions": ("bucket_time", "copy"),
    "silver_market_context_mentions": ("bucket_time", "copy"),
}

ALL_GOLD_TABLE_LAYOUTS = {
    "gold_factor_buckets_daily": ("bucket_time", "copy"),
    "gold_asset_factor_panel_daily": ("bucket_time", "copy"),
    "gold_factor_crossover_links_daily": ("bucket_time", "copy"),
    "gold_asset_factor_crossover_links_daily": ("bucket_time", "copy"),
}

INDEXED_PAYLOAD_TABLES = {"doc_review_daily", "doc_detail_daily"}


def _parse_date(value: str) -> date:
    return datetime.strptime(value, DATE_FORMAT).date()


def _iter_dates(start_date: str, end_date: str | None = None) -> list[str]:
    start = _parse_date(start_date)
    end = _parse_date(end_date or start_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    out: list[str] = []
    cursor = start
    while cursor <= end:
        out.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return out


def _normalize_store(store: str | Path) -> str:
    raw = str(store).rstrip("/")
    if raw.startswith("http:/") and not raw.startswith("http://"):
        return "http://" + raw[len("http:/") :]
    if raw.startswith("https:/") and not raw.startswith("https://"):
        return "https://" + raw[len("https:/") :]
    return raw


def _is_remote_store(store: str) -> bool:
    parsed = urllib.parse.urlparse(store)
    return parsed.scheme in {"http", "https"}


def _read_manifest(store: str) -> dict[str, object]:
    if _is_remote_store(store):
        with urllib.request.urlopen(f"{store}/manifest.json") as response:
            return json.loads(response.read().decode("utf-8"))
    path = Path(store) / "manifest.json"
    return json.loads(path.read_text())


def _manifest_fingerprint(store: str) -> str:
    payload = _read_manifest(store)
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def materialized_dates(store: str | Path) -> list[str]:
    payload = _read_manifest(_normalize_store(store))
    return sorted(str(value) for value in payload.get("materialized_dates", []))


def _partition_file(store: str, table: str, partition_column: str, day: str) -> str:
    if _is_remote_store(store):
        return f"{store}/{table}/{partition_column}={day}/part-000.parquet"
    return str(Path(store) / table / f"{partition_column}={day}" / "part-000.parquet")


def _partition_index_file(store: str, table: str, partition_column: str, day: str) -> str:
    if _is_remote_store(store):
        return f"{store}/{table}/{partition_column}={day}/index.json"
    return str(Path(store) / table / f"{partition_column}={day}" / "index.json")


def _read_partition_index(store: str, table: str, partition_column: str, day: str) -> dict[str, object] | None:
    path = _partition_index_file(store, table, partition_column, day)
    try:
        if _is_remote_store(store):
            with urllib.request.urlopen(path) as response:
                return json.loads(response.read().decode("utf-8"))
        file_path = Path(path)
        if not file_path.exists():
            return None
        return json.loads(file_path.read_text())
    except Exception:
        return None


def _existing_partition_files(store: str, table: str, partition_column: str, days: list[str]) -> list[str]:
    if table in INDEXED_PAYLOAD_TABLES:
        out: list[str] = []
        for day in days:
            manifest = _read_partition_index(store, table, partition_column, day)
            if manifest is None:
                fallback = _partition_file(store, table, partition_column, day)
                if _is_remote_store(store) or Path(fallback).exists():
                    out.append(fallback)
                continue
            for chunk in manifest.get("chunks", []):
                file_name = str(chunk.get("file_name"))
                if _is_remote_store(store):
                    out.append(f"{store}/{table}/{partition_column}={day}/{file_name}")
                else:
                    out.append(str(Path(store) / table / f"{partition_column}={day}" / file_name))
        return out
    if _is_remote_store(store):
        return [_partition_file(store, table, partition_column, day) for day in days]
    out: list[str] = []
    root = Path(store)
    for day in days:
        path = root / table / f"{partition_column}={day}" / "part-000.parquet"
        if path.exists():
            out.append(str(path))
    return out


def _read_parquet_expr(paths: list[str]) -> str:
    return f"read_parquet({json.dumps(paths)}, union_by_name=true)"


def _load_httpfs(con: duckdb.DuckDBPyConnection, store: str) -> None:
    if _is_remote_store(store):
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
        con.execute("SET enable_http_metadata_cache = true;")


def initialize_v3_query_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS graph_doc_nodes_daily (
            doc_id UBIGINT,
            event_time TIMESTAMP,
            partition_date DATE,
            source_domain VARCHAR,
            document_identifier VARCHAR,
            title VARCHAR,
            source_type VARCHAR,
            source_priority INTEGER,
            market_context_text VARCHAR,
            market_context_score DOUBLE,
            tone DOUBLE,
            geo_labels_json VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_review_daily (
            doc_id UBIGINT,
            partition_date DATE,
            document_identifier VARCHAR,
            title VARCHAR,
            summary_text VARCHAR,
            relevant_text VARCHAR,
            metadata_json VARCHAR,
            gkg_extras VARCHAR,
            quotations VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_detail_daily (
            doc_id UBIGINT,
            partition_date DATE,
            body_text VARCHAR,
            sharing_image VARCHAR,
            related_images VARCHAR,
            social_image_embeds VARCHAR,
            social_video_embeds VARCHAR,
            amounts VARCHAR,
            dates VARCHAR,
            gcam VARCHAR,
            translation_info VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_payload_daily (
            doc_id UBIGINT,
            partition_date DATE,
            document_identifier VARCHAR,
            title VARCHAR,
            summary_text VARCHAR,
            body_text VARCHAR,
            relevant_text VARCHAR,
            metadata_json VARCHAR,
            gkg_extras VARCHAR,
            sharing_image VARCHAR,
            related_images VARCHAR,
            social_image_embeds VARCHAR,
            social_video_embeds VARCHAR,
            quotations VARCHAR,
            amounts VARCHAR,
            dates VARCHAR,
            gcam VARCHAR,
            translation_info VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_review_daily_file_index (
            partition_date DATE,
            file_path VARCHAR,
            min_doc_id UBIGINT,
            max_doc_id UBIGINT,
            row_count BIGINT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_detail_daily_file_index (
            partition_date DATE,
            file_path VARCHAR,
            min_doc_id UBIGINT,
            max_doc_id UBIGINT,
            row_count BIGINT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS bronze_candidates (
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
            title VARCHAR,
            summary_text VARCHAR,
            body_text VARCHAR,
            relevant_text VARCHAR,
            metadata_json VARCHAR,
            gkg_extras VARCHAR,
            sharing_image VARCHAR,
            related_images VARCHAR,
            social_image_embeds VARCHAR,
            social_video_embeds VARCHAR,
            quotations VARCHAR,
            amounts VARCHAR,
            dates VARCHAR,
            gcam VARCHAR,
            translation_info VARCHAR,
            source_type VARCHAR,
            source_priority INTEGER,
            market_context_text VARCHAR,
            market_context_score DOUBLE,
            tone DOUBLE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS silver_event_graph (
            event_time TIMESTAMP,
            cluster_id UBIGINT,
            doc_id UBIGINT,
            factor_ids VARCHAR,
            factor_labels VARCHAR,
            asset_ids VARCHAR,
            asset_labels VARCHAR,
            geo_ids VARCHAR,
            geo_labels VARCHAR,
            source_id UBIGINT,
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
        CREATE TABLE IF NOT EXISTS silver_factor_mentions (
            bucket_time DATE,
            event_time TIMESTAMP,
            doc_id UBIGINT,
            cluster_id UBIGINT,
            factor_id UINTEGER,
            factor_label VARCHAR,
            geo_id UBIGINT,
            geo_label VARCHAR,
            source_id UBIGINT,
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
        CREATE TABLE IF NOT EXISTS silver_asset_factor_mentions (
            bucket_time DATE,
            event_time TIMESTAMP,
            doc_id UBIGINT,
            cluster_id UBIGINT,
            factor_id UINTEGER,
            factor_label VARCHAR,
            asset_id UBIGINT,
            asset_label VARCHAR,
            geo_id UBIGINT,
            geo_label VARCHAR,
            source_id UBIGINT,
            source_domain VARCHAR,
            tone DOUBLE,
            novelty DOUBLE,
            source_weight DOUBLE,
            classification_confidence DOUBLE,
            asset_factor_relevance DOUBLE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS silver_market_context_mentions (
            bucket_time DATE,
            event_time TIMESTAMP,
            doc_id UBIGINT,
            cluster_id UBIGINT,
            factor_label VARCHAR,
            asset_label VARCHAR,
            source_domain VARCHAR,
            source_type VARCHAR,
            source_priority INTEGER,
            market_context_text VARCHAR,
            market_context_score DOUBLE,
            classification_confidence DOUBLE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS graph_build_partitions (
            partition_date DATE,
            source_glob VARCHAR,
            processed_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS gold_factor_buckets_daily (
            bucket_time DATE,
            factor_id UINTEGER,
            factor_label VARCHAR,
            geo_id UBIGINT,
            geo_label VARCHAR,
            doc_count BIGINT,
            mention_count BIGINT,
            unique_sources BIGINT,
            geo_count BIGINT,
            tone_mean DOUBLE,
            tone_zscore_30d DOUBLE,
            avg_abs_tone DOUBLE,
            novelty_mean DOUBLE,
            negative_tail_count BIGINT,
            positive_tail_count BIGINT,
            source_dispersion DOUBLE,
            confidence_mean DOUBLE,
            first_seen TIMESTAMP,
            last_seen TIMESTAMP,
            narrative_score DOUBLE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS gold_asset_factor_panel_daily (
            bucket_time DATE,
            asset_id UBIGINT,
            asset_label VARCHAR,
            factor_id UINTEGER,
            factor_label VARCHAR,
            geo_id UBIGINT,
            geo_label VARCHAR,
            doc_count BIGINT,
            mention_count BIGINT,
            unique_sources BIGINT,
            geo_count BIGINT,
            tone_mean DOUBLE,
            tone_zscore_30d DOUBLE,
            avg_abs_tone DOUBLE,
            novelty_mean DOUBLE,
            event_intensity DOUBLE,
            source_dispersion DOUBLE,
            confidence DOUBLE,
            narrative_score DOUBLE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS gold_factor_crossover_links_daily (
            prior_bucket_time DATE,
            bucket_time DATE,
            factor_id UINTEGER,
            factor_label VARCHAR,
            geo_id UBIGINT,
            geo_label VARCHAR,
            prior_doc_count BIGINT,
            doc_count BIGINT,
            prior_narrative_score DOUBLE,
            narrative_score DOUBLE,
            doc_count_delta BIGINT,
            narrative_score_delta DOUBLE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS gold_asset_factor_crossover_links_daily (
            prior_bucket_time DATE,
            bucket_time DATE,
            asset_id UBIGINT,
            asset_label VARCHAR,
            factor_id UINTEGER,
            factor_label VARCHAR,
            geo_id UBIGINT,
            geo_label VARCHAR,
            prior_doc_count BIGINT,
            doc_count BIGINT,
            prior_narrative_score DOUBLE,
            narrative_score DOUBLE,
            doc_count_delta BIGINT,
            narrative_score_delta DOUBLE
        )
        """
    )


def _replace_relation_with_view(con: duckdb.DuckDBPyConnection, relation_name: str, select_sql: str) -> None:
    row = con.execute(
        """
        SELECT table_type
        FROM information_schema.tables
        WHERE table_name = ?
        LIMIT 1
        """,
        [relation_name],
    ).fetchone()
    if row is not None:
        relation_type = str(row[0]).upper()
        if relation_type == "VIEW":
            con.execute(f"DROP VIEW {relation_name}")
        else:
            con.execute(f"DROP TABLE {relation_name}")
    con.execute(f"CREATE VIEW {relation_name} AS {select_sql}")


def _replace_copy_table_view(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    paths: list[str],
) -> None:
    if not paths:
        return
    read_expr = _read_parquet_expr(paths)
    if table_name == "silver_event_graph":
        select_sql = f"""
            SELECT
                CAST(event_time AS TIMESTAMP) AS event_time,
                cluster_id,
                doc_id,
                factor_ids,
                factor_labels,
                asset_ids,
                asset_labels,
                geo_ids,
                geo_labels,
                source_id,
                source_domain,
                tone,
                novelty,
                source_weight,
                classification_confidence,
                model_version,
                prompt_version,
                CAST(created_at AS TIMESTAMP) AS created_at
            FROM {read_expr}
        """
    elif table_name == "silver_factor_mentions":
        select_sql = f"""
            SELECT
                bucket_time,
                CAST(event_time AS TIMESTAMP) AS event_time,
                doc_id,
                cluster_id,
                factor_id,
                factor_label,
                geo_id,
                geo_label,
                source_id,
                source_domain,
                tone,
                novelty,
                source_weight,
                classification_confidence
            FROM {read_expr}
        """
    elif table_name == "silver_asset_factor_mentions":
        select_sql = f"""
            SELECT
                bucket_time,
                CAST(event_time AS TIMESTAMP) AS event_time,
                doc_id,
                cluster_id,
                factor_id,
                factor_label,
                asset_id,
                asset_label,
                geo_id,
                geo_label,
                source_id,
                source_domain,
                tone,
                novelty,
                source_weight,
                classification_confidence,
                asset_factor_relevance
            FROM {read_expr}
        """
    elif table_name == "silver_market_context_mentions":
        select_sql = f"""
            SELECT
                bucket_time,
                CAST(event_time AS TIMESTAMP) AS event_time,
                doc_id,
                cluster_id,
                factor_label,
                asset_label,
                source_domain,
                source_type,
                source_priority,
                market_context_text,
                market_context_score,
                classification_confidence
            FROM {read_expr}
        """
    elif table_name == "gold_factor_buckets_daily":
        select_sql = f"""
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
                tone_zscore_30d,
                avg_abs_tone,
                novelty_mean,
                negative_tail_count,
                positive_tail_count,
                source_dispersion,
                confidence_mean,
                CAST(first_seen AS TIMESTAMP) AS first_seen,
                CAST(last_seen AS TIMESTAMP) AS last_seen,
                narrative_score
            FROM {read_expr}
        """
    elif table_name == "gold_asset_factor_panel_daily":
        select_sql = f"""
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
                tone_zscore_30d,
                avg_abs_tone,
                novelty_mean,
                event_intensity,
                source_dispersion,
                confidence,
                narrative_score
            FROM {read_expr}
        """
    elif table_name == "graph_doc_nodes_daily":
        select_sql = f"""
            SELECT
                doc_id,
                CAST(event_time AS TIMESTAMP) AS event_time,
                partition_date,
                source_domain,
                document_identifier,
                title,
                source_type,
                source_priority,
                market_context_text,
                market_context_score,
                tone,
                geo_labels_json
            FROM {read_expr}
        """
    elif table_name == "doc_payload_daily":
        select_sql = f"SELECT * FROM {read_expr}"
    elif table_name == "doc_review_daily":
        select_sql = f"SELECT * FROM {read_expr}"
    elif table_name == "doc_detail_daily":
        select_sql = f"SELECT * FROM {read_expr}"
    else:
        select_sql = f"SELECT * FROM {read_expr}"
    _replace_relation_with_view(con, table_name, select_sql)


def _replace_bronze_candidates_view(con: duckdb.DuckDBPyConnection, store: str, requested_dates: list[str]) -> None:
    graph_paths = _existing_partition_files(store, "graph_doc_nodes_daily", "partition_date", requested_dates)
    if not graph_paths:
        return
    graph_expr = _read_parquet_expr(graph_paths)
    _replace_relation_with_view(
        con,
        "bronze_candidates",
        f"""
        SELECT
            g.doc_id,
            CAST(NULL AS VARCHAR) AS record_datetime,
            CAST(g.event_time AS TIMESTAMP) AS event_time,
            g.partition_date,
            g.source_domain,
            g.document_identifier,
            CAST(NULL AS VARCHAR) AS v2_themes,
            CAST(NULL AS VARCHAR) AS v2_tone,
            CAST(NULL AS VARCHAR) AS v2_locations,
            CAST(NULL AS VARCHAR) AS v2_persons,
            CAST(NULL AS VARCHAR) AS v2_organizations,
            CAST(NULL AS VARCHAR) AS all_names,
            g.title,
            CAST(NULL AS VARCHAR) AS summary_text,
            CAST(NULL AS VARCHAR) AS body_text,
            CAST(NULL AS VARCHAR) AS relevant_text,
            CAST(NULL AS VARCHAR) AS metadata_json,
            CAST(NULL AS VARCHAR) AS gkg_extras,
            CAST(NULL AS VARCHAR) AS sharing_image,
            CAST(NULL AS VARCHAR) AS related_images,
            CAST(NULL AS VARCHAR) AS social_image_embeds,
            CAST(NULL AS VARCHAR) AS social_video_embeds,
            CAST(NULL AS VARCHAR) AS quotations,
            CAST(NULL AS VARCHAR) AS amounts,
            CAST(NULL AS VARCHAR) AS dates,
            CAST(NULL AS VARCHAR) AS gcam,
            CAST(NULL AS VARCHAR) AS translation_info,
            g.source_type,
            g.source_priority,
            g.market_context_text,
            g.market_context_score,
            g.tone
        FROM {graph_expr} AS g
        """,
    )


def _load_graph_build_partitions(con: duckdb.DuckDBPyConnection, store: str, dates: list[str]) -> None:
    if not dates:
        return
    con.executemany(
        "INSERT INTO graph_build_partitions VALUES (?, ?, current_timestamp::TIMESTAMP)",
        [(day, store) for day in dates],
    )


def _load_payload_file_index(
    con: duckdb.DuckDBPyConnection,
    store: str,
    table_name: str,
    partition_column: str,
    dates: list[str],
) -> None:
    index_table = f"{table_name}_file_index"
    con.execute(f"DELETE FROM {index_table}")
    rows: list[tuple[str, str, int, int, int]] = []
    for day in dates:
        manifest = _read_partition_index(store, table_name, partition_column, day)
        if manifest is None:
            fallback = _partition_file(store, table_name, partition_column, day)
            if _is_remote_store(store) or Path(fallback).exists():
                rows.append((day, fallback, 0, 2**64 - 1, 0))
            continue
        for chunk in manifest.get("chunks", []):
            file_name = str(chunk.get("file_name"))
            if _is_remote_store(store):
                file_path = f"{store}/{table_name}/{partition_column}={day}/{file_name}"
            else:
                file_path = str(Path(store) / table_name / f"{partition_column}={day}" / file_name)
            rows.append(
                (
                    day,
                    file_path,
                    int(chunk.get("min_doc_id", 0)),
                    int(chunk.get("max_doc_id", 0)),
                    int(chunk.get("row_count", 0)),
                )
            )
    if rows:
        con.executemany(
            f"INSERT INTO {index_table} VALUES (?, ?, ?, ?, ?)",
            rows,
        )


def _cache_key(
    store: str,
    requested_dates: list[str],
    gold_dates: list[str],
    *,
    gold_scope: str,
    load_profile: str,
) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "store": store,
        "manifest_fingerprint": _manifest_fingerprint(store),
        "requested_dates": requested_dates,
        "gold_dates": gold_dates,
        "gold_scope": gold_scope,
        "load_profile": load_profile,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _cache_paths(cache_key: str) -> tuple[Path, Path]:
    db_path = DEFAULT_QUERY_CACHE_ROOT / f"{cache_key}.duckdb"
    meta_path = DEFAULT_QUERY_CACHE_ROOT / f"{cache_key}.json"
    return db_path, meta_path


def _write_cache_metadata(
    meta_path: Path,
    *,
    store: str,
    requested_dates: list[str],
    gold_dates: list[str],
    gold_scope: str,
    load_profile: str,
) -> None:
    meta_path.write_text(
        json.dumps(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "store": store,
                "manifest_fingerprint": _manifest_fingerprint(store),
                "requested_dates": requested_dates,
                "gold_dates": gold_dates,
                "gold_scope": gold_scope,
                "load_profile": load_profile,
                "built_at_utc": datetime.utcnow().isoformat() + "Z",
            },
            indent=2,
            sort_keys=True,
        )
    )


def _build_cached_db(
    db_path: Path,
    store: str,
    requested_dates: list[str],
    gold_dates: list[str],
    *,
    gold_scope: str,
    load_profile: str,
) -> None:
    if let_parent := db_path.parent:
        let_parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".tmp.duckdb")
    if tmp_path.exists():
        tmp_path.unlink()
    con = duckdb.connect(str(tmp_path))
    try:
        _load_httpfs(con, store)
        initialize_v3_query_schema(con)
        if load_profile == "full":
            _replace_bronze_candidates_view(con, store, requested_dates)
            for table_name, (partition_column, scope) in WINDOW_TABLE_LAYOUTS.items():
                dates = requested_dates if scope == "copy" else materialized_dates(store)
                paths = _existing_partition_files(store, table_name, partition_column, dates)
                _replace_copy_table_view(con, table_name, paths)
                if table_name in INDEXED_PAYLOAD_TABLES:
                    _load_payload_file_index(con, store, table_name, partition_column, dates)
        for table_name, (partition_column, _) in ALL_GOLD_TABLE_LAYOUTS.items():
            paths = _existing_partition_files(store, table_name, partition_column, gold_dates)
            _replace_copy_table_view(con, table_name, paths)
        _load_graph_build_partitions(con, store, gold_dates)
        con.close()
        if db_path.exists():
            db_path.unlink()
        os.replace(tmp_path, db_path)
    finally:
        try:
            con.close()
        except Exception:
            pass
        if tmp_path.exists():
            tmp_path.unlink()


@contextmanager
def resolve_query_db(
    store: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    gold_scope: str = "window",
    load_profile: str = "full",
) -> Iterator[Path]:
    normalized = _normalize_store(store)
    if normalized.endswith(".duckdb") and not _is_remote_store(normalized):
        yield Path(normalized)
        return

    all_dates = materialized_dates(normalized)
    requested_dates = _iter_dates(start_date, end_date) if start_date else all_dates
    if gold_scope not in {"window", "all"}:
        raise ValueError("gold_scope must be 'window' or 'all'")
    if load_profile not in {"full", "gold_only"}:
        raise ValueError("load_profile must be 'full' or 'gold_only'")
    gold_dates = all_dates if gold_scope == "all" else requested_dates
    cache_key = _cache_key(
        normalized,
        requested_dates,
        gold_dates,
        gold_scope=gold_scope,
        load_profile=load_profile,
    )
    db_path, meta_path = _cache_paths(cache_key)
    if not db_path.exists():
        _build_cached_db(
            db_path,
            normalized,
            requested_dates,
            gold_dates,
            gold_scope=gold_scope,
            load_profile=load_profile,
        )
        _write_cache_metadata(
            meta_path,
            store=normalized,
            requested_dates=requested_dates,
            gold_dates=gold_dates,
            gold_scope=gold_scope,
            load_profile=load_profile,
        )
    yield db_path
