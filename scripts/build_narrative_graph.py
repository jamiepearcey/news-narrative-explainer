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
import html
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable
from urllib.parse import urlparse

import duckdb
from narrative_text_matching import ASSET_TEXT_PATTERNS, asset_cues, factor_cues, match_count

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


TEXT_TITLE_CANDIDATES = (
    "title",
    "article_title",
    "headline",
    "source_title",
)
TEXT_SUMMARY_CANDIDATES = (
    "summary",
    "snippet",
    "description",
    "excerpt",
    "lead",
    "lead_text",
)
TEXT_BODY_CANDIDATES = (
    "text",
    "article_text",
    "body_text",
    "content",
    "translated_text",
    "full_text",
)
METADATA_CANDIDATES = ("metadata_json",)
GKG_EXTRAS_CANDIDATES = ("gkg_extras", "extras")
SHARING_IMAGE_CANDIDATES = ("sharing_image", "SharingImage")
RELATED_IMAGES_CANDIDATES = ("related_images", "RelatedImages")
SOCIAL_IMAGE_EMBEDS_CANDIDATES = ("social_image_embeds", "SocialImageEmbeds")
SOCIAL_VIDEO_EMBEDS_CANDIDATES = ("social_video_embeds", "SocialVideoEmbeds")
QUOTATIONS_CANDIDATES = ("quotations", "Quotations")
AMOUNTS_CANDIDATES = ("amounts", "Amounts")
DATES_CANDIDATES = ("dates", "Dates")
GCAM_CANDIDATES = ("gcam", "GCAM")
TRANSLATION_INFO_CANDIDATES = ("translation_info", "TranslationInfo")
ASSET_CONTEXT_REQUIRED = {
    "WTI",
    "Brent",
    "HG",
    "BDI",
    "Gold",
    "BTC",
    "FXI",
    "NG",
    "TTF",
    "XLE",
    "XME",
    "GDX",
    "CAD",
    "FCX",
    "BHP",
    "RIO",
    "COIN",
    "NDX",
    "SPX",
}
MARKET_WRAP_DOMAINS = {
    "reuters.com",
    "wsj.com",
    "barrons.com",
    "marketwatch.com",
    "apnews.com",
    "ft.com",
    "finance.yahoo.com",
    "investopedia.com",
    "business-standard.com",
    "cnbcafrica.com",
    "moneycontrol.com",
}
COMMODITY_SPECIALIST_DOMAINS = {
    "oilandgas360.com",
    "kitco.com",
    "mining.com",
    "oilprice.com",
}
MARKET_SENTENCE_TERMS = [
    "YIELD",
    "TREASURY",
    "DOLLAR",
    "USD",
    "NASDAQ",
    "STOCK",
    "EQUITY",
    "OIL",
    "GOLD",
    "COPPER",
    "INFLATION",
    "FED",
    "RATE",
    "RISK OFF",
    "SAFE HAVEN",
    "REAL YIELD",
    "TERM PREMIUM",
    "AI",
    "SEMI",
    "S&P",
    "WALL STREET",
    "QQQ",
    "NASDAQ 100",
]
MARKET_CONTEXT_BATCH_SIZE = 2_000


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


def input_columns(con: duckdb.DuckDBPyConnection, input_glob: str) -> set[str]:
    cursor = con.execute(
        f"SELECT * FROM read_parquet({json.dumps(input_glob)}, union_by_name=true) LIMIT 0"
    )
    return {column[0].lower() for column in cursor.description}


def first_present(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate.lower() in columns:
            return candidate
    return None


def sql_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def optional_text_expr(column_name: str | None) -> str:
    if column_name is None:
        return "NULL"
    return f"NULLIF(trim(CAST({sql_identifier(column_name)} AS VARCHAR)), '')"


def optional_expr_for(columns: set[str], candidates: tuple[str, ...]) -> str:
    return optional_text_expr(first_present(columns, candidates))


def normalize_html_text(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = html.unescape(raw) if "&" in raw else raw
    text = " ".join(text.split())
    return text or None


def sql_contains_any(column_expr: str, terms: Iterable[str]) -> str:
    predicates = [
        f"position({sql_string_literal(term)} IN upper(COALESCE({column_expr}, ''))) > 0"
        for term in terms
    ]
    return "(" + " OR ".join(predicates) + ")" if predicates else "FALSE"


def classify_source_type(source_domain: str | None, title: str | None, document_identifier: str | None) -> tuple[str, int]:
    domain = (source_domain or "").lower()
    title_upper = (title or "").upper()
    url = (document_identifier or "").lower()
    if domain in MARKET_WRAP_DOMAINS:
        return "market_wrap", 5
    if domain in COMMODITY_SPECIALIST_DOMAINS:
        return "commodity_specialist", 4
    if any(
        token in title_upper
        for token in (
            "FINANCIAL COMPARISON",
            " VS. ",
            " VS ",
            "COMPARE",
            "EARNINGS PREVIEW",
            "QUARTER ENDED",
            "EBIT MARGIN",
            "(NASDAQ:",
            "(NYSE:",
        )
    ) or "/markets/stocks/" in url:
        return "company_specific", 2
    if any(token in title_upper for token in ("ETF", "YIELDS", "TREASURY", "DOLLAR", "NASDAQ", "MARKETS")):
        return "market_wrap", 4
    if any(token in title_upper for token in ("CRUDE", "OIL", "GOLD", "COPPER", "BULLION", "MINING")):
        return "commodity_specialist", 3
    if any(token in title_upper for token in ("INC.", " INC ", "CORP", "EARNINGS", "PLACEMENT", "GUIDANCE")):
        return "company_specific", 2
    return "general_news", 1


def split_sentences(text: str | None) -> list[str]:
    cleaned = normalize_html_text(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\s+\|\|\s+|\n+", cleaned)
    sentences: list[str] = []
    for part in parts:
        candidate = part.strip()
        if len(candidate) >= 40:
            sentences.append(candidate)
    return sentences


def extract_market_context_text(
    title: str | None,
    summary_text: str | None,
    body_text: str | None,
    relevant_text: str | None,
) -> tuple[str | None, float]:
    sentences = [*split_sentences(title), *split_sentences(summary_text), *split_sentences(body_text)]
    if not sentences:
        sentences = split_sentences(relevant_text)
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for sentence in sentences:
        normalized = sentence.upper()
        score = sum(1 for term in MARKET_SENTENCE_TERMS if term in normalized)
        if score <= 0 or sentence in seen:
            continue
        seen.add(sentence)
        scored.append((score, sentence))
    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    kept = [sentence for _, sentence in scored[:5]]
    if not kept:
        return None, 0.0
    return " || ".join(kept), float(sum(score for score, _ in scored[:5]))


def relevant_text_expr(title_expr: str, summary_expr: str, body_expr: str) -> str:
    return f"""
        NULLIF(
            trim(
                concat_ws(
                    ' || ',
                    COALESCE({title_expr}, ''),
                    COALESCE({summary_expr}, ''),
                    COALESCE(substr({body_expr}, 1, 4000), ''),
                    COALESCE(all_names, ''),
                    COALESCE(v2_organizations, ''),
                    COALESCE(v2_persons, ''),
                    COALESCE(v2_themes, ''),
                    COALESCE(v2_locations, '')
                )
            ),
            ''
        )
    """


def initialize_schema(con: duckdb.DuckDBPyConnection) -> None:
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
        CREATE TABLE IF NOT EXISTS factor_dictionary (
            factor_id INTEGER,
            factor_label VARCHAR,
            factor_group VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS source_dictionary (
            source_id INTEGER,
            source_domain VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS asset_dictionary (
            asset_id INTEGER,
            asset_label VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS geo_dictionary (
            geo_id INTEGER,
            geo_label VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS silver_event_graph (
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
        CREATE TABLE IF NOT EXISTS silver_factor_mentions (
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
        CREATE TABLE IF NOT EXISTS silver_asset_factor_mentions (
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
        CREATE TABLE IF NOT EXISTS gold_factor_buckets_daily_base (
            bucket_time DATE,
            factor_id INTEGER,
            factor_label VARCHAR,
            geo_id INTEGER,
            geo_label VARCHAR,
            doc_count INTEGER,
            mention_count INTEGER,
            unique_sources INTEGER,
            geo_count INTEGER,
            tone_mean DOUBLE,
            avg_abs_tone DOUBLE,
            novelty_mean DOUBLE,
            negative_tail_count INTEGER,
            positive_tail_count INTEGER,
            confidence_mean DOUBLE,
            first_seen TIMESTAMP,
            last_seen TIMESTAMP,
            source_dispersion DOUBLE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS gold_factor_buckets_daily (
            bucket_time DATE,
            factor_id INTEGER,
            factor_label VARCHAR,
            geo_id INTEGER,
            geo_label VARCHAR,
            doc_count INTEGER,
            mention_count INTEGER,
            unique_sources INTEGER,
            geo_count INTEGER,
            tone_mean DOUBLE,
            tone_zscore_30d DOUBLE,
            avg_abs_tone DOUBLE,
            novelty_mean DOUBLE,
            negative_tail_count INTEGER,
            positive_tail_count INTEGER,
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
        CREATE TABLE IF NOT EXISTS gold_asset_factor_panel_daily_base (
            bucket_time DATE,
            factor_id INTEGER,
            factor_label VARCHAR,
            asset_id INTEGER,
            asset_label VARCHAR,
            geo_id INTEGER,
            geo_label VARCHAR,
            doc_count INTEGER,
            mention_count INTEGER,
            unique_sources INTEGER,
            geo_count INTEGER,
            tone_mean DOUBLE,
            avg_abs_tone DOUBLE,
            novelty_mean DOUBLE,
            confidence DOUBLE,
            asset_factor_relevance_mean DOUBLE,
            source_dispersion DOUBLE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS gold_asset_factor_panel_daily (
            bucket_time DATE,
            asset_id INTEGER,
            asset_label VARCHAR,
            factor_id INTEGER,
            factor_label VARCHAR,
            geo_id INTEGER,
            geo_label VARCHAR,
            doc_count INTEGER,
            mention_count INTEGER,
            unique_sources INTEGER,
            geo_count INTEGER,
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
            factor_id INTEGER,
            factor_label VARCHAR,
            geo_id INTEGER,
            geo_label VARCHAR,
            prior_doc_count INTEGER,
            doc_count INTEGER,
            prior_narrative_score DOUBLE,
            narrative_score DOUBLE,
            doc_count_delta INTEGER,
            narrative_score_delta DOUBLE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS gold_asset_factor_crossover_links_daily (
            prior_bucket_time DATE,
            bucket_time DATE,
            asset_id INTEGER,
            asset_label VARCHAR,
            factor_id INTEGER,
            factor_label VARCHAR,
            geo_id INTEGER,
            geo_label VARCHAR,
            prior_doc_count INTEGER,
            doc_count INTEGER,
            prior_narrative_score DOUBLE,
            narrative_score DOUBLE,
            doc_count_delta INTEGER,
            narrative_score_delta DOUBLE
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
            "INSERT INTO bronze_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            "INSERT INTO silver_asset_factor_mentions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            asset_factor_mentions,
        )
        asset_factor_mentions.clear()


def asset_factor_relevance(text: str | None, asset_label: str | None, factor_label: str | None) -> float:
    asset_hits = match_count(text, asset_cues(asset_label))
    factor_hits = match_count(text, factor_cues(factor_label))
    if asset_hits == 0 and factor_hits == 0:
        return 0.0
    return float((asset_hits * 2) + factor_hits)


def normalize_bronze_text_fields(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        """
        SELECT doc_id, title, summary_text, gkg_extras
        FROM bronze_candidates
        WHERE
            position('&' IN COALESCE(title, '')) > 0
            OR position('&' IN COALESCE(summary_text, '')) > 0
            OR position('&' IN COALESCE(gkg_extras, '')) > 0
        """
    ).fetchall()
    if not rows:
        return
    updates = [
        (
            normalize_html_text(title),
            normalize_html_text(summary_text),
            normalize_html_text(gkg_extras),
            doc_id,
        )
        for doc_id, title, summary_text, gkg_extras in rows
    ]
    con.executemany(
        """
        UPDATE bronze_candidates
        SET
            title = ?,
            summary_text = ?,
            gkg_extras = ?
        WHERE doc_id = ?
        """,
        updates,
    )


def update_bronze_market_context(con: duckdb.DuckDBPyConnection) -> None:
    title_market_wrap_predicate = sql_contains_any(
        "title",
        (
            "ETF",
            "YIELDS",
            "TREASURY",
            "DOLLAR",
            "NASDAQ",
            "MARKETS",
        ),
    )
    title_commodity_predicate = sql_contains_any(
        "title",
        ("CRUDE", "OIL", "GOLD", "COPPER", "BULLION", "MINING"),
    )
    title_company_predicate = sql_contains_any(
        "title",
        (
            "FINANCIAL COMPARISON",
            " VS. ",
            " VS ",
            "COMPARE",
            "EARNINGS PREVIEW",
            "QUARTER ENDED",
            "EBIT MARGIN",
            "(NASDAQ:",
            "(NYSE:",
            "INC.",
            " INC ",
            "CORP",
            "EARNINGS",
            "PLACEMENT",
            "GUIDANCE",
        ),
    )
    market_relevance_predicate = sql_contains_any(
        "relevant_text",
        MARKET_SENTENCE_TERMS,
    )
    con.execute(
        f"""
        UPDATE bronze_candidates
        SET
            source_type = CASE
                WHEN source_domain IN ({", ".join(sql_string_literal(value) for value in sorted(MARKET_WRAP_DOMAINS))})
                    THEN 'market_wrap'
                WHEN source_domain IN ({", ".join(sql_string_literal(value) for value in sorted(COMMODITY_SPECIALIST_DOMAINS))})
                    THEN 'commodity_specialist'
                WHEN {title_company_predicate}
                    OR position('/markets/stocks/' IN lower(COALESCE(document_identifier, ''))) > 0
                    THEN 'company_specific'
                WHEN {title_market_wrap_predicate}
                    THEN 'market_wrap'
                WHEN {title_commodity_predicate}
                    THEN 'commodity_specialist'
                ELSE 'general_news'
            END,
            source_priority = CASE
                WHEN source_domain IN ({", ".join(sql_string_literal(value) for value in sorted(MARKET_WRAP_DOMAINS))})
                    THEN 5
                WHEN source_domain IN ({", ".join(sql_string_literal(value) for value in sorted(COMMODITY_SPECIALIST_DOMAINS))})
                    THEN 4
                WHEN {title_company_predicate}
                    OR position('/markets/stocks/' IN lower(COALESCE(document_identifier, ''))) > 0
                    THEN 2
                WHEN {title_market_wrap_predicate}
                    THEN 4
                WHEN {title_commodity_predicate}
                    THEN 3
                ELSE 1
            END
        """
    )
    cursor = con.execute(
        f"""
        SELECT
            doc_id,
            title,
            summary_text,
            CASE
                WHEN length(COALESCE(body_text, '')) > 2000 THEN substr(body_text, 1, 2000)
                ELSE body_text
            END AS body_text,
            CASE
                WHEN length(COALESCE(relevant_text, '')) > 4000 THEN substr(relevant_text, 1, 4000)
                ELSE relevant_text
            END AS relevant_text
        FROM bronze_candidates
        WHERE source_priority >= 3
          AND (
              {market_relevance_predicate}
              OR source_type IN ('market_wrap', 'commodity_specialist')
          )
        """
    )
    updates = []
    while True:
        rows = cursor.fetchmany(MARKET_CONTEXT_BATCH_SIZE)
        if not rows:
            break
        for doc_id, title, summary_text, body_text, relevant_text in rows:
            market_context_text, market_context_score = extract_market_context_text(
                title,
                summary_text,
                body_text,
                relevant_text,
            )
            updates.append((market_context_text, market_context_score, doc_id))
    if not updates:
        return
    con.executemany(
        """
        UPDATE bronze_candidates
        SET
            market_context_text = ?,
            market_context_score = ?
        WHERE doc_id = ?
        """,
        updates,
    )


def update_asset_factor_relevance(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        """
        SELECT
            m.doc_id,
            b.relevant_text,
            m.asset_label,
            m.factor_label
        FROM silver_asset_factor_mentions m
        JOIN bronze_candidates b USING (doc_id)
        GROUP BY 1, 2, 3, 4
        """
    ).fetchall()
    if not rows:
        return
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE asset_factor_relevance_updates (
            doc_id UBIGINT,
            asset_label VARCHAR,
            factor_label VARCHAR,
            asset_factor_relevance DOUBLE
        )
        """
    )
    updates = [
        (
            doc_id,
            asset_label,
            factor_label,
            asset_factor_relevance(relevant_text, asset_label, factor_label),
        )
        for doc_id, relevant_text, asset_label, factor_label in rows
    ]
    con.executemany(
        """
        INSERT INTO asset_factor_relevance_updates VALUES (?, ?, ?, ?)
        """,
        updates,
    )
    con.execute(
        """
        UPDATE silver_asset_factor_mentions AS m
        SET asset_factor_relevance = u.asset_factor_relevance
        FROM asset_factor_relevance_updates AS u
        WHERE m.doc_id = u.doc_id
          AND m.asset_label = u.asset_label
          AND m.factor_label = u.factor_label
        """
    )


def sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def partition_dates_in_input(con: duckdb.DuckDBPyConnection, input_glob: str) -> list[str]:
    rows = con.execute(
        f"""
        SELECT DISTINCT CAST(try_cast(partition_date AS DATE) AS VARCHAR) AS partition_date
        FROM read_parquet({json.dumps(input_glob)}, union_by_name=true)
        WHERE try_cast(partition_date AS DATE) IS NOT NULL
        ORDER BY 1
        """
    ).fetchall()
    return [str(row[0]) for row in rows if row[0]]


def processed_partition_dates(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute("SELECT CAST(partition_date AS VARCHAR) FROM graph_build_partitions").fetchall()
    return {str(row[0]) for row in rows if row[0]}


def append_dictionary_values(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    id_column: str,
    value_column: str,
    select_sql: str,
) -> None:
    con.execute(
        f"""
        INSERT INTO {table_name}
        SELECT
            COALESCE((SELECT MAX({id_column}) FROM {table_name}), 0)
                + row_number() OVER (ORDER BY candidate_value)::INTEGER AS {id_column},
            candidate_value AS {value_column}
        FROM (
            SELECT DISTINCT candidate_value
            FROM ({select_sql}) AS candidate_values
            WHERE candidate_value IS NOT NULL
              AND candidate_value NOT IN (SELECT {value_column} FROM {table_name})
        )
        """
    )


def materialize_daily_rollups(con: duckdb.DuckDBPyConnection, processed_dates: list[str]) -> None:
    if not processed_dates:
        return
    date_list = ", ".join(sql_string_literal(value) for value in processed_dates)
    con.execute(
        f"""
        DELETE FROM gold_factor_buckets_daily_base
        WHERE bucket_time IN ({date_list})
        """
    )
    con.execute(
        f"""
        INSERT INTO gold_factor_buckets_daily_base
        SELECT
            bucket_time,
            factor_id,
            factor_label,
            geo_id,
            geo_label,
            COUNT(DISTINCT doc_id)::INTEGER AS doc_count,
            COUNT(*)::INTEGER AS mention_count,
            COUNT(DISTINCT source_id)::INTEGER AS unique_sources,
            COUNT(DISTINCT geo_id)::INTEGER AS geo_count,
            AVG(tone) AS tone_mean,
            AVG(abs(COALESCE(tone, 0.0))) AS avg_abs_tone,
            AVG(novelty) AS novelty_mean,
            COUNT(DISTINCT CASE WHEN tone <= -5 THEN doc_id ELSE NULL END)::INTEGER AS negative_tail_count,
            COUNT(DISTINCT CASE WHEN tone >= 5 THEN doc_id ELSE NULL END)::INTEGER AS positive_tail_count,
            AVG(classification_confidence) AS confidence_mean,
            MIN(event_time) AS first_seen,
            MAX(event_time) AS last_seen,
            CASE
                WHEN COUNT(DISTINCT doc_id) = 0 THEN NULL
                ELSE CAST(COUNT(DISTINCT source_id) AS DOUBLE) / CAST(COUNT(DISTINCT doc_id) AS DOUBLE)
            END AS source_dispersion
        FROM silver_factor_mentions
        WHERE bucket_time IN ({date_list})
        GROUP BY 1, 2, 3, 4, 5
        """
    )
    con.execute(
        f"""
        DELETE FROM gold_asset_factor_panel_daily_base
        WHERE bucket_time IN ({date_list})
        """
    )
    con.execute(
        f"""
        INSERT INTO gold_asset_factor_panel_daily_base
        SELECT
            bucket_time,
            factor_id,
            factor_label,
            asset_id,
            asset_label,
            geo_id,
            geo_label,
            COUNT(DISTINCT doc_id)::INTEGER AS doc_count,
            COUNT(*)::INTEGER AS mention_count,
            COUNT(DISTINCT source_id)::INTEGER AS unique_sources,
            COUNT(DISTINCT geo_id)::INTEGER AS geo_count,
            AVG(tone) AS tone_mean,
            AVG(abs(COALESCE(tone, 0.0))) AS avg_abs_tone,
            AVG(novelty) AS novelty_mean,
            AVG(classification_confidence) AS confidence,
            AVG(asset_factor_relevance) AS asset_factor_relevance_mean,
            CASE
                WHEN COUNT(DISTINCT doc_id) = 0 THEN NULL
                ELSE CAST(COUNT(DISTINCT source_id) AS DOUBLE) / CAST(COUNT(DISTINCT doc_id) AS DOUBLE)
            END AS source_dispersion
        FROM silver_asset_factor_mentions
        WHERE bucket_time IN ({date_list})
        GROUP BY 1, 2, 3, 4, 5, 6, 7
        """
    )
    con.execute(
        f"""
        DELETE FROM gold_factor_buckets_daily
        WHERE bucket_time IN ({date_list})
        """
    )
    con.execute(
        f"""
        INSERT INTO gold_factor_buckets_daily
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
            doc_count,
            mention_count,
            unique_sources,
            geo_count,
            tone_mean,
            CASE
                WHEN tone_mean_30d_std IS NULL OR tone_mean_30d_std = 0 THEN NULL
                ELSE (tone_mean - tone_mean_30d_avg) / tone_mean_30d_std
            END AS tone_zscore_30d,
            avg_abs_tone,
            novelty_mean,
            negative_tail_count,
            positive_tail_count,
            source_dispersion,
            confidence_mean,
            first_seen,
            last_seen,
            CAST(doc_count AS DOUBLE)
                * (0.5 + COALESCE(source_dispersion, 0.0))
                * (1.0 + (COALESCE(avg_abs_tone, 0.0) / 5.0)) AS narrative_score
        FROM roll
        WHERE bucket_time IN ({date_list})
        """
    )
    con.execute(
        f"""
        DELETE FROM gold_asset_factor_panel_daily
        WHERE bucket_time IN ({date_list})
        """
    )
    con.execute(
        f"""
        INSERT INTO gold_asset_factor_panel_daily
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
            doc_count,
            mention_count,
            unique_sources,
            geo_count,
            tone_mean,
            CASE
                WHEN tone_mean_30d_std IS NULL OR tone_mean_30d_std = 0 THEN NULL
                ELSE (tone_mean - tone_mean_30d_avg) / tone_mean_30d_std
            END AS tone_zscore_30d,
            avg_abs_tone,
            novelty_mean,
            CAST(doc_count AS DOUBLE)
                * COALESCE(source_dispersion, 0.0)
                * (0.5 + COALESCE(asset_factor_relevance_mean, 0.0) / 8.0) AS event_intensity,
            source_dispersion,
            confidence,
            CAST(doc_count AS DOUBLE)
                * (0.5 + COALESCE(source_dispersion, 0.0))
                * (0.5 + COALESCE(asset_factor_relevance_mean, 0.0) / 8.0)
                * (1.0 + (COALESCE(avg_abs_tone, 0.0) / 5.0)) AS narrative_score
        FROM roll
        WHERE bucket_time IN ({date_list})
        """
    )


def materialize_crossover_links(con: duckdb.DuckDBPyConnection, processed_dates: list[str]) -> None:
    if not processed_dates:
        return
    affected_dates = sorted(
        {
            row[0]
            for row in con.execute(
                f"""
                WITH raw_dates AS (
                    SELECT CAST(bucket_time AS DATE) AS bucket_time
                    FROM (VALUES {", ".join(f"({sql_string_literal(value)})" for value in processed_dates)}) AS v(bucket_time)
                )
                SELECT CAST(bucket_time AS VARCHAR)
                FROM raw_dates
                UNION
                SELECT CAST((CAST(bucket_time AS DATE) + INTERVAL 1 DAY) AS VARCHAR)
                FROM raw_dates
                """
            ).fetchall()
            if row[0]
        }
    )
    if not affected_dates:
        return
    affected_list = ", ".join(sql_string_literal(value) for value in affected_dates)
    con.execute(
        f"""
        DELETE FROM gold_factor_crossover_links_daily
        WHERE bucket_time IN ({affected_list})
        """
    )
    con.execute(
        f"""
        INSERT INTO gold_factor_crossover_links_daily
        SELECT
            prev.bucket_time AS prior_bucket_time,
            curr.bucket_time,
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
        JOIN gold_factor_buckets_daily prev
          ON prev.factor_id = curr.factor_id
         AND prev.geo_id = curr.geo_id
         AND prev.bucket_time = curr.bucket_time - INTERVAL 1 DAY
        WHERE curr.bucket_time IN ({affected_list})
        """
    )
    con.execute(
        f"""
        DELETE FROM gold_asset_factor_crossover_links_daily
        WHERE bucket_time IN ({affected_list})
        """
    )
    con.execute(
        f"""
        INSERT INTO gold_asset_factor_crossover_links_daily
        SELECT
            prev.bucket_time AS prior_bucket_time,
            curr.bucket_time,
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
        JOIN gold_asset_factor_panel_daily prev
          ON prev.asset_id = curr.asset_id
         AND prev.factor_id = curr.factor_id
         AND prev.geo_id = curr.geo_id
         AND prev.bucket_time = curr.bucket_time - INTERVAL 1 DAY
        WHERE curr.bucket_time IN ({affected_list})
        """
    )


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
    input_dates = partition_dates_in_input(con, input_glob)
    if not input_dates:
        con.close()
        return
    already_processed = processed_partition_dates(con)
    dates_to_process = [value for value in input_dates if value not in already_processed]
    if not dates_to_process:
        con.close()
        return
    date_filter = ", ".join(sql_string_literal(value) for value in dates_to_process)
    columns = input_columns(con, input_glob)
    raw_title_expr = optional_text_expr(first_present(columns, TEXT_TITLE_CANDIDATES))
    summary_expr = optional_text_expr(first_present(columns, TEXT_SUMMARY_CANDIDATES))
    body_expr = optional_text_expr(first_present(columns, TEXT_BODY_CANDIDATES))
    metadata_expr = optional_expr_for(columns, METADATA_CANDIDATES)
    gkg_extras_expr = optional_expr_for(columns, GKG_EXTRAS_CANDIDATES)
    page_title_expr = f"""
        NULLIF(
            trim(
                regexp_replace(
                    replace(
                        replace(
                            replace(
                                regexp_extract(COALESCE({gkg_extras_expr}, ''), '(?is)<PAGE_TITLE>(.*?)</PAGE_TITLE>', 1),
                                '&amp;',
                                '&'
                            ),
                            '&#xA0;',
                            ' '
                        ),
                        '&#x2026;',
                        '...'
                    ),
                    '\\s+',
                    ' ',
                    'g'
                )
            ),
            ''
        )
    """
    title_expr = f"COALESCE({raw_title_expr}, {page_title_expr})"
    relevant_expr = relevant_text_expr(title_expr, summary_expr, body_expr)
    sharing_image_expr = optional_expr_for(columns, SHARING_IMAGE_CANDIDATES)
    related_images_expr = optional_expr_for(columns, RELATED_IMAGES_CANDIDATES)
    social_image_embeds_expr = optional_expr_for(columns, SOCIAL_IMAGE_EMBEDS_CANDIDATES)
    social_video_embeds_expr = optional_expr_for(columns, SOCIAL_VIDEO_EMBEDS_CANDIDATES)
    quotations_expr = optional_expr_for(columns, QUOTATIONS_CANDIDATES)
    amounts_expr = optional_expr_for(columns, AMOUNTS_CANDIDATES)
    dates_expr = optional_expr_for(columns, DATES_CANDIDATES)
    gcam_expr = optional_expr_for(columns, GCAM_CANDIDATES)
    translation_info_expr = optional_expr_for(columns, TRANSLATION_INFO_CANDIDATES)
    if con.execute("SELECT COUNT(*) FROM factor_dictionary").fetchone()[0] == 0:
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
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE asset_rule_patterns (
            asset_label VARCHAR,
            pattern VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE asset_context_required (
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
    con.executemany(
        "INSERT INTO asset_rule_patterns VALUES (?, ?)",
        [
            (asset_label, pattern.upper())
            for asset_label, patterns in ASSET_TEXT_PATTERNS.items()
            for pattern in [*patterns, asset_label]
        ],
    )
    con.executemany(
        "INSERT INTO asset_context_required VALUES (?)",
        [(asset_label,) for asset_label in sorted(ASSET_CONTEXT_REQUIRED)],
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE bronze_stage AS
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
            {title_expr} AS title,
            {summary_expr} AS summary_text,
            {body_expr} AS body_text,
            {relevant_expr} AS relevant_text,
            {metadata_expr} AS metadata_json,
            {gkg_extras_expr} AS gkg_extras,
            {sharing_image_expr} AS sharing_image,
            {related_images_expr} AS related_images,
            {social_image_embeds_expr} AS social_image_embeds,
            {social_video_embeds_expr} AS social_video_embeds,
            {quotations_expr} AS quotations,
            {amounts_expr} AS amounts,
            {dates_expr} AS dates,
            {gcam_expr} AS gcam,
            {translation_info_expr} AS translation_info,
            CAST(NULL AS VARCHAR) AS source_type,
            CAST(NULL AS INTEGER) AS source_priority,
            CAST(NULL AS VARCHAR) AS market_context_text,
            CAST(NULL AS DOUBLE) AS market_context_score,
            try_cast(regexp_extract(COALESCE(v2_tone, ''), '^\\s*(-?[0-9]+(?:\\.[0-9]+)?)', 1) AS DOUBLE) AS tone
        FROM read_parquet({json.dumps(input_glob)}, union_by_name=true)
        WHERE try_cast(partition_date AS DATE) IN ({date_filter})
        """
    )
    con.execute(
        """
        INSERT INTO bronze_candidates
        SELECT s.*
        FROM bronze_stage s
        LEFT JOIN bronze_candidates b USING (doc_id)
        WHERE b.doc_id IS NULL
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE stage_doc_ids AS
        SELECT doc_id
        FROM bronze_stage
        """
    )
    normalize_bronze_text_fields(con)
    update_bronze_market_context(con)
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
            ) AS match_text,
            upper(
                concat_ws(
                    ' | ',
                    COALESCE(title, ''),
                    COALESCE(summary_text, ''),
                    COALESCE(body_text, ''),
                    COALESCE(relevant_text, ''),
                    COALESCE(v2_themes, ''),
                    COALESCE(v2_persons, ''),
                    COALESCE(v2_organizations, ''),
                    COALESCE(all_names, ''),
                    COALESCE(v2_locations, ''),
                    COALESCE(gkg_extras, ''),
                    COALESCE(document_identifier, '')
                )
            ) AS asset_match_text
        FROM bronze_candidates
        WHERE doc_id IN (SELECT doc_id FROM stage_doc_ids)
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
        LEFT JOIN asset_context_required r
          ON r.asset_label = a.asset_label
        WHERE r.asset_label IS NULL

        UNION

        SELECT DISTINCT
            m.doc_id,
            m.factor_id,
            m.factor_label,
            a.asset_label
        FROM matched_factors m
        JOIN factor_rule_assets a
          ON a.factor_id = m.factor_id
        JOIN asset_context_required r
          ON r.asset_label = a.asset_label
        JOIN bronze_enriched b
          ON b.doc_id = m.doc_id
        JOIN asset_rule_patterns p
          ON p.asset_label = a.asset_label
         AND contains(b.asset_match_text, p.pattern)
        """
    )
    append_dictionary_values(
        con,
        "source_dictionary",
        "source_id",
        "source_domain",
        """
        SELECT DISTINCT b.source_domain AS candidate_value
        FROM bronze_enriched b
        JOIN matched_factors m USING (doc_id)
        """,
    )
    append_dictionary_values(
        con,
        "geo_dictionary",
        "geo_id",
        "geo_label",
        """
        SELECT DISTINCT unnest(b.geo_labels) AS candidate_value
        FROM bronze_enriched b
        JOIN matched_factors m USING (doc_id)
        """,
    )
    append_dictionary_values(
        con,
        "asset_dictionary",
        "asset_id",
        "asset_label",
        """
        SELECT DISTINCT asset_label AS candidate_value
        FROM matched_assets
        """,
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
        WHERE b.doc_id NOT IN (SELECT doc_id FROM silver_event_graph)
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
        WHERE NOT EXISTS (
            SELECT 1
            FROM silver_factor_mentions existing
            WHERE existing.doc_id = b.doc_id
              AND existing.factor_id = m.factor_id
              AND existing.geo_id = d.geo_id
        )
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
            least(0.95, 0.55 + (0.08 * f.factor_count)) AS classification_confidence,
            0.0 AS asset_factor_relevance
        FROM bronze_enriched b
        JOIN factor_rollup f USING (doc_id)
        JOIN matched_assets m USING (doc_id)
        JOIN asset_dictionary a USING (asset_label)
        JOIN source_dictionary s USING (source_domain)
        CROSS JOIN UNNEST(b.geo_labels) AS g(geo_label)
        JOIN geo_dictionary d USING (geo_label)
        WHERE NOT EXISTS (
            SELECT 1
            FROM silver_asset_factor_mentions existing
            WHERE existing.doc_id = b.doc_id
              AND existing.factor_id = m.factor_id
              AND existing.asset_id = a.asset_id
              AND existing.geo_id = d.geo_id
        )
        """
    )
    con.execute(
        """
        INSERT INTO silver_market_context_mentions
        SELECT DISTINCT
            CAST(b.event_time AS DATE) AS bucket_time,
            b.event_time,
            b.doc_id,
            CAST(hash(b.source_domain || '|' || b.document_identifier) AS UBIGINT) AS cluster_id,
            m.factor_label,
            a.asset_label,
            b.source_domain,
            b.source_type,
            b.source_priority,
            b.market_context_text,
            b.market_context_score,
            least(0.95, 0.55 + (0.08 * f.factor_count)) AS classification_confidence
        FROM bronze_enriched b
        JOIN factor_rollup f USING (doc_id)
        JOIN matched_assets m USING (doc_id)
        JOIN asset_dictionary a USING (asset_label)
        WHERE b.market_context_text IS NOT NULL
          AND NOT EXISTS (
              SELECT 1
              FROM silver_market_context_mentions existing
              WHERE existing.doc_id = b.doc_id
                AND existing.factor_label = m.factor_label
                AND existing.asset_label = a.asset_label
          )
        """
    )
    update_asset_factor_relevance(con)
    materialize_daily_rollups(con, dates_to_process)
    materialize_crossover_links(con, dates_to_process)
    con.executemany(
        "INSERT INTO graph_build_partitions VALUES (?, ?, current_timestamp::TIMESTAMP)",
        [(value, input_glob) for value in dates_to_process],
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
