#!/usr/bin/env python3
"""Minimal stdio MCP wrapper for the standalone narrative explainer."""

from __future__ import annotations

import json
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from narrative_text_matching import asset_cues, factor_cues, match_count, matched_cues
from parquet_narrative_store import resolve_query_db
from query_narrative_graph import query_explain_move, query_supporting_docs, run_query


SERVER_NAME = "news-narrative-explainer"
SERVER_VERSION = "0.1.0"
DEFAULT_DB = str(Path(__file__).resolve().parents[1] / "data" / "narrative_graph_parquet")
IMPACT_RULES_PATH = Path(__file__).resolve().parents[1] / "config" / "asset_impact_mechanisms.json"
IMPACT_RULES: dict[str, dict[str, dict[str, Any]]] = json.loads(IMPACT_RULES_PATH.read_text())
STRICT_GENERIC_FACTORS = {
    "civil_unrest",
    "crypto_regulation",
    "elections_policy",
    "growth_activity",
    "labour_market",
}
MACRO_ASSETS = {"US2Y", "US10Y", "DXY", "NDX", "SPX"}
INDEX_ASSETS = {"NDX", "SPX"}
FACTOR_CANDIDATE_LIMIT = 10
FACTOR_REGIME_LABELS = {
    "banking_stress": "financial-stress repricing",
    "central_bank_policy": "Fed/policy repricing",
    "copper": "copper-specific supply-demand repricing",
    "crypto_regulation": "crypto policy repricing",
    "earnings": "earnings/valuation repricing",
    "fiscal_debt": "duration/fiscal-supply repricing",
    "gold_precious": "precious-metals positioning unwind",
    "growth_activity": "growth-sensitive de-rating",
    "inflation": "inflation-relief repricing",
    "interest_rates": "rate-path repricing",
    "labour_market": "labor/Fed repricing",
    "oil": "oil/geopolitical premium repricing",
    "sanctions_trade": "trade-and-sanctions repricing",
    "shipping_disruption": "supply-chain chokepoint repricing",
    "war_conflict": "geopolitical-risk premium repricing",
}
SPECIFIC_NARRATIVE_RULES = {
    "central_bank_policy": [
        (["HAWKISH", "FED"], "hawkish Fed repricing"),
        (["FED PATH"], "Fed-path repricing"),
        (["FED", "POLICY"], "Fed policy repricing"),
        (["ECB", "POLICY"], "ECB policy repricing"),
    ],
    "interest_rates": [
        (["FRONT-END", "RATE"], "front-end rate repricing"),
        (["2-YEAR", "YIELD"], "front-end Treasury repricing"),
        (["10-YEAR", "YIELD"], "long-end Treasury repricing"),
        (["RATE SENSITIVITY"], "rate-sensitivity repricing"),
    ],
    "growth_activity": [
        (["TECH SELLOFF"], "tech-led growth scare"),
        (["SOFT LANDING"], "soft-landing repricing"),
        (["PMI"], "PMI/growth repricing"),
        (["MANUFACTURING"], "manufacturing-growth repricing"),
        (["GROWTH"], "growth-sensitive repricing"),
    ],
    "elections_policy": [
        (["ELECTION", "UNCERTAINTY"], "election uncertainty"),
        (["ELECTION"], "election-policy noise"),
        (["TARIFF"], "tariff-policy repricing"),
    ],
    "labour_market": [
        (["PAYROLL"], "payrolls/Fed repricing"),
        (["JOBS"], "jobs/Fed repricing"),
        (["WAGES"], "wage/Fed repricing"),
        (["LABOR"], "labor/Fed repricing"),
    ],
    "gold_precious": [
        (["DOLLAR", "FED"], "dollar/Fed pressure on gold"),
        (["REAL-RATE"], "real-rate pressure on gold"),
        (["BULLION", "DOLLAR"], "dollar pressure on bullion"),
    ],
    "oil": [
        (["RED SEA"], "Red Sea oil-risk repricing"),
        (["HORMUZ"], "Hormuz supply-risk repricing"),
        (["MIDDLE EAST"], "Middle East oil-risk repricing"),
        (["OIL SLUMP"], "oil-led disinflation repricing"),
    ],
    "shipping_disruption": [
        (["RED SEA"], "Red Sea shipping disruption"),
        (["TANKER"], "tanker-route disruption"),
        (["SHIPPING"], "shipping-route disruption"),
    ],
    "war_conflict": [
        (["MIDDLE EAST"], "Middle East geopolitical repricing"),
        (["RED SEA"], "Red Sea conflict repricing"),
        (["SAFE HAVEN"], "safe-haven risk repricing"),
    ],
}
ASSET_OVERLAY_LABELS = {
    "BTC": "Hedge overlay",
    "DXY": "Macro overlay",
    "Gold": "Hedge overlay",
    "HG": "Asset overlay",
    "NDX": "Equity overlay",
    "SPX": "Equity overlay",
    "US10Y": "Macro overlay",
    "US2Y": "Macro overlay",
    "WTI": "Asset overlay",
}
ASSET_FACTOR_SELECTION_BONUS = {
    "US2Y": {
        "central_bank_policy": 18.0,
        "inflation": 16.0,
        "interest_rates": 14.0,
        "labour_market": 8.0,
        "banking_stress": -6.0,
    },
    "US10Y": {
        "fiscal_debt": 14.0,
        "central_bank_policy": 14.0,
        "inflation": 16.0,
        "interest_rates": 14.0,
        "labour_market": 4.0,
    },
    "DXY": {
        "central_bank_policy": 26.0,
        "interest_rates": 18.0,
        "inflation": 10.0,
        "war_conflict": 6.0,
        "sanctions_trade": 4.0,
        "elections_policy": -20.0,
        "civil_unrest": -32.0,
        "labour_market": -8.0,
    },
    "NDX": {
        "earnings": 18.0,
        "interest_rates": 16.0,
        "central_bank_policy": 14.0,
        "growth_activity": 8.0,
        "mergers_acquisitions": 4.0,
    },
    "SPX": {
        "central_bank_policy": 20.0,
        "interest_rates": 16.0,
        "growth_activity": 12.0,
        "labour_market": 10.0,
        "earnings": 6.0,
        "elections_policy": -16.0,
        "restructuring_fraud": -18.0,
        "mergers_acquisitions": -10.0,
    },
}
ASSET_DISAMBIGUATION_TERMS = {
    "Gold": ["BULLION", "PRICE", "ETF", "OUNCE", "METAL", "XAU", "SAFE HAVEN", "DOLLAR", "RATES"],
    "NDX": ["NASDAQ", "NASDAQ 100", "NASDAQ-100", "NDX", "QQQ", "WALL STREET", "TECH STOCKS", "TECH SELLOFF", "CHIP STOCKS", "MEGA CAP", "SEMIS", "SEMICONDUCTOR", "INDEX", "STOCKS SLUMP"],
    "SPX": ["S&P 500", "SP 500", "SPX", "WALL STREET", "INDEX", "US STOCKS", "U.S. STOCKS", "EQUITIES", "STOCKS SLUMP", "STOCKS FALL"],
    "DXY": ["DOLLAR INDEX", "GREENBACK", "FX", "CURRENCY", "FED", "USD"],
}
ASSET_EXCLUSION_TERMS = {
    "Gold": ["INC", "ANNOUNCES", "PLACEMENT", "MINE", "MINING", "EXPLORATION"],
    "NDX": ["FINANCIAL COMPARISON", "QUARTER ENDED", "EBIT MARGIN", "REVENUE TRENDS", "SHARES OF"],
    "SPX": ["S&P GLOBAL", "S AND P GLOBAL", "SHARES OF"],
    "DXY": ["DOLLAR GENERAL", "DOLLAR TREE"],
}
ASSET_EXCLUSION_REGEX = {
    "NDX": [r"\bNASDAQ:[A-Z0-9.]+", r"\bNYSE:[A-Z0-9.]+", r"\([A-Z]+:[A-Z0-9.]+\)"],
    "SPX": [r"\bNASDAQ:[A-Z0-9.]+", r"\bNYSE:[A-Z0-9.]+", r"\([A-Z]+:[A-Z0-9.]+\)"],
}
PROXY_CHANNEL_RULES = {
    "US2Y": {
        "inflation": ["INFLATION", "CPI", "PCE", "OIL", "ENERGY", "PRICE PRESSURE"],
        "central_bank_policy": ["FED", "FOMC", "RATE", "POLICY", "CUT", "HIKE"],
        "interest_rates": ["YIELD", "RATE", "TREASURY", "FRONT END", "POLICY"],
        "labour_market": ["JOBS", "PAYROLLS", "UNEMPLOYMENT", "WAGES", "LABOR"],
        "growth_activity": ["PMI", "ACTIVITY", "MANUFACTURING", "GROWTH"],
    },
    "US10Y": {
        "inflation": ["INFLATION", "BREAKEVEN", "OIL", "ENERGY", "PRICE PRESSURE"],
        "central_bank_policy": ["FED", "FOMC", "RATE", "POLICY"],
        "interest_rates": ["YIELD", "TREASURY", "TERM PREMIUM", "RATES"],
        "growth_activity": ["GROWTH", "PMI", "ACTIVITY", "MANUFACTURING"],
        "fiscal_debt": ["AUCTION", "SUPPLY", "DEFICIT", "DEBT", "TREASURY ISSUANCE"],
    },
    "DXY": {
        "central_bank_policy": ["FED", "ECB", "BOJ", "RATE", "POLICY", "DIFFERENTIAL"],
        "interest_rates": ["YIELD", "RATES", "DIFFERENTIAL", "TREASURY"],
        "inflation": ["INFLATION", "PRICE PRESSURE", "PCE", "CPI"],
        "war_conflict": ["SAFE HAVEN", "RISK OFF", "GEOPOLITICAL", "CONFLICT"],
        "growth_activity": ["US EXCEPTIONALISM", "GROWTH", "ACTIVITY", "PMI"],
    },
    "NDX": {
        "interest_rates": ["REAL YIELD", "DISCOUNT RATE", "RATE", "YIELD"],
        "central_bank_policy": ["FED", "RATE", "POLICY", "TIGHTER CONDITIONS"],
        "growth_activity": ["RISK APPETITE", "GROWTH", "ACTIVITY", "SOFT LANDING"],
        "earnings": ["EARNINGS", "GUIDANCE", "MEGA CAP", "AI", "SEMIS"],
        "mergers_acquisitions": ["DEAL", "ACQUISITION", "MERGER", "TECH"],
    },
    "SPX": {
        "interest_rates": ["REAL YIELD", "DISCOUNT RATE", "RATE", "YIELD", "TREASURY"],
        "central_bank_policy": ["FED", "RATE", "POLICY", "TIGHTER CONDITIONS", "MONETARY POLICY"],
        "growth_activity": ["RISK APPETITE", "GROWTH", "ACTIVITY", "SOFT LANDING", "ECONOMY"],
        "earnings": ["EARNINGS", "GUIDANCE", "PROFITS", "MARGINS", "VALUATION"],
        "labour_market": ["JOBS", "PAYROLLS", "WAGES", "LABOR", "UNEMPLOYMENT"],
    },
}
ASSET_PROXY_MARKET_TERMS = {
    "US2Y": ["YIELD", "TREASURY", "RATES", "BOND", "FRONT END", "2-YEAR", "TWO-YEAR", "RATE HIKE", "RATE CUT"],
    "US10Y": ["YIELD", "TREASURY", "RATES", "BOND", "TERM PREMIUM", "10-YEAR", "TEN-YEAR", "AUCTION"],
    "DXY": ["DOLLAR", "USD", "GREENBACK", "CURRENCY", "FX", "EXCHANGE RATE"],
    "NDX": ["NASDAQ", "TECH", "MEGA CAP", "SEMIS", "SEMICONDUCTOR", "STOCKS", "EQUITIES", "AI"],
    "SPX": ["S&P 500", "SPX", "WALL STREET", "US STOCKS", "U.S. STOCKS", "EQUITIES", "INDEX"],
}
TRANSMISSION_TARGETS = {
    "oil": ["WTI", "US2Y", "US10Y", "DXY", "Gold", "NDX"],
    "war_conflict": ["WTI", "Gold", "DXY", "US2Y", "US10Y"],
    "shipping_disruption": ["WTI", "US2Y", "US10Y", "DXY"],
    "inflation": ["US2Y", "US10Y", "DXY", "Gold", "NDX"],
    "central_bank_policy": ["US2Y", "US10Y", "DXY", "Gold", "NDX"],
    "gold_precious": ["Gold", "DXY", "US10Y"],
    "growth_activity": ["NDX", "US10Y", "DXY", "HG"],
    "fiscal_debt": ["US10Y", "DXY"],
    "interest_rates": ["US2Y", "US10Y", "DXY", "Gold", "NDX"],
}
TRANSMISSION_CHAINS = {
    "oil": "oil/geopolitical unwind -> lower oil -> lower inflation pressure -> lower yields; in parallel, stronger dollar / Fed restraint can weigh on gold and limit equity relief",
    "war_conflict": "less conflict premium -> lower oil and weaker safe-haven demand -> lower inflation pressure -> lower yields",
    "shipping_disruption": "fewer shipping bottlenecks -> lower energy inflation pressure -> lower yields",
    "inflation": "softer inflation narrative -> lower yields -> valuation relief unless another overlay blocks risk-on",
    "central_bank_policy": "hawkish Fed repricing / relative policy support -> stronger dollar and pressure on duration and gold",
    "gold_precious": "dollar / real-rate pressure -> weaker precious metals even if bonds rally",
}
CORE_CROSS_ASSET_SET = {"WTI", "Gold", "US2Y", "US10Y", "DXY", "NDX", "SPX"}
SIMILAR_DAY_FACTORS = (
    "central_bank_policy",
    "war_conflict",
    "oil",
    "inflation",
    "interest_rates",
    "shipping_disruption",
    "growth_activity",
    "gold_precious",
    "labour_market",
    "sanctions_trade",
)


def _json_dumps(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def _write_message(payload: dict[str, Any]) -> None:
    body = _json_dumps(payload)
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        key, _, value = line.decode("ascii").partition(":")
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = sys.stdin.buffer.read(length)
    return json.loads(body.decode("utf-8"))


def _tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "explain_move",
            "description": "Explain which news factors were most active for an asset in a date window.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "asset_label": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                    "db": {"type": "string", "default": DEFAULT_DB},
                },
                "required": ["asset_label"],
            },
        },
        {
            "name": "summarize_narrative",
            "description": "Produce a concise deterministic text summary for an asset move from local news factors.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "asset_label": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                    "db": {"type": "string", "default": DEFAULT_DB},
                },
                "required": ["asset_label"],
            },
        },
        {
            "name": "supporting_docs",
            "description": "Return supporting document URLs for an asset and optional factor in a date window.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "asset_label": {"type": "string"},
                    "factor_label": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                    "db": {"type": "string", "default": DEFAULT_DB},
                },
                "required": ["asset_label"],
            },
        },
        {
            "name": "explain_day",
            "description": "Rank the dominant narratives for a day across a universe of assets using direct and indirect evidence.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "universe": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "default": 5},
                    "db": {"type": "string", "default": DEFAULT_DB},
                },
                "required": ["date", "universe"],
            },
        },
        {
            "name": "explain_cross_asset_move",
            "description": "Compare multiple assets on a date and report shared narratives, conflicting narratives, missing evidence, and confidence by asset.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "assets": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "default": 5},
                    "db": {"type": "string", "default": DEFAULT_DB},
                },
                "required": ["date", "assets"],
            },
        },
        {
            "name": "build_narrative_frame",
            "description": "Build a structured narrative frame with human-facing regime, transmission, blocking overlay, weakest asset, unresolved areas, and consistency warnings.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "universe": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "default": 5},
                    "db": {"type": "string", "default": DEFAULT_DB},
                },
                "required": ["date", "universe"],
            },
        },
        {
            "name": "find_contradictory_assets",
            "description": "Find the assets whose narratives most contradict the dominant day narrative and explain the required overlay.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "universe": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "default": 5},
                    "db": {"type": "string", "default": DEFAULT_DB},
                },
                "required": ["date", "universe"],
            },
        },
        {
            "name": "explain_asset_via_day_context",
            "description": "Explain an asset using both direct evidence and indirect day-level macro context.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "asset_label": {"type": "string"},
                    "universe": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "default": 5},
                    "db": {"type": "string", "default": DEFAULT_DB},
                },
                "required": ["date", "asset_label"],
            },
        },
        {
            "name": "query_duckdb",
            "description": "Run a guarded read-only DuckDB query for edge-case inspection when the existing explanation helpers are insufficient.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
                    "db": {"type": "string", "default": DEFAULT_DB},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
                "required": ["sql"],
            },
        },
        {
            "name": "similar_days",
            "description": "Find prior local days with the most similar factor mix to a chosen date.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                    "db": {"type": "string", "default": DEFAULT_DB},
                },
                "required": ["date"],
            },
        },
        {
            "name": "intraday_evolution",
            "description": "Describe how the narrative evolved through the day when intraday event buckets are sufficiently populated.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                    "db": {"type": "string", "default": DEFAULT_DB},
                },
                "required": ["date"],
            },
        },
    ]


def _text_content(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _json_content(payload: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload, indent=2, default=str)}]}


_DISALLOWED_SQL_PATTERNS = [
    r";",
    r"\bATTACH\b",
    r"\bCOPY\b",
    r"\bCREATE\b",
    r"\bDELETE\b",
    r"\bDETACH\b",
    r"\bDROP\b",
    r"\bEXPORT\b",
    r"\bINSERT\b",
    r"\bINSTALL\b",
    r"\bLOAD\b",
    r"\bPRAGMA\b",
    r"\bREPLACE\b",
    r"\bSET\b",
    r"\bUPDATE\b",
    r"\bVACUUM\b",
]


def _normalize_query_sql(sql: str, limit: int) -> str:
    normalized = sql.strip().rstrip(";")
    if not normalized:
        raise ValueError("sql must not be empty")
    upper = normalized.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise ValueError("only SELECT or WITH queries are allowed")
    for pattern in _DISALLOWED_SQL_PATTERNS:
        if re.search(pattern, upper):
            raise ValueError(f"disallowed SQL pattern: {pattern}")
    if re.search(r"\bLIMIT\s+\d+\b", upper):
        return normalized
    return f"{normalized}\nLIMIT {max(1, min(limit, 200))}"


def _clean_text(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned or None


def _clip_text(text: str | None, limit: int = 240) -> str | None:
    cleaned = _clean_text(text)
    if not cleaned:
        return None
    if len(cleaned) <= limit:
        return cleaned
    clipped = cleaned[:limit].rsplit(" ", 1)[0].strip()
    return clipped + "..."


def _doc_evidence_line(doc: dict[str, Any]) -> str:
    title = _clean_text(doc.get("title"))
    evidence = _clip_text(doc.get("evidence_text") or doc.get("relevant_text"), 220)
    parts: list[str] = []
    if title:
        parts.append(title)
    if evidence and evidence != title:
        parts.append(evidence)
    if not parts:
        parts.append(doc["document_identifier"])
    return " | ".join(parts)


def _source_reference(doc: dict[str, Any]) -> str:
    source_domain = _clean_text(doc.get("source_domain")) or "unknown-source"
    document_identifier = _clean_text(doc.get("document_identifier"))
    if document_identifier:
        return f"{source_domain} | {document_identifier}"
    return source_domain


def _reference_key(url: str | None, label: str | None) -> str | None:
    if url:
        return url
    if label:
        return label
    return None


def _ensure_reference(
    reference_map: dict[str, str],
    reference_lines: list[str],
    *,
    url: str | None,
    label: str | None,
) -> str | None:
    key = _reference_key(url, label)
    if not key:
        return None
    ref_id = reference_map.get(key)
    if ref_id:
        return ref_id
    ref_id = f"S{len(reference_map) + 1}"
    reference_map[key] = ref_id
    if url and label:
        reference_lines.append(f"[{ref_id}] {label} | {url}")
    elif url:
        reference_lines.append(f"[{ref_id}] {url}")
    else:
        reference_lines.append(f"[{ref_id}] {label}")
    return ref_id


def _doc_reference_label(doc: dict[str, Any]) -> str | None:
    source_domain = _clean_text(doc.get("source_domain")) or "unknown-source"
    title = _clean_text(doc.get("title"))
    if title:
        return f"{source_domain} | {title}"
    return source_domain


def _doc_reference_id(doc: dict[str, Any], reference_map: dict[str, str], reference_lines: list[str]) -> str | None:
    return _ensure_reference(
        reference_map,
        reference_lines,
        url=_clean_text(doc.get("document_identifier")),
        label=_doc_reference_label(doc),
    )


def _append_reference_block(lines: list[str], reference_lines: list[str]) -> list[str]:
    if not reference_lines:
        return lines
    return [*lines, "References:", *reference_lines]


def _doc_mentions_asset_context(doc: dict[str, Any], asset_label: str) -> bool:
    cues = asset_cues(asset_label)
    title = doc.get("title")
    evidence = doc.get("evidence_text") or doc.get("relevant_text")
    return (match_count(title, cues) + match_count(evidence, cues)) > 0


def _proxy_channel_terms(asset_label: str, factor_label: str) -> list[str]:
    return list(PROXY_CHANNEL_RULES.get(asset_label, {}).get(factor_label, []))


def _proxy_market_terms(asset_label: str) -> list[str]:
    return list(ASSET_PROXY_MARKET_TERMS.get(asset_label, []))


def _asset_context_is_ambiguous(asset_label: str, asset_hits: list[str]) -> bool:
    normalized_hits = set(asset_hits)
    if asset_label == "Gold" and normalized_hits == {"GOLD"}:
        return True
    if asset_label == "DXY" and normalized_hits <= {"DOLLAR", "USD"}:
        return True
    if asset_label == "NDX" and normalized_hits <= {"NASDAQ"}:
        return True
    if asset_label == "SPX" and normalized_hits <= {"SPX"}:
        return True
    return False


def _asset_disambiguation_hits(text: str, asset_label: str) -> list[str]:
    return matched_cues(text, list(ASSET_DISAMBIGUATION_TERMS.get(asset_label, [])))


def _asset_exclusion_hits(text: str, asset_label: str) -> list[str]:
    return matched_cues(text, list(ASSET_EXCLUSION_TERMS.get(asset_label, [])))


def _asset_raw_exclusion_hits(text: str, asset_label: str) -> list[str]:
    if not text:
        return []
    hits: list[str] = []
    for pattern in ASSET_EXCLUSION_REGEX.get(asset_label, []):
        if re.search(pattern, text, re.IGNORECASE):
            hits.append(pattern)
    return hits


def _doc_mentions_proxy_context(doc: dict[str, Any], asset_label: str, factor_label: str) -> bool:
    terms = _proxy_channel_terms(asset_label, factor_label)
    if not terms:
        return False
    text = " ".join(
        filter(
            None,
            [
                _clean_text(doc.get("title")),
                _clean_text(doc.get("summary_text")),
                _clean_text(doc.get("evidence_text")),
                _clean_text(doc.get("relevant_text")),
            ],
        )
    )
    hits = matched_cues(text, terms)
    factor_hits = matched_cues(text, factor_cues(factor_label))
    market_hits = matched_cues(text, _proxy_market_terms(asset_label))
    return bool(hits) and bool(market_hits) and bool(hits or factor_hits)


def _impact_rule(asset_label: str, factor_label: str) -> dict[str, Any] | None:
    return IMPACT_RULES.get(asset_label, {}).get(factor_label)


def _doc_match_text(doc: dict[str, Any]) -> str:
    return " ".join(
        filter(
            None,
            [
                _clean_text(doc.get("title")),
                _clean_text(doc.get("summary_text")),
                _clean_text(doc.get("market_context_text")),
                _clean_text(doc.get("evidence_text")),
                _clean_text(doc.get("relevant_text")),
            ],
        )
    )


def _dxy_policy_overlap_hits(text: str) -> list[str]:
    return matched_cues(
        text,
        _proxy_channel_terms("DXY", "central_bank_policy")
        + _proxy_channel_terms("DXY", "interest_rates"),
    )


def _factor_doc_strength(doc: dict[str, Any], asset_label: str, factor_label: str) -> dict[str, Any]:
    rule = _impact_rule(asset_label, factor_label) or {}
    text = _doc_match_text(doc)
    title = _clean_text(doc.get("title"))
    asset_hits = matched_cues(text, asset_cues(asset_label))
    factor_hits = matched_cues(text, factor_cues(factor_label))
    mechanism_hits = matched_cues(text, list(rule.get("mechanism_terms", [])))
    proxy_hits = matched_cues(text, _proxy_channel_terms(asset_label, factor_label))
    proxy_market_hits = matched_cues(text, _proxy_market_terms(asset_label))
    visible_text = " ".join(
        filter(
            None,
            [
                _clean_text(doc.get("title")),
                _clean_text(doc.get("summary_text")),
                _clean_text(doc.get("evidence_text")),
            ],
        )
    )
    disambiguation_hits = _asset_disambiguation_hits(visible_text, asset_label)
    exclusion_hits = _asset_exclusion_hits(visible_text, asset_label)
    exclusion_hits.extend(_asset_raw_exclusion_hits(visible_text, asset_label))
    title_asset_hits = matched_cues(title, asset_cues(asset_label))
    title_factor_hits = matched_cues(title, factor_cues(factor_label))
    dxy_policy_overlap_hits = _dxy_policy_overlap_hits(text) if asset_label == "DXY" else []
    dxy_conflict_channel_hits = [
        hit for hit in [*mechanism_hits, *proxy_hits]
        if hit in {"SAFE HAVEN", "GEOPOLITICAL", "CONFLICT", "RISK OFF"}
    ]
    dxy_policy_penalty = 0.0
    if asset_label == "DXY" and factor_label == "war_conflict":
        if len(dxy_policy_overlap_hits) >= 2 and len(dxy_conflict_channel_hits) <= 2:
            dxy_policy_penalty = 14.0
        elif len(dxy_policy_overlap_hits) > len(dxy_conflict_channel_hits):
            dxy_policy_penalty = 8.0
    score = (
        (len(asset_hits) * 3.0)
        + (len(factor_hits) * 2.0)
        + (len(mechanism_hits) * 2.0)
        + (len(proxy_hits) * 2.5)
        + (len(proxy_market_hits) * 2.5)
        + (len(disambiguation_hits) * 2.5)
        + (len(title_asset_hits) * 2.0)
        + (len(title_factor_hits) * 2.0)
        + (4.0 if doc.get("source_type") == "market_wrap" else 2.5 if doc.get("source_type") == "commodity_specialist" else 0.0)
        + (float(doc.get("market_context_score") or 0.0) * 0.3)
        + float(doc.get("relevance_score") or 0.0)
        + float(doc.get("classification_confidence") or 0.0)
        - dxy_policy_penalty
    )
    direct_asset_context_ok = (
        bool(asset_hits)
        and not exclusion_hits
        and (not _asset_context_is_ambiguous(asset_label, asset_hits) or bool(disambiguation_hits))
    )
    direct_supported = direct_asset_context_ok and (bool(factor_hits) or len(mechanism_hits) >= 2)
    proxy_supported = (
        not direct_supported
        and bool(proxy_hits)
        and bool(proxy_market_hits)
        and (bool(factor_hits) or len(mechanism_hits) >= 1 or len(proxy_hits) >= 2)
    )
    if asset_label == "DXY" and factor_label == "war_conflict":
        conflict_channel_ok = len(dxy_conflict_channel_hits) >= 2
        policy_dominant = len(dxy_policy_overlap_hits) >= 2 and len(dxy_policy_overlap_hits) > len(dxy_conflict_channel_hits)
        if policy_dominant:
            direct_supported = False
            proxy_supported = False
        elif direct_supported and not conflict_channel_ok:
            direct_supported = False
        elif proxy_supported and not conflict_channel_ok:
            proxy_supported = False
    if asset_label in INDEX_ASSETS and proxy_supported:
        proxy_supported = (
            not exclusion_hits
            and (
                doc.get("source_type") == "market_wrap"
                or float(doc.get("market_context_score") or 0.0) >= 4.0
            )
            and (bool(disambiguation_hits) or len(proxy_market_hits) >= 2)
        )
    supported = direct_supported or proxy_supported
    grade = "weak"
    if supported and score >= 24.0:
        grade = "strong"
    elif supported and score >= 16.0:
        grade = "moderate"
    evidence_mode = "direct" if direct_supported else "proxy" if proxy_supported else "weak"
    return {
        "score": round(score, 3),
        "grade": grade,
        "supported": supported,
        "evidence_mode": evidence_mode,
        "asset_hits": asset_hits,
        "factor_hits": factor_hits,
        "mechanism_hits": mechanism_hits,
        "proxy_hits": proxy_hits,
        "proxy_market_hits": proxy_market_hits,
        "disambiguation_hits": disambiguation_hits,
        "exclusion_hits": exclusion_hits,
        "dxy_policy_overlap_hits": dxy_policy_overlap_hits,
        "dxy_conflict_channel_hits": dxy_conflict_channel_hits,
    }


def _factor_metric(row: dict[str, Any], primary: str, secondary: str | None = None) -> Any:
    value = row.get(primary)
    if value is None and secondary is not None:
        value = row.get(secondary)
    return value


def _factor_summary_line(row: dict[str, Any]) -> str:
    return (
        f"- {row['factor_label']}: "
        f"docs={_factor_metric(row, 'doc_count', 'news_count')}, "
        f"mentions={row.get('mention_count')}, "
        f"sources={_factor_metric(row, 'avg_unique_sources', 'unique_sources')}, "
        f"geo={_factor_metric(row, 'avg_geo_count', 'geo_count')}, "
        f"tone={_factor_metric(row, 'avg_tone_mean', 'tone_mean')}, "
        f"score={_factor_metric(row, 'avg_narrative_score', 'narrative_score')}"
    )


def _top_factor_docs(
    db: Path,
    asset_label: str,
    start_date: str | None,
    end_date: str | None,
    factors: list[dict[str, Any]],
    per_factor: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for factor in factors:
        factor_label = factor["factor_label"]
        docs = query_supporting_docs(
            db,
            asset_label=asset_label,
            factor_label=factor_label,
            start_date=start_date,
            end_date=end_date,
            limit=max(per_factor * 3, 6),
        )
        grouped[factor_label] = [
            doc
            for doc in docs
            if (
                (
                    float(doc.get("relevance_score") or 0.0) > 0.0
                    and _doc_mentions_asset_context(doc, asset_label)
                )
                or _doc_mentions_proxy_context(doc, asset_label, factor_label)
            )
        ][:per_factor]
    return grouped


def _best_positive_docs(grouped_docs: dict[str, list[dict[str, Any]]], limit: int = 3) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for docs in grouped_docs.values():
        scored.extend(docs)
    scored.sort(
        key=lambda row: (
            float(row.get("relevance_score") or 0.0),
            float(row.get("classification_confidence") or 0.0),
            row.get("event_time"),
        ),
        reverse=True,
    )
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc in scored:
        key = str(doc.get("document_identifier") or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
        if len(deduped) >= limit:
            break
    return deduped


def _dedupe_factor_docs(asset_label: str, grouped_docs: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    best_assignments: dict[str, tuple[str, dict[str, Any], float]] = {}
    for factor_label, docs in grouped_docs.items():
        for doc in docs:
            key = str(doc.get("document_identifier") or "")
            if not key:
                continue
            strength = _factor_doc_strength(doc, asset_label, factor_label)
            mechanism_bonus = len(strength.get("mechanism_hits") or []) * 6.0
            factor_bonus = len(strength.get("factor_hits") or []) * 2.0
            generic_penalty = 12.0 if factor_label in STRICT_GENERIC_FACTORS and asset_label in MACRO_ASSETS else 0.0
            assignment_score = float(strength["score"]) + mechanism_bonus + factor_bonus - generic_penalty + (
                8.0 if strength["evidence_mode"] == "direct" else 4.0 if strength["evidence_mode"] == "proxy" else 0.0
            )
            current = best_assignments.get(key)
            if current is None or assignment_score > current[2]:
                best_assignments[key] = (factor_label, doc, assignment_score)

    deduped: dict[str, list[dict[str, Any]]] = {factor_label: [] for factor_label in grouped_docs}
    for factor_label, doc, _ in best_assignments.values():
        deduped.setdefault(factor_label, []).append(doc)
    return deduped


def _supported_factor_blocks(
    asset_label: str,
    factors: list[dict[str, Any]],
    factor_docs: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    supported_blocks: list[dict[str, Any]] = []
    weak_labels: list[str] = []
    for factor in factors:
        factor_label = factor["factor_label"]
        docs = factor_docs.get(factor_label, [])
        scored_docs: list[dict[str, Any]] = []
        for doc in docs:
            strength = _factor_doc_strength(doc, asset_label, factor_label)
            if not strength["supported"] or strength["grade"] == "weak":
                continue
            enriched = dict(doc)
            enriched["impact_strength"] = strength
            scored_docs.append(enriched)
        scored_docs.sort(
            key=lambda row: (
                2 if row["impact_strength"]["grade"] == "strong" else 1,
                float(row["impact_strength"]["score"]),
                float(row.get("classification_confidence") or 0.0),
            ),
            reverse=True,
        )
        best_doc = scored_docs[0] if scored_docs else None
        best_strength = best_doc["impact_strength"] if best_doc else None
        include_generic = False
        if best_strength:
            include_generic = (
                factor_label not in STRICT_GENERIC_FACTORS
                or (
                    best_strength["grade"] == "strong"
                    and float(best_strength["score"]) >= 24.0
                    and (
                        best_strength["mechanism_hits"]
                        or best_strength["factor_hits"]
                        or len(scored_docs) >= 2
                    )
                )
            )
        if scored_docs and include_generic:
            narrative_label = _specific_narrative_label(factor_label, scored_docs[:2])
            supported_blocks.append(
                {
                    "factor": factor,
                    "rule": _impact_rule(asset_label, factor_label),
                    "docs": scored_docs[:2],
                    "best_grade": best_strength["grade"],
                    "best_score": best_strength["score"],
                    "source_confidence": _source_confidence_for_docs(scored_docs[:2]),
                    "narrative_label": narrative_label,
                    "narrative_provenance": _narrative_provenance_from_docs(scored_docs[:2], factor_label),
                }
            )
        else:
            weak_labels.append(factor_label)
    supported_blocks.sort(
        key=lambda block: (
            _supported_block_rank(asset_label, block),
            float(block.get("best_score") or 0.0),
            float(block.get("factor", {}).get("adjusted_narrative_score") or 0.0),
        ),
        reverse=True,
    )
    return supported_blocks, weak_labels


def _supported_block_rank(asset_label: str, block: dict[str, Any]) -> float:
    factor = block.get("factor", {})
    factor_label = str(factor.get("factor_label") or "")
    best_score = float(block.get("best_score") or 0.0)
    adjusted_score = float(factor.get("adjusted_narrative_score") or 0.0)
    docs = list(block.get("docs") or [])
    best_doc = docs[0] if docs else {}
    evidence_mode = str(best_doc.get("impact_strength", {}).get("evidence_mode") or "")
    direct_bonus = 12.0 if evidence_mode == "direct" else 6.0 if evidence_mode == "proxy" else 0.0
    impact_rule_bonus = 8.0 if block.get("rule") else 0.0
    market_wrap_bonus = sum(
        3.0 if doc.get("source_type") == "market_wrap" else 1.5 if doc.get("source_type") == "commodity_specialist" else 0.0
        for doc in docs
    )
    market_context_bonus = sum(float(doc.get("market_context_score") or 0.0) * 0.25 for doc in docs[:2])
    proxy_preference_bonus = 6.0 if factor_label in PROXY_CHANNEL_RULES.get(asset_label, {}) else 0.0
    asset_factor_bonus = ASSET_FACTOR_SELECTION_BONUS.get(asset_label, {}).get(factor_label, 0.0)
    narrative_label = str(block.get("narrative_label") or "")
    specificity_bonus = 0.0
    if narrative_label and narrative_label != _factor_regime_label(factor_label):
        specificity_bonus = 8.0 if factor_label in STRICT_GENERIC_FACTORS else 3.0
    generic_penalty = 0.0
    if factor_label in STRICT_GENERIC_FACTORS:
        generic_penalty = 18.0 if asset_label in MACRO_ASSETS else 8.0
    return (
        best_score
        + (adjusted_score * 0.02)
        + direct_bonus
        + impact_rule_bonus
        + market_wrap_bonus
        + market_context_bonus
        + proxy_preference_bonus
        + asset_factor_bonus
        + specificity_bonus
        - generic_penalty
    )


def _source_confidence_level(
    *,
    unique_sources: int,
    market_wrap_docs: int,
    specialist_docs: int,
    direct_docs: int,
    proxy_docs: int,
) -> str:
    evidence_bonus = direct_docs * 2 + proxy_docs
    if unique_sources >= 2 and (market_wrap_docs >= 1 or specialist_docs >= 1 or evidence_bonus >= 2):
        return "high"
    if unique_sources >= 2 or market_wrap_docs >= 1 or specialist_docs >= 1 or evidence_bonus >= 1:
        return "medium"
    return "low"


def _source_confidence_for_docs(docs: list[dict[str, Any]]) -> str:
    unique_sources = {
        _clean_text(doc.get("source_domain"))
        for doc in docs
        if _clean_text(doc.get("source_domain"))
    }
    market_wrap_docs = sum(1 for doc in docs if doc.get("source_type") == "market_wrap")
    specialist_docs = sum(1 for doc in docs if doc.get("source_type") == "commodity_specialist")
    direct_docs = sum(
        1
        for doc in docs
        if str(doc.get("impact_strength", {}).get("evidence_mode") or "") == "direct"
    )
    proxy_docs = sum(
        1
        for doc in docs
        if str(doc.get("impact_strength", {}).get("evidence_mode") or "") == "proxy"
    )
    return _source_confidence_level(
        unique_sources=len(unique_sources),
        market_wrap_docs=market_wrap_docs,
        specialist_docs=specialist_docs,
        direct_docs=direct_docs,
        proxy_docs=proxy_docs,
    )


def _source_weighting_summary_for_docs(docs: list[dict[str, Any]]) -> str:
    unique_sources = {
        _clean_text(doc.get("source_domain"))
        for doc in docs
        if _clean_text(doc.get("source_domain"))
    }
    market_wrap_docs = sum(1 for doc in docs if doc.get("source_type") == "market_wrap")
    specialist_docs = sum(1 for doc in docs if doc.get("source_type") == "commodity_specialist")
    direct_docs = sum(
        1
        for doc in docs
        if str(doc.get("impact_strength", {}).get("evidence_mode") or "") == "direct"
    )
    proxy_docs = sum(
        1
        for doc in docs
        if str(doc.get("impact_strength", {}).get("evidence_mode") or "") == "proxy"
    )
    return (
        f"unique_sources={len(unique_sources)}, market_wrap_docs={market_wrap_docs}, "
        f"specialist_docs={specialist_docs}, direct_docs={direct_docs}, proxy_docs={proxy_docs}"
    )


def _state_primary_source_confidence(state: dict[str, Any]) -> str:
    supported_blocks = list(state.get("supported_blocks") or [])
    if not supported_blocks:
        return "low"
    return str(supported_blocks[0].get("source_confidence") or "low")


def _state_primary_evidence_mode(state: dict[str, Any]) -> str:
    supported_blocks = list(state.get("supported_blocks") or [])
    if not supported_blocks:
        return "unresolved"
    docs = list(supported_blocks[0].get("docs") or [])
    if not docs:
        return "unresolved"
    return str(docs[0].get("impact_strength", {}).get("evidence_mode") or "unresolved")


def _state_primary_narrative_label(state: dict[str, Any]) -> str:
    supported_blocks = list(state.get("supported_blocks") or [])
    if not supported_blocks:
        return "an unresolved indirect macro-wrap explanation"
    block = supported_blocks[0]
    return str(block.get("narrative_label") or _factor_regime_label(block["factor"]["factor_label"]))


def _narrative_provenance_from_docs(docs: list[dict[str, Any]], factor_label: str) -> str:
    if not docs:
        return "unresolved"
    narrative_label = _specific_narrative_label(factor_label, docs)
    if narrative_label == _factor_regime_label(factor_label):
        return "taxonomy-fallback"
    evidence_modes = {
        str(doc.get("impact_strength", {}).get("evidence_mode") or "")
        for doc in docs[:2]
    }
    if "direct" in evidence_modes:
        return "direct-text"
    if "proxy" in evidence_modes:
        return "proxy-text"
    return "text-derived"


def _state_primary_narrative_provenance(state: dict[str, Any]) -> str:
    supported_blocks = list(state.get("supported_blocks") or [])
    if not supported_blocks:
        return "unresolved"
    return str(supported_blocks[0].get("narrative_provenance") or "unresolved")


def _preferred_day_context_row(
    asset_label: str,
    impact_rows: list[dict[str, Any]],
    excluded_factor: str | None = None,
) -> dict[str, Any] | None:
    for row in impact_rows:
        if excluded_factor and row.get("factor_label") == excluded_factor:
            continue
        if asset_label not in set(row.get("explained_assets") or []):
            continue
        if row.get("narrative_provenance") == "taxonomy-fallback":
            continue
        return row
    return None


def _preferred_narrative_for_state(
    state: dict[str, Any],
    impact_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    label = _state_primary_narrative_label(state)
    original_label = label
    provenance = _state_primary_narrative_provenance(state)
    original_provenance = provenance
    factor_label = str(state.get("primary_factor") or "")
    source = "state-primary"
    substitution_reason = "none"
    replaced = "none"
    substitution_confidence = "none"
    supporting_urls: list[str] = []
    if provenance == "taxonomy-fallback" and impact_rows:
        context_row = _preferred_day_context_row(
            state["asset_label"],
            impact_rows,
            excluded_factor=factor_label or None,
        )
        if context_row is not None:
            label = str(context_row.get("narrative_label") or label)
            provenance = str(context_row.get("narrative_provenance") or provenance)
            factor_label = str(context_row.get("factor_label") or factor_label)
            source = "day-context"
            substitution_reason = f"{original_provenance} replaced by {provenance} day narrative"
            replaced = f"{original_label} -> {label}"
            substitution_confidence = str(context_row.get("source_confidence") or "medium")
            supporting_urls = list(context_row.get("supporting_urls") or [])[:3]
    return {
        "label": label,
        "provenance": provenance,
        "factor_label": factor_label,
        "source": source,
        "context_substitution": "yes" if source == "day-context" else "no",
        "substitution_reason": substitution_reason,
        "replaced": replaced,
        "substitution_confidence": substitution_confidence,
        "supporting_urls": supporting_urls,
    }


def _factor_impact_line(asset_label: str, factor_row: dict[str, Any], rule: dict[str, Any] | None) -> str:
    impact_path = None
    if rule:
        impact_path = rule.get("impact_path")
    if not impact_path:
        impact_path = f"{factor_row['factor_label']} appears linked to {asset_label} in the stored news set."
    return f"- {factor_row['factor_label']}: {impact_path}"


def _specific_narrative_label(factor_label: str, docs: list[dict[str, Any]]) -> str:
    text = " ".join(
        filter(
            None,
            [
                piece
                for doc in docs[:2]
                for piece in [
                    _clean_text(doc.get("title")),
                    _clean_text(doc.get("summary_text")),
                    _clean_text(doc.get("market_context_text")),
                    _clean_text(doc.get("evidence_text")),
                ]
                if piece
            ],
        )
    ).upper()
    for required_terms, label in SPECIFIC_NARRATIVE_RULES.get(factor_label, []):
        if all(term in text for term in required_terms):
            return label
    return _factor_regime_label(factor_label)


def _factor_evidence_line(
    doc: dict[str, Any],
    reference_map: dict[str, str],
    reference_lines: list[str],
) -> str:
    strength = doc["impact_strength"]
    matched = (
        strength["mechanism_hits"][:3]
        or strength["proxy_market_hits"][:3]
        or strength["proxy_hits"][:3]
        or strength["factor_hits"][:3]
        or strength["asset_hits"][:3]
    )
    matched_text = ", ".join(matched) if matched else "text overlap"
    ref_id = _doc_reference_id(doc, reference_map, reference_lines)
    ref_text = f"[{ref_id}] " if ref_id else ""
    return (
        f"  evidence={strength['evidence_mode']}/{strength['grade']} score={strength['score']} cues={matched_text} | "
        f"{doc['event_time']} | {ref_text}source={_source_reference(doc)} | {_doc_evidence_line(doc)}"
    )


def _asset_state(
    db: Path,
    asset_label: str,
    date: str,
    limit: int,
) -> dict[str, Any]:
    payload = query_explain_move(
        db,
        asset_label=asset_label,
        start_date=date,
        end_date=date,
        limit=max(limit, FACTOR_CANDIDATE_LIMIT),
    )
    factors = payload["top_narratives"][:FACTOR_CANDIDATE_LIMIT]
    factor_docs = _top_factor_docs(
        db=db,
        asset_label=asset_label,
        start_date=date,
        end_date=date,
        factors=factors,
    )
    factor_docs = _dedupe_factor_docs(asset_label, factor_docs)
    supported_blocks, weak_labels = _supported_factor_blocks(asset_label, factors, factor_docs)
    primary_factor = supported_blocks[0]["factor"]["factor_label"] if supported_blocks else None
    return {
        "asset_label": asset_label,
        "payload": payload,
        "factors": factors[:3],
        "candidate_factors": factors,
        "factor_docs": factor_docs,
        "supported_blocks": supported_blocks,
        "weak_labels": weak_labels,
        "confidence": _confidence_label(supported_blocks, weak_labels),
        "primary_factor": primary_factor,
    }


def _block_weight(block: dict[str, Any]) -> float:
    factor = block.get("factor", {})
    factor_label = str(factor.get("factor_label") or "")
    narrative_label = str(block.get("narrative_label") or "")
    specificity_bonus = 0.0
    if narrative_label and narrative_label != _factor_regime_label(factor_label):
        specificity_bonus = 2.0 if factor_label in STRICT_GENERIC_FACTORS else 0.75
    return (
        float(block.get("best_score") or 0.0)
        + float(factor.get("adjusted_narrative_score") or 0.0) / 100.0
        + specificity_bonus
    )


def _day_states(db: Path, date: str, universe: list[str], limit: int) -> list[dict[str, Any]]:
    return [_asset_state(db, asset_label, date, limit) for asset_label in universe]


def _rank_day_narratives(states: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ranked: dict[str, dict[str, Any]] = {}
    for state in states:
        for block in state["supported_blocks"]:
            factor_label = block["factor"]["factor_label"]
            entry = ranked.setdefault(
                factor_label,
                {
                    "factor_label": factor_label,
                    "regime_label": _factor_regime_label(factor_label),
                    "narrative_label": block.get("narrative_label") or _factor_regime_label(factor_label),
                    "narrative_provenance": block.get("narrative_provenance") or "unresolved",
                    "assets": [],
                    "sources": set(),
                    "urls": [],
                    "docs": [],
                    "score": 0.0,
                },
            )
            if state["asset_label"] not in entry["assets"]:
                entry["assets"].append(state["asset_label"])
            entry["score"] += _block_weight(block)
            for doc in block["docs"]:
                source = _clean_text(doc.get("source_domain"))
                url = _clean_text(doc.get("document_identifier"))
                if source:
                    entry["sources"].add(source)
                if url and url not in entry["urls"]:
                    entry["urls"].append(url)
                entry["docs"].append(doc)
    rows = []
    for entry in ranked.values():
        rows.append(
            {
                "factor_label": entry["factor_label"],
                "regime_label": entry["regime_label"],
                "narrative_label": entry["narrative_label"],
                "narrative_provenance": entry["narrative_provenance"],
                "assets": sorted(entry["assets"]),
                "source_diversity": len(entry["sources"]),
                "supporting_urls": entry["urls"][:3],
                "source_confidence": _source_confidence_for_docs(entry["docs"]),
                "source_weighting": _source_weighting_summary_for_docs(entry["docs"]),
                "score": round(entry["score"], 3),
            }
        )
    rows.sort(key=lambda row: (row["score"], len(row["assets"]), row["source_diversity"]), reverse=True)
    return rows[:limit]


def _supported_asset_labels(states: list[dict[str, Any]]) -> set[str]:
    return {state["asset_label"] for state in states if state["supported_blocks"]}


def _market_impact_rows(states: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    evidence_rows = _rank_day_narratives(states, max(limit * 2, 10))
    observed_assets = {state["asset_label"] for state in states}
    supported_assets = _supported_asset_labels(states)
    rows: list[dict[str, Any]] = []
    for row in evidence_rows:
        targets = set(TRANSMISSION_TARGETS.get(row["factor_label"], row["assets"]))
        explained_assets = sorted(targets.intersection(observed_assets))
        supported_overlap = sorted(targets.intersection(supported_assets))
        impact_score = (
            len(explained_assets) * 10.0
            + len(supported_overlap) * 5.0
            + float(row["source_diversity"]) * 2.0
            + float(row["score"])
        )
        rows.append(
            {
                **row,
                "explained_assets": explained_assets,
                "supported_overlap": supported_overlap,
                "impact_score": round(impact_score, 3),
            }
        )
    rows.sort(
        key=lambda row: (row["impact_score"], len(row["explained_assets"]), row["source_diversity"], row["score"]),
        reverse=True,
    )
    return rows[:limit]


def _explanation_fit_rows(states: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    impact_rows = _market_impact_rows(states, max(limit * 2, 8))
    if not impact_rows:
        return []
    top_rows = impact_rows[: min(len(impact_rows), 6)]
    preferred_by_asset = {
        state["asset_label"]: _preferred_narrative_for_state(state, impact_rows)
        for state in states
    }
    supported_assets = {state["asset_label"] for state in states if state["supported_blocks"]}
    all_assets = {state["asset_label"] for state in states}
    combos: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, ...]] = set()
    for combo_size in range(1, min(3, len(top_rows)) + 1):
        for subset in combinations(top_rows, combo_size):
            factor_key = tuple(sorted(row["factor_label"] for row in subset))
            if factor_key in seen_keys:
                continue
            seen_keys.add(factor_key)
            explained_assets: set[str] = set()
            direct_support: set[str] = set()
            sources = 0
            score = 0.0
            supporting_urls: list[str] = []
            labels: list[str] = []
            provenances: list[str] = []
            confidences: list[str] = []
            for row in subset:
                explained_assets.update(row["explained_assets"])
                direct_support.update(row["supported_overlap"])
                sources += int(row["source_diversity"])
                score += float(row["score"])
                labels.append(row["narrative_label"])
                provenances.append(row["narrative_provenance"])
                confidences.append(row["source_confidence"])
                for url in row["supporting_urls"]:
                    if url not in supporting_urls:
                        supporting_urls.append(url)
            matched_assets = {
                asset_label
                for asset_label, preferred in preferred_by_asset.items()
                if preferred.get("factor_label") in factor_key or asset_label in explained_assets
            }
            contradictions = sorted(supported_assets - matched_assets)
            unresolved = sorted(all_assets - explained_assets - matched_assets)
            fit_score = (
                len(matched_assets) * 25.0
                + len(explained_assets.intersection(all_assets)) * 8.0
                + len(direct_support) * 5.0
                + min(sources, 12) * 1.5
                + score
                - len(contradictions) * 15.0
                - len(unresolved) * 6.0
                - max(0, combo_size - 1) * 4.0
            )
            combos.append(
                {
                    "factor_labels": [row["factor_label"] for row in subset],
                    "narrative_labels": labels,
                    "narrative_provenances": provenances,
                    "source_confidences": confidences,
                    "source_weightings": [row.get("source_weighting", "unknown") for row in subset],
                    "explained_assets": sorted(explained_assets.intersection(all_assets)),
                    "matched_assets": sorted(matched_assets),
                    "direct_support": sorted(direct_support),
                    "contradictions": contradictions,
                    "unresolved": unresolved,
                    "supporting_urls": supporting_urls[:5],
                    "fit_score": round(fit_score, 3),
                }
            )
    combos.sort(
        key=lambda row: (
            row["fit_score"],
            len(row["matched_assets"]),
            -len(row["contradictions"]),
            -len(row["unresolved"]),
            len(row["direct_support"]),
        ),
        reverse=True,
    )
    return combos[:limit]


def _transmission_rows(states: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    impact_rows = _market_impact_rows(states, max(limit * 2, 10))
    rows: list[dict[str, Any]] = []
    for row in impact_rows:
        chain = TRANSMISSION_CHAINS.get(row["factor_label"])
        if not chain:
            continue
        rows.append({**row, "chain": chain})
    return rows[:limit]


def _failure_mode_for_factor(factor_label: str) -> str:
    if factor_label == "oil":
        return (
            "If stronger direct evidence shows rates, dollar, or equities were driven by a non-oil macro shock, "
            "then the oil/geopolitical unwind is only a coincident headline, not the first causal link."
        )
    if factor_label == "inflation":
        return (
            "If new evidence shows the inflation/rates repricing was entirely downstream of oil normalization, "
            "then this is not an independent regime but a transmission step."
        )
    if factor_label == "gold_precious":
        return (
            "If direct bullion-market evidence is displaced by stronger dollar/Fed or conflict-specific evidence, "
            "then precious-metals positioning unwind is incomplete as the primary explanation."
        )
    if factor_label == "war_conflict":
        return (
            "If the conflict headlines fail to map to actual commodity, safe-haven, or rates transmission, "
            "then war coverage is just high-volume background noise."
        )
    if factor_label == "earnings":
        return (
            "If direct index-level evidence shows the move was macro rather than company or sector specific, "
            "then earnings is too narrow to explain the broader asset move."
        )
    return (
        "If stronger direct asset-specific evidence emerges for another factor, "
        "then this explanation should be downgraded to an overlay or discarded."
    )


def _failure_mode_for_asset_state(state: dict[str, Any]) -> str:
    asset_label = state["asset_label"]
    primary = state.get("primary_factor")
    if not primary:
        return (
            f"If direct {asset_label} evidence emerges from asset-specific or macro-wrap articles, "
            "then the current unresolved classification is incomplete."
        )
    if asset_label == "NDX":
        return (
            "If direct NDX evidence shows the move was driven by a specific mega-cap, AI, semis, or earnings story, "
            "then the current macro-contradiction framing is incomplete."
        )
    if asset_label == "DXY":
        return (
            "If stronger FX-market evidence ties the dollar to explicit relative-policy or safe-haven headlines, "
            "then the current unresolved dollar classification is too conservative."
        )
    if asset_label in {"US2Y", "US10Y"}:
        return (
            "If direct Treasury-market evidence shows traders reacted primarily to Fed-path, auction, or labor headlines, "
            "then the move is less second-order than the current transmission-heavy reading implies."
        )
    return _failure_mode_for_factor(primary)


def _weakest_fitting_core_asset(states: list[dict[str, Any]], matched_assets: set[str] | None = None) -> str | None:
    matched_assets = matched_assets or set()
    core_states = [state for state in states if state["asset_label"] in CORE_CROSS_ASSET_SET]
    if not core_states:
        return None

    def rank_tuple(state: dict[str, Any]) -> tuple[int, int, int, int, str]:
        confidence = str(state.get("confidence") or "")
        direct_low = 1 if "direct=low" in confidence else 0
        overall_low = 1 if "overall=low" in confidence else 0
        matched = 0 if state["asset_label"] in matched_assets else 1
        no_support = 1 if not state.get("supported_blocks") else 0
        return (direct_low, overall_low, matched, no_support, state["asset_label"])

    weakest = max(core_states, key=rank_tuple)
    if "direct=low" not in str(weakest.get("confidence") or "") and weakest["asset_label"] in matched_assets:
        return None
    return str(weakest["asset_label"])


def _normalized_fit_score(best_fit: dict[str, Any] | None, states: list[dict[str, Any]]) -> float:
    if not best_fit or not states:
        return 0.0
    asset_count = max(len(states), 1)
    max_fit = asset_count * 38.0 + 25.0
    min_fit = -asset_count * 21.0
    raw = float(best_fit.get("fit_score", 0.0))
    normalized = (raw - min_fit) / max(max_fit - min_fit, 1.0)
    return round(max(0.0, min(1.0, normalized)), 2)


def _contradiction_score(best_fit: dict[str, Any] | None, states: list[dict[str, Any]]) -> float:
    if not best_fit or not states:
        return 1.0 if states else 0.0
    total_assets = max(len(states), 1)
    contradiction_ratio = len(best_fit.get("contradictions", [])) / total_assets
    unresolved_ratio = len(best_fit.get("unresolved", [])) / total_assets
    direct_gap = max(0, total_assets - len(best_fit.get("direct_support", []))) / total_assets
    score = contradiction_ratio * 0.6 + unresolved_ratio * 0.25 + direct_gap * 0.15
    return round(max(0.0, min(1.0, score)), 2)


def _unsupported_assets(states: list[dict[str, Any]], best_fit: dict[str, Any] | None = None) -> list[str]:
    unsupported = {state["asset_label"] for state in states if not state["supported_blocks"]}
    if best_fit:
        unsupported.update(best_fit.get("unresolved", []))
    return sorted(unsupported)


def _trust_lines(
    states: list[dict[str, Any]],
    best_fit: dict[str, Any] | None = None,
    weakest_core: str | None = None,
) -> list[str]:
    fit_confidence = _normalized_fit_score(best_fit, states)
    contradiction = _contradiction_score(best_fit, states)
    unsupported = _unsupported_assets(states, best_fit)
    lines = [
        "Trust summary:",
        f"fit_confidence={fit_confidence}",
        f"contradiction_score={contradiction}",
        f"unsupported_assets={', '.join(unsupported) if unsupported else 'none'}",
    ]
    if weakest_core:
        lines.append(f"weakest_core_asset={weakest_core}")
    return lines


def _cannot_answer_lines(states: list[dict[str, Any]], best_fit: dict[str, Any] | None = None) -> list[str]:
    unsupported = _unsupported_assets(states, best_fit)
    if not unsupported:
        return ["Unsupported / cannot answer: none."]
    details: list[str] = []
    for state in states:
        if state["asset_label"] not in unsupported:
            continue
        details.append(f"{state['asset_label']} ({_state_primary_evidence_mode(state)})")
    if best_fit:
        matched = set(best_fit.get("matched_assets", []))
        for asset_label in best_fit.get("unresolved", []):
            if asset_label not in matched and all(not entry.startswith(f"{asset_label} ") for entry in details):
                details.append(f"{asset_label} (combination unresolved)")
    return [
        "Unsupported / cannot answer: "
        + "; ".join(details)
        + "."
    ]


def _day_summary_lines(date: str, states: list[dict[str, Any]], limit: int) -> list[str]:
    evidence_ranked = _rank_day_narratives(states, limit)
    if not evidence_ranked:
        return [f"No supported day-level narratives were verified from stored text for {date}."]
    impact_ranked = _market_impact_rows(states, limit)
    fit_ranked = _explanation_fit_rows(states, limit)
    transmission_ranked = _transmission_rows(states, limit)
    weakest_core = _weakest_fitting_core_asset(
        states,
        set(fit_ranked[0]["matched_assets"]) if fit_ranked else set(),
    )
    reference_map: dict[str, str] = {}
    reference_lines: list[str] = []
    lines = [f"Top narratives for {date} across {', '.join(state['asset_label'] for state in states)}:"]
    lines.append("Explanatory fit ranking:")
    for index, row in enumerate(fit_ranked, start=1):
        refs: list[str] = []
        for url in row["supporting_urls"]:
            ref_id = _ensure_reference(reference_map, reference_lines, url=url, label=None)
            if ref_id:
                refs.append(f"[{ref_id}]")
        lines.append(
            f"{index}. {' + '.join(row['narrative_labels'])}: fit_score={row['fit_score']}, "
            f"provenance={', '.join(row['narrative_provenances'])}, "
            f"source_confidence={', '.join(row['source_confidences'])}, "
            f"source_weighting={' | '.join(row['source_weightings'])}, "
            f"matched_assets={', '.join(row['matched_assets']) if row['matched_assets'] else 'none'}, "
            f"contradictions={', '.join(row['contradictions']) if row['contradictions'] else 'none'}, "
            f"unresolved={', '.join(row['unresolved']) if row['unresolved'] else 'none'}, "
            f"supporting_sources={' '.join(refs) if refs else 'none'}"
        )
    lines.append("Evidence strength ranking:")
    for index, row in enumerate(evidence_ranked, start=1):
        refs: list[str] = []
        for url in row["supporting_urls"]:
            ref_id = _ensure_reference(reference_map, reference_lines, url=url, label=None)
            if ref_id:
                refs.append(f"[{ref_id}]")
        urls = ", ".join(refs) if refs else "none"
        lines.append(
            f"{index}. {row['narrative_label']} ({row['factor_label']}): score={row['score']}, "
            f"provenance={row['narrative_provenance']}, "
            f"source_diversity={row['source_diversity']}, source_confidence={row['source_confidence']}, "
            f"source_weighting={row['source_weighting']}, "
            f"affected_assets={', '.join(row['assets'])}, "
            f"supporting_sources={urls}"
        )
    lines.append("Market impact ranking:")
    for index, row in enumerate(impact_ranked, start=1):
        lines.append(
            f"{index}. {row['narrative_label']} ({row['factor_label']}): impact_score={row['impact_score']}, "
            f"provenance={row['narrative_provenance']}, "
            f"source_confidence={row['source_confidence']}, "
            f"explained_assets={', '.join(row['explained_assets'])}, "
            f"direct_support={', '.join(row['supported_overlap']) if row['supported_overlap'] else 'none'}"
        )
    lines.append("Transmission ranking:")
    if transmission_ranked:
        for index, row in enumerate(transmission_ranked, start=1):
            lines.append(
                f"{index}. {row['narrative_label']} ({row['factor_label']}): provenance={row['narrative_provenance']}; {row['chain']}"
            )
    else:
        lines.append("1. No transmission chain cleared the evidence threshold.")
    lines.extend(_trust_lines(states, fit_ranked[0] if fit_ranked else None, weakest_core))
    lines.extend(_cannot_answer_lines(states, fit_ranked[0] if fit_ranked else None))
    lines.append(
        "Failure mode: "
        + _failure_mode_for_factor((fit_ranked[0]["factor_labels"][0] if fit_ranked else impact_ranked[0]["factor_label"]))
    )
    return _append_reference_block(lines, reference_lines)


def _cross_asset_lines(date: str, states: list[dict[str, Any]], limit: int) -> list[str]:
    ranked = _rank_day_narratives(states, limit)
    impact_rows = _market_impact_rows(states, max(limit * 2, 10))
    fit_rows = _explanation_fit_rows(states, max(limit, 3))
    shared = [row for row in ranked if len(row["assets"]) >= 2]
    best_fit = fit_rows[0] if fit_rows else None
    fit_matched = set(best_fit["matched_assets"]) if best_fit else set()
    conflicting = [state for state in states if state["asset_label"] not in fit_matched]
    missing = [state["asset_label"] for state in states if not state["supported_blocks"]]
    reference_map: dict[str, str] = {}
    reference_lines: list[str] = []
    weakest_core = _weakest_fitting_core_asset(states, fit_matched)
    lines = [f"Cross-asset move for {date} across {', '.join(state['asset_label'] for state in states)}:"]
    if best_fit:
        lines.append(
            "Best explanatory combination: "
            + f"{' + '.join(best_fit['narrative_labels'])} "
            + f"(evidence=combination-fit, fit_score={best_fit['fit_score']}, "
            + f"provenance={', '.join(best_fit['narrative_provenances'])}, "
            + f"source_confidence={', '.join(best_fit['source_confidences'])}, "
            + f"source_weighting={' | '.join(best_fit['source_weightings'])}, "
            + f"matched_assets={', '.join(best_fit['matched_assets']) if best_fit['matched_assets'] else 'none'}, "
            + f"contradictions={', '.join(best_fit['contradictions']) if best_fit['contradictions'] else 'none'}, "
            + f"unresolved={', '.join(best_fit['unresolved']) if best_fit['unresolved'] else 'none'})."
        )
    if shared:
        lines.append(
            "Shared narratives: "
            + "; ".join(
                (
                    lambda refs: f"{row['narrative_label']} affecting {', '.join(row['assets'])} "
                    f"(provenance={row['narrative_provenance']})"
                    + (f" {' '.join(refs)}" if refs else "")
                )(
                    [
                        f"[{ref_id}]"
                        for url in row["supporting_urls"]
                        if (ref_id := _ensure_reference(reference_map, reference_lines, url=url, label=None))
                    ]
                )
                for row in shared[:limit]
            )
            + "."
        )
    else:
        lines.append("Shared narratives: none cleared the evidence threshold across multiple assets.")
    if conflicting:
        lines.append(
            "Conflicting narratives: "
            + "; ".join(
                (
                    lambda preferred: f"{state['asset_label']} needed {preferred['label']} "
                    f"(evidence={_state_primary_evidence_mode(state)}, provenance={preferred['provenance']}, "
                    f"source={preferred['source']}, "
                    f"context_substitution={preferred['context_substitution']}, "
                    f"substitution_reason={preferred['substitution_reason']}, "
                    f"replaced={preferred['replaced']}, "
                    f"substitution_confidence={preferred['substitution_confidence']}, "
                    f"source_confidence={_state_primary_source_confidence(state)})"
                )(_preferred_narrative_for_state(state, impact_rows))
                for state in conflicting[:limit]
            )
            + "."
        )
    else:
        if weakest_core == "NDX":
            lines.append(
                "Conflicting narratives: no contradiction breaks the combined explanation, "
                "but NDX remains the weakest-fitting core asset because lower yields would normally be supportive."
            )
        elif weakest_core:
            lines.append(
                "Conflicting narratives: no contradiction breaks the combined explanation, "
                f"but {weakest_core} remains the weakest-fitting core asset."
            )
        else:
            lines.append("Conflicting narratives: no contradiction breaks the combined explanation.")
    lines.extend(_trust_lines(states, best_fit, weakest_core))
    lines.extend(_cannot_answer_lines(states, best_fit))
    lines.append(
        "Confidence by asset: "
        + "; ".join(f"{state['asset_label']}={state['confidence']}" for state in states)
        + "."
    )
    missing_assets = [state for state in states if not state["supported_blocks"]]
    if missing_assets:
        lines.append("Failure mode: " + _failure_mode_for_asset_state(missing_assets[0]))
    elif shared:
        lines.append("Failure mode: " + _failure_mode_for_factor(shared[0]["factor_label"]))
    elif states:
        lines.append("Failure mode: " + _failure_mode_for_asset_state(states[0]))
    return _append_reference_block(lines, reference_lines)


def _contradictory_asset_lines(date: str, states: list[dict[str, Any]], limit: int) -> list[str]:
    ranked = _rank_day_narratives(states, limit)
    fit_rows = _explanation_fit_rows(states, max(limit, 3))
    if not ranked and not fit_rows:
        return [f"No dominant day narrative was verified from stored text for {date}."]
    impact_rows = _market_impact_rows(states, max(limit * 2, 10))
    reference_map: dict[str, str] = {}
    reference_lines: list[str] = []
    dominant_fit = fit_rows[0] if fit_rows else None
    dominant = ranked[0] if ranked else None
    dominant_factors = set(dominant_fit["factor_labels"]) if dominant_fit else ({dominant["factor_label"]} if dominant else set())
    matched_assets = set(dominant_fit["matched_assets"]) if dominant_fit else set()
    contradictions = [state for state in states if state["asset_label"] not in matched_assets]
    lines = [
        (
            f"Dominant narrative for {date}: {' + '.join(dominant_fit['narrative_labels'])} "
            f"({', '.join(dominant_fit['factor_labels'])}) matching {', '.join(dominant_fit['matched_assets']) if dominant_fit['matched_assets'] else 'none'} "
            f"(fit_score={dominant_fit['fit_score']}, provenance={', '.join(dominant_fit['narrative_provenances'])}, "
            f"source_confidence={', '.join(dominant_fit['source_confidences'])})."
            if dominant_fit
            else f"Dominant narrative for {date}: {dominant['narrative_label']} ({dominant['factor_label']}) affecting {', '.join(dominant['assets'])} "
            f"(provenance={dominant['narrative_provenance']}, source_confidence={dominant['source_confidence']})."
        ),
        "Expected responses: assets explained by the best-fit narrative combination should align first; others need an overlay or remain unresolved.",
    ]
    if contradictions:
        for state in contradictions[:limit]:
            preferred = _preferred_narrative_for_state(state, impact_rows)
            preferred_refs = [
                f"[{ref_id}]"
                for url in preferred.get("supporting_urls", [])
                if (ref_id := _ensure_reference(reference_map, reference_lines, url=url, label=None))
            ]
            preferred_refs_text = f" {' '.join(preferred_refs)}" if preferred_refs else ""
            lines.append(
                f"- {state['asset_label']}: contradiction to dominant narrative. "
                f"Required overlay={preferred['label']}{preferred_refs_text}. "
                f"Overlay evidence={_state_primary_evidence_mode(state)}. "
                f"Overlay provenance={preferred['provenance']}. "
                f"Overlay source={preferred['source']}. "
                f"Overlay context_substitution={preferred['context_substitution']}. "
                f"Overlay substitution_reason={preferred['substitution_reason']}. "
                f"Overlay replaced={preferred['replaced']}. "
                f"Overlay substitution_confidence={preferred['substitution_confidence']}. "
                f"Overlay source_confidence={_state_primary_source_confidence(state)}. "
                f"Confidence={state['confidence']}."
            )
            lines.append(f"  Failure mode: {_failure_mode_for_asset_state(state)}")
    else:
        lines.append("No strong contradictions were found relative to the dominant narrative.")
        if dominant_fit:
            lines.append("Failure mode: " + _failure_mode_for_factor(dominant_fit["factor_labels"][0]))
        elif dominant:
            lines.append("Failure mode: " + _failure_mode_for_factor(dominant["factor_label"]))
    return _append_reference_block(lines, reference_lines)


def _asset_via_day_context_lines(
    date: str,
    asset_state: dict[str, Any],
    day_states: list[dict[str, Any]],
    limit: int,
) -> list[str]:
    other_states = [state for state in day_states if state["asset_label"] != asset_state["asset_label"]]
    ranked_day = _rank_day_narratives(day_states, limit)
    impact_rows = _market_impact_rows(day_states, max(limit * 2, 10))
    preferred = _preferred_narrative_for_state(asset_state, impact_rows)
    reference_map: dict[str, str] = {}
    reference_lines: list[str] = []
    lines = [f"Asset day-context explanation for {asset_state['asset_label']} on {date}:"]
    if asset_state["supported_blocks"]:
        direct = asset_state["supported_blocks"][0]
        ref_tokens = []
        for doc in direct["docs"][:2]:
            ref_id = _doc_reference_id(doc, reference_map, reference_lines)
            ref_tokens.append(f"{_source_reference(doc)}{f' [{ref_id}]' if ref_id else ''}")
        lines.append(
            "Direct evidence: "
            + f"{direct.get('narrative_label') or _factor_regime_label(direct['factor']['factor_label'])} via "
            + ", ".join(ref_tokens)
            + f" (evidence={_state_primary_evidence_mode(asset_state)}, provenance={_state_primary_narrative_provenance(asset_state)}, "
            + f"source_confidence={_state_primary_source_confidence(asset_state)})"
            + "."
        )
    else:
        lines.append("Direct evidence: no strong asset-specific explanation was verified from stored text.")
    lines.append(
        "Context substitution: "
        + f"context_substitution={preferred['context_substitution']} "
        + f"(source={preferred['source']}, provenance={preferred['provenance']}, "
        + f"substitution_reason={preferred['substitution_reason']}, replaced={preferred['replaced']}, "
        + f"substitution_confidence={preferred['substitution_confidence']})."
    )
    if preferred["source"] == "day-context":
        preferred_refs = [
            f"[{ref_id}]"
            for url in preferred.get("supporting_urls", [])
            if (ref_id := _ensure_reference(reference_map, reference_lines, url=url, label=None))
        ]
        lines.append(
            "Preferred day-context narrative: "
            + f"{preferred['label']} (provenance={preferred['provenance']}, source={preferred['source']}, "
            + f"context_substitution={preferred['context_substitution']}, "
            + f"substitution_reason={preferred['substitution_reason']}, "
            + f"replaced={preferred['replaced']}, "
            + f"substitution_confidence={preferred['substitution_confidence']})"
            + (f" {' '.join(preferred_refs)}." if preferred_refs else ".")
        )
    if ranked_day:
        indirect = [row for row in ranked_day if asset_state["asset_label"] not in row["assets"]]
        if indirect:
            row = indirect[0]
            refs = []
            for url in row["supporting_urls"]:
                ref_id = _ensure_reference(reference_map, reference_lines, url=url, label=None)
                if ref_id:
                    refs.append(f"[{ref_id}]")
            lines.append(
                "Indirect day context: "
                + f"{row['narrative_label']} was strongly evidenced elsewhere in the market via "
                + ", ".join(row["assets"])
                + f", with provenance={row['narrative_provenance']}, source_confidence={row['source_confidence']} "
                + f"and sources {' '.join(refs) if refs else 'none'}."
            )
        else:
            lines.append("Indirect day context: the asset was already aligned with the dominant supported day narrative.")
    supporting_others = [state["asset_label"] for state in other_states if state["supported_blocks"]]
    if supporting_others:
        lines.append("Macro-wrap proxies with stronger evidence: " + ", ".join(supporting_others[:limit]) + ".")
    lines.append("Confidence: " + asset_state["confidence"] + ".")
    lines.append("Failure mode: " + _failure_mode_for_asset_state(asset_state))
    return _append_reference_block(lines, reference_lines)


def _humanize_narrative_label(label: str) -> str:
    replacements = {
        "Fed policy repricing": "Fed/dollar restraint",
        "Middle East geopolitical repricing": "Middle East risk unwind",
        "trade-and-sanctions repricing": "trade/sanctions pressure on energy",
        "oil/geopolitical premium repricing": "oil/geopolitical premium unwind",
        "inflation-relief repricing": "inflation-relief impulse",
        "rate-path repricing": "policy-rate path repricing",
        "jobs/Fed repricing": "labor/Fed repricing",
    }
    return replacements.get(label, label)


def _humanize_combination(labels: list[str]) -> str:
    return " + ".join(_humanize_narrative_label(label) for label in labels)


def _economic_hypothesis_from_fit_row(row: dict[str, Any], weakest_core: str | None) -> tuple[str, str]:
    factors = set(row.get("factor_labels", []))
    if factors & {"war_conflict", "oil", "shipping_disruption", "sanctions_trade"}:
        return (
            "Oil/geopolitical risk unwind",
            "Oil/geopolitical risk-premium unwind drove the day, with Fed/dollar restraint limiting the equity response."
            if "central_bank_policy" in factors or weakest_core == "NDX"
            else "Oil/geopolitical risk-premium unwind drove the day."
        )
    if factors & {"central_bank_policy", "interest_rates", "labour_market"}:
        return (
            "Fed repricing",
            "Fed policy repricing was the primary driver, with oil weakness acting mainly through inflation expectations."
        )
    if factors & {"growth_activity", "earnings", "mergers_acquisitions"} or weakest_core == "NDX":
        return (
            "Tech-specific de-rating",
            "Tech-specific de-rating dominated equity behaviour independently of the broader macro backdrop."
        )
    if "inflation" in factors:
        return (
            "Inflation relief",
            "Disinflationary news dominated the session, pulling yields lower and supporting only a partial risk response."
        )
    if "gold_precious" in factors:
        return (
            "Safe-haven unwind",
            "Safe-haven demand faded, weakening defensive assets alongside the broader macro adjustment."
        )
    return ("Mixed macro repricing", f"The session reflected a mixed macro repricing centered on {_humanize_combination(row.get('narrative_labels', []))}.")


def _hypothesis_confidence(rank: int, row: dict[str, Any], unsupported_assets: list[str]) -> str:
    if rank == 1 and not unsupported_assets:
        return "High"
    if rank == 1:
        return "High"
    if rank == 2:
        return "Medium"
    return "Medium-Low"


def _hypothesis_explains_summary(item: dict[str, Any]) -> list[str]:
    hypothesis = str(item.get("hypothesis") or "")
    if hypothesis == "Oil/geopolitical risk unwind":
        return ["WTI", "yields", "inflation"]
    if hypothesis == "Fed repricing":
        return ["dollar", "rates"]
    if hypothesis == "Tech-specific de-rating":
        return ["NDX"]
    if hypothesis == "Inflation relief":
        return ["yields", "inflation"]
    if hypothesis == "Safe-haven unwind":
        return ["Gold", "defensive flows"]
    return list(item.get("affected_assets") or [])[:3]


def _hypothesis_weakness_summary(item: dict[str, Any], weakest_core: str | None) -> str:
    hypothesis = str(item.get("hypothesis") or "")
    if hypothesis == "Oil/geopolitical risk unwind":
        return "Does not fully explain NDX." if weakest_core == "NDX" else "Does not fully explain the weakest equity leg."
    if hypothesis == "Fed repricing":
        return "Under-explains oil."
    if hypothesis == "Tech-specific de-rating":
        return "Does not explain bonds or oil."
    if hypothesis == "Inflation relief":
        return "Too clean to explain weaker commodities and incomplete equity relief on its own."
    if hypothesis == "Safe-haven unwind":
        return "Does not explain the full rates and commodity move on its own."
    return "Does not explain the full cross-asset pattern on its own."


def _top_competing_hypotheses(
    fit_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    impact_rows: list[dict[str, Any]],
    weakest_core: str | None,
    unsupported_assets: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in fit_rows:
        label, thesis = _economic_hypothesis_from_fit_row(row, weakest_core)
        entry = grouped.get(label)
        if entry is None or float(row.get("fit_score") or 0.0) > float(entry.get("market_fit_score") or 0.0):
            grouped[label] = {
                "hypothesis": label,
                "thesis": thesis,
                "market_fit_score": row.get("fit_score"),
                "model_composition": row.get("narrative_labels", []),
                "affected_assets": row.get("matched_assets", []),
                "contradictions": row.get("contradictions", []),
                "unresolved_assets": row.get("unresolved", []),
                "source_weight_breakdown": row.get("source_weightings", []),
            }
    supplemental_rows: list[dict[str, Any]] = []
    for row in evidence_rows:
        supplemental_rows.append(
            {
                "factor_labels": [row.get("factor_label")],
                "narrative_labels": [row.get("narrative_label")],
                "fit_score": row.get("score"),
                "matched_assets": row.get("assets", []),
                "contradictions": [],
                "unresolved": [],
                "source_weightings": [row.get("source_weighting", "unknown")],
            }
        )
    for row in impact_rows:
        supplemental_rows.append(
            {
                "factor_labels": [row.get("factor_label")],
                "narrative_labels": [row.get("narrative_label")],
                "fit_score": row.get("impact_score"),
                "matched_assets": row.get("explained_assets", []),
                "contradictions": [],
                "unresolved": [],
                "source_weightings": [f"direct_support={', '.join(row.get('supported_overlap', [])) or 'none'}"],
            }
        )
    for row in supplemental_rows:
        label, thesis = _economic_hypothesis_from_fit_row(row, weakest_core)
        if label in grouped:
            continue
        grouped[label] = {
            "hypothesis": label,
            "thesis": thesis,
            "market_fit_score": row.get("fit_score"),
            "model_composition": row.get("narrative_labels", []),
            "affected_assets": row.get("matched_assets", []),
            "contradictions": row.get("contradictions", []),
            "unresolved_assets": row.get("unresolved", []),
            "source_weight_breakdown": row.get("source_weightings", []),
        }
    ranked = sorted(grouped.values(), key=lambda item: float(item.get("market_fit_score") or 0.0), reverse=True)
    output: list[dict[str, Any]] = []
    for idx, item in enumerate(ranked[:3], start=1):
        reason = ""
        if idx == 1:
            reason = (
                "Best explains WTI, Treasury yields, and the inflation channel while remaining consistent with the stronger dollar and only partial equity participation."
            )
        elif item["hypothesis"] == "Fed repricing":
            reason = "Explains the dollar and rates well but struggles to account for the full scale of the oil move."
        elif item["hypothesis"] == "Tech-specific de-rating":
            reason = "Helps explain Nasdaq weakness but cannot explain the broader cross-asset pattern on its own."
        elif item["hypothesis"] == "Inflation relief":
            reason = "Fits the bond move, but on its own it is too clean for a session with weaker commodities and incomplete equity relief."
        else:
            reason = "Captures part of the session, but not the full cross-asset move as cleanly as the winner."
        output.append(
            {
                **item,
                "rank": idx,
                "confidence": _hypothesis_confidence(idx, item, unsupported_assets),
                "explains": _hypothesis_explains_summary(item),
                "weakness": _hypothesis_weakness_summary(item, weakest_core),
                "why_it_fits": reason,
                "winner": idx == 1,
            }
        )
    return output


def _preferred_transmission_row(best_fit: dict[str, Any] | None, transmission_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not transmission_rows:
        return None
    factor_labels = set(best_fit.get("factor_labels", [])) if best_fit else set()

    def rank(row: dict[str, Any]) -> tuple[int, float]:
        factor = row.get("factor_label")
        impact = float(row.get("impact_score") or 0.0)
        if factor in factor_labels and factor in {"war_conflict", "oil", "shipping_disruption", "sanctions_trade"}:
            return (4, impact)
        if factor in factor_labels and factor == "central_bank_policy":
            return (3, impact)
        if factor in {"war_conflict", "oil", "shipping_disruption", "sanctions_trade"}:
            return (2, impact)
        if factor == "central_bank_policy":
            return (1, impact)
        return (0, impact)

    return max(transmission_rows, key=rank)


def _blocking_overlay(best_fit: dict[str, Any] | None, transmission_rows: list[dict[str, Any]], weakest_core: str | None) -> str | None:
    factor_labels = set(best_fit.get("factor_labels", [])) if best_fit else set()
    if weakest_core == "NDX" and "central_bank_policy" in factor_labels:
        return "stronger dollar and continued Fed-related policy restraint blocked a clean risk-on session"
    if "central_bank_policy" in factor_labels:
        return "Fed/dollar restraint limited broader risk-on confirmation"
    if any(row.get("factor_label") == "central_bank_policy" for row in transmission_rows):
        return "Fed/dollar restraint remained an overlay on top of the primary regime"
    return None


def _consistency_warnings(
    best_fit: dict[str, Any] | None,
    transmission_rows: list[dict[str, Any]],
    weakest_core: str | None,
    unsupported_assets: list[str],
) -> list[str]:
    warnings: list[str] = []
    first_row = _preferred_transmission_row(best_fit, transmission_rows)
    first_chain = str(first_row.get("chain") or "") if first_row else ""
    if weakest_core == "NDX" and "lower yields" in first_chain:
        warnings.append(
            "Do not describe NDX weakness as duration pressure when lower-yield transmission dominates; prefer dollar/policy restraint unless direct tech evidence overrides it."
        )
    if "Gold" in unsupported_assets:
        warnings.append("Do not present gold as strongly explained; direct support is weak in the current local slice.")
    if weakest_core == "NDX":
        warnings.append("Treat the equity-tech leg as weaker than the oil/rates leg unless direct tech-specific evidence appears.")
    return warnings


def _confidence_summary(best_fit: dict[str, Any] | None, unsupported_assets: list[str], weakest_core: str | None) -> str:
    if not best_fit:
        return "Low confidence: no best-fit combination cleared the local evidence threshold."
    if unsupported_assets:
        if weakest_core == "NDX":
            return "Higher confidence on the oil/rates leg than on the equity-tech leg; the weakest core asset is NDX."
        return "Confidence is moderate: the primary regime fits well, but some assets remain unresolved in the local slice."
    return "Confidence is relatively high: the primary regime fits broadly and unresolved assets are limited."


def _narrative_frame_payload(date: str, states: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    fit_rows = _explanation_fit_rows(states, limit)
    evidence_rows = _rank_day_narratives(states, limit)
    impact_rows = _market_impact_rows(states, limit)
    transmission_rows = _transmission_rows(states, limit)
    best_fit = fit_rows[0] if fit_rows else None
    best_evidence = evidence_rows[0] if evidence_rows else None
    weakest_core = _weakest_fitting_core_asset(states, set(best_fit["matched_assets"]) if best_fit else set())
    unsupported_assets = _unsupported_assets(states, best_fit)
    first_row = _preferred_transmission_row(best_fit, transmission_rows)
    overlay = _blocking_overlay(best_fit, transmission_rows, weakest_core)
    primary_regime = _humanize_combination(best_fit["narrative_labels"]) if best_fit else "unresolved"
    first_link = None
    transmission_chain = None
    if first_row:
        first_link = first_row["chain"].split("->", 1)[0].strip()
        transmission_chain = first_row["chain"]
    weakest_reason = None
    if weakest_core:
        weakest_state = next((state for state in states if state["asset_label"] == weakest_core), None)
        if weakest_state:
            weakest_reason = _failure_mode_for_asset_state(weakest_state)
    reference_urls: list[str] = []
    for url in (best_fit.get("supporting_urls", []) if best_fit else []):
        if url not in reference_urls:
            reference_urls.append(url)
    for row in impact_rows[:3]:
        for url in row.get("supporting_urls", []):
            if url not in reference_urls:
                reference_urls.append(url)
    dominant_summary = None
    if best_fit and first_row:
        dominant_summary = (
            f"For {date}, the dominant market narrative was {first_row['chain']}"
            + (f", while {overlay}." if overlay else ".")
        )
    elif best_fit:
        dominant_summary = (
            f"For {date}, the dominant market narrative combined {primary_regime}."
        )
    best_explanation_summary = None
    if best_fit and first_row:
        best_explanation_summary = (
            f"The single best explanation for the cross-asset move was {first_row['chain']}"
            + (f", while {overlay}." if overlay else ".")
        )
    elif best_fit:
        best_explanation_summary = (
            f"The single best explanation for the cross-asset move was the combined {primary_regime} regime."
        )
    parallel_channels: list[dict[str, Any]] = []
    if first_row:
        parallel_channels.append(
            {
                "channel": "primary_transmission",
                "factor_label": first_row["factor_label"],
                "narrative_label": first_row["narrative_label"],
                "chain": first_row["chain"],
                "source_confidence": first_row["source_confidence"],
            }
        )
    if overlay:
        parallel_channels.append(
            {
                "channel": "blocking_overlay",
                "factor_label": "central_bank_policy" if any(row.get("factor_label") == "central_bank_policy" for row in transmission_rows) else None,
                "narrative_label": "Fed/dollar restraint",
                "chain": overlay,
                "source_confidence": next(
                    (row.get("source_confidence") for row in transmission_rows if row.get("factor_label") == "central_bank_policy"),
                    "medium",
                ),
            }
        )
    if "Gold" in unsupported_assets:
        parallel_channels.append(
            {
                "channel": "weakly_resolved_parallel_leg",
                "factor_label": "gold_precious",
                "narrative_label": "dollar / real-rate pressure on gold",
                "chain": TRANSMISSION_CHAINS["gold_precious"],
                "source_confidence": next(
                    (row.get("source_confidence") for row in transmission_rows if row.get("factor_label") == "gold_precious"),
                    "low",
                ),
            }
        )
    competing_hypotheses = _top_competing_hypotheses(fit_rows, evidence_rows, impact_rows, weakest_core, unsupported_assets)
    return {
        "date": date,
        "primary_regime": primary_regime,
        "primary_regime_raw": best_fit["narrative_labels"] if best_fit else [],
        "first_link": first_link,
        "transmission_chain": transmission_chain,
        "blocking_overlay": overlay,
        "weakest_asset": weakest_core,
        "weakest_asset_reason": weakest_reason,
        "unresolved_assets": unsupported_assets,
        "consistency_warnings": _consistency_warnings(best_fit, transmission_rows, weakest_core, unsupported_assets),
        "confidence_summary": _confidence_summary(best_fit, unsupported_assets, weakest_core),
        "diagnostics": {
            "fit_metric": _normalized_fit_score(best_fit, states),
            "contradiction_metric": _contradiction_score(best_fit, states),
            "matched_assets": best_fit.get("matched_assets", []) if best_fit else [],
            "contradictions": best_fit.get("contradictions", []) if best_fit else [],
            "unresolved": best_fit.get("unresolved", []) if best_fit else [],
        },
        "dominant_narrative": {
            "question": "What was the dominant market narrative today?",
            "label": primary_regime,
            "raw_labels": best_fit["narrative_labels"] if best_fit else [],
            "summary": dominant_summary,
            "causal_chain": transmission_chain,
            "first_link": first_link,
            "blocking_overlay": overlay,
            "affected_assets": best_fit.get("matched_assets", []) if best_fit else [],
            "weakest_asset": weakest_core,
            "unresolved_assets": unsupported_assets,
            "fit_metric": _normalized_fit_score(best_fit, states),
            "contradiction_metric": _contradiction_score(best_fit, states),
            "source_weight_breakdown": best_fit.get("source_weightings", []) if best_fit else [],
            "supporting_references": reference_urls[:5],
        },
        "best_explanation": {
            "question": "What was the single best explanation for today's cross-asset moves?",
            "label": primary_regime,
            "raw_labels": best_fit["narrative_labels"] if best_fit else [],
            "summary": best_explanation_summary,
            "best_fit_combination": " + ".join(best_fit["narrative_labels"]) if best_fit else None,
            "market_fit_score": best_fit.get("fit_score") if best_fit else None,
            "evidence_strength_score": best_evidence.get("score") if best_evidence else None,
            "transmission_plausibility": transmission_chain,
            "parallel_channels": parallel_channels,
            "weakest_asset": weakest_core,
            "unresolved_assets": unsupported_assets,
            "source_weight_breakdown": best_fit.get("source_weightings", []) if best_fit else [],
            "supporting_references": reference_urls[:5],
        },
        "top_competing_hypotheses": competing_hypotheses,
        "rankings": {
            "evidence_strength": [
                {
                    "narrative_label": row["narrative_label"],
                    "factor_label": row["factor_label"],
                    "score": row["score"],
                    "source_confidence": row["source_confidence"],
                    "source_weighting": row["source_weighting"],
                    "affected_assets": row["assets"],
                    "supporting_urls": row["supporting_urls"],
                }
                for row in evidence_rows
            ],
            "market_impact": [
                {
                    "narrative_label": row["narrative_label"],
                    "factor_label": row["factor_label"],
                    "impact_score": row["impact_score"],
                    "source_confidence": row["source_confidence"],
                    "explained_assets": row["explained_assets"],
                    "direct_support": row["supported_overlap"],
                }
                for row in impact_rows
            ],
            "transmission_plausibility": [
                {
                    "narrative_label": row["narrative_label"],
                    "factor_label": row["factor_label"],
                    "chain": row["chain"],
                    "source_confidence": row["source_confidence"],
                }
                for row in transmission_rows
            ],
        },
        "market_impact_rows": [
            {
                "narrative_label": row["narrative_label"],
                "factor_label": row["factor_label"],
                "impact_score": row["impact_score"],
                "explained_assets": row["explained_assets"],
                "source_confidence": row["source_confidence"],
            }
            for row in impact_rows
        ],
        "transmission_rows": [
            {
                "narrative_label": row["narrative_label"],
                "factor_label": row["factor_label"],
                "chain": row["chain"],
                "source_confidence": row["source_confidence"],
            }
            for row in transmission_rows
        ],
        "supporting_references": reference_urls[:8],
    }


def _factor_regime_label(factor_label: str) -> str:
    return FACTOR_REGIME_LABELS.get(factor_label, factor_label.replace("_", " "))


def _overlay_label(asset_label: str) -> str:
    return ASSET_OVERLAY_LABELS.get(asset_label, "Secondary overlay")


def _confidence_label(supported_blocks: list[dict[str, Any]], weak_labels: list[str]) -> str:
    def level(score: int) -> str:
        if score >= 2:
            return "high"
        if score == 1:
            return "medium"
        return "low"

    direct_score = sum(
        1
        for block in supported_blocks
        if block.get("docs") and block["docs"][0]["impact_strength"].get("evidence_mode") == "direct"
    )
    proxy_score = sum(
        1
        for block in supported_blocks
        if block.get("docs") and block["docs"][0]["impact_strength"].get("evidence_mode") == "proxy"
    )
    transmission_score = sum(
        1 for block in supported_blocks if block.get("factor", {}).get("factor_label") in TRANSMISSION_CHAINS
    )
    direct_level = level(direct_score)
    proxy_level = level(proxy_score)
    transmission_level = level(transmission_score)
    overall_numeric = 0
    if direct_score >= 2:
        overall_numeric = 2
    elif direct_score >= 1 or (proxy_score >= 2 and transmission_score >= 1):
        overall_numeric = 1
    elif proxy_score >= 1 or transmission_score >= 1:
        overall_numeric = 1
    overall_level = level(overall_numeric)
    suffix = "; lower on generic graph-ranked factors without text confirmation" if weak_labels else ""
    return (
        f"direct={direct_level}, proxy/channel={proxy_level}, "
        f"transmission={transmission_level}, overall={overall_level}{suffix}"
    )


def _primary_regime_line(supported_blocks: list[dict[str, Any]], weak_labels: list[str]) -> str:
    if not supported_blocks:
        if weak_labels:
            return (
                "Primary regime: unresolved from stored text; the graph ranked "
                + ", ".join(weak_labels[:3])
                + " but the article text did not verify them strongly."
            )
        return "Primary regime: unresolved from stored text."
    primary = supported_blocks[0]
    factor_label = primary["factor"]["factor_label"]
    rule = primary.get("rule") or {}
    path = rule.get("impact_path") or f"{factor_label} appears linked in the stored news set."
    return (
        f"Primary regime: {primary.get('narrative_label') or _factor_regime_label(factor_label)} "
        f"(provenance={primary.get('narrative_provenance') or 'unresolved'}). {path}"
    )


def _overlay_line(asset_label: str, supported_blocks: list[dict[str, Any]]) -> str:
    label = _overlay_label(asset_label)
    if len(supported_blocks) < 2:
        return f"{label}: no second text-backed overlay cleared the evidence threshold."
    secondary = supported_blocks[1]
    factor_label = secondary["factor"]["factor_label"]
    rule = secondary.get("rule") or {}
    path = rule.get("impact_path") or f"{factor_label} appears linked in the stored news set."
    return (
        f"{label}: {secondary.get('narrative_label') or _factor_regime_label(factor_label)} "
        f"(provenance={secondary.get('narrative_provenance') or 'unresolved'}). {path}"
    )


def _contradiction_line(weak_labels: list[str]) -> str:
    if weak_labels:
        listed = ", ".join(weak_labels[:3])
        return (
            "Contradiction: graph-ranked factors "
            + listed
            + " were too generic or too weakly evidenced in stored text to treat as primary explanations."
        )
    return "Contradiction: no major graph-ranked contradiction survived the text-verification filter."


def _tool_explain_move(arguments: dict[str, Any]) -> dict[str, Any]:
    store = Path(arguments.get("db", DEFAULT_DB))
    with resolve_query_db(store, arguments.get("start_date"), arguments.get("end_date")) as db:
        payload = query_explain_move(
            db,
            asset_label=arguments["asset_label"],
            start_date=arguments.get("start_date"),
            end_date=arguments.get("end_date"),
            limit=int(arguments.get("limit", 10)),
        )
    return _text_content(json.dumps(payload, indent=2, default=str))


def _tool_supporting_docs(arguments: dict[str, Any]) -> dict[str, Any]:
    store = Path(arguments.get("db", DEFAULT_DB))
    with resolve_query_db(store, arguments.get("start_date"), arguments.get("end_date")) as db:
        payload = query_supporting_docs(
            db,
            asset_label=arguments["asset_label"],
            factor_label=arguments.get("factor_label"),
            start_date=arguments.get("start_date"),
            end_date=arguments.get("end_date"),
            limit=int(arguments.get("limit", 10)),
        )
    return _text_content(json.dumps(payload, indent=2, default=str))


def _tool_summarize_narrative(arguments: dict[str, Any]) -> dict[str, Any]:
    store = Path(arguments.get("db", DEFAULT_DB))
    with resolve_query_db(store, arguments.get("start_date"), arguments.get("end_date")) as db:
        payload = query_explain_move(
            db,
            asset_label=arguments["asset_label"],
            start_date=arguments.get("start_date"),
            end_date=arguments.get("end_date"),
            limit=max(int(arguments.get("limit", 5)), FACTOR_CANDIDATE_LIMIT),
        )
        factors = payload["top_narratives"][:FACTOR_CANDIDATE_LIMIT]
        if not factors:
            summary = f"No narratives found for {payload['asset_label']} in the requested window."
            return _text_content(summary)

        factor_docs = _top_factor_docs(
            db=db,
            asset_label=payload["asset_label"],
            start_date=arguments.get("start_date"),
            end_date=arguments.get("end_date"),
            factors=factors,
        )
        factor_docs = _dedupe_factor_docs(payload["asset_label"], factor_docs)
        supported_blocks, weak_labels = _supported_factor_blocks(payload["asset_label"], factors, factor_docs)
        top_factor_labels = ", ".join(f["factor_label"] for f in factors)
        reference_map: dict[str, str] = {}
        reference_lines: list[str] = []
        summary_lines = [
            f"Graph-ranked factors for {payload['asset_label']}: {top_factor_labels}.",
            _primary_regime_line(supported_blocks, weak_labels),
            _overlay_line(payload["asset_label"], supported_blocks),
            _contradiction_line(weak_labels),
            "Confidence: " + _confidence_label(supported_blocks, weak_labels) + ".",
        ]
        if supported_blocks:
            supported_labels = ", ".join(
                f"{block.get('narrative_label') or block['factor']['factor_label']} ({block['factor']['factor_label']})"
                for block in supported_blocks
            )
            summary_lines.append(
                f"Most defensible text-backed explanations for {payload['asset_label']}: {supported_labels}."
            )
            summary_lines.append("Impact paths:")
            for block in supported_blocks:
                summary_lines.append(
                    _factor_impact_line(payload["asset_label"], block["factor"], block.get("rule"))
                )
                summary_lines.append(
                    f"  metrics: {_factor_summary_line(block['factor'])[2:]} | "
                    f"source_confidence={block['source_confidence']}"
                )
                for doc in block["docs"]:
                    summary_lines.append(_factor_evidence_line(doc, reference_map, reference_lines))
        else:
            summary_lines.append(
                "No strong asset-specific explanation was verified from the stored article text in this window."
            )
        if weak_labels:
            summary_lines.append(
                "Graph-ranked but weakly evidenced in stored text: " + ", ".join(weak_labels) + "."
            )
        return _text_content("\n".join(_append_reference_block(summary_lines, reference_lines)))


def _tool_explain_day(arguments: dict[str, Any]) -> dict[str, Any]:
    date = arguments["date"]
    universe = list(arguments["universe"])
    limit = int(arguments.get("limit", 5))
    store = Path(arguments.get("db", DEFAULT_DB))
    with resolve_query_db(store, date, date) as db:
        states = _day_states(db, date, universe, limit)
    return _text_content("\n".join(_day_summary_lines(date, states, limit)))


def _tool_explain_cross_asset_move(arguments: dict[str, Any]) -> dict[str, Any]:
    date = arguments["date"]
    assets = list(arguments["assets"])
    limit = int(arguments.get("limit", 5))
    store = Path(arguments.get("db", DEFAULT_DB))
    with resolve_query_db(store, date, date) as db:
        states = _day_states(db, date, assets, limit)
    return _text_content("\n".join(_cross_asset_lines(date, states, limit)))


def _tool_build_narrative_frame(arguments: dict[str, Any]) -> dict[str, Any]:
    date = arguments["date"]
    universe = list(arguments["universe"])
    limit = int(arguments.get("limit", 5))
    store = Path(arguments.get("db", DEFAULT_DB))
    with resolve_query_db(store, date, date) as db:
        states = _day_states(db, date, universe, limit)
        return _json_content(_narrative_frame_payload(date, states, limit))


def _tool_find_contradictory_assets(arguments: dict[str, Any]) -> dict[str, Any]:
    date = arguments["date"]
    universe = list(arguments["universe"])
    limit = int(arguments.get("limit", 5))
    store = Path(arguments.get("db", DEFAULT_DB))
    with resolve_query_db(store, date, date) as db:
        states = _day_states(db, date, universe, limit)
    return _text_content("\n".join(_contradictory_asset_lines(date, states, limit)))


def _tool_explain_asset_via_day_context(arguments: dict[str, Any]) -> dict[str, Any]:
    date = arguments["date"]
    asset_label = arguments["asset_label"]
    universe = list(arguments.get("universe") or [asset_label, "WTI", "Gold", "US2Y", "US10Y", "DXY", "NDX"])
    if asset_label not in universe:
        universe.insert(0, asset_label)
    limit = int(arguments.get("limit", 5))
    store = Path(arguments.get("db", DEFAULT_DB))
    with resolve_query_db(store, date, date) as db:
        states = _day_states(db, date, universe, limit)
        asset_state = next(state for state in states if state["asset_label"] == asset_label)
        return _text_content("\n".join(_asset_via_day_context_lines(date, asset_state, states, limit)))


def _tool_query_duckdb(arguments: dict[str, Any]) -> dict[str, Any]:
    store = Path(arguments.get("db", DEFAULT_DB))
    sql = str(arguments["sql"])
    limit = int(arguments.get("limit", 50))
    normalized = _normalize_query_sql(sql, limit)
    with resolve_query_db(store, arguments.get("start_date"), arguments.get("end_date")) as db:
        rows = run_query(db, normalized)
        return _json_content(
            {
                "db": str(db),
                "sql": normalized,
                "row_count": len(rows),
                "rows": rows,
            }
        )


def _tool_similar_days(arguments: dict[str, Any]) -> dict[str, Any]:
    store = Path(arguments.get("db", DEFAULT_DB))
    date = str(arguments["date"])
    limit = max(1, min(int(arguments.get("limit", 5)), 20))
    factor_list = ", ".join(f"'{factor}'" for factor in SIMILAR_DAY_FACTORS)
    sql = f"""
    WITH daily_factor AS (
        SELECT
            bucket_time,
            factor_label,
            SUM(narrative_score) AS score
        FROM gold_factor_buckets_daily
        WHERE factor_label IN ({factor_list})
        GROUP BY 1, 2
    ),
    daily_vec AS (
        SELECT
            bucket_time,
            {", ".join(
                f"SUM(CASE WHEN factor_label = '{factor}' THEN score ELSE 0 END) AS {factor}"
                for factor in SIMILAR_DAY_FACTORS
            )}
        FROM daily_factor
        GROUP BY 1
    ),
    base_day AS (
        SELECT * FROM daily_vec WHERE bucket_time = CAST('{date}' AS DATE)
    )
    SELECT
        d.bucket_time,
        SQRT(
            {" + ".join(
                f"POW(d.{factor} - b.{factor}, 2)"
                for factor in SIMILAR_DAY_FACTORS
            )}
        ) AS distance
    FROM daily_vec d
    CROSS JOIN base_day b
    WHERE d.bucket_time < CAST('{date}' AS DATE)
    ORDER BY distance ASC, d.bucket_time DESC
    LIMIT {limit}
    """
    with resolve_query_db(store) as db:
        rows = run_query(db, sql)
        if not rows:
            return _text_content(f"No prior local days were available for similarity comparison before {date}.")
        lines = [f"Similar days for {date}:"]
        for index, row in enumerate(rows, start=1):
            lines.append(f"{index}. {row['bucket_time']}: distance={round(float(row['distance']), 3)}")
        lines.append(
            "Caveat: similarity is based on daily factor-mix distance across the local narrative graph, not on price paths."
        )
        return _text_content("\n".join(lines))


def _tool_intraday_evolution(arguments: dict[str, Any]) -> dict[str, Any]:
    store = Path(arguments.get("db", DEFAULT_DB))
    date = str(arguments["date"])
    limit = max(1, min(int(arguments.get("limit", 5)), 20))
    with resolve_query_db(store, date, date) as db:
        hour_rows = run_query(
            db,
            """
            SELECT
                DATE_TRUNC('hour', event_time) AS hour_bucket,
                COUNT(*) AS event_count
            FROM silver_event_graph
            WHERE CAST(event_time AS DATE) = CAST(? AS DATE)
            GROUP BY 1
            ORDER BY 1
            """,
            [date],
        )
        if len(hour_rows) < 2:
            only = hour_rows[0]["hour_bucket"] if hour_rows else "none"
            return _text_content(
                f"Intraday evolution for {date}: cannot answer reliably from the current local data.\n"
                f"Observed intraday buckets={len(hour_rows)}; bucket_hours={only}.\n"
                "Reason: there are not enough distinct intraday event buckets to infer a sequence of narrative transitions."
            )
        top_rows = run_query(
            db,
            """
            WITH hourly AS (
                SELECT
                    DATE_TRUNC('hour', event_time) AS hour_bucket,
                    factor_label,
                    COUNT(*) AS mention_count,
                    AVG(tone) AS tone_mean
                FROM silver_factor_mentions
                WHERE CAST(event_time AS DATE) = CAST(? AS DATE)
                GROUP BY 1, 2
            ),
            ranked AS (
                SELECT
                    hour_bucket,
                    factor_label,
                    mention_count,
                    tone_mean,
                    ROW_NUMBER() OVER (PARTITION BY hour_bucket ORDER BY mention_count DESC, factor_label ASC) AS rn
                FROM hourly
            )
            SELECT hour_bucket, factor_label, mention_count, tone_mean
            FROM ranked
            WHERE rn <= ?
            ORDER BY hour_bucket, rn
            """,
            [date, limit],
        )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in top_rows:
            grouped.setdefault(str(row["hour_bucket"]), []).append(row)
        lines = [f"Intraday evolution for {date}:"]
        for hour, rows in grouped.items():
            parts = [
                f"{row['factor_label']} mentions={row['mention_count']} tone={round(float(row['tone_mean'] or 0.0), 3)}"
                for row in rows
            ]
            lines.append(f"- {hour}: " + "; ".join(parts))
        lines.append(
            "Caveat: this reads narrative intensity through hourly factor mentions only; it does not infer price causality by itself."
        )
        return _text_content("\n".join(lines))


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "explain_move":
        return _tool_explain_move(arguments)
    if name == "summarize_narrative":
        return _tool_summarize_narrative(arguments)
    if name == "supporting_docs":
        return _tool_supporting_docs(arguments)
    if name == "explain_day":
        return _tool_explain_day(arguments)
    if name == "explain_cross_asset_move":
        return _tool_explain_cross_asset_move(arguments)
    if name == "build_narrative_frame":
        return _tool_build_narrative_frame(arguments)
    if name == "find_contradictory_assets":
        return _tool_find_contradictory_assets(arguments)
    if name == "explain_asset_via_day_context":
        return _tool_explain_asset_via_day_context(arguments)
    if name == "query_duckdb":
        return _tool_query_duckdb(arguments)
    if name == "similar_days":
        return _tool_similar_days(arguments)
    if name == "intraday_evolution":
        return _tool_intraday_evolution(arguments)
    raise ValueError(f"unknown tool: {name}")


def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    req_id = request.get("id")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2025-06-18",
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "capabilities": {"tools": {}},
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": _tool_specs()}}
    if method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        try:
            result = call_tool(tool_name, arguments)
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as exc:  # pragma: no cover
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(exc)},
            }
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def main() -> int:
    while True:
        request = _read_message()
        if request is None:
            return 0
        response = handle_request(request)
        if response is not None:
            _write_message(response)


if __name__ == "__main__":
    raise SystemExit(main())
