"""Shared lexical matching helpers for v3 narrative evidence selection."""

from __future__ import annotations

import json
import string
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = ROOT / "config" / "news_narrative_taxonomy.json"

with TAXONOMY_PATH.open("r", encoding="utf-8") as handle:
    _TAXONOMY = json.load(handle)

FACTOR_PATTERNS_BY_LABEL = {
    factor["label"]: factor.get("patterns", []) for factor in _TAXONOMY.get("factors", [])
}

ASSET_TEXT_PATTERNS = {
    "WTI": ["WTI", "CRUDE", "OIL", "OPEC", "BARREL", "TANKER", "RED SEA"],
    "Brent": ["BRENT", "CRUDE", "OIL", "OPEC", "BARREL", "TANKER", "RED SEA"],
    "HG": ["COPPER", "COMEX COPPER", "LME COPPER", "SMELTER", "MINE SUPPLY"],
    "FXI": ["FXI", "CHINA", "CHINESE", "HONG KONG", "CSI 300", "MAINLAND"],
    "BTC": ["BITCOIN", "BTC", "CRYPTO", "TOKEN", "STABLECOIN", "EXCHANGE", "ETF"],
    "BDI": ["BALTIC DRY", "DRY BULK", "BULK CARRIER", "FREIGHT", "SHIPPING"],
    "Gold": ["GOLD", "BULLION", "XAU", "PRECIOUS METAL", "SAFE HAVEN"],
    "DXY": ["DOLLAR", "USD", "GREENBACK", "US DOLLAR"],
    "US2Y": ["2Y", "2-YEAR", "TWO-YEAR", "SHORT-DATED TREASURY", "FRONT-END YIELD"],
    "US10Y": ["10Y", "10-YEAR", "TEN-YEAR", "LONG-DATED TREASURY", "BENCHMARK YIELD"],
    "NDX": ["NASDAQ 100", "NASDAQ-100", "NDX", "QQQ", "NASDAQ FALLS", "NASDAQ DROPS", "NASDAQ SLIDES", "TECH SELLOFF", "CHIP STOCKS"],
    "SPX": ["S&P 500", "SP 500", "SPX", "WALL STREET", "US STOCKS", "U.S. STOCKS", "STOCKS SLUMP", "STOCKS FALL"],
}

_PUNCT_TRANSLATION = str.maketrans({character: " " for character in string.punctuation})


def normalize_for_match(text: str | None) -> str:
    if not text:
        return ""
    cleaned = text.upper().replace("_", " ").translate(_PUNCT_TRANSLATION)
    return " ".join(cleaned.split())


def match_count(text: str | None, cues: list[str]) -> int:
    normalized = normalize_for_match(text)
    if not normalized:
        return 0
    tokens = set(normalized.split())
    hits = 0
    for cue in cues:
        normalized_cue = normalize_for_match(cue)
        if not normalized_cue:
            continue
        if " " in normalized_cue:
            matched = f" {normalized_cue} " in f" {normalized} "
        else:
            matched = normalized_cue in tokens
        if matched:
            hits += 1
    return hits


def matched_cues(text: str | None, cues: list[str]) -> list[str]:
    normalized = normalize_for_match(text)
    if not normalized:
        return []
    tokens = set(normalized.split())
    hits: list[str] = []
    seen: set[str] = set()
    for cue in cues:
        normalized_cue = normalize_for_match(cue)
        if not normalized_cue or normalized_cue in seen:
            continue
        if " " in normalized_cue:
            matched = f" {normalized_cue} " in f" {normalized} "
        else:
            matched = normalized_cue in tokens
        if matched:
            hits.append(normalized_cue)
            seen.add(normalized_cue)
    return hits


def factor_cues(factor_label: str | None) -> list[str]:
    if not factor_label:
        return []
    cues = list(FACTOR_PATTERNS_BY_LABEL.get(factor_label, []))
    cues.append(factor_label.replace("_", " "))
    return cues


def asset_cues(asset_label: str | None) -> list[str]:
    if not asset_label:
        return []
    cues = list(ASSET_TEXT_PATTERNS.get(asset_label, []))
    cues.append(asset_label)
    return cues
