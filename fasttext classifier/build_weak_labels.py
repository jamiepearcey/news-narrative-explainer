#!/usr/bin/env python3
"""Build a conservative weak-labeled corpus for fastText."""

from __future__ import annotations

import csv
import json
import random
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
DATA_DIR = WORK_DIR / "data"
FEEDBACK_DIR = WORK_DIR / "feedback"
RESULTS_DIR = WORK_DIR / "results"
INPUT_PARQUET_GLOB = str(PROJECT_ROOT / "data/gdelt_candidates_20d_full/**/*.parquet")
RAW_EXPORT_CSV = DATA_DIR / "corpus_projection.csv"
WEAK_LABEL_CSV = DATA_DIR / "weak_labels.csv"
TRAIN_TXT = DATA_DIR / "train.txt"
VALID_TXT = DATA_DIR / "valid.txt"
SUMMARY_JSON = RESULTS_DIR / "weak_label_summary.json"
REVIEW_QUEUE_CSV = FEEDBACK_DIR / "review_queue.csv"
FEEDBACK_LABELS_CSV = FEEDBACK_DIR / "labeled_feedback.csv"
SOURCE_PROFILES_CSV = RESULTS_DIR / "source_probe_profiles.csv"
DOMAIN_SCORES_CSV = RESULTS_DIR / "effective_domain_scores.csv"

KEEP_LABELS = {
    "keep_finance",
    "keep_macro",
    "keep_geopolitics",
    "keep_company_event",
}

DROP_LABELS = {
    "drop_sports",
    "drop_entertainment",
    "drop_lifestyle",
    "drop_local_crime",
    "drop_low_quality",
    "drop_press_release",
}

ALL_LABELS = KEEP_LABELS | DROP_LABELS

PRESS_RELEASE_DOMAINS = {
    "prnewswire.com",
    "businesswire.com",
    "globenewswire.com",
    "newsfilecorp.com",
    "accessnewswire.com",
    "openpr.com",
}

PRESS_RELEASE_HINT_DOMAINS = {
    "itnewsonline.com",
    "manilatimes.net",
    "searchlight.vc",
    "interfax.com.ua",
    "en.acnnewswire.com",
    "pr.com",
}

LOW_QUALITY_DOMAINS = {
    "drudge.com",
    "zazoom.it",
    "river949.com.au",
}

REVIEW_DEPRIORITIZE_DOMAINS = {
    "mediaite.com",
    "inewsgr.com",
    "newspim.com",
    "sbctv.gr",
}

FINANCE_TERMS = {
    "oil", "crude", "wti", "brent", "gas", "lng", "gold", "silver", "copper",
    "yield", "yields", "bond", "bonds", "treasury", "treasuries", "stock",
    "stocks", "shares", "equity", "equities", "forex", "currency", "dollar",
    "euro", "yen", "inflation", "cpi", "ppi", "rates", "rate", "fed", "ecb",
    "boe", "boj", "opec", "tariff", "sanction", "sanctions", "export",
    "import", "shipping", "freight", "earnings", "guidance", "revenue",
    "profit", "loss", "acquisition", "merger", "ipo", "bank", "banking",
    "credit", "loan", "liquidity", "default", "refinery", "pipeline",
}

MACRO_TERMS = {
    "fed", "federal reserve", "ecb", "bank of england", "boj", "central bank",
    "interest rates", "rate cut", "rate hike", "inflation", "recession",
    "gdp", "employment", "payrolls", "treasury", "bond yields", "fiscal",
    "deficit", "stimulus", "tariff", "sanctions", "trade war",
}

GEOPOLITICS_TERMS = {
    "israel", "iran", "ukraine", "russia", "china", "taiwan", "lebanon",
    "syria", "hormuz", "red sea", "missile", "drone", "military", "war",
    "attack", "ceasefire", "sanctions", "opec", "embargo", "nuclear",
    "shipping lane", "strait",
}

COMPANY_EVENT_TERMS = {
    "earnings", "guidance", "quarter", "quarterly", "ceo", "cfo", "board",
    "acquisition", "merger", "deal", "stake", "investment", "funding", "ipo",
    "listing", "delisting", "bankruptcy", "lawsuit", "probe", "recall",
    "plant", "factory", "production", "dividend", "buyback", "layoffs",
}

EQUITY_MARKET_TERMS = {
    "nasdaq", "nyse", "shares", "share price", "stock price", "etf", "insider",
    "price target", "dividend", "holdings", "positions", "stake", "trading down",
    "trading up", "analyst", "downgraded", "upgraded", "earnings", "equity",
    "securities", "investors", "portfolio", "fund", "funds",
}

STRONG_COMPANY_EVENT_PHRASES = {
    "earnings guidance",
    "releases earnings results",
    "beats estimates",
    "fda feedback",
    "fda approval",
    "acquired",
    "acquires",
    "acquisition",
    "fundraise",
    "convertible bond",
    "equity fundraise",
    "merger arbitrage",
    "named official marketing partner",
    "make-or-break moment",
    "stock soaring",
    "shares are sliding",
}

IPO_MARKET_STORY_PHRASES = {
    "stock at ipo",
    "buy spacex",
    "buy open ai",
    "buy openai",
    "before the ipo",
    "ai boom valuations",
    "big test of ai boom valuations",
    "ipo valuations",
    "ipo playbook",
    "draw more orders than shares available",
    "etf makes it easier",
}

MARKET_GEOPOLITICS_PHRASES = {
    "hormuz",
    "strait of hormuz",
    "sanctioned tanker",
    "sanctions",
    "maritime blockade",
    "oil flows",
    "fuel trade restrictions",
    "oil tanker",
    "shipping insurance",
    "crude exports",
}

ENERGY_MARKET_THEME_MARKERS = {
    "ENV_OIL",
    "WB_507_ENERGY_AND_EXTRACTIVES",
    "WB_539_OIL_AND_GAS_POLICY_STRATEGY_AND_INSTITUTIONS",
    "WB_548_PPP_IN_OIL_AND_GAS",
    "WB_698_TRADE",
    "WB_2298_REFINERIES",
}

SPORTS_TERMS = {
    "fc", "goal", "goals", "match", "striker", "premier league", "nba",
    "nfl", "nhl", "cricket", "tennis", "olympics", "championship", "coach",
    "midfielder", "tournament", "season opener", "world cup",
}

ENTERTAINMENT_TERMS = {
    "celebrity", "movie", "film", "actor", "actress", "box office", "album",
    "song", "music", "streaming", "netflix", "hollywood", "trailer",
    "festival", "red carpet", "tv show",
}

LIFESTYLE_TERMS = {
    "recipe", "fashion", "beauty", "diet", "travel tips", "wedding", "style",
    "viral video", "home decor", "astrology", "horoscope", "wellness",
    "restaurant", "cake", "luxury dining",
}

LOCAL_CRIME_TERMS = {
    "charged", "arrested", "police", "sheriff", "deputy", "medicaid fraud",
    "county jail", "homicide", "shooting", "burglary", "stolen", "suspect",
    "court appearance", "local authorities",
}

PRESS_RELEASE_TERMS = {
    "press release", "globenewswire", "business wire", "pr newswire",
    "access newswire", "newsfile", "announces", "announced today", "investor brand network",
    "announces closing of", "eqs-news",
}

ARCHIVE_URL_PATTERNS = {
    "/tag/",
    "/category/",
    "/archives/",
}

PRESS_RELEASE_URL_PATTERNS = {
    "/press-release/",
    "/news-releases/",
    "/globenewswire/",
    "/tmt-newswire/",
}

KEEP_THEME_PREFIXES = {
    "ECON_", "EPU_", "WB_332_", "WB_450_", "WB_507_", "WB_508_", "WB_509_",
    "WB_698_", "WB_1104_", "WB_1921_", "WB_1973_", "WB_2936_", "WB_625_",
    "TAX_ECON_", "MARITIME", "ENV_NUCLEARPOWER",
}

MACRO_THEME_MARKERS = {
    "ECON_STOCKMARKET", "ECON_DEBT", "ECON_TAXATION", "EPU_ECONOMY",
    "EPU_POLICY", "WB_450_DEBT", "WB_1104_MACROECONOMIC_VULNERABILITY_AND_DEBT",
}

GEO_THEME_MARKERS = {
    "ARMEDCONFLICT", "MILITARY", "DRONES", "PROPAGANDA", "MARITIME_PIRACY",
    "MANMADE_DISASTER_NUCLEAR_ACCIDENT",
}

DROP_THEME_MARKERS = {
    "SOC_GENERALCRIME", "GENERAL_HEALTH", "MEDICAL", "TOURISM",
}

GENERIC_THEME_NOISE = {
    "AFFECT",
    "EDUCATION",
    "GENERAL_GOVERNMENT",
    "GENERAL_HEALTH",
    "LEADER",
    "LEGISLATION",
    "MEDIA_MSM",
    "MEDIA_SOCIAL",
    "SCIENCE",
    "SOC_INNOVATION",
    "USPEC_POLITICS_GENERAL1",
}


@dataclass
class LabelDecision:
    label: str
    confidence: float
    reasons: list[str]
    finance_cluster_score: float


def ensure_dirs() -> None:
    for path in [DATA_DIR, FEEDBACK_DIR, RESULTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def export_projection() -> None:
    query = f"""
    COPY (
        SELECT
            partition_date,
            lower(regexp_replace(regexp_extract(document_identifier, 'https?://([^/]+)', 1), '^www\\.', '')) AS source_domain,
            source_common_name,
            document_identifier,
            coalesce(
                nullif(title, ''),
                nullif(regexp_extract(metadata_json, '<PAGE_TITLE>([^<]+)</PAGE_TITLE>', 1), '')
            ) AS resolved_title,
            coalesce(summary, '') AS summary,
            substr(coalesce(text, ''), 1, 4000) AS text_excerpt,
            coalesce(v2_themes, '') AS v2_themes
        FROM read_parquet('{INPUT_PARQUET_GLOB}')
    ) TO '{RAW_EXPORT_CSV.as_posix()}' (FORMAT CSV, HEADER TRUE);
    """
    subprocess.run(["duckdb", "-c", query], check=True, capture_output=True, text=True)


def load_source_profiles() -> dict[str, dict[str, str]]:
    if not SOURCE_PROFILES_CSV.exists():
        return {}
    with SOURCE_PROFILES_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {normalize_domain(row["source_domain"]): row for row in reader}


def load_domain_scores() -> dict[str, dict[str, str]]:
    if not DOMAIN_SCORES_CSV.exists():
        return {}
    with DOMAIN_SCORES_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {normalize_domain(row["source_domain"]): row for row in reader}


def load_feedback() -> dict[str, dict[str, str]]:
    if not FEEDBACK_LABELS_CSV.exists():
        return {}
    with FEEDBACK_LABELS_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            row["document_identifier"]: row
            for row in reader
            if row.get("document_identifier") and row.get("label") in ALL_LABELS
        }


def normalize_text(*parts: str) -> str:
    text = " ".join(part for part in parts if part).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_domain(raw_domain: str) -> str:
    domain = (raw_domain or "").strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]
    if ":" in domain:
        domain = domain.split(":", 1)[0]
    return domain


def tokenize_themes(raw_themes: str) -> set[str]:
    out: set[str] = set()
    for item in raw_themes.split(";"):
        token = item.split(",", 1)[0].strip()
        if token:
            out.add(token)
    return out


def count_term_hits(text: str, terms: Iterable[str]) -> int:
    return sum(1 for term in terms if term in text)


def finance_cluster_score(text: str, themes: set[str]) -> float:
    finance_hits = count_term_hits(text, FINANCE_TERMS)
    theme_hits = sum(
        1
        for theme in themes
        if theme in MACRO_THEME_MARKERS
        or theme in GEO_THEME_MARKERS
        or any(theme.startswith(prefix) for prefix in KEEP_THEME_PREFIXES)
    )
    score = min(1.0, 0.12 * finance_hits + 0.08 * theme_hits)
    return round(score, 4)


def source_profile_flags(source_domain: str, source_profile: dict[str, str] | None) -> tuple[bool, bool, bool]:
    if source_domain in PRESS_RELEASE_DOMAINS:
        return False, False, False
    if source_domain in LOW_QUALITY_DOMAINS:
        return False, False, True
    if not source_profile:
        return False, False, False
    enough_sample = int(source_profile.get("enough_sample") or 0) == 1
    market_relevance_rate = float(source_profile.get("market_relevance_rate") or 0.0)
    industry_signal_rate = float(source_profile.get("industry_signal_rate") or 0.0)
    junk_rate = float(source_profile.get("junk_rate") or 0.0)
    trusted = enough_sample and (market_relevance_rate >= 0.45 or industry_signal_rate >= 0.35)
    industry_useful = enough_sample and (industry_signal_rate >= 0.2 or source_profile.get("source_profile") == "industry_useful")
    junk_heavy = enough_sample and junk_rate >= 0.6 and market_relevance_rate < 0.2 and industry_signal_rate < 0.15
    return trusted, industry_useful, junk_heavy


def choose_label(
    source_domain: str,
    document_identifier: str,
    title: str,
    summary: str,
    text_excerpt: str,
    themes: set[str],
    source_profile: dict[str, str] | None,
    domain_score: dict[str, str] | None,
) -> LabelDecision | None:
    url = document_identifier.lower()
    text = normalize_text(title, summary, text_excerpt, url.replace("/", " "))
    trusted_source, industry_useful_source, junk_heavy_source = source_profile_flags(source_domain, source_profile)
    effective_archetype = (domain_score or {}).get("effective_archetype", "")
    effective_score_0_10 = float((domain_score or {}).get("effective_score_0_10") or 0.0)
    cluster_score = finance_cluster_score(text, themes)
    finance_hits = count_term_hits(text, FINANCE_TERMS)
    macro_hits = count_term_hits(text, MACRO_TERMS)
    geo_hits = count_term_hits(text, GEOPOLITICS_TERMS)
    company_hits = count_term_hits(text, COMPANY_EVENT_TERMS)
    equity_hits = count_term_hits(text, EQUITY_MARKET_TERMS)
    sports_hits = count_term_hits(text, SPORTS_TERMS)
    entertainment_hits = count_term_hits(text, ENTERTAINMENT_TERMS)
    lifestyle_hits = count_term_hits(text, LIFESTYLE_TERMS)
    crime_hits = count_term_hits(text, LOCAL_CRIME_TERMS)
    press_hits = count_term_hits(text, PRESS_RELEASE_TERMS)
    macro_theme_hits = sum(1 for theme in themes if theme in MACRO_THEME_MARKERS or theme.startswith("EPU_"))
    geo_theme_hits = sum(1 for theme in themes if theme in GEO_THEME_MARKERS)
    keep_theme_hits = sum(1 for theme in themes if any(theme.startswith(prefix) for prefix in KEEP_THEME_PREFIXES))
    energy_market_theme_hits = sum(1 for theme in themes if theme in ENERGY_MARKET_THEME_MARKERS)
    drop_theme_hits = sum(1 for theme in themes if theme in DROP_THEME_MARKERS)

    reasons: list[str] = []
    text_length = len(normalize_text(title, summary, text_excerpt))
    strong_company_event_hits = count_term_hits(text, STRONG_COMPANY_EVENT_PHRASES)
    market_geopolitics_hits = count_term_hits(text, MARKET_GEOPOLITICS_PHRASES)
    ipo_market_story_hits = count_term_hits(text, IPO_MARKET_STORY_PHRASES)
    pure_ipo_market_story = (
        ("ipo" in text or "initial public offering" in text)
        and strong_company_event_hits <= 1
        and ipo_market_story_hits >= 1
        and finance_hits >= 1
        and company_hits <= 1
    )

    mirror_press_release = (
        source_domain in PRESS_RELEASE_HINT_DOMAINS
        and (
            press_hits >= 1
            or "globenewswire" in url
            or "newswire" in url
            or any(pattern in url for pattern in PRESS_RELEASE_URL_PATTERNS)
        )
    )
    if (
        source_domain in PRESS_RELEASE_DOMAINS
        or mirror_press_release
        or (
            effective_archetype == "press_release_or_mirror"
            and (
                press_hits >= 1
                or any(
                    marker in text
                    for marker in (
                        "announces",
                        "participate in",
                        "voting results",
                        "initial public offering",
                        "announces closing of",
                        "company announcement",
                        "eqs-news",
                    )
                )
            )
        )
        or press_hits >= 2
        or any(pattern in url for pattern in PRESS_RELEASE_URL_PATTERNS)
    ):
        confidence = min(0.99, 0.88 + 0.03 * press_hits)
        return LabelDecision("drop_press_release", round(confidence, 4), ["press_release_pattern"], cluster_score)

    if source_domain in LOW_QUALITY_DOMAINS or (
        junk_heavy_source
        and cluster_score < 0.1
        and finance_hits == 0
        and macro_hits == 0
        and geo_hits == 0
        and company_hits == 0
        and keep_theme_hits == 0
        and text_length < 220
    ):
        confidence = min(0.99, 0.82 + 0.08 * (1 if source_domain in LOW_QUALITY_DOMAINS else 0))
        return LabelDecision("drop_low_quality", round(confidence, 4), ["low_quality_source"], cluster_score)

    if (
        re.fullmatch(r"\d{2}/\d{2}/\d{4}\s*-\s*.+", title.strip())
        or any(pattern in url for pattern in ARCHIVE_URL_PATTERNS)
    ):
        return LabelDecision("drop_low_quality", 0.97, ["archive_or_index_page"], cluster_score)

    # Sports leakage was too noisy in manual review. Preserve ambiguous sports
    # rows for review unless the evidence is unambiguously sports-only.
    if (
        sports_hits >= 4
        and finance_hits == 0
        and macro_hits == 0
        and geo_hits == 0
        and company_hits == 0
        and keep_theme_hits == 0
        and cluster_score < 0.05
    ):
        confidence = min(0.98, 0.84 + 0.02 * sports_hits)
        return LabelDecision("drop_sports", round(confidence, 4), ["sports_terms_strict"], cluster_score)

    if entertainment_hits >= 2 and finance_hits == 0 and keep_theme_hits == 0:
        confidence = min(0.98, 0.84 + 0.03 * entertainment_hits)
        return LabelDecision("drop_entertainment", round(confidence, 4), ["entertainment_terms"], cluster_score)

    if lifestyle_hits >= 2 and finance_hits == 0 and keep_theme_hits == 0:
        confidence = min(0.98, 0.83 + 0.03 * lifestyle_hits)
        return LabelDecision("drop_lifestyle", round(confidence, 4), ["lifestyle_terms"], cluster_score)

    if (
        crime_hits >= 2
        and cluster_score < 0.15
        and finance_hits == 0
        and macro_hits == 0
        and company_hits == 0
        and not trusted_source
        and not industry_useful_source
    ):
        confidence = min(0.97, 0.82 + 0.03 * crime_hits)
        return LabelDecision("drop_local_crime", round(confidence, 4), ["local_crime_terms"], cluster_score)

    if (
        (company_hits >= 1 and strong_company_event_hits >= 1)
        or (
            strong_company_event_hits >= 1
            and cluster_score >= 0.45
            and (
                trusted_source
                or industry_useful_source
                or effective_score_0_10 >= 6.0
                or keep_theme_hits >= 2
            )
        )
        or (
            company_hits >= 2
            and (finance_hits >= 1 or equity_hits >= 1 or trusted_source or industry_useful_source or cluster_score >= 0.25)
        )
    ) and not pure_ipo_market_story and not (
        source_domain in PRESS_RELEASE_DOMAINS
        or source_domain in PRESS_RELEASE_HINT_DOMAINS
        or effective_archetype == "press_release_or_mirror"
        or press_hits >= 1
    ):
        reasons.extend(["company_event_terms"])
        if strong_company_event_hits >= 1:
            reasons.append("strong_company_event_phrase")
        confidence = min(
            0.99,
            0.79
            + 0.04 * company_hits
            + 0.03 * finance_hits
            + 0.03 * equity_hits
            + 0.03 * strong_company_event_hits
            + 0.03 * int(trusted_source or industry_useful_source),
        )
        return LabelDecision("keep_company_event", round(confidence, 4), reasons, cluster_score)

    if (
        (geo_hits + geo_theme_hits >= 2 and market_geopolitics_hits >= 1)
        or (market_geopolitics_hits >= 2 and finance_hits >= 1)
    ):
        reasons.extend(["geopolitics_terms", "market_geopolitics_phrase"])
        confidence = min(
            0.99,
            0.8
            + 0.03 * geo_hits
            + 0.02 * geo_theme_hits
            + 0.03 * market_geopolitics_hits
            + 0.03 * int(finance_hits >= 1 or macro_hits >= 1),
        )
        return LabelDecision("keep_geopolitics", round(confidence, 4), reasons, cluster_score)

    if (
        pure_ipo_market_story
        and not (
            source_domain in PRESS_RELEASE_DOMAINS
            or source_domain in PRESS_RELEASE_HINT_DOMAINS
            or effective_archetype == "press_release_or_mirror"
            or press_hits >= 1
        )
        and (
            trusted_source
            or industry_useful_source
            or effective_score_0_10 >= 6.0
            or keep_theme_hits >= 1
        )
    ):
        reasons.extend(["ipo_market_story", "finance_terms"])
        confidence = min(
            0.99,
            0.78
            + 0.03 * finance_hits
            + 0.02 * ipo_market_story_hits
            + 0.03 * int(trusted_source or industry_useful_source)
            + 0.02 * int(keep_theme_hits >= 1),
        )
        return LabelDecision("keep_finance", round(confidence, 4), reasons, cluster_score)

    if (
        energy_market_theme_hits >= 2
        and (
            market_geopolitics_hits >= 1
            or finance_hits >= 1
            or keep_theme_hits >= 2
        )
        and cluster_score <= 0.3
        and finance_hits == 0
        and company_hits == 0
        and equity_hits == 0
        and source_domain not in PRESS_RELEASE_DOMAINS
        and source_domain not in PRESS_RELEASE_HINT_DOMAINS
        and effective_archetype not in {"press_release_or_mirror", "low_quality_or_scraper"}
        and press_hits == 0
    ):
        reasons.extend(["energy_trade_themes"])
        if market_geopolitics_hits >= 1:
            reasons.append("market_geopolitics_phrase")
        confidence = min(
            0.99,
            0.76
            + 0.025 * energy_market_theme_hits
            + 0.02 * keep_theme_hits
            + 0.03 * int(finance_hits >= 1 or market_geopolitics_hits >= 1)
            + 0.02 * int(trusted_source or industry_useful_source),
        )
        label = "keep_geopolitics" if market_geopolitics_hits >= 1 else "keep_macro"
        return LabelDecision(label, round(confidence, 4), reasons, cluster_score)

    if (
        effective_archetype == "market_blog_or_stock_blurb"
        and (finance_hits + equity_hits + company_hits >= 3)
        and (finance_hits >= 1 or equity_hits >= 2 or "nasdaq" in text or "nyse" in text)
    ):
        reasons.extend(["domain_market_blog_bias", "equity_market_terms"])
        confidence = min(
            0.99,
            0.81
            + 0.025 * finance_hits
            + 0.025 * equity_hits
            + 0.02 * company_hits
            + 0.02 * int(effective_score_0_10 >= 5.0),
        )
        return LabelDecision("keep_finance", round(confidence, 4), reasons, cluster_score)

    if macro_hits + macro_theme_hits >= 3 and cluster_score >= 0.3:
        reasons.extend(["macro_terms", "macro_themes"])
        confidence = min(0.99, 0.8 + 0.03 * macro_hits + 0.03 * macro_theme_hits + 0.04 * int(trusted_source))
        return LabelDecision("keep_macro", round(confidence, 4), reasons, cluster_score)

    if geo_hits + geo_theme_hits >= 3 and (cluster_score >= 0.25 or finance_hits >= 1):
        reasons.extend(["geopolitics_terms", "geo_themes"])
        confidence = min(0.99, 0.79 + 0.03 * geo_hits + 0.03 * geo_theme_hits + 0.04 * int(trusted_source))
        return LabelDecision("keep_geopolitics", round(confidence, 4), reasons, cluster_score)

    if finance_hits + keep_theme_hits >= 4 or (cluster_score >= 0.45 and (trusted_source or industry_useful_source)):
        reasons.extend(["finance_terms", "keep_themes"])
        confidence = min(0.99, 0.8 + 0.025 * finance_hits + 0.02 * keep_theme_hits + 0.05 * int(trusted_source or industry_useful_source))
        return LabelDecision("keep_finance", round(confidence, 4), reasons, cluster_score)

    if drop_theme_hits >= 2 and cluster_score < 0.1 and junk_heavy_source and not industry_useful_source:
        confidence = min(0.95, 0.77 + 0.03 * drop_theme_hits)
        return LabelDecision("drop_low_quality", round(confidence, 4), ["drop_themes_no_finance"], cluster_score)

    return None


def filtered_themes_for_fasttext(themes: set[str]) -> list[str]:
    filtered: list[str] = []
    for theme in sorted(themes):
        if theme in GENERIC_THEME_NOISE:
            continue
        if theme in MACRO_THEME_MARKERS or theme in GEO_THEME_MARKERS:
            filtered.append(theme)
            continue
        if any(theme.startswith(prefix) for prefix in KEEP_THEME_PREFIXES):
            filtered.append(theme)
            continue
        if theme.startswith("ENV_") or theme.startswith("SLFID_") or theme.startswith("TAX_ECON_"):
            filtered.append(theme)
    return filtered[:12]


def text_for_fasttext(title: str, summary: str, text_excerpt: str, themes: set[str]) -> str:
    theme_text = " ".join(filtered_themes_for_fasttext(themes))
    natural_text = " || ".join(part for part in [title, summary, text_excerpt[:1500]] if part)
    raw = " || ".join(part for part in [natural_text, theme_text] if part)
    clean = re.sub(r"\s+", " ", raw.replace("\n", " ").replace("\r", " ")).strip()
    return clean


def record_to_train_line(label: str, text: str) -> str:
    sanitized = text.replace("__label__", "label").strip()
    return f"__label__{label} {sanitized}"


def build_corpus() -> None:
    ensure_dirs()
    if not RAW_EXPORT_CSV.exists():
        export_projection()

    source_profiles = load_source_profiles()
    domain_scores = load_domain_scores()
    feedback = load_feedback()
    rng = random.Random(20260629)
    summary_counts: Counter[str] = Counter()
    feedback_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    review_queue_rows = 0
    fieldnames = [
        "partition_date",
        "source_domain",
        "document_identifier",
        "label",
        "confidence",
        "label_source",
        "finance_cluster_score",
        "source_profile",
        "market_relevance_rate",
        "industry_signal_rate",
        "junk_rate",
        "reasons",
        "title",
        "summary",
        "text",
        "natural_text_length",
        "summary_length",
        "text_excerpt_length",
        "filtered_theme_count",
        "finance_hits",
        "macro_hits",
        "geo_hits",
        "company_hits",
        "equity_hits",
        "sports_hits",
        "entertainment_hits",
        "lifestyle_hits",
        "crime_hits",
        "press_hits",
        "keep_theme_hits",
        "macro_theme_hits",
        "geo_theme_hits",
        "drop_theme_hits",
    ]

    with (
        RAW_EXPORT_CSV.open("r", encoding="utf-8") as input_handle,
        WEAK_LABEL_CSV.open("w", newline="", encoding="utf-8") as weak_handle,
        REVIEW_QUEUE_CSV.open("w", newline="", encoding="utf-8") as review_handle,
        TRAIN_TXT.open("w", encoding="utf-8") as train_handle,
        VALID_TXT.open("w", encoding="utf-8") as valid_handle,
    ):
        reader = csv.DictReader(input_handle)
        weak_writer = csv.DictWriter(weak_handle, fieldnames=fieldnames)
        review_writer = csv.DictWriter(review_handle, fieldnames=fieldnames)
        weak_writer.writeheader()
        review_writer.writeheader()

        for row in reader:
            source_domain = normalize_domain(row["source_domain"])
            source_profile = source_profiles.get(source_domain)
            domain_score = domain_scores.get(source_domain)
            title = row["resolved_title"] or ""
            summary = row["summary"] or ""
            text_excerpt = row["text_excerpt"] or ""
            themes = tokenize_themes(row["v2_themes"] or "")
            natural_text = normalize_text(title, summary, text_excerpt)
            natural_text_length = len(natural_text)
            summary_length = len(normalize_text(summary))
            text_excerpt_length = len(normalize_text(text_excerpt))
            filtered_theme_count = len(filtered_themes_for_fasttext(themes))
            finance_hits = count_term_hits(natural_text, FINANCE_TERMS)
            macro_hits = count_term_hits(natural_text, MACRO_TERMS)
            geo_hits = count_term_hits(natural_text, GEOPOLITICS_TERMS)
            company_hits = count_term_hits(natural_text, COMPANY_EVENT_TERMS)
            equity_hits = count_term_hits(natural_text, EQUITY_MARKET_TERMS)
            sports_hits = count_term_hits(natural_text, SPORTS_TERMS)
            entertainment_hits = count_term_hits(natural_text, ENTERTAINMENT_TERMS)
            lifestyle_hits = count_term_hits(natural_text, LIFESTYLE_TERMS)
            crime_hits = count_term_hits(natural_text, LOCAL_CRIME_TERMS)
            press_hits = count_term_hits(natural_text, PRESS_RELEASE_TERMS)
            keep_theme_hits = sum(1 for theme in themes if any(theme.startswith(prefix) for prefix in KEEP_THEME_PREFIXES))
            macro_theme_hits = sum(1 for theme in themes if theme in MACRO_THEME_MARKERS or theme.startswith("EPU_"))
            geo_theme_hits = sum(1 for theme in themes if theme in GEO_THEME_MARKERS)
            drop_theme_hits = sum(1 for theme in themes if theme in DROP_THEME_MARKERS)
            fasttext_text = text_for_fasttext(title, summary, text_excerpt, themes)
            if len(fasttext_text) < 40:
                continue

            feedback_row = feedback.get(row["document_identifier"])
            if feedback_row:
                label = feedback_row["label"]
                confidence = 1.0
                reasons = ["human_feedback"]
                cluster_score = finance_cluster_score(normalize_text(title, summary, text_excerpt), themes)
                label_source = "human_feedback"
                feedback_counts[label] += 1
            else:
                decision = choose_label(
                    source_domain=source_domain,
                    document_identifier=row["document_identifier"],
                    title=title,
                    summary=summary,
                    text_excerpt=text_excerpt,
                    themes=themes,
                    source_profile=source_profile,
                    domain_score=domain_score,
                )
                if decision is None:
                    review_writer.writerow(
                        {
                            "partition_date": row["partition_date"],
                            "source_domain": source_domain,
                            "document_identifier": row["document_identifier"],
                            "label": "review",
                            "confidence": "0.0000",
                            "label_source": "review_fallback",
                            "finance_cluster_score": f"{finance_cluster_score(normalize_text(title, summary, text_excerpt), themes):.4f}",
                            "source_profile": (source_profile or {}).get("source_profile", ""),
                            "market_relevance_rate": (source_profile or {}).get("market_relevance_rate", ""),
                            "industry_signal_rate": (source_profile or {}).get("industry_signal_rate", ""),
                            "junk_rate": (source_profile or {}).get("junk_rate", ""),
                            "reasons": "ambiguous_or_industry_preserve",
                            "title": title,
                            "summary": summary[:500],
                            "text": fasttext_text[:1800],
                            "natural_text_length": str(natural_text_length),
                            "summary_length": str(summary_length),
                            "text_excerpt_length": str(text_excerpt_length),
                            "filtered_theme_count": str(filtered_theme_count),
                            "finance_hits": str(finance_hits),
                            "macro_hits": str(macro_hits),
                            "geo_hits": str(geo_hits),
                            "company_hits": str(company_hits),
                            "equity_hits": str(equity_hits),
                            "sports_hits": str(sports_hits),
                            "entertainment_hits": str(entertainment_hits),
                            "lifestyle_hits": str(lifestyle_hits),
                            "crime_hits": str(crime_hits),
                            "press_hits": str(press_hits),
                            "keep_theme_hits": str(keep_theme_hits),
                            "macro_theme_hits": str(macro_theme_hits),
                            "geo_theme_hits": str(geo_theme_hits),
                            "drop_theme_hits": str(drop_theme_hits),
                        }
                    )
                    review_queue_rows += 1
                    continue
                label = decision.label
                confidence = decision.confidence
                reasons = decision.reasons
                cluster_score = decision.finance_cluster_score
                label_source = "weak_rules"

            weak_row = {
                "partition_date": row["partition_date"],
                "source_domain": source_domain,
                "document_identifier": row["document_identifier"],
                "label": label,
                "confidence": f"{confidence:.4f}",
                "label_source": label_source,
                "finance_cluster_score": f"{cluster_score:.4f}",
                "source_profile": (source_profile or {}).get("source_profile", ""),
                "market_relevance_rate": (source_profile or {}).get("market_relevance_rate", ""),
                "industry_signal_rate": (source_profile or {}).get("industry_signal_rate", ""),
                "junk_rate": (source_profile or {}).get("junk_rate", ""),
                "reasons": "|".join(reasons),
                "title": title,
                "summary": summary[:500],
                "text": fasttext_text[:1800],
                "natural_text_length": str(natural_text_length),
                "summary_length": str(summary_length),
                "text_excerpt_length": str(text_excerpt_length),
                "filtered_theme_count": str(filtered_theme_count),
                "finance_hits": str(finance_hits),
                "macro_hits": str(macro_hits),
                "geo_hits": str(geo_hits),
                "company_hits": str(company_hits),
                "equity_hits": str(equity_hits),
                "sports_hits": str(sports_hits),
                "entertainment_hits": str(entertainment_hits),
                "lifestyle_hits": str(lifestyle_hits),
                "crime_hits": str(crime_hits),
                "press_hits": str(press_hits),
                "keep_theme_hits": str(keep_theme_hits),
                "macro_theme_hits": str(macro_theme_hits),
                "geo_theme_hits": str(geo_theme_hits),
                "drop_theme_hits": str(drop_theme_hits),
            }
            weak_writer.writerow(weak_row)
            summary_counts[label] += 1

            if confidence < 0.85:
                review_writer.writerow(weak_row)
                review_queue_rows += 1

            line = record_to_train_line(label, fasttext_text)
            if rng.random() < 0.1:
                valid_handle.write(line + "\n")
                split_counts["valid"] += 1
            else:
                train_handle.write(line + "\n")
                split_counts["train"] += 1

    summary = {
        "raw_export_csv": str(RAW_EXPORT_CSV),
        "source_profiles_csv": str(SOURCE_PROFILES_CSV),
        "weak_label_csv": str(WEAK_LABEL_CSV),
        "train_txt": str(TRAIN_TXT),
        "valid_txt": str(VALID_TXT),
        "review_queue_csv": str(REVIEW_QUEUE_CSV),
        "labeled_rows": sum(summary_counts.values()),
        "label_counts": dict(sorted(summary_counts.items())),
        "feedback_override_counts": dict(sorted(feedback_counts.items())),
        "split_counts": dict(split_counts),
        "review_queue_rows": review_queue_rows,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    build_corpus()
