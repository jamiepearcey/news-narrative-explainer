#!/usr/bin/env python3
"""Build a domain-level audit with proposed source scores from corpus samples."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
RESULTS_DIR = WORK_DIR / "results"
V3_RESULTS_DIR = PROJECT_ROOT / "v3" / "results"
SCORED_WEAK_LABELS_CSV = RESULTS_DIR / "scored_weak_labels.csv"
DOMAIN_STORY_SAMPLES_CSV = RESULTS_DIR / "domain_story_samples.csv"
SOURCE_PROFILES_CSV = RESULTS_DIR / "source_probe_profiles.csv"
EXTERNAL_QUALITY_CSV = V3_RESULTS_DIR / "source_quality_external.csv"
OUTPUT_CSV = RESULTS_DIR / "domain_score_audit.csv"
SUMMARY_JSON = RESULTS_DIR / "domain_score_audit_summary.json"

LOW_QUALITY_DOMAINS = {
    "zazoom.it",
    "river949.com.au",
    "drudge.com",
}

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

HIGH_VALUE_TERMS = (
    "oil",
    "crude",
    "gas",
    "lng",
    "shipping",
    "freight",
    "yield",
    "inflation",
    "tariff",
    "sanction",
    "unemployment",
    "rates",
    "pipeline",
    "refinery",
    "diesel",
    "biorefinery",
    "earnings",
    "merger",
    "ipo",
    "exports",
    "imports",
    "nuclear",
    "power",
    "energy",
    "data center",
)

LOW_VALUE_TERMS = (
    "gossip",
    "wedding",
    "matrimonio",
    "molest",
    "crime",
    "podcast",
    "celebrity",
    "festival",
    "mural",
    "blaze",
)

PREMIUM_REFERENCE_DOMAINS = {
    "reuters.com",
    "bloomberg.com",
    "apnews.com",
    "ft.com",
    "wsj.com",
    "wsj.net",
}

OFFICIAL_REFERENCE_DOMAIN_SUFFIXES = (
    ".gov",
    ".gov.uk",
    ".europa.eu",
)

STOCK_BLOG_TERMS = (
    "issues earnings results",
    "releases earnings guidance",
    "sells",
    "shares of stock",
    "critical analysis",
    "to watch now",
    "price target",
    "analyst",
    "nasdaq:",
    "nyse:",
    "otcmkts:",
)

POLITICS_TERMS = (
    "president",
    "minister",
    "party",
    "leadership",
    "government",
    "council",
    "political",
)

BUSINESS_DOMAIN_HINTS = (
    "finance",
    "finanz",
    "finanzen",
    "financial",
    "market",
    "markets",
    "money",
    "business",
    "economy",
    "economic",
    "economics",
    "borsa",
    "bourse",
    "boerse",
    "bolsa",
    "capital",
    "stocks",
    "equity",
    "invest",
)

BUSINESS_SAMPLE_TERMS = (
    "stock",
    "stocks",
    "share",
    "shares",
    "bond",
    "bonds",
    "yield",
    "forex",
    "currency",
    "rates",
    "inflation",
    "gdp",
    "ipo",
    "market",
    "markets",
    "economy",
    "exports",
    "imports",
    "crude",
    "oil",
    "gold",
    "silver",
    "treasury",
)

ARCHETYPE_SCORE_BANDS = {
    "premium_primary": (9.0, 10.0),
    "specialist_trade": (7.0, 9.0),
    "mainstream_business_or_general": (5.5, 8.5),
    "market_blog_or_stock_blurb": (2.5, 5.0),
    "mixed_politics_or_general_aggregator": (2.0, 4.5),
    "press_release_or_mirror": (0.5, 2.0),
    "low_quality_or_scraper": (0.0, 1.0),
}

AGGREGATOR_NETWORK_SUFFIXES = (
    "news.net",
    "sun.com",
    "star.com",
    "globe.com",
    "leader.com",
    "post.com",
    "herald.com",
)

SYNTHETIC_GEO_NEWS_SUFFIXES = (
    "nationalnews",
    "national",
    "news",
    "times",
    "telegraph",
    "statesman",
    "echo",
    "standard",
    "mirror",
    "source",
    "guardian",
    "independent",
    "globe",
    "herald",
    "leader",
    "sun",
    "star",
)


def normalize_domain(raw: str) -> str:
    value = (raw or "").strip().lower()
    if value.startswith("www."):
        value = value[4:]
    if ":" in value:
        value = value.split(":", 1)[0]
    return value


def normalize_title(raw: str) -> str:
    value = (raw or "").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def canonicalize_title(raw: str) -> str:
    value = normalize_title(raw).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def title_value_score(title: str) -> int:
    text = title.lower()
    score = sum(1 for term in HIGH_VALUE_TERMS if term in text)
    score -= sum(1 for term in LOW_VALUE_TERMS if term in text)
    return score


def is_synthetic_geo_news_domain(domain: str) -> bool:
    host = domain.split(".", 1)[0]
    if host.startswith(("the", "my", "our")):
        return False
    if "-" in host or len(host) < 10:
        return False
    return any(host.endswith(suffix) and host != suffix for suffix in SYNTHETIC_GEO_NEWS_SUFFIXES)


def classify_archetype(
    domain: str,
    profile: dict[str, str] | None,
    external: dict[str, str] | None,
    counters: Counter[str],
    sample_titles: list[str],
    sample_urls: list[str],
) -> tuple[str, str]:
    sample_text = " || ".join(title.lower() for title in sample_titles)
    external_score = float((external or {}).get("final_source_quality_score") or 0.0) * 10.0
    market_relevance = float((profile or {}).get("market_relevance_rate") or 0.0)
    industry_signal = float((profile or {}).get("industry_signal_rate") or 0.0)
    junk_rate = float((profile or {}).get("junk_rate") or 0.0)
    source_profile = (profile or {}).get("source_profile") or ""
    rows = max(1, counters["rows"])
    hard_negative_rate = counters["hard_negative_rows"] / rows

    stock_blog_hits = sum(1 for term in STOCK_BLOG_TERMS if term in sample_text)
    politics_hits = sum(1 for term in POLITICS_TERMS if term in sample_text)
    business_sample_hits = sum(1 for term in BUSINESS_SAMPLE_TERMS if term in sample_text)
    high_value_score = sum(max(0, title_value_score(title)) for title in sample_titles)
    low_value_score = sum(1 for title in sample_titles if title_value_score(title) < 0)
    market_sample_count = sum(1 for title in sample_titles if title_value_score(title) >= 1)
    politics_sample_count = sum(1 for title in sample_titles if sum(1 for term in POLITICS_TERMS if term in title.lower()) >= 1)
    business_domain_hint = any(term in domain for term in BUSINESS_DOMAIN_HINTS)
    is_official_reference = domain.endswith(OFFICIAL_REFERENCE_DOMAIN_SUFFIXES) or domain in {
        "state.gov",
        "federalregister.gov",
        "ecb.europa.eu",
        "federalreserve.gov",
        "imf.org",
        "worldbank.org",
    }
    is_aggregator_network = domain.endswith(AGGREGATOR_NETWORK_SUFFIXES) or domain.endswith(".wn.com")
    canonical_titles = [canonicalize_title(title) for title in sample_titles if canonicalize_title(title)]
    unique_canonical_titles = set(canonical_titles)
    duplicate_sample_titles = len(canonical_titles) >= 3 and len(unique_canonical_titles) <= 2
    synthetic_geo_news = is_synthetic_geo_news_domain(domain)
    synthetic_news_id_urls = sum(
        1 for url in sample_urls if re.search(r"/news/\d+$", (url or "").rstrip("/"))
    )

    if domain in LOW_QUALITY_DOMAINS:
        return "low_quality_or_scraper", "explicit_low_quality_domain"
    if domain in PRESS_RELEASE_DOMAINS:
        return "press_release_or_mirror", "press_release_wire"
    if domain in PRESS_RELEASE_HINT_DOMAINS:
        return "press_release_or_mirror", "press_release_mirror_or_hint"
    if domain in PREMIUM_REFERENCE_DOMAINS or is_official_reference:
        return "premium_primary", "premium_reference_domain_or_external"
    if is_aggregator_network and external_score < 6.0:
        return "mixed_politics_or_general_aggregator", "aggregator_network_domain"
    if synthetic_geo_news and synthetic_news_id_urls >= 2 and external_score < 6.0:
        return "mixed_politics_or_general_aggregator", "synthetic_geo_news_id_url_pattern"
    if synthetic_geo_news and duplicate_sample_titles and external_score < 6.0:
        return "mixed_politics_or_general_aggregator", "synthetic_geo_news_with_duplicate_samples"
    if synthetic_geo_news and external_score == 0.0 and source_profile == "market_relevant":
        return "mixed_politics_or_general_aggregator", "synthetic_geo_news_market_profile"
    if junk_rate >= 0.5 or low_value_score >= 2:
        return "low_quality_or_scraper", "junk_or_low_value_samples"
    if stock_blog_hits >= 3 and industry_signal < 0.3 and external_score < 6.0:
        return "market_blog_or_stock_blurb", "stock_blurb_sample_pattern"
    if politics_sample_count >= 2 and market_sample_count <= 1:
        return "mixed_politics_or_general_aggregator", "politics_dominant_sample_mix"
    if politics_hits >= 3 and high_value_score <= 2 and market_relevance < 0.65:
        return "mixed_politics_or_general_aggregator", "politics_heavy_samples"
    if source_profile == "industry_useful" and industry_signal >= 0.45 and high_value_score >= 4:
        return "specialist_trade", "industry_useful_with_relevant_samples"
    if market_sample_count == 1 and politics_sample_count >= 1 and market_relevance < 0.75:
        return "mixed_politics_or_general_aggregator", "single_relevant_sample_only"
    if market_sample_count <= 1 and external_score == 0.0 and source_profile == "market_relevant":
        return "mixed_politics_or_general_aggregator", "weak_sample_mix_vs_profile"
    if external_score >= 6.0 and market_sample_count >= 1 and junk_rate < 0.35:
        return "mainstream_business_or_general", "external_quality_with_market_sample"
    if business_domain_hint and market_sample_count >= 1 and junk_rate < 0.35 and politics_sample_count <= 1:
        return "mainstream_business_or_general", "business_domain_with_market_sample"
    if business_sample_hits >= 3 and market_relevance >= 0.45 and junk_rate < 0.25:
        return "mainstream_business_or_general", "business_sample_pattern"
    if market_relevance >= 0.7 and high_value_score >= 4 and hard_negative_rate < 0.08:
        return "mainstream_business_or_general", "high_market_relevance_with_signal"
    if source_profile == "industry_useful" and market_sample_count >= 2:
        return "specialist_trade", "industry_useful_profile"
    if source_profile == "market_relevant":
        return "mainstream_business_or_general", "market_relevant_profile"
    return "mixed_politics_or_general_aggregator", "fallback_mixed_domain"


def load_source_profiles() -> dict[str, dict[str, str]]:
    if not SOURCE_PROFILES_CSV.exists():
        return {}
    with SOURCE_PROFILES_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {normalize_domain(row["source_domain"]): row for row in reader}


def load_external_quality() -> dict[str, dict[str, str]]:
    if not EXTERNAL_QUALITY_CSV.exists():
        return {}
    with EXTERNAL_QUALITY_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {normalize_domain(row["source_domain"]): row for row in reader}


def proposed_score_0_10(
    domain: str,
    profile: dict[str, str] | None,
    external: dict[str, str] | None,
    counters: Counter[str],
    sample_titles: list[str],
    sample_urls: list[str],
) -> tuple[float, str]:
    archetype, archetype_basis = classify_archetype(
        domain,
        profile,
        external,
        counters,
        sample_titles,
        sample_urls,
    )
    external_score = float((external or {}).get("final_source_quality_score") or 0.0) * 10.0
    market_relevance = float((profile or {}).get("market_relevance_rate") or 0.0)
    industry_signal = float((profile or {}).get("industry_signal_rate") or 0.0)
    junk_rate = float((profile or {}).get("junk_rate") or 0.0)
    novelty_rate = float((profile or {}).get("novelty_rate") or 0.0)
    source_profile = (profile or {}).get("source_profile") or ""
    business_domain_hint = any(term in domain for term in BUSINESS_DOMAIN_HINTS)
    market_sample_count = sum(1 for title in sample_titles if title_value_score(title) >= 1)
    politics_sample_count = sum(
        1
        for title in sample_titles
        if sum(1 for term in POLITICS_TERMS if term in title.lower()) >= 1
    )

    total_rows = max(1, counters["rows"])
    keep_rate = counters["keep_rows"] / total_rows
    drop_rate = counters["drop_rows"] / total_rows
    review_rate = counters["review_rows"] / total_rows
    hard_negative_rate = counters["hard_negative_rows"] / total_rows
    title_signal_score = 0.0
    if sample_titles:
        title_signal_score = sum(title_value_score(title) for title in sample_titles) / len(sample_titles)
    band_floor, band_ceiling = ARCHETYPE_SCORE_BANDS[archetype]
    band_mid = (band_floor + band_ceiling) / 2.0
    corpus_adjustment = (
        (1.8 * market_relevance)
        + (1.2 * industry_signal)
        + (0.6 * novelty_rate)
        + (0.8 * keep_rate)
        - (0.9 * drop_rate)
        - (1.0 * junk_rate)
        - (0.9 * hard_negative_rate)
        + min(0.8, 0.2 * title_signal_score)
    )
    score = band_mid + corpus_adjustment - 1.0
    if external_score > 0:
        score = (0.7 * score) + (0.3 * external_score)
    if review_rate > 0.75 and keep_rate < 0.1:
        score -= 0.6
        archetype_basis += "|mostly_ambiguous"
    if counters["sample_low_value_titles"] >= 2:
        score -= 0.8
        archetype_basis += "|sample_noise"
    if counters["sample_high_value_titles"] >= 2:
        score += 0.4
        archetype_basis += "|sample_market_signal"
    if (
        archetype == "mainstream_business_or_general"
        and (
            "business_domain_with_market_sample" in archetype_basis
            or "business_sample_pattern" in archetype_basis
        )
        and external_score < 6.0
        and source_profile != "industry_useful"
    ):
        score = min(score, 6.5)
        archetype_basis += "|capped_auto_business_promotion"
    if (
        archetype == "mainstream_business_or_general"
        and "market_relevant_profile" in archetype_basis
        and external_score < 6.0
        and market_relevance < 0.7
    ):
        score = min(score, 6.5)
        archetype_basis += "|capped_generic_market_profile"
    if (
        archetype == "mainstream_business_or_general"
        and "external_quality_with_market_sample" in archetype_basis
        and source_profile in {"mixed_or_sparse", ""}
        and not business_domain_hint
        and market_sample_count <= 1
    ):
        score = min(score, 6.75)
        archetype_basis += "|capped_sparse_external_match"
    if (
        archetype == "mainstream_business_or_general"
        and "external_quality_with_market_sample" in archetype_basis
        and source_profile in {"mixed_or_sparse", ""}
        and politics_sample_count >= 2
    ):
        score = min(score, 6.5)
        archetype_basis += "|capped_sparse_politics_mix"
    if (
        archetype == "mainstream_business_or_general"
        and "market_relevant_profile" in archetype_basis
        and external_score == 0.0
        and not business_domain_hint
        and politics_sample_count >= 2
    ):
        score = min(score, 6.5)
        archetype_basis += "|capped_politics_heavy_market_profile"
    if (
        archetype == "mainstream_business_or_general"
        and "external_quality_with_market_sample" in archetype_basis
        and source_profile == "mixed_or_sparse"
        and market_sample_count <= 1
        and politics_sample_count >= 2
    ):
        score = min(score, 6.5)
        archetype_basis += "|capped_geopolitics_over_business"
    score = max(band_floor, min(band_ceiling, round(score, 2)))
    return score, f"{archetype}|{archetype_basis}"


def main() -> None:
    source_profiles = load_source_profiles()
    external_quality = load_external_quality()

    counters_by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    sample_titles_by_domain: dict[str, list[tuple[str, str]]] = defaultdict(list)

    with SCORED_WEAK_LABELS_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            domain = normalize_domain(row.get("source_domain") or "")
            counters = counters_by_domain[domain]
            counters["rows"] += 1
            label = row["label"]
            if label.startswith("keep_"):
                counters["keep_rows"] += 1
            elif label.startswith("drop_"):
                counters["drop_rows"] += 1
            elif label == "review":
                counters["review_rows"] += 1

            predicted_label = row.get("predicted_label") or ""
            decision_band = row.get("decision_band") or ""
            if label.startswith("drop_") and predicted_label.startswith("keep_") and decision_band != "review":
                counters["hard_negative_rows"] += 1

    if DOMAIN_STORY_SAMPLES_CSV.exists():
        with DOMAIN_STORY_SAMPLES_CSV.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                domain = normalize_domain(row.get("source_domain") or "")
                title = normalize_title(row.get("title") or "")
                url = row.get("document_identifier") or ""
                if not domain or not title:
                    continue
                sample_titles_by_domain[domain].append((title, url))
                counters = counters_by_domain[domain]
                title_score = title_value_score(title)
                if title_score > 0:
                    counters["sample_high_value_titles"] += 1
                if title_score < 0:
                    counters["sample_low_value_titles"] += 1

    fieldnames = [
        "source_domain",
        "rows",
        "keep_rows",
        "drop_rows",
        "review_rows",
        "hard_negative_rows",
        "source_profile",
        "market_relevance_rate",
        "industry_signal_rate",
        "junk_rate",
        "novelty_rate",
        "external_status",
        "external_score_0_10",
        "current_external_quality_score_0_1",
        "proposed_archetype",
        "proposed_score_0_10",
        "proposed_score_0_1",
        "proposal_basis",
        "sample_title_1",
        "sample_url_1",
        "sample_title_2",
        "sample_url_2",
        "sample_title_3",
        "sample_url_3",
    ]

    rows_out: list[dict[str, str]] = []
    for domain, counters in counters_by_domain.items():
        profile = source_profiles.get(domain)
        external = external_quality.get(domain)
        chosen = sample_titles_by_domain.get(domain, [])[:3]

        proposed_score, basis = proposed_score_0_10(
            domain=domain,
            profile=profile,
            external=external,
            counters=counters,
            sample_titles=[title for title, _url in chosen],
            sample_urls=[url for _title, url in chosen],
        )
        external_quality_score_0_1 = float((external or {}).get("final_source_quality_score") or 0.0)
        row = {
            "source_domain": domain,
            "rows": str(counters["rows"]),
            "keep_rows": str(counters["keep_rows"]),
            "drop_rows": str(counters["drop_rows"]),
            "review_rows": str(counters["review_rows"]),
            "hard_negative_rows": str(counters["hard_negative_rows"]),
            "source_profile": (profile or {}).get("source_profile", ""),
            "market_relevance_rate": (profile or {}).get("market_relevance_rate", ""),
            "industry_signal_rate": (profile or {}).get("industry_signal_rate", ""),
            "junk_rate": (profile or {}).get("junk_rate", ""),
            "novelty_rate": (profile or {}).get("novelty_rate", ""),
            "external_status": (external or {}).get("external_status", ""),
            "external_score_0_10": f"{external_quality_score_0_1 * 10.0:.2f}",
            "current_external_quality_score_0_1": f"{external_quality_score_0_1:.4f}",
            "proposed_archetype": basis.split("|", 1)[0],
            "proposed_score_0_10": f"{proposed_score:.2f}",
            "proposed_score_0_1": f"{proposed_score / 10.0:.4f}",
            "proposal_basis": basis,
            "sample_title_1": chosen[0][0] if len(chosen) > 0 else "",
            "sample_url_1": chosen[0][1] if len(chosen) > 0 else "",
            "sample_title_2": chosen[1][0] if len(chosen) > 1 else "",
            "sample_url_2": chosen[1][1] if len(chosen) > 1 else "",
            "sample_title_3": chosen[2][0] if len(chosen) > 2 else "",
            "sample_url_3": chosen[2][1] if len(chosen) > 2 else "",
        }
        rows_out.append(row)

    rows_out.sort(
        key=lambda row: (
            -int(row["rows"]),
            -float(row["proposed_score_0_10"]),
            row["source_domain"],
        )
    )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {
        "output_csv": str(OUTPUT_CSV),
        "rows": len(rows_out),
        "top_domains_by_rows": [
            {
                "source_domain": row["source_domain"],
                "rows": int(row["rows"]),
                "proposed_score_0_10": float(row["proposed_score_0_10"]),
                "proposal_basis": row["proposal_basis"],
            }
            for row in rows_out[:20]
        ],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
