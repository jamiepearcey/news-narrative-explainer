#!/usr/bin/env python3
"""Enrich source quality inventory with external MBFC and Ad Fontes data."""

from __future__ import annotations

import csv
import json
import re
import threading
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests


PROJECT_ROOT = Path("/Users/jamiepearcey/projects")
RESULTS_DIR = PROJECT_ROOT / "research/news-narrative-explainer/v3/results"
INPUT_CSV = RESULTS_DIR / "source_score_inventory.csv"
OUTPUT_CSV = RESULTS_DIR / "source_quality_external.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "source_quality_external_summary.json"
ERROR_LOG = RESULTS_DIR / "source_quality_external_errors.json"
EVENT_LEDGER = PROJECT_ROOT / "research/news-narrative-explainer/data/source_quality_events.jsonl"

MBFC_SITEMAP = "https://mediabiasfactcheck.com/page-sitemap.xml"
ADFONTES_SITEMAP = "https://adfontesmedia.com/post-sitemap.xml"
ALLSIDES_SITEMAP = "https://www.allsides.com/sitemap.xml"

USER_AGENT = "Mozilla/5.0 (compatible; source-quality-builder/1.0)"
TIMEOUT = 20
MAX_WORKERS = 12
TOP_DOMAIN_LIMIT = 150
MAX_MBFC_URLS = 180
MAX_ADFONTES_URLS = 180

MULTIPART_SUFFIXES = {
    "co.uk",
    "com.au",
    "com.br",
    "com.cn",
    "com.tw",
    "co.jp",
    "co.in",
    "co.nz",
    "org.uk",
    "gov.uk",
    "europa.eu",
}

LOW_VALUE_LINK_HOSTS = {
    "adfontesmedia.com",
    "mediabiasfactcheck.com",
    "facebook.com",
    "x.com",
    "twitter.com",
    "youtube.com",
    "linkedin.com",
    "instagram.com",
    "wikipedia.org",
    "wordpress.org",
}

DOMAIN_ALIASES = {
    "wsj.com": ["wall-street-journal", "wsj"],
    "ft.com": ["financial-times", "ft"],
    "apnews.com": ["associated-press", "ap-news", "apnews"],
    "nytimes.com": ["new-york-times", "nyt"],
    "abcnews.go.com": ["abc-news", "abcnews"],
    "foxnews.com": ["fox-news", "foxnews"],
    "npr.org": ["npr"],
    "cnn.com": ["cnn"],
    "bbc.com": ["bbc", "bbc-news"],
    "bbc.co.uk": ["bbc", "bbc-news"],
    "bloomberg.com": ["bloomberg"],
    "reuters.com": ["reuters"],
    "cnbc.com": ["cnbc"],
    "marketwatch.com": ["marketwatch"],
    "barrons.com": ["barrons"],
    "seekingalpha.com": ["seeking-alpha", "seekingalpha"],
    "investopedia.com": ["investopedia"],
    "finance.yahoo.com": ["yahoo", "yahoo-finance"],
    "yahoo.com": ["yahoo"],
    "theglobeandmail.com": ["globe-and-mail", "globeandmail"],
    "business-standard.com": ["business-standard"],
    "channelnewsasia.com": ["channel-news-asia", "cna"],
    "kitco.com": ["kitco"],
    "oilprice.com": ["oilprice"],
    "argusmedia.com": ["argus-media", "argus"],
}

MBFC_FACTUAL_SCORES = {
    "very high": 1.0,
    "high": 0.9,
    "mostly factual": 0.75,
    "mixed": 0.5,
    "low": 0.25,
    "very low": 0.1,
    "questionable": 0.05,
    "satire": 0.0,
}

REFERENCE_SIGNAL_WEIGHTS = {
    "llm_judged_article_useful": 0.2,
    "frequently_cited_by_high_quality_sources": 0.5,
    "often_contradicted_by_later_reporting": -1.0,
    "user_selected_as_evidence": 0.3,
    "user_ignored_or_dismissed": -0.1,
    "original_reporting": 0.5,
    "duplicate_of_another_article": -0.3,
    "produced_hallucination_in_summary": -1.0,
}

SIGNAL_COLUMN_NAMES = {
    "llm_judged_article_useful": "signal_useful_count",
    "frequently_cited_by_high_quality_sources": "signal_cited_by_hq_count",
    "often_contradicted_by_later_reporting": "signal_contradicted_count",
    "user_selected_as_evidence": "signal_user_selected_count",
    "user_ignored_or_dismissed": "signal_user_dismissed_count",
    "original_reporting": "signal_original_reporting_count",
    "duplicate_of_another_article": "signal_duplicate_count",
    "produced_hallucination_in_summary": "signal_hallucination_count",
}

TIER_10_DOMAINS = {
    "reuters.com",
    "bloomberg.com",
    "apnews.com",
    "ft.com",
    "wsj.com",
    "federalreserve.gov",
    "treasury.gov",
    "sec.gov",
    "ecb.europa.eu",
    "bankofengland.co.uk",
    "opec.org",
    "iea.org",
    "imf.org",
    "worldbank.org",
}

TIER_89_DOMAINS = {
    "cnbc.com",
    "marketwatch.com",
    "morningstar.com",
    "nikkei.com",
    "economist.com",
    "bbc.com",
    "bbc.co.uk",
    "abcnews.go.com",
    "cnn.com",
    "npr.org",
}

TIER_57_DOMAINS = {
    "argusmedia.com",
    "oilprice.com",
    "kitco.com",
    "rigzone.com",
    "worldoil.com",
    "shipandbunker.com",
    "gcaptain.com",
    "hellenicshippingnews.com",
    "theglobeandmail.com",
    "financialpost.com",
    "livemint.com",
    "business-standard.com",
    "channelnewsasia.com",
    "moneycontrol.com",
    "benzinga.com",
    "seekingalpha.com",
    "nasdaq.com",
    "bullionvault.com",
    "afr.com",
    "borsaitaliana.it",
}

TIER_14_DOMAINS = {
    "prnewswire.com",
    "openpr.com",
    "financialcontent.com",
    "tickerreport.com",
}


@dataclass
class MbfcRecord:
    source_domain: str
    root_domain: str
    page_url: str
    bias_rating: str | None
    factual_reporting: str | None
    credibility: str | None
    media_type: str | None
    traffic_popularity: str | None


@dataclass
class AdFontesRecord:
    source_domain: str
    root_domain: str
    page_url: str
    reliability: float | None
    bias: float | None


@dataclass
class SourceQualityEvent:
    source_domain: str
    signal: str
    count: int


SCRAPE_ERRORS: list[dict[str, str]] = []


def normalize_host(host: str) -> str:
    return host.lower().strip().removeprefix("www.")


def root_domain(host: str) -> str:
    host = normalize_host(host)
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    last_two = ".".join(parts[-2:])
    last_three = ".".join(parts[-3:])
    if last_two in MULTIPART_SUFFIXES and len(parts) >= 3:
        return ".".join(parts[-3:])
    if last_three in MULTIPART_SUFFIXES and len(parts) >= 4:
        return ".".join(parts[-4:])
    return ".".join(parts[-2:])


def domain_from_url(url: str) -> str | None:
    try:
        host = urlparse(url).hostname
    except ValueError:
        return None
    return normalize_host(host) if host else None


def read_inventory() -> list[dict[str, str]]:
    with INPUT_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    for row in rows:
        row["root_domain"] = root_domain(row["source_domain"])
    return rows


def load_source_quality_events() -> dict[str, Counter[str]]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    if not EVENT_LEDGER.exists():
        return counters
    with EVENT_LEDGER.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                record_error("event_ledger_parse", f"{EVENT_LEDGER}:{line_number}", error)
                continue
            source_domain = normalize_host(str(payload.get("source_domain", "")))
            signal = str(payload.get("signal", "")).strip()
            if not source_domain or signal not in REFERENCE_SIGNAL_WEIGHTS:
                continue
            count = int(payload.get("count", 1) or 1)
            counters[source_domain][signal] += count
    return counters


def slugify_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def candidate_slug_tokens(inventory: list[dict[str, str]]) -> set[str]:
    rows = sorted(inventory, key=lambda row: int(row["row_count"]), reverse=True)[:TOP_DOMAIN_LIMIT]
    tokens: set[str] = set()
    for row in rows:
        domain = row["source_domain"]
        root = row["root_domain"]
        base = root.split(".")[0]
        tokens.add(slugify_token(base))
        tokens.add(slugify_token(domain.split(".")[0]))
        for alias in DOMAIN_ALIASES.get(domain, []):
            tokens.add(alias)
        for alias in DOMAIN_ALIASES.get(root, []):
            tokens.add(alias)
    return {token for token in tokens if token and len(token) >= 2}


def url_matches_tokens(url: str, tokens: set[str]) -> bool:
    slug = url.rstrip("/").split("/")[-1].lower()
    return any(token in slug for token in tokens)


def fetch_xml_urls(url: str) -> list[str]:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [el.text for el in root.findall(".//sm:loc", ns) if el.text]


def fetch_text(url: str) -> str:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    response.raise_for_status()
    return response.text


def record_error(stage: str, url: str, error: Exception) -> None:
    SCRAPE_ERRORS.append({"stage": stage, "url": url, "error": repr(error)})


def scrape_mbfc_page(url: str) -> MbfcRecord | None:
    try:
        html = fetch_text(url)
    except Exception as error:
        record_error("mbfc_fetch", url, error)
        return None
    source_match = re.search(r'Source:\s*<a href="([^"]+)"', html, re.I)
    source_url = source_match.group(1) if source_match else None
    source_domain = domain_from_url(source_url) if source_url else None
    if not source_domain:
        return None
    bias_match = re.search(r'Bias Rating","value":"([^"]+)"', html, re.I)
    factual_match = re.search(r'Factual Reporting","value":"([^"]+)"', html, re.I)
    credibility_match = re.search(
        r"MBFC Credibility Rating:\s*<span[^>]*><strong>([^<]+)</strong>",
        html,
        re.I,
    )
    media_type_match = re.search(r"Media Type:\s*<strong>([^<]+)", html, re.I)
    traffic_match = re.search(r"Traffic/Popularity:\s*<strong>([^<]+)", html, re.I)
    return MbfcRecord(
        source_domain=source_domain,
        root_domain=root_domain(source_domain),
        page_url=url,
        bias_rating=bias_match.group(1).strip() if bias_match else None,
        factual_reporting=factual_match.group(1).strip() if factual_match else None,
        credibility=credibility_match.group(1).strip() if credibility_match else None,
        media_type=media_type_match.group(1).strip() if media_type_match else None,
        traffic_popularity=traffic_match.group(1).strip() if traffic_match else None,
    )


def candidate_domains_from_html(html: str) -> list[str]:
    counts: Counter[str] = Counter()
    for href in re.findall(r'href="(https?://[^"]+)"', html, re.I):
        host = domain_from_url(href)
        if not host:
            continue
        if host in LOW_VALUE_LINK_HOSTS:
            continue
        if any(host.endswith(f".{base}") for base in LOW_VALUE_LINK_HOSTS):
            continue
        counts[host] += 1
    return [host for host, _count in counts.most_common()]


def scrape_adfontes_page(url: str, known_domains: set[str], known_roots: set[str]) -> AdFontesRecord | None:
    try:
        html = fetch_text(url)
    except Exception as error:
        record_error("adfontes_fetch", url, error)
        return None
    reliability_match = re.search(r"<strong>Reliability:\s*([0-9.]+)</strong>", html, re.I)
    bias_match = re.search(r"<strong>Bias:\s*(-?[0-9.]+)</strong>", html, re.I)
    domains = candidate_domains_from_html(html)
    chosen_domain = None
    for domain in domains:
        if domain in known_domains or root_domain(domain) in known_roots:
            chosen_domain = domain
            break
    if not chosen_domain and domains:
        chosen_domain = domains[0]
    if not chosen_domain:
        return None
    return AdFontesRecord(
        source_domain=chosen_domain,
        root_domain=root_domain(chosen_domain),
        page_url=url,
        reliability=float(reliability_match.group(1)) if reliability_match else None,
        bias=float(bias_match.group(1)) if bias_match else None,
    )


def scrape_allsides_candidates() -> dict[str, str]:
    urls = fetch_xml_urls(ALLSIDES_SITEMAP)
    out: dict[str, str] = {}
    for url in urls:
        if "/news-source/" not in url:
            continue
        slug = url.rstrip("/").split("/")[-1]
        out[slug] = url
    return out


def factual_score(value: str | None) -> float | None:
    if not value:
        return None
    return MBFC_FACTUAL_SCORES.get(value.strip().lower())


def adfontes_quality_score(reliability: float | None) -> float | None:
    if reliability is None:
        return None
    return max(0.0, min(1.0, reliability / 64.0))


def tier_assignment(source_domain: str, source_type: str) -> tuple[str, float, str]:
    domain = normalize_host(source_domain)
    if domain in TIER_10_DOMAINS or domain.endswith(".gov"):
        return ("tier_10", 1.0, "explicit_tier_10_domain")
    if domain in TIER_89_DOMAINS:
        return ("tier_8_9", 0.88, "explicit_tier_8_9_domain")
    if domain in TIER_57_DOMAINS:
        return ("tier_5_7", 0.65, "explicit_tier_5_7_domain")
    if domain in TIER_14_DOMAINS:
        return ("tier_1_4", 0.2, "explicit_tier_1_4_domain")
    if source_type == "market_wrap":
        return ("tier_8_9", 0.82, "source_type_market_wrap")
    if source_type == "commodity_specialist":
        return ("tier_5_7", 0.68, "source_type_commodity_specialist")
    if source_type == "company_specific":
        return ("tier_5_7", 0.55, "source_type_company_specific")
    return ("tier_0_unknown", 0.0, "unknown_domain_fallback")


def external_reference_score(
    static_prior: float,
    mbfc_score: float | None,
    adfontes_score: float | None,
) -> tuple[float, str]:
    if mbfc_score is not None and adfontes_score is not None:
        return (0.6 * mbfc_score + 0.4 * adfontes_score, "external_mbfc+adfontes")
    if mbfc_score is not None:
        return (mbfc_score, "external_mbfc_only")
    if adfontes_score is not None:
        return (adfontes_score, "external_adfontes_only")
    return (static_prior, "static_prior_only")


def dynamic_signal_features(event_counts: Counter[str]) -> tuple[dict[str, int], float, int, int, int]:
    signal_columns = {
        column_name: int(event_counts.get(signal, 0))
        for signal, column_name in SIGNAL_COLUMN_NAMES.items()
    }
    positive = 0
    negative = 0
    total = 0
    reference_adjustment = 0.0
    for signal, count in event_counts.items():
        if signal not in REFERENCE_SIGNAL_WEIGHTS:
            continue
        total += count
        if REFERENCE_SIGNAL_WEIGHTS[signal] >= 0:
            positive += count
        else:
            negative += count
        reference_adjustment += REFERENCE_SIGNAL_WEIGHTS[signal] * count
    return signal_columns, total, positive, negative, round(reference_adjustment, 4)


def final_quality_score(
    external_score: float,
    dynamic_adjustment: float,
) -> float:
    # Keep the default score usable today, but persist raw counts so this can
    # be rebalanced later without rebuilding the source universe.
    return max(0.0, min(1.1, external_score + (0.05 * dynamic_adjustment)))


def build_external_maps(inventory: list[dict[str, str]]) -> tuple[dict[str, MbfcRecord], dict[str, AdFontesRecord], dict[str, str]]:
    known_domains = {row["source_domain"] for row in inventory}
    known_roots = {row["root_domain"] for row in inventory}
    tokens = candidate_slug_tokens(inventory)

    mbfc_urls = [
        url for url in fetch_xml_urls(MBFC_SITEMAP) if url.count("/") > 3 and url_matches_tokens(url, tokens)
    ][:MAX_MBFC_URLS]
    adfontes_urls = [
        url
        for url in fetch_xml_urls(ADFONTES_SITEMAP)
        if "bias-and-reliability" in url and url_matches_tokens(url, tokens)
    ][:MAX_ADFONTES_URLS]
    allsides_candidates = scrape_allsides_candidates()

    mbfc_records: dict[str, MbfcRecord] = {}
    adfontes_records: dict[str, AdFontesRecord] = {}
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        mbfc_futures = {
            executor.submit(scrape_mbfc_page, url): url
            for url in mbfc_urls
        }
        for future in as_completed(mbfc_futures):
            record = future.result()
            if not record:
                continue
            with lock:
                mbfc_records.setdefault(record.source_domain, record)

        adfontes_futures = {
            executor.submit(scrape_adfontes_page, url, known_domains, known_roots): url
            for url in adfontes_urls
        }
        for future in as_completed(adfontes_futures):
            record = future.result()
            if not record:
                continue
            with lock:
                existing = adfontes_records.get(record.source_domain)
                if existing is None or (record.reliability or 0.0) > (existing.reliability or 0.0):
                    adfontes_records[record.source_domain] = record

    return mbfc_records, adfontes_records, allsides_candidates


def enrich_inventory() -> list[dict[str, object]]:
    inventory = read_inventory()
    mbfc_map, adfontes_map, allsides_candidates = build_external_maps(inventory)
    event_map = load_source_quality_events()
    mbfc_root_map = {record.root_domain: record for record in mbfc_map.values()}
    adfontes_root_map = {record.root_domain: record for record in adfontes_map.values()}

    enriched: list[dict[str, object]] = []
    for row in inventory:
        domain = row["source_domain"]
        root = row["root_domain"]
        heuristic = float(row["current_actual_score"])
        tier_label, static_prior, tier_basis = tier_assignment(domain, row["source_type"])
        mbfc = mbfc_map.get(domain) or mbfc_root_map.get(root)
        adfontes = adfontes_map.get(domain) or adfontes_root_map.get(root)
        mbfc_score = factual_score(mbfc.factual_reporting) if mbfc else None
        adfontes_score = adfontes_quality_score(adfontes.reliability) if adfontes else None
        external_score, score_basis = external_reference_score(static_prior, mbfc_score, adfontes_score)
        signal_columns, signal_total, signal_positive, signal_negative, reference_adjustment = dynamic_signal_features(
            event_map.get(domain, Counter())
        )
        final_score = final_quality_score(external_score, reference_adjustment)
        normalized_name = re.sub(r"[^a-z0-9]+", "-", domain.lower()).strip("-")
        allsides_url = allsides_candidates.get(f"{normalized_name}-media-bias")
        enriched.append(
            {
                "source_domain": domain,
                "source_type": row["source_type"],
                "min_source_priority": int(row["min_source_priority"]),
                "max_source_priority": int(row["max_source_priority"]),
                "row_count": int(row["row_count"]),
                "day_count": int(row["day_count"]),
                "heuristic_score": heuristic,
                "tier_label": tier_label,
                "tier_basis": tier_basis,
                "static_prior_score": round(static_prior, 4),
                "mbfc_url": mbfc.page_url if mbfc else "",
                "mbfc_bias_rating": mbfc.bias_rating if mbfc else "",
                "mbfc_factual_reporting": mbfc.factual_reporting if mbfc else "",
                "mbfc_credibility": mbfc.credibility if mbfc else "",
                "mbfc_quality_score": round(mbfc_score, 4) if mbfc_score is not None else "",
                "adfontes_url": adfontes.page_url if adfontes else "",
                "adfontes_reliability": round(adfontes.reliability, 2) if adfontes and adfontes.reliability is not None else "",
                "adfontes_bias": round(adfontes.bias, 2) if adfontes and adfontes.bias is not None else "",
                "adfontes_quality_score": round(adfontes_score, 4) if adfontes_score is not None else "",
                "allsides_url": allsides_url or "",
                "allsides_bias_rating": "",
                "external_match_count": int(bool(mbfc)) + int(bool(adfontes)),
                "external_reference_score": round(external_score, 4),
                **signal_columns,
                "signal_total_count": signal_total,
                "signal_positive_count": signal_positive,
                "signal_negative_count": signal_negative,
                "reference_dynamic_adjustment": reference_adjustment,
                "score_basis": score_basis,
                "final_source_quality_score": round(final_score, 4),
                "external_status": "externally_matched" if score_basis != "static_prior_only" else "static_prior_only",
            }
        )
    enriched.sort(key=lambda row: (-int(row["row_count"]), row["source_domain"]))
    return enriched


def write_outputs(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "source_domain",
        "source_type",
        "min_source_priority",
        "max_source_priority",
        "row_count",
        "day_count",
        "heuristic_score",
        "tier_label",
        "tier_basis",
        "static_prior_score",
        "mbfc_url",
        "mbfc_bias_rating",
        "mbfc_factual_reporting",
        "mbfc_credibility",
        "mbfc_quality_score",
        "adfontes_url",
        "adfontes_reliability",
        "adfontes_bias",
        "adfontes_quality_score",
        "allsides_url",
        "allsides_bias_rating",
        "external_match_count",
        "external_reference_score",
        *SIGNAL_COLUMN_NAMES.values(),
        "signal_total_count",
        "signal_positive_count",
        "signal_negative_count",
        "reference_dynamic_adjustment",
        "score_basis",
        "final_source_quality_score",
        "external_status",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "rows": len(rows),
        "externally_matched_rows": sum(1 for row in rows if row["external_status"] == "externally_matched"),
        "static_prior_only_rows": sum(1 for row in rows if row["external_status"] == "static_prior_only"),
        "mbfc_matches": sum(1 for row in rows if row["mbfc_url"]),
        "adfontes_matches": sum(1 for row in rows if row["adfontes_url"]),
        "allsides_url_candidates": sum(1 for row in rows if row["allsides_url"]),
        "scrape_errors": len(SCRAPE_ERRORS),
        "score_basis": dict(Counter(row["score_basis"] for row in rows)),
        "tier_label_counts": dict(Counter(row["tier_label"] for row in rows)),
        "signal_column_names": SIGNAL_COLUMN_NAMES,
        "signal_totals": {
            column_name: sum(int(row[column_name]) for row in rows)
            for column_name in SIGNAL_COLUMN_NAMES.values()
        },
        "domains_with_events": sum(1 for row in rows if int(row["signal_total_count"]) > 0),
        "top_50": rows[:50],
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ERROR_LOG.write_text(json.dumps(SCRAPE_ERRORS, indent=2), encoding="utf-8")


def main() -> None:
    rows = enrich_inventory()
    write_outputs(rows)
    print(json.dumps({"csv": str(OUTPUT_CSV), "summary": str(OUTPUT_SUMMARY), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
