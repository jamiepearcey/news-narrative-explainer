#!/usr/bin/env python3
"""Prioritize review rows and hard negatives for the next labeling pass."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
FEEDBACK_DIR = WORK_DIR / "feedback"
RESULTS_DIR = WORK_DIR / "results"
REVIEW_QUEUE_CSV = FEEDBACK_DIR / "review_queue.csv"
SCORED_WEAK_LABELS_CSV = RESULTS_DIR / "scored_weak_labels.csv"
PRIORITIZED_REVIEW_CSV = FEEDBACK_DIR / "prioritized_review_queue.csv"
HARD_NEGATIVES_CSV = FEEDBACK_DIR / "hard_negative_candidates.csv"
SUMMARY_JSON = RESULTS_DIR / "review_queue_prioritization_summary.json"
TOP_REVIEW_LIMIT = 5000
TOP_HARD_NEGATIVE_LIMIT = 2000

LOW_QUALITY_DOMAINS = {
    "zazoom.it",
    "river949.com.au",
    "drudge.com",
}

REVIEW_DEPRIORITIZE_DOMAINS = {
    "mediaite.com",
    "inewsgr.com",
    "newspim.com",
    "sbctv.gr",
}

PRESS_RELEASE_MIRROR_DOMAINS = {
    "itnewsonline.com",
    "manilatimes.net",
    "searchlight.vc",
    "interfax.com.ua",
    "en.acnnewswire.com",
    "pr.com",
    "briefingwire.com",
    "openpr.com",
}

LOW_VALUE_URL_PATTERNS = (
    "/gossip/",
    "/crime/",
    "/tv/",
    "/entertainment/",
    "/celebrity/",
    "/koinonia/",
)

LOW_VALUE_TITLE_TERMS = (
    "gossip",
    "wedding",
    "matrimonio",
    "molest",
    "podcast",
    "invited",
    "crime",
    "world environment day",
    "awareness campaign",
    "mural",
    "blaze",
)

HIGH_VALUE_MARKET_TERMS = (
    "oil",
    "crude",
    "gas",
    "lng",
    "shipping",
    "freight",
    "yield",
    "yields",
    "inflation",
    "tariff",
    "tariffs",
    "sanction",
    "sanctions",
    "unemployment",
    "rates",
    "rate cut",
    "rate hike",
    "pipeline",
    "refinery",
    "diesel",
    "biorefinery",
    "merger",
    "earnings",
    "ipo",
    "imports",
    "exports",
)


def canonical_document_id(raw: str) -> str:
    value = (raw or "").strip().lower()
    value = re.sub(r"^https?://www\.", "https://", value)
    value = value.replace(":443/", "/")
    value = value.replace("http://", "https://", 1)
    return value


def canonical_title(raw: str) -> str:
    value = (raw or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+[|\-–]\s+[^|\-–]{0,80}$", "", value)
    return value


def review_priority(row: dict[str, str]) -> tuple[float, float, float, str]:
    source_domain = (row.get("source_domain") or "").split(":", 1)[0]
    document_identifier = row.get("document_identifier") or ""
    title_text = (row.get("title") or "").lower()
    url_text = document_identifier.lower()
    finance_cluster_score = float(row.get("finance_cluster_score") or 0.0)
    market_relevance_rate = float(row.get("market_relevance_rate") or 0.0)
    industry_signal_rate = float(row.get("industry_signal_rate") or 0.0)
    junk_rate = float(row.get("junk_rate") or 0.0)
    natural_text_length = int(row.get("natural_text_length") or 0)
    summary_length = int(row.get("summary_length") or 0)
    text_excerpt_length = int(row.get("text_excerpt_length") or 0)
    filtered_theme_count = int(row.get("filtered_theme_count") or 0)
    finance_hits = int(row.get("finance_hits") or 0)
    macro_hits = int(row.get("macro_hits") or 0)
    geo_hits = int(row.get("geo_hits") or 0)
    company_hits = int(row.get("company_hits") or 0)
    sports_hits = int(row.get("sports_hits") or 0)
    entertainment_hits = int(row.get("entertainment_hits") or 0)
    lifestyle_hits = int(row.get("lifestyle_hits") or 0)
    crime_hits = int(row.get("crime_hits") or 0)
    press_hits = int(row.get("press_hits") or 0)
    keep_theme_hits = int(row.get("keep_theme_hits") or 0)
    macro_theme_hits = int(row.get("macro_theme_hits") or 0)
    geo_theme_hits = int(row.get("geo_theme_hits") or 0)
    drop_theme_hits = int(row.get("drop_theme_hits") or 0)
    source_profile = row.get("source_profile") or ""
    natural_signal_hits = finance_hits + macro_hits + geo_hits + company_hits
    theme_signal_hits = keep_theme_hits + macro_theme_hits + geo_theme_hits
    signal_hits = natural_signal_hits + theme_signal_hits
    junk_hits = sports_hits + entertainment_hits + lifestyle_hits + crime_hits + drop_theme_hits + press_hits
    text_bonus = min(0.5, natural_text_length / 600.0)
    summary_bonus = 0.25 if summary_length >= 120 else 0.0
    excerpt_bonus = 0.2 if text_excerpt_length >= 180 else 0.0
    thin_text_penalty = 0.45 if natural_text_length < 90 and summary_length == 0 and text_excerpt_length < 40 else 0.0
    thin_theme_penalty = 0.25 if filtered_theme_count >= 8 and natural_text_length < 120 else 0.0
    positive_signal_bonus = min(1.2, 0.32 * natural_signal_hits + 0.06 * theme_signal_hits)
    junk_signal_penalty = min(1.4, 0.25 * junk_hits)
    sparse_penalty = 0.2 if source_profile == "mixed_or_sparse" and signal_hits <= 1 else 0.0
    market_profile_bonus = 0.18 if source_profile == "market_relevant" else 0.0
    theme_only_penalty = 0.6 if natural_signal_hits == 0 and theme_signal_hits >= 3 and source_profile == "mixed_or_sparse" else 0.0
    no_lexical_signal_penalty = 0.35 if natural_signal_hits == 0 and finance_cluster_score < 0.3 else 0.0
    weak_lexical_penalty = 0.45 if natural_signal_hits == 1 and theme_signal_hits >= 4 else 0.0
    high_value_market_bonus = 0.25 * sum(1 for term in HIGH_VALUE_MARKET_TERMS if term in title_text or term in url_text)
    pure_geo_penalty = 0.55 if geo_hits > 0 and finance_hits == 0 and macro_hits == 0 and company_hits == 0 and high_value_market_bonus == 0 else 0.0
    low_value_penalty = 0.0
    if any(pattern in document_identifier.lower() for pattern in LOW_VALUE_URL_PATTERNS):
        low_value_penalty += 0.8
    if any(term in title_text for term in LOW_VALUE_TITLE_TERMS):
        low_value_penalty += 0.8
    if crime_hits > 0:
        low_value_penalty += 0.6
    domain_penalty = 0.0
    if source_domain in LOW_QUALITY_DOMAINS:
        domain_penalty += 1.2
    if source_domain in REVIEW_DEPRIORITIZE_DOMAINS:
        domain_penalty += 0.6
    if source_domain in PRESS_RELEASE_MIRROR_DOMAINS and finance_hits <= 1:
        domain_penalty += 0.8
    priority = (
        (1.2 * finance_cluster_score)
        + (1.0 * industry_signal_rate)
        + (0.8 * market_relevance_rate)
        - (0.7 * junk_rate)
        + positive_signal_bonus
        + min(0.9, high_value_market_bonus)
        - junk_signal_penalty
        + text_bonus
        + summary_bonus
        + excerpt_bonus
        - thin_text_penalty
        - thin_theme_penalty
        - theme_only_penalty
        - no_lexical_signal_penalty
        - weak_lexical_penalty
        - pure_geo_penalty
        - low_value_penalty
        - domain_penalty
        - sparse_penalty
        + market_profile_bonus
        + (0.2 if source_profile == "industry_useful" else 0.0)
    )
    return (-priority, -industry_signal_rate, -market_relevance_rate, row.get("document_identifier", ""))


def hard_negative_priority(row: dict[str, str]) -> tuple[float, float, str]:
    predicted_score = float(row.get("predicted_score") or 0.0)
    finance_cluster_score = float(row.get("finance_cluster_score") or 0.0)
    return (-predicted_score, -finance_cluster_score, row.get("document_identifier", ""))


def main() -> None:
    prioritized_rows: list[dict[str, str]] = []
    with REVIEW_QUEUE_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    rows.sort(key=review_priority)
    seen_document_ids: set[str] = set()
    seen_titles: set[str] = set()
    for row in rows:
        canonical_id = canonical_document_id(row.get("document_identifier") or "")
        canonical_row_title = canonical_title(row.get("title") or "")
        if canonical_id in seen_document_ids:
            continue
        if canonical_row_title and canonical_row_title in seen_titles:
            continue
        seen_document_ids.add(canonical_id)
        if canonical_row_title:
            seen_titles.add(canonical_row_title)
        prioritized_rows.append(row)
        if len(prioritized_rows) >= TOP_REVIEW_LIMIT:
            break

    hard_negative_rows: list[dict[str, str]] = []
    with SCORED_WEAK_LABELS_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row["label"]
            predicted_label = row["predicted_label"]
            decision_band = row["decision_band"]
            if not label.startswith("drop_"):
                continue
            if predicted_label.startswith("drop_"):
                continue
            if decision_band == "review":
                continue
            row["error_type"] = "drop_misclassified_as_keep"
            hard_negative_rows.append(row)
    hard_negative_rows.sort(key=hard_negative_priority)
    hard_negative_rows = hard_negative_rows[:TOP_HARD_NEGATIVE_LIMIT]

    for path, rows_to_write in [
        (PRIORITIZED_REVIEW_CSV, prioritized_rows),
        (HARD_NEGATIVES_CSV, hard_negative_rows),
    ]:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_to_write[0].keys()) if rows_to_write else [])
            if rows_to_write:
                writer.writeheader()
                writer.writerows(rows_to_write)

    review_profile_counts = Counter(row.get("source_profile", "") for row in prioritized_rows)
    hard_negative_counts = Counter((row["label"], row["predicted_label"]) for row in hard_negative_rows)
    summary = {
        "prioritized_review_rows": len(prioritized_rows),
        "hard_negative_rows": len(hard_negative_rows),
        "prioritized_review_csv": str(PRIORITIZED_REVIEW_CSV),
        "hard_negative_csv": str(HARD_NEGATIVES_CSV),
        "review_profile_counts": dict(sorted(review_profile_counts.items())),
        "top_hard_negative_confusions": [
            {"truth": truth, "predicted": predicted, "count": count}
            for (truth, predicted), count in hard_negative_counts.most_common(20)
        ],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
