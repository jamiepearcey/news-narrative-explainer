#!/usr/bin/env python3
"""Materialize explicit per-domain story samples from the scored corpus."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
RESULTS_DIR = WORK_DIR / "results"
SCORED_WEAK_LABELS_CSV = RESULTS_DIR / "scored_weak_labels.csv"
OUTPUT_CSV = RESULTS_DIR / "domain_story_samples.csv"
SUMMARY_JSON = RESULTS_DIR / "domain_story_samples_summary.json"

HIGH_VALUE_TERMS = (
    "oil", "crude", "gas", "lng", "shipping", "freight", "yield", "inflation",
    "tariff", "sanction", "unemployment", "rates", "pipeline", "refinery",
    "diesel", "biorefinery", "earnings", "merger", "ipo", "exports", "imports",
    "nuclear", "power", "energy", "data center",
)

LOW_VALUE_TERMS = (
    "gossip", "wedding", "matrimonio", "molest", "crime", "podcast", "celebrity",
    "festival", "mural", "blaze", "invited",
)

SAMPLES_PER_DOMAIN = 3


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


def sample_priority(row: dict[str, str]) -> tuple[float, float, float, str]:
    title = normalize_title(row.get("title") or "")
    title_l = title.lower()
    nat_hits = sum(int(row.get(k) or 0) for k in ["finance_hits", "macro_hits", "geo_hits", "company_hits"])
    theme_hits = sum(int(row.get(k) or 0) for k in ["keep_theme_hits", "macro_theme_hits", "geo_theme_hits"])
    finance_cluster = float(row.get("finance_cluster_score") or 0.0)
    predicted_score = float(row.get("predicted_score") or 0.0)
    title_value = sum(1 for term in HIGH_VALUE_TERMS if term in title_l) - sum(1 for term in LOW_VALUE_TERMS if term in title_l)
    score = (
        (0.8 * title_value)
        + (0.7 * nat_hits)
        + (0.15 * theme_hits)
        + (1.4 * finance_cluster)
        + (0.2 * predicted_score)
    )
    return (-score, -nat_hits, -predicted_score, title)


def main() -> None:
    rows_by_domain: dict[str, list[dict[str, str]]] = defaultdict(list)

    with SCORED_WEAK_LABELS_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            domain = normalize_domain(row.get("source_domain") or "")
            if not domain:
                continue
            rows_by_domain[domain].append(row)

    fieldnames = [
        "source_domain",
        "sample_rank",
        "title",
        "document_identifier",
        "label",
        "predicted_label",
        "predicted_score",
        "decision_band",
        "source_profile",
        "finance_cluster_score",
        "finance_hits",
        "macro_hits",
        "geo_hits",
        "company_hits",
        "keep_theme_hits",
        "macro_theme_hits",
        "geo_theme_hits",
        "summary",
    ]

    rows_out: list[dict[str, str]] = []
    summary_domains: list[dict[str, str | int]] = []
    for domain, rows in rows_by_domain.items():
        rows.sort(key=sample_priority)
        seen_titles: set[str] = set()
        chosen: list[dict[str, str]] = []
        for row in rows:
            title = normalize_title(row.get("title") or "")
            title_key = title.lower()
            if not title or title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            chosen.append(row)
            if len(chosen) >= SAMPLES_PER_DOMAIN:
                break

        for idx, row in enumerate(chosen, start=1):
            rows_out.append(
                {
                    "source_domain": domain,
                    "sample_rank": str(idx),
                    "title": normalize_title(row.get("title") or ""),
                    "document_identifier": row.get("document_identifier") or "",
                    "label": row.get("label") or "",
                    "predicted_label": row.get("predicted_label") or "",
                    "predicted_score": row.get("predicted_score") or "",
                    "decision_band": row.get("decision_band") or "",
                    "source_profile": row.get("source_profile") or "",
                    "finance_cluster_score": row.get("finance_cluster_score") or "",
                    "finance_hits": row.get("finance_hits") or "",
                    "macro_hits": row.get("macro_hits") or "",
                    "geo_hits": row.get("geo_hits") or "",
                    "company_hits": row.get("company_hits") or "",
                    "keep_theme_hits": row.get("keep_theme_hits") or "",
                    "macro_theme_hits": row.get("macro_theme_hits") or "",
                    "geo_theme_hits": row.get("geo_theme_hits") or "",
                    "summary": (row.get("summary") or "")[:400],
                }
            )
        summary_domains.append(
            {
                "source_domain": domain,
                "corpus_rows": len(rows),
                "samples_emitted": len(chosen),
            }
        )

    rows_out.sort(key=lambda row: (row["source_domain"], int(row["sample_rank"])))
    summary_domains.sort(key=lambda row: (-int(row["corpus_rows"]), row["source_domain"]))

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {
        "output_csv": str(OUTPUT_CSV),
        "sample_rows": len(rows_out),
        "unique_domains": len(summary_domains),
        "samples_per_domain_target": SAMPLES_PER_DOMAIN,
        "top_domains_by_rows": summary_domains[:20],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
