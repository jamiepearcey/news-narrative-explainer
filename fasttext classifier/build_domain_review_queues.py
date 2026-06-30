#!/usr/bin/env python3
"""Create targeted domain review queues from the domain score audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
RESULTS_DIR = WORK_DIR / "results"
AUDIT_CSV = RESULTS_DIR / "domain_score_audit.csv"
EFFECTIVE_CSV = RESULTS_DIR / "effective_domain_scores.csv"
REVIEW_DIR = RESULTS_DIR / "domain_review_queues"
SUMMARY_JSON = REVIEW_DIR / "summary.json"


def load_rows() -> list[dict[str, str]]:
    source = EFFECTIVE_CSV if EFFECTIVE_CSV.exists() else AUDIT_CSV
    with source.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_int(row: dict[str, str], *keys: str) -> int:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return int(float(value))
    return 0


def row_float(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return 0.0


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    if not rows:
        raise SystemExit("domain_score_audit.csv is empty")

    fieldnames = list(rows[0].keys())
    unresolved_rows = [
        row for row in rows
        if row.get("score_source", "heuristic_audit") != "reviewed_override"
    ]

    top_volume = sorted(
        unresolved_rows,
        key=lambda row: (-row_int(row, "rows"), row["source_domain"]),
    )[:500]
    suspicious_high = [
        row for row in unresolved_rows
        if row_float(row, "effective_score_0_10", "proposed_score_0_10") >= 7.0
        and (row.get("effective_archetype") or row.get("proposed_archetype")) in {
            "mainstream_business_or_general",
            "specialist_trade",
        }
        and (
            "sample_market_signal" not in row["proposal_basis"]
            or row_float(row, "external_score_0_10") == 0.0
        )
        and row.get("score_source", "heuristic_audit") != "reviewed_override"
    ]
    mixed_domains = [
        row for row in unresolved_rows
        if (row.get("effective_archetype") or row.get("proposed_archetype")) == "mixed_politics_or_general_aggregator"
        and row_int(row, "rows") >= 25
    ]
    stock_blurbs = [
        row for row in unresolved_rows
        if (row.get("effective_archetype") or row.get("proposed_archetype")) == "market_blog_or_stock_blurb"
    ]
    likely_good = [
        row for row in unresolved_rows
        if row_float(row, "effective_score_0_10", "proposed_score_0_10") >= 8.0
        and (row.get("effective_archetype") or row.get("proposed_archetype")) in {
            "premium_primary",
            "specialist_trade",
            "mainstream_business_or_general",
        }
    ][:300]

    outputs = {
        "top_volume_review.csv": top_volume,
        "suspicious_high_scores.csv": suspicious_high,
        "mixed_domains_review.csv": mixed_domains,
        "stock_blurb_domains_review.csv": stock_blurbs,
        "likely_good_reference.csv": likely_good,
    }

    for filename, queue_rows in outputs.items():
        write_csv(REVIEW_DIR / filename, queue_rows, fieldnames)

    summary = {
        "review_dir": str(REVIEW_DIR),
        "source_csv": str(EFFECTIVE_CSV if EFFECTIVE_CSV.exists() else AUDIT_CSV),
        "queues": {name: len(queue_rows) for name, queue_rows in outputs.items()},
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
