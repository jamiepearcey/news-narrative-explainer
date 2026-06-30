#!/usr/bin/env python3
"""Summarize the fully finalized domain-score universe."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
RESULTS_DIR = WORK_DIR / "results"
EFFECTIVE_CSV = RESULTS_DIR / "effective_domain_scores.csv"
OVERRIDES_CSV = RESULTS_DIR / "reviewed_domain_overrides.csv"
SUMMARY_JSON = RESULTS_DIR / "final_domain_scores_summary.json"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def score_band(score: float) -> str:
    if score >= 9.0:
        return "9.0-10.0"
    if score >= 7.0:
        return "7.0-8.9"
    if score >= 5.0:
        return "5.0-6.9"
    if score >= 3.0:
        return "3.0-4.9"
    return "0.0-2.9"


def main() -> None:
    effective_rows = load_csv(EFFECTIVE_CSV)
    override_rows = load_csv(OVERRIDES_CSV)

    archetypes = Counter(row["effective_archetype"] for row in effective_rows)
    score_bands = Counter(score_band(float(row["effective_score_0_10"])) for row in effective_rows)
    review_statuses = Counter(row["review_status"] for row in override_rows)
    top_domains = sorted(
        effective_rows,
        key=lambda row: (-float(row["effective_score_0_10"]), row["source_domain"]),
    )[:25]

    summary = {
        "effective_rows": len(effective_rows),
        "override_rows": len(override_rows),
        "review_status_counts": dict(sorted(review_statuses.items())),
        "archetype_counts": dict(sorted(archetypes.items())),
        "score_band_counts": dict(sorted(score_bands.items())),
        "top_scored_domains": [
            {
                "source_domain": row["source_domain"],
                "effective_archetype": row["effective_archetype"],
                "effective_score_0_10": row["effective_score_0_10"],
                "review_status": row["review_status"],
            }
            for row in top_domains
        ],
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
