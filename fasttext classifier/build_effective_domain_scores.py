#!/usr/bin/env python3
"""Merge heuristic domain audit scores with manually reviewed overrides."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
RESULTS_DIR = WORK_DIR / "results"
AUDIT_CSV = RESULTS_DIR / "domain_score_audit.csv"
OVERRIDES_CSV = RESULTS_DIR / "reviewed_domain_overrides.csv"
OUTPUT_CSV = RESULTS_DIR / "effective_domain_scores.csv"
SUMMARY_JSON = RESULTS_DIR / "effective_domain_scores_summary.json"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    audit_rows = load_csv(AUDIT_CSV)
    override_rows = load_csv(OVERRIDES_CSV) if OVERRIDES_CSV.exists() else []
    overrides = {row["source_domain"]: row for row in override_rows}

    fieldnames = [
        "source_domain",
        "effective_archetype",
        "effective_score_0_10",
        "effective_score_0_1",
        "score_source",
        "heuristic_archetype",
        "heuristic_score_0_10",
        "review_status",
        "reviewed_archetype",
        "reviewed_score_0_10",
        "proposal_basis",
        "sample_title_1",
        "sample_title_2",
        "sample_title_3",
    ]
    passthrough_fieldnames = [
        name
        for name in audit_rows[0].keys()
        if name not in {
            "source_domain",
            "proposed_archetype",
            "proposed_score_0_10",
            "proposed_score_0_1",
        }
        and name not in fieldnames
    ]
    fieldnames.extend(passthrough_fieldnames)

    rows_out: list[dict[str, str]] = []
    reviewed_count = 0
    for row in audit_rows:
        domain = row["source_domain"]
        override = overrides.get(domain)
        if override:
            reviewed_count += 1
            effective_archetype = override["reviewed_archetype"]
            effective_score_0_10 = float(override["reviewed_score_0_10"])
            score_source = "reviewed_override"
            review_status = override["review_status"]
        else:
            effective_archetype = row["proposed_archetype"]
            effective_score_0_10 = float(row["proposed_score_0_10"])
            score_source = "heuristic_audit"
            review_status = ""

        out_row = {
            "source_domain": domain,
            "effective_archetype": effective_archetype,
            "effective_score_0_10": f"{effective_score_0_10:.2f}",
            "effective_score_0_1": f"{effective_score_0_10 / 10.0:.4f}",
            "score_source": score_source,
            "heuristic_archetype": row["proposed_archetype"],
            "heuristic_score_0_10": row["proposed_score_0_10"],
            "review_status": review_status,
            "reviewed_archetype": override["reviewed_archetype"] if override else "",
            "reviewed_score_0_10": override["reviewed_score_0_10"] if override else "",
            "proposal_basis": row["proposal_basis"],
            "sample_title_1": row["sample_title_1"],
            "sample_title_2": row["sample_title_2"],
            "sample_title_3": row["sample_title_3"],
        }
        for name in passthrough_fieldnames:
            out_row[name] = row.get(name, "")
        rows_out.append(out_row)

    rows_out.sort(key=lambda row: (-float(row["effective_score_0_10"]), row["source_domain"]))

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {
        "output_csv": str(OUTPUT_CSV),
        "rows": len(rows_out),
        "reviewed_override_rows": reviewed_count,
        "heuristic_rows": len(rows_out) - reviewed_count,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
