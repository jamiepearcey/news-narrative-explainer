#!/usr/bin/env python3
"""Finalize all remaining heuristic domain scores into reviewed overrides."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
RESULTS_DIR = WORK_DIR / "results"
AUDIT_CSV = RESULTS_DIR / "domain_score_audit.csv"
OVERRIDES_CSV = RESULTS_DIR / "reviewed_domain_overrides.csv"
SUMMARY_JSON = RESULTS_DIR / "finalize_remaining_domain_overrides_summary.json"

FIELDNAMES = [
    "source_domain",
    "reviewed_archetype",
    "reviewed_score_0_10",
    "review_status",
    "rationale",
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_rationale(row: dict[str, str]) -> str:
    archetype = row["proposed_archetype"]
    profile = row.get("source_profile") or "no_profile"
    basis = row.get("proposal_basis") or "audit"
    sample_titles = [
        row.get("sample_title_1", "").strip(),
        row.get("sample_title_2", "").strip(),
        row.get("sample_title_3", "").strip(),
    ]
    sample_titles = [title for title in sample_titles if title]
    if sample_titles:
        sample_note = f"sampled_titles={min(3, len(sample_titles))}"
    else:
        sample_note = "sampled_titles=0"
    return (
        "Bulk-finalized from the sampled-story audit and retained current effective score; "
        f"archetype={archetype}, profile={profile}, basis={basis}, {sample_note}."
    )


def main() -> None:
    audit_rows = load_csv(AUDIT_CSV)
    override_rows = load_csv(OVERRIDES_CSV) if OVERRIDES_CSV.exists() else []
    overrides = {}
    for row in override_rows:
        normalized = dict(row)
        rationale = normalized.get("rationale", "")
        if rationale.startswith("Bulk-finalized from the sampled-story audit"):
            normalized["review_status"] = "reviewed_bulk_finalized"
        elif normalized.get("review_status") in {"reviewed", "reviewed_manual", ""}:
            normalized["review_status"] = "reviewed_manual"
        if normalized["review_status"] == "reviewed_manual":
            overrides[normalized["source_domain"]] = normalized

    generated = 0
    for row in audit_rows:
        domain = row["source_domain"]
        if domain in overrides:
            continue
        overrides[domain] = {
            "source_domain": domain,
            "reviewed_archetype": row["proposed_archetype"],
            "reviewed_score_0_10": row["proposed_score_0_10"],
            "review_status": "reviewed_bulk_finalized",
            "rationale": make_rationale(row),
        }
        generated += 1

    rows_out = sorted(overrides.values(), key=lambda row: row["source_domain"])
    with OVERRIDES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {
        "audit_rows": len(audit_rows),
        "overrides_written": len(rows_out),
        "generated_new_overrides": generated,
        "existing_reviewed_overrides": len(override_rows),
        "output_csv": str(OVERRIDES_CSV),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
