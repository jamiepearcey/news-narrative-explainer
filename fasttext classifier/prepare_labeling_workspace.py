#!/usr/bin/env python3
"""Prepare and refresh the labeling workspace in one command."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activate-next", action="store_true")
    parser.add_argument("--force-activate", action="store_true")
    parser.add_argument("--hard-cases-csv")
    parser.add_argument("--eval-csv")
    parser.add_argument("--packets-dir")
    parser.add_argument("--responses-dir")
    parser.add_argument("--status-json")
    parser.add_argument("--selection-json")
    parser.add_argument("--active-brief-json")
    parser.add_argument("--active-brief-md")
    parser.add_argument("--review-sheet-json")
    parser.add_argument("--review-sheet-md")
    parser.add_argument("--review-sheet-csv")
    parser.add_argument("--shortlist-json")
    parser.add_argument("--shortlist-md")
    parser.add_argument("--shortlist-csv")
    parser.add_argument("--shortlist-response-json")
    parser.add_argument("--shortlist-response-md")
    parser.add_argument("--shortlist-limit", type=int)
    parser.add_argument("--manifest-json")
    parser.add_argument("--manifest-md")
    parser.add_argument("--manifest-csv")
    return parser.parse_args()


def run_json(script: str, *extra: str) -> dict[str, object]:
    cmd = ["python3", str(WORK_DIR / script), *extra]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def main() -> None:
    args = parse_args()

    manifest_extra: list[str] = []
    if args.packets_dir:
        manifest_extra.extend(["--packets-dir", args.packets_dir])
    if args.responses_dir:
        manifest_extra.extend(["--responses-dir", args.responses_dir])
    if args.manifest_json:
        manifest_extra.extend(["--manifest-json", args.manifest_json])
    if args.manifest_md:
        manifest_extra.extend(["--manifest-md", args.manifest_md])
    if args.manifest_csv:
        manifest_extra.extend(["--manifest-csv", args.manifest_csv])

    status_extra: list[str] = []
    if args.hard_cases_csv:
        status_extra.extend(["--hard-cases-csv", args.hard_cases_csv])
    if args.eval_csv:
        status_extra.extend(["--eval-csv", args.eval_csv])
    if args.packets_dir:
        status_extra.extend(["--packets-dir", args.packets_dir])
    if args.responses_dir:
        status_extra.extend(["--responses-dir", args.responses_dir])
    if args.status_json:
        status_extra.extend(["--output-json", args.status_json])

    brief_extra: list[str] = []
    if args.status_json:
        brief_extra.extend(["--status-json", args.status_json])
    if args.active_brief_json:
        brief_extra.extend(["--output-json", args.active_brief_json])
    if args.active_brief_md:
        brief_extra.extend(["--output-md", args.active_brief_md])

    review_sheet_extra: list[str] = []
    if args.status_json:
        review_sheet_extra.extend(["--status-json", args.status_json])
    if args.review_sheet_json:
        review_sheet_extra.extend(["--output-json", args.review_sheet_json])
    if args.review_sheet_md:
        review_sheet_extra.extend(["--output-md", args.review_sheet_md])
    if args.review_sheet_csv:
        review_sheet_extra.extend(["--output-csv", args.review_sheet_csv])
    if args.shortlist_json:
        review_sheet_extra.extend(["--output-shortlist-json", args.shortlist_json])
    if args.shortlist_md:
        review_sheet_extra.extend(["--output-shortlist-md", args.shortlist_md])
    if args.shortlist_csv:
        review_sheet_extra.extend(["--output-shortlist-csv", args.shortlist_csv])
    if args.shortlist_response_json:
        review_sheet_extra.extend(["--output-shortlist-response-json", args.shortlist_response_json])
    if args.shortlist_response_md:
        review_sheet_extra.extend(["--output-shortlist-response-md", args.shortlist_response_md])
    if args.shortlist_limit is not None:
        review_sheet_extra.extend(["--shortlist-limit", str(args.shortlist_limit)])

    manifest = run_json("build_labeling_response_manifest.py", *manifest_extra)
    status_before = run_json("labeling_status.py", *status_extra)

    selection = None
    active_response_batches = [
        batch
        for batch in status_before.get("packet_batches", [])
        if batch.get("response_present")
    ]
    if args.activate_next and (args.force_activate or not active_response_batches):
        extra = ["--activate-response"]
        if args.status_json:
            extra.extend(["--status-json", args.status_json])
        if args.selection_json:
            extra.extend(["--selection-json", args.selection_json])
        if args.force_activate:
            extra.append("--force")
        selection = run_json("select_next_labeling_batch.py", *extra)

    status_after = run_json("labeling_status.py", *status_extra)
    active_brief = run_json("build_active_labeling_brief.py", *brief_extra)
    review_sheet = run_json("build_labeling_review_sheet.py", *review_sheet_extra)

    summary = {
        "manifest_batches": manifest["batch_count"],
        "status_before_next_batch": status_before.get("next_batch"),
        "active_response_batches_before": [
            batch["batch"] for batch in active_response_batches
        ],
        "selection": selection,
        "status_after_next_batch": status_after.get("next_batch"),
        "active_batch": active_brief["active_batch"],
        "active_remaining_rows": active_brief["remaining_rows"],
        "active_response_path": active_brief["response_path"],
        "review_sheet_rows": review_sheet["rows"],
        "review_sheet_remaining_rows": review_sheet["remaining_rows"],
        "shortlist_rows": review_sheet["shortlist_rows"],
        "activate_next": args.activate_next,
        "force_activate": args.force_activate,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
