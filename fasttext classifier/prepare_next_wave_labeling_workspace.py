#!/usr/bin/env python3
"""Prepare the next-wave eval-labeling workspace in one command."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
FEEDBACK_DIR = WORK_DIR / "feedback"
RESULTS_DIR = WORK_DIR / "results"

NEXT_WAVE_CSV = FEEDBACK_DIR / "gpt_eval_next_wave.csv"
NEXT_WAVE_PACKETS_DIR = FEEDBACK_DIR / "next_wave_labeling_packets"
NEXT_WAVE_RESPONSES_DIR = FEEDBACK_DIR / "next_wave_labeling_responses"
NEXT_WAVE_STATUS_JSON = RESULTS_DIR / "next_wave_labeling_status.json"
NEXT_WAVE_SELECTION_JSON = RESULTS_DIR / "next_wave_next_batch.json"
NEXT_WAVE_ACTIVE_BRIEF_JSON = RESULTS_DIR / "next_wave_active_labeling_brief.json"
NEXT_WAVE_ACTIVE_BRIEF_MD = RESULTS_DIR / "next_wave_active_labeling_brief.md"
NEXT_WAVE_REVIEW_SHEET_JSON = RESULTS_DIR / "next_wave_active_labeling_review_sheet.json"
NEXT_WAVE_REVIEW_SHEET_MD = RESULTS_DIR / "next_wave_active_labeling_review_sheet.md"
NEXT_WAVE_REVIEW_SHEET_CSV = RESULTS_DIR / "next_wave_active_labeling_review_sheet.csv"
NEXT_WAVE_SHORTLIST_JSON = RESULTS_DIR / "next_wave_active_labeling_shortlist.json"
NEXT_WAVE_SHORTLIST_MD = RESULTS_DIR / "next_wave_active_labeling_shortlist.md"
NEXT_WAVE_SHORTLIST_CSV = RESULTS_DIR / "next_wave_active_labeling_shortlist.csv"
NEXT_WAVE_SHORTLIST_RESPONSE_JSON = RESULTS_DIR / "next_wave_active_labeling_shortlist.response.json"
NEXT_WAVE_SHORTLIST_RESPONSE_MD = RESULTS_DIR / "next_wave_active_labeling_shortlist.response.md"
NEXT_WAVE_MANIFEST_JSON = NEXT_WAVE_RESPONSES_DIR / "manifest.json"
NEXT_WAVE_MANIFEST_MD = NEXT_WAVE_RESPONSES_DIR / "manifest.md"
NEXT_WAVE_MANIFEST_CSV = NEXT_WAVE_RESPONSES_DIR / "manifest.csv"
NEXT_WAVE_PACKET_SUMMARY_JSON = RESULTS_DIR / "gpt_eval_next_wave_packets_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activate-next", action="store_true")
    parser.add_argument("--force-activate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=24)
    return parser.parse_args()


def run_json(script: str, *extra: str) -> dict[str, object]:
    cmd = ["python3", str(WORK_DIR / script), *extra]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def main() -> None:
    args = parse_args()

    packet_summary = run_json(
        "build_gpt_labeling_packets.py",
        "--input-csv",
        str(NEXT_WAVE_CSV),
        "--output-dir",
        str(NEXT_WAVE_PACKETS_DIR),
        "--batch-size",
        str(args.batch_size),
        "--summary-json",
        str(NEXT_WAVE_PACKET_SUMMARY_JSON),
    )

    prepare_extra = [
        "--hard-cases-csv",
        str(NEXT_WAVE_CSV),
        "--packets-dir",
        str(NEXT_WAVE_PACKETS_DIR),
        "--responses-dir",
        str(NEXT_WAVE_RESPONSES_DIR),
        "--status-json",
        str(NEXT_WAVE_STATUS_JSON),
        "--selection-json",
        str(NEXT_WAVE_SELECTION_JSON),
        "--active-brief-json",
        str(NEXT_WAVE_ACTIVE_BRIEF_JSON),
        "--active-brief-md",
        str(NEXT_WAVE_ACTIVE_BRIEF_MD),
        "--review-sheet-json",
        str(NEXT_WAVE_REVIEW_SHEET_JSON),
        "--review-sheet-md",
        str(NEXT_WAVE_REVIEW_SHEET_MD),
        "--review-sheet-csv",
        str(NEXT_WAVE_REVIEW_SHEET_CSV),
        "--shortlist-json",
        str(NEXT_WAVE_SHORTLIST_JSON),
        "--shortlist-md",
        str(NEXT_WAVE_SHORTLIST_MD),
        "--shortlist-csv",
        str(NEXT_WAVE_SHORTLIST_CSV),
        "--shortlist-response-json",
        str(NEXT_WAVE_SHORTLIST_RESPONSE_JSON),
        "--shortlist-response-md",
        str(NEXT_WAVE_SHORTLIST_RESPONSE_MD),
        "--manifest-json",
        str(NEXT_WAVE_MANIFEST_JSON),
        "--manifest-md",
        str(NEXT_WAVE_MANIFEST_MD),
        "--manifest-csv",
        str(NEXT_WAVE_MANIFEST_CSV),
    ]
    if args.activate_next:
        prepare_extra.append("--activate-next")
    if args.force_activate:
        prepare_extra.append("--force-activate")

    workspace_summary = run_json("prepare_labeling_workspace.py", *prepare_extra)
    status_summary = run_json(
        "labeling_status.py",
        "--hard-cases-csv",
        str(NEXT_WAVE_CSV),
        "--packets-dir",
        str(NEXT_WAVE_PACKETS_DIR),
        "--responses-dir",
        str(NEXT_WAVE_RESPONSES_DIR),
        "--output-json",
        str(NEXT_WAVE_STATUS_JSON),
    )

    summary = {
        "next_wave_csv": str(NEXT_WAVE_CSV),
        "packets_dir": str(NEXT_WAVE_PACKETS_DIR),
        "responses_dir": str(NEXT_WAVE_RESPONSES_DIR),
        "packet_summary": packet_summary,
        "workspace_summary": workspace_summary,
        "status_summary": {
            "hard_case_rows": status_summary.get("hard_case_rows"),
            "hard_case_unlabeled_rows": status_summary.get("hard_case_unlabeled_rows"),
            "packet_batches": len(status_summary.get("packet_batches", [])),
            "next_batch": status_summary.get("next_batch"),
        },
        "review_sheet_md": str(NEXT_WAVE_REVIEW_SHEET_MD),
        "review_sheet_json": str(NEXT_WAVE_REVIEW_SHEET_JSON),
        "review_sheet_csv": str(NEXT_WAVE_REVIEW_SHEET_CSV),
        "shortlist_md": str(NEXT_WAVE_SHORTLIST_MD),
        "shortlist_json": str(NEXT_WAVE_SHORTLIST_JSON),
        "shortlist_csv": str(NEXT_WAVE_SHORTLIST_CSV),
        "shortlist_response_json": str(NEXT_WAVE_SHORTLIST_RESPONSE_JSON),
        "shortlist_response_md": str(NEXT_WAVE_SHORTLIST_RESPONSE_MD),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
