#!/usr/bin/env python3
"""Apply the active next-wave shortlist response and refresh workspace artifacts."""

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
NEXT_WAVE_APPLY_SHORTLIST_SUMMARY_JSON = RESULTS_DIR / "next_wave_apply_shortlist_response_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activate-next", action="store_true")
    parser.add_argument("--force-activate", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--sync-response-file", action="store_true")
    parser.add_argument("--hard-cases-csv", default=str(NEXT_WAVE_CSV))
    parser.add_argument("--packets-dir", default=str(NEXT_WAVE_PACKETS_DIR))
    parser.add_argument("--responses-dir", default=str(NEXT_WAVE_RESPONSES_DIR))
    parser.add_argument("--status-json", default=str(NEXT_WAVE_STATUS_JSON))
    parser.add_argument("--selection-json", default=str(NEXT_WAVE_SELECTION_JSON))
    parser.add_argument("--active-brief-json", default=str(NEXT_WAVE_ACTIVE_BRIEF_JSON))
    parser.add_argument("--active-brief-md", default=str(NEXT_WAVE_ACTIVE_BRIEF_MD))
    parser.add_argument("--review-sheet-json", default=str(NEXT_WAVE_REVIEW_SHEET_JSON))
    parser.add_argument("--review-sheet-md", default=str(NEXT_WAVE_REVIEW_SHEET_MD))
    parser.add_argument("--review-sheet-csv", default=str(NEXT_WAVE_REVIEW_SHEET_CSV))
    parser.add_argument("--shortlist-json", default=str(NEXT_WAVE_SHORTLIST_JSON))
    parser.add_argument("--shortlist-md", default=str(NEXT_WAVE_SHORTLIST_MD))
    parser.add_argument("--shortlist-csv", default=str(NEXT_WAVE_SHORTLIST_CSV))
    parser.add_argument("--shortlist-response-json", default=str(NEXT_WAVE_SHORTLIST_RESPONSE_JSON))
    parser.add_argument("--shortlist-response-md", default=str(NEXT_WAVE_SHORTLIST_RESPONSE_MD))
    parser.add_argument("--apply-summary-json", default=str(NEXT_WAVE_APPLY_SHORTLIST_SUMMARY_JSON))
    return parser.parse_args()


def run_json(script: str, *extra: str) -> dict[str, object]:
    cmd = ["python3", str(WORK_DIR / script), *extra]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def main() -> None:
    args = parse_args()
    hard_cases_csv = str(Path(args.hard_cases_csv))
    packets_dir = str(Path(args.packets_dir))
    responses_dir = str(Path(args.responses_dir))
    status_json = str(Path(args.status_json))

    status_before = run_json(
        "labeling_status.py",
        "--hard-cases-csv",
        hard_cases_csv,
        "--packets-dir",
        packets_dir,
        "--responses-dir",
        responses_dir,
        "--output-json",
        status_json,
    )

    shortlist_extra = [
        "--status-json",
        status_json,
        "--shortlist-response-json",
        str(Path(args.shortlist_response_json)),
        "--summary-json",
        str(Path(args.apply_summary_json)),
    ]
    if args.strict:
        shortlist_extra.append("--strict")
    if args.sync_response_file:
        shortlist_extra.append("--sync-response-file")

    shortlist_apply = run_json("apply_active_shortlist_response.py", *shortlist_extra)

    workspace_extra = [
        "--hard-cases-csv",
        hard_cases_csv,
        "--packets-dir",
        packets_dir,
        "--responses-dir",
        responses_dir,
        "--status-json",
        status_json,
        "--selection-json",
        str(Path(args.selection_json)),
        "--active-brief-json",
        str(Path(args.active_brief_json)),
        "--active-brief-md",
        str(Path(args.active_brief_md)),
        "--review-sheet-json",
        str(Path(args.review_sheet_json)),
        "--review-sheet-md",
        str(Path(args.review_sheet_md)),
        "--review-sheet-csv",
        str(Path(args.review_sheet_csv)),
        "--shortlist-json",
        str(Path(args.shortlist_json)),
        "--shortlist-md",
        str(Path(args.shortlist_md)),
        "--shortlist-csv",
        str(Path(args.shortlist_csv)),
        "--shortlist-response-json",
        str(Path(args.shortlist_response_json)),
        "--shortlist-response-md",
        str(Path(args.shortlist_response_md)),
    ]
    if args.activate_next:
        workspace_extra.append("--activate-next")
    if args.force_activate:
        workspace_extra.append("--force-activate")

    workspace_summary = run_json("prepare_labeling_workspace.py", *workspace_extra)

    status_after = run_json(
        "labeling_status.py",
        "--hard-cases-csv",
        hard_cases_csv,
        "--packets-dir",
        packets_dir,
        "--responses-dir",
        responses_dir,
        "--output-json",
        status_json,
    )

    summary = {
        "status_before_next_batch": status_before.get("next_batch"),
        "status_after_next_batch": status_after.get("next_batch"),
        "shortlist_apply": shortlist_apply,
        "workspace_summary": workspace_summary,
        "active_remaining_rows_after": workspace_summary.get("active_remaining_rows"),
        "shortlist_rows_after": workspace_summary.get("shortlist_rows"),
        "activate_next": args.activate_next,
        "force_activate": args.force_activate,
        "strict": args.strict,
        "sync_response_file": args.sync_response_file,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
