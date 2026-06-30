#!/usr/bin/env python3
"""Finalize one next-wave labeling run: import, merge, refresh queue, reevaluate."""

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
NEXT_WAVE_SELECTION_JSON = RESULTS_DIR / "next_wave_next_batch.json"
NEXT_WAVE_IMPORT_SUMMARY_JSON = RESULTS_DIR / "next_wave_import_gpt_packet_responses_summary.json"
NEXT_WAVE_MERGE_SUMMARY_JSON = RESULTS_DIR / "next_wave_merge_gpt_eval_labels_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activate-next", action="store_true")
    parser.add_argument("--force-activate", action="store_true")
    return parser.parse_args()


def run_json(script: str, *extra: str) -> dict[str, object]:
    cmd = ["python3", str(WORK_DIR / script), *extra]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def main() -> None:
    args = parse_args()

    status_before = run_json(
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

    import_summary = run_json(
        "import_gpt_packet_responses.py",
        "--packets-dir",
        str(NEXT_WAVE_PACKETS_DIR),
        "--responses-dir",
        str(NEXT_WAVE_RESPONSES_DIR),
        "--summary-json",
        str(NEXT_WAVE_IMPORT_SUMMARY_JSON),
    )

    merge_summary = run_json(
        "merge_gpt_eval_labels.py",
        "--input-csv",
        str(NEXT_WAVE_CSV),
        "--packet-dir",
        str(NEXT_WAVE_PACKETS_DIR),
        "--hard-cases-csv",
        str(NEXT_WAVE_CSV),
        "--summary-json",
        str(NEXT_WAVE_MERGE_SUMMARY_JSON),
    )

    workspace_extra = [
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
    ]
    if args.activate_next:
        workspace_extra.append("--activate-next")
    if args.force_activate:
        workspace_extra.append("--force-activate")

    workspace_summary = run_json("prepare_labeling_workspace.py", *workspace_extra)

    status_after = run_json(
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
    active_brief = run_json(
        "build_active_labeling_brief.py",
        "--status-json",
        str(NEXT_WAVE_STATUS_JSON),
        "--output-json",
        str(NEXT_WAVE_ACTIVE_BRIEF_JSON),
        "--output-md",
        str(NEXT_WAVE_ACTIVE_BRIEF_MD),
    )
    eval_summary = run_json("evaluate_gpt_eval_set.py")

    summary = {
        "status_before_next_batch": status_before.get("next_batch"),
        "status_after_next_batch": status_after.get("next_batch"),
        "import_processed_batches": import_summary.get("processed_batches"),
        "import_updated_rows": import_summary.get("updated_rows"),
        "merge_inserted_rows": merge_summary.get("inserted_rows"),
        "merge_updated_rows": merge_summary.get("updated_rows"),
        "merge_hard_case_updates": merge_summary.get("hard_case_updates"),
        "workspace_summary": workspace_summary,
        "active_batch_after": active_brief.get("active_batch"),
        "active_remaining_rows_after": active_brief.get("remaining_rows"),
        "eval_rows": eval_summary.get("matched_rows"),
        "eval_fine_accuracy": (eval_summary.get("model_metrics") or {}).get("fine_accuracy"),
        "eval_coarse_accuracy": (eval_summary.get("model_metrics") or {}).get("coarse_accuracy"),
        "activate_next": args.activate_next,
        "force_activate": args.force_activate,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
