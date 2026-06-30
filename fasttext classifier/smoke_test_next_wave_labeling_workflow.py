#!/usr/bin/env python3
"""Run a tempdir smoke test for the dedicated next-wave labeling workflow."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
FEEDBACK_DIR = WORK_DIR / "feedback"
RESULTS_DIR = WORK_DIR / "results"

NEXT_WAVE_CSV = FEEDBACK_DIR / "gpt_eval_next_wave.csv"
PACKETS_DIR = FEEDBACK_DIR / "next_wave_labeling_packets"
RESPONSES_DIR = FEEDBACK_DIR / "next_wave_labeling_responses"
EVAL_CSV = FEEDBACK_DIR / "gpt_labeled_eval_set.csv"


def run(cmd: list[str]) -> dict[str, object]:
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def make_synthetic_response(rows_csv: Path, output_json: Path, limit: int = 3) -> None:
    with rows_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    payload: list[dict[str, str]] = []
    for row in rows[:limit]:
        predicted_label = row.get("predicted_label") or row.get("weak_label") or "keep_macro"
        payload.append(
            {
                "document_identifier": row["document_identifier"],
                "label": predicted_label,
                "confidence": "0.78",
                "notes": "synthetic next-wave smoke test label",
            }
        )
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_synthetic_shortlist_response(source_json: Path, output_json: Path, limit: int = 2) -> None:
    rows = json.loads(source_json.read_text(encoding="utf-8"))
    payload: list[dict[str, str]] = []
    for row in rows[:limit]:
        payload.append(
            {
                "document_identifier": row["document_identifier"],
                "label": "keep_macro",
                "confidence": "0.81",
                "notes": "synthetic shortlist smoke test label",
            }
        )
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="next-wave-labeling-smoke-") as tempdir_raw:
        tempdir = Path(tempdir_raw)
        feedback_dir = tempdir / "feedback"
        packets_dir = feedback_dir / "next_wave_labeling_packets"
        responses_dir = feedback_dir / "next_wave_labeling_responses"
        results_dir = tempdir / "results"
        feedback_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(NEXT_WAVE_CSV, feedback_dir / "gpt_eval_next_wave.csv")
        shutil.copy(EVAL_CSV, feedback_dir / "gpt_labeled_eval_set.csv")
        shutil.copytree(PACKETS_DIR, packets_dir)
        shutil.copytree(RESPONSES_DIR, responses_dir)

        synthetic_response = responses_dir / "batch_001.response.json"
        make_synthetic_response(packets_dir / "batch_001.rows.csv", synthetic_response)

        import_summary = run(
            [
                "python3",
                str(WORK_DIR / "import_gpt_packet_responses.py"),
                "--packets-dir",
                str(packets_dir),
                "--responses-dir",
                str(responses_dir),
                "--summary-json",
                str(results_dir / "next_wave_import_summary.json"),
            ]
        )

        merge_summary = run(
            [
                "python3",
                str(WORK_DIR / "merge_gpt_eval_labels.py"),
                "--input-csv",
                str(feedback_dir / "gpt_eval_next_wave.csv"),
                "--packet-dir",
                str(packets_dir),
                "--hard-cases-csv",
                str(feedback_dir / "gpt_eval_next_wave.csv"),
                "--output-csv",
                str(feedback_dir / "gpt_labeled_eval_set.csv"),
                "--summary-json",
                str(results_dir / "next_wave_merge_summary.json"),
            ]
        )

        status_summary = run(
            [
                "python3",
                str(WORK_DIR / "labeling_status.py"),
                "--hard-cases-csv",
                str(feedback_dir / "gpt_eval_next_wave.csv"),
                "--eval-csv",
                str(feedback_dir / "gpt_labeled_eval_set.csv"),
                "--packets-dir",
                str(packets_dir),
                "--responses-dir",
                str(responses_dir),
                "--output-json",
                str(results_dir / "next_wave_labeling_status.json"),
            ]
        )

        active_brief = run(
            [
                "python3",
                str(WORK_DIR / "build_active_labeling_brief.py"),
                "--status-json",
                str(results_dir / "next_wave_labeling_status.json"),
                "--output-json",
                str(results_dir / "next_wave_active_labeling_brief.json"),
                "--output-md",
                str(results_dir / "next_wave_active_labeling_brief.md"),
            ]
        )

        review_sheet = run(
            [
                "python3",
                str(WORK_DIR / "build_labeling_review_sheet.py"),
                "--status-json",
                str(results_dir / "next_wave_labeling_status.json"),
                "--output-json",
                str(results_dir / "next_wave_active_labeling_review_sheet.json"),
                "--output-md",
                str(results_dir / "next_wave_active_labeling_review_sheet.md"),
                "--output-csv",
                str(results_dir / "next_wave_active_labeling_review_sheet.csv"),
                "--output-shortlist-json",
                str(results_dir / "next_wave_active_labeling_shortlist.json"),
                "--output-shortlist-md",
                str(results_dir / "next_wave_active_labeling_shortlist.md"),
                "--output-shortlist-csv",
                str(results_dir / "next_wave_active_labeling_shortlist.csv"),
                "--output-shortlist-response-json",
                str(results_dir / "next_wave_active_labeling_shortlist.response.json"),
                "--output-shortlist-response-md",
                str(results_dir / "next_wave_active_labeling_shortlist.response.md"),
                "--shortlist-limit",
                "5",
            ]
        )

        shortlist_response = results_dir / "next_wave_active_labeling_shortlist.response.json"
        make_synthetic_shortlist_response(shortlist_response, shortlist_response)

        shortlist_apply = run(
            [
                "python3",
                str(WORK_DIR / "apply_active_shortlist_response.py"),
                "--status-json",
                str(results_dir / "next_wave_labeling_status.json"),
                "--shortlist-response-json",
                str(shortlist_response),
                "--summary-json",
                str(results_dir / "next_wave_apply_shortlist_summary.json"),
                "--sync-response-file",
            ]
        )

        shortlist_finalize = run(
            [
                "python3",
                str(WORK_DIR / "finalize_next_wave_shortlist_run.py"),
                "--hard-cases-csv",
                str(feedback_dir / "gpt_eval_next_wave.csv"),
                "--packets-dir",
                str(packets_dir),
                "--responses-dir",
                str(responses_dir),
                "--status-json",
                str(results_dir / "next_wave_labeling_status.json"),
                "--selection-json",
                str(results_dir / "next_wave_next_batch.json"),
                "--active-brief-json",
                str(results_dir / "next_wave_active_labeling_brief.json"),
                "--active-brief-md",
                str(results_dir / "next_wave_active_labeling_brief.md"),
                "--review-sheet-json",
                str(results_dir / "next_wave_active_labeling_review_sheet.json"),
                "--review-sheet-md",
                str(results_dir / "next_wave_active_labeling_review_sheet.md"),
                "--review-sheet-csv",
                str(results_dir / "next_wave_active_labeling_review_sheet.csv"),
                "--shortlist-json",
                str(results_dir / "next_wave_active_labeling_shortlist.json"),
                "--shortlist-md",
                str(results_dir / "next_wave_active_labeling_shortlist.md"),
                "--shortlist-csv",
                str(results_dir / "next_wave_active_labeling_shortlist.csv"),
                "--shortlist-response-json",
                str(results_dir / "next_wave_active_labeling_shortlist.response.json"),
                "--shortlist-response-md",
                str(results_dir / "next_wave_active_labeling_shortlist.response.md"),
                "--apply-summary-json",
                str(results_dir / "next_wave_apply_shortlist_summary.json"),
                "--strict",
                "--sync-response-file",
            ]
        )

        summary = {
            "tempdir": str(tempdir),
            "synthetic_response": str(synthetic_response),
            "import_summary": import_summary,
            "merge_summary": merge_summary,
            "status_summary": {
                "hard_case_labeled_rows": status_summary["hard_case_labeled_rows"],
                "canonical_eval_rows": status_summary["canonical_eval_rows"],
                "merged_from_hard_cases": status_summary["merged_from_hard_cases"],
                "next_batch": status_summary["next_batch"],
                "batch_001": next(
                    batch for batch in status_summary["packet_batches"] if batch["batch"] == "batch_001"
                ),
            },
            "active_brief": {
                "active_batch": active_brief["active_batch"],
                "completed_rows": active_brief["completed_rows"],
                "remaining_rows": active_brief["remaining_rows"],
                "response_path": active_brief["response_path"],
            },
            "review_sheet": {
                "rows": review_sheet["rows"],
                "remaining_rows": review_sheet["remaining_rows"],
                "focus_counts": review_sheet["focus_counts"],
                "shortlist_rows": review_sheet["shortlist_rows"],
                "first_document": (review_sheet["review_rows"] or [{}])[0].get("document_identifier"),
            },
            "shortlist_apply": shortlist_apply,
            "shortlist_finalize": shortlist_finalize,
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
