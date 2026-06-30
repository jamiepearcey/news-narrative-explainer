#!/usr/bin/env python3
"""Run a tempdir smoke test for packet import, merge, and status."""

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
PACKETS_DIR = FEEDBACK_DIR / "labeling_packets"
RESPONSES_DIR = FEEDBACK_DIR / "labeling_responses"
HARD_CASES_CSV = FEEDBACK_DIR / "gpt_eval_hard_cases.csv"
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
                "confidence": "0.75",
                "notes": "synthetic smoke test label",
            }
        )
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="labeling-smoke-") as tempdir_raw:
        tempdir = Path(tempdir_raw)
        feedback_dir = tempdir / "feedback"
        packets_dir = feedback_dir / "labeling_packets"
        responses_dir = feedback_dir / "labeling_responses"
        results_dir = tempdir / "results"
        feedback_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(HARD_CASES_CSV, feedback_dir / "gpt_eval_hard_cases.csv")
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
                str(results_dir / "import_summary.json"),
            ]
        )

        merge_summary = run(
            [
                "python3",
                str(WORK_DIR / "merge_gpt_eval_labels.py"),
                "--input-csv",
                str(feedback_dir / "gpt_eval_hard_cases.csv"),
                "--packet-dir",
                str(packets_dir),
                "--hard-cases-csv",
                str(feedback_dir / "gpt_eval_hard_cases.csv"),
                "--output-csv",
                str(feedback_dir / "gpt_labeled_eval_set.csv"),
                "--summary-json",
                str(results_dir / "merge_summary.json"),
            ]
        )

        status_summary = run(
            [
                "python3",
                str(WORK_DIR / "labeling_status.py"),
                "--hard-cases-csv",
                str(feedback_dir / "gpt_eval_hard_cases.csv"),
                "--eval-csv",
                str(feedback_dir / "gpt_labeled_eval_set.csv"),
                "--packets-dir",
                str(packets_dir),
                "--responses-dir",
                str(responses_dir),
                "--output-json",
                str(results_dir / "labeling_status.json"),
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
                "batch_001": next(
                    batch for batch in status_summary["packet_batches"] if batch["batch"] == "batch_001"
                ),
            },
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
