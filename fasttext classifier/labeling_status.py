#!/usr/bin/env python3
"""Report labeling packet and eval-set completion status."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
FEEDBACK_DIR = WORK_DIR / "feedback"
RESULTS_DIR = WORK_DIR / "results"
HARD_CASES_CSV = FEEDBACK_DIR / "gpt_eval_hard_cases.csv"
EVAL_CSV = FEEDBACK_DIR / "gpt_labeled_eval_set.csv"
PACKETS_DIR = FEEDBACK_DIR / "labeling_packets"
RESPONSES_DIR = FEEDBACK_DIR / "labeling_responses"
OUTPUT_JSON = RESULTS_DIR / "labeling_status.json"
VALID_LABELS = {
    "keep_finance",
    "keep_macro",
    "keep_geopolitics",
    "keep_company_event",
    "drop_sports",
    "drop_entertainment",
    "drop_lifestyle",
    "drop_local_crime",
    "drop_low_quality",
    "drop_press_release",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-cases-csv", default=str(HARD_CASES_CSV))
    parser.add_argument("--eval-csv", default=str(EVAL_CSV))
    parser.add_argument("--packets-dir", default=str(PACKETS_DIR))
    parser.add_argument("--responses-dir", default=str(RESPONSES_DIR))
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def packet_status(packets_dir: Path, responses_dir: Path) -> list[dict[str, object]]:
    batches: list[dict[str, object]] = []
    for rows_path in sorted(packets_dir.glob("batch_*.rows.csv")):
        stem = rows_path.name.removesuffix(".rows.csv")
        template_path = packets_dir / f"{stem}.template.json"
        prompt_path = packets_dir / f"{stem}.prompt.md"
        response_candidates = [
            responses_dir / f"{stem}.response.json",
            responses_dir / f"{stem}.response.md",
            responses_dir / f"{stem}.response.txt",
        ]
        response_path = next((path for path in response_candidates if path.exists()), None)
        stub_response_path = responses_dir / f"{stem}.response.stub.json"
        rows = read_csv(rows_path)
        template_rows = json.loads(template_path.read_text(encoding="utf-8")) if template_path.exists() else []
        completed = sum(1 for row in template_rows if (row.get("label") or "").strip())
        invalid = sum(
            1
            for row in template_rows
            if (row.get("label") or "").strip() and (row.get("label") or "").strip() not in VALID_LABELS
        )
        batches.append(
            {
                "batch": stem,
                "rows": len(rows),
                "template_rows": len(template_rows),
                "completed_template_rows": completed,
                "remaining_template_rows": max(len(template_rows) - completed, 0),
                "invalid_template_rows": invalid,
                "strata_counts": dict(sorted(Counter(row["stratum"] for row in rows).items())),
                "prompt_path": str(prompt_path),
                "template_path": str(template_path),
                "rows_path": str(rows_path),
                "expected_response_json": str(responses_dir / f"{stem}.response.json"),
                "stub_response_json": str(stub_response_path),
                "response_path": str(response_path) if response_path else "",
                "response_present": bool(response_path),
                "stub_response_present": stub_response_path.exists(),
            }
        )
    return batches


def main() -> None:
    args = parse_args()
    hard_cases_csv = Path(args.hard_cases_csv)
    eval_csv = Path(args.eval_csv)
    packets_dir = Path(args.packets_dir)
    responses_dir = Path(args.responses_dir)
    output_json = Path(args.output_json)

    hard_case_rows = read_csv(hard_cases_csv)
    eval_rows = read_csv(eval_csv)
    batches = packet_status(packets_dir, responses_dir)

    hard_case_labeled = sum(1 for row in hard_case_rows if (row.get("label") or "").strip())
    hard_case_invalid = sum(
        1
        for row in hard_case_rows
        if (row.get("label") or "").strip() and (row.get("label") or "").strip() not in VALID_LABELS
    )
    eval_ids = {row["document_identifier"] for row in eval_rows if row.get("document_identifier")}
    hard_case_ids = {row["document_identifier"] for row in hard_case_rows if row.get("document_identifier")}
    merged_from_hard_cases = len(eval_ids & hard_case_ids)

    summary = {
        "hard_cases_csv": str(hard_cases_csv),
        "eval_csv": str(eval_csv),
        "responses_dir": str(responses_dir),
        "hard_case_rows": len(hard_case_rows),
        "hard_case_labeled_rows": hard_case_labeled,
        "hard_case_unlabeled_rows": len(hard_case_rows) - hard_case_labeled,
        "hard_case_invalid_rows": hard_case_invalid,
        "canonical_eval_rows": len(eval_rows),
        "canonical_eval_strata_counts": dict(sorted(Counter(row["stratum"] for row in eval_rows).items())),
        "hard_case_strata_counts": dict(sorted(Counter(row["stratum"] for row in hard_case_rows).items())),
        "merged_from_hard_cases": merged_from_hard_cases,
        "unmerged_hard_case_rows": len(hard_case_ids - eval_ids),
        "packet_batches": batches,
    }
    next_batch = None
    remaining_batches = [
        batch for batch in batches if int(batch.get("remaining_template_rows") or 0) > 0
    ]
    if remaining_batches:
        next_batch = sorted(
            remaining_batches,
            key=lambda batch: (
                1 if batch.get("response_present") else 0,
                -int(batch.get("remaining_template_rows") or 0),
                str(batch.get("batch") or ""),
            ),
        )[0]["batch"]
    summary["next_batch"] = next_batch
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
