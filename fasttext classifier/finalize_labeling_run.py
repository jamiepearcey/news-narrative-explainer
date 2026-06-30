#!/usr/bin/env python3
"""Finalize one labeling run: import, merge, advance, and reevaluate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--advance-force", action="store_true")
    return parser.parse_args()


def run_json(script: str, *extra: str) -> dict[str, object]:
    cmd = ["python3", str(WORK_DIR / script), *extra]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def main() -> None:
    args = parse_args()

    status_before = run_json("labeling_status.py")
    import_summary = run_json("import_gpt_packet_responses.py")
    merge_summary = run_json("merge_gpt_eval_labels.py")

    advance_extra: list[str] = []
    if args.advance_force:
        advance_extra.append("--force")
    advance_summary = run_json("advance_labeling_queue.py", *advance_extra)

    status_after = run_json("labeling_status.py")
    active_brief = run_json("build_active_labeling_brief.py")
    eval_summary = run_json("evaluate_gpt_eval_set.py")

    summary = {
        "status_before_next_batch": status_before.get("next_batch"),
        "status_after_next_batch": status_after.get("next_batch"),
        "import_processed_batches": import_summary.get("processed_batches"),
        "import_updated_rows": import_summary.get("updated_rows"),
        "merge_inserted_rows": merge_summary.get("inserted_rows"),
        "merge_updated_rows": merge_summary.get("updated_rows"),
        "merge_hard_case_updates": merge_summary.get("hard_case_updates"),
        "advance_summary": advance_summary,
        "active_batch_after": active_brief.get("active_batch"),
        "active_remaining_rows_after": active_brief.get("remaining_rows"),
        "eval_rows": eval_summary.get("matched_rows"),
        "eval_fine_accuracy": (eval_summary.get("model_metrics") or {}).get("fine_accuracy"),
        "eval_coarse_accuracy": (eval_summary.get("model_metrics") or {}).get("coarse_accuracy"),
        "advance_force": args.advance_force,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
