#!/usr/bin/env python3
"""Advance the active labeling queue when the current batch is complete."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
RESULTS_DIR = WORK_DIR / "results"
STATUS_JSON = RESULTS_DIR / "labeling_status.json"
ACTIVE_BRIEF_JSON = RESULTS_DIR / "active_labeling_brief.json"
SELECTION_JSON = RESULTS_DIR / "next_labeling_batch.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_json(cmd: list[str]) -> dict[str, object]:
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def refresh_status() -> dict[str, object]:
    return run_json(["python3", str(WORK_DIR / "labeling_status.py")])


def build_brief() -> dict[str, object]:
    return run_json(["python3", str(WORK_DIR / "build_active_labeling_brief.py")])


def select_next(activate: bool) -> dict[str, object]:
    cmd = ["python3", str(WORK_DIR / "select_next_labeling_batch.py")]
    if activate:
        cmd.append("--activate-response")
    return run_json(cmd)


def main() -> None:
    args = parse_args()
    status = refresh_status()
    brief = build_brief()

    active_batch = str(brief["active_batch"])
    remaining_rows = int(brief["remaining_rows"])
    activated_next = None

    if remaining_rows == 0 or args.force:
        activated_next = select_next(activate=True)
        status = refresh_status()
        brief = build_brief()

    summary = {
        "active_batch_before": active_batch,
        "remaining_rows_before": remaining_rows,
        "force": args.force,
        "activated_next": activated_next,
        "status_json": str(STATUS_JSON),
        "active_brief_json": str(ACTIVE_BRIEF_JSON),
        "selection_json": str(SELECTION_JSON),
        "active_batch_after": brief["active_batch"],
        "remaining_rows_after": brief["remaining_rows"],
        "next_batch_after": status.get("next_batch"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
