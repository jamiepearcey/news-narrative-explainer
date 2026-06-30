#!/usr/bin/env python3
"""Select the next labeling batch and optionally activate its response file."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
RESULTS_DIR = WORK_DIR / "results"
DEFAULT_STATUS_JSON = RESULTS_DIR / "labeling_status.json"
DEFAULT_SELECTION_JSON = RESULTS_DIR / "next_labeling_batch.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-json", default=str(DEFAULT_STATUS_JSON))
    parser.add_argument("--selection-json", default=str(DEFAULT_SELECTION_JSON))
    parser.add_argument("--activate-response", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def batch_priority(batch: dict[str, object]) -> tuple[int, int, str]:
    remaining = int(batch.get("remaining_template_rows") or 0)
    response_present = 1 if batch.get("response_present") else 0
    return (response_present, -remaining, str(batch.get("batch") or ""))


def main() -> None:
    args = parse_args()
    status_json = Path(args.status_json)
    selection_json = Path(args.selection_json)

    status = json.loads(status_json.read_text(encoding="utf-8"))
    batches = status.get("packet_batches", [])
    if not batches:
        raise SystemExit("No packet batches found in status file")

    candidates = [
        batch
        for batch in batches
        if int(batch.get("remaining_template_rows") or 0) > 0
    ]
    if not candidates:
        raise SystemExit("No remaining labeling work found")

    selected = sorted(candidates, key=batch_priority)[0]

    activated_response = None
    if args.activate_response:
        response_path = Path(selected["expected_response_json"])
        stub_path = Path(selected["stub_response_json"])
        if response_path.exists() and not args.force:
            activated_response = str(response_path)
        else:
            response_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(stub_path, response_path)
            activated_response = str(response_path)

    summary = {
        "selected_batch": selected["batch"],
        "remaining_template_rows": selected["remaining_template_rows"],
        "prompt_path": selected["prompt_path"],
        "template_path": selected["template_path"],
        "rows_path": selected["rows_path"],
        "expected_response_json": selected["expected_response_json"],
        "stub_response_json": selected["stub_response_json"],
        "response_present": selected["response_present"],
        "stub_response_present": selected["stub_response_present"],
        "activated_response": activated_response,
        "activate_response": args.activate_response,
        "force": args.force,
    }
    selection_json.parent.mkdir(parents=True, exist_ok=True)
    selection_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
