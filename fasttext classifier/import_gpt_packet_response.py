#!/usr/bin/env python3
"""Import a GPT/manual JSON response into a packet template."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
FEEDBACK_DIR = WORK_DIR / "feedback"
DEFAULT_TEMPLATE_JSON = FEEDBACK_DIR / "labeling_packets/batch_001.template.json"

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
    parser.add_argument("--template-json", default=str(DEFAULT_TEMPLATE_JSON))
    parser.add_argument("--response-file", required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def extract_json_payload(text: str) -> list[dict[str, object]]:
    stripped = text.strip()
    if stripped.startswith("["):
        payload = json.loads(stripped)
    else:
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if not match:
            raise SystemExit("Could not find JSON array in response file")
        payload = json.loads(match.group(1))
    if not isinstance(payload, list):
        raise SystemExit("Expected top-level JSON list")
    return payload


def main() -> None:
    args = parse_args()
    template_json = Path(args.template_json)
    response_file = Path(args.response_file)

    template_rows = json.loads(template_json.read_text(encoding="utf-8"))
    response_rows = extract_json_payload(response_file.read_text(encoding="utf-8"))

    template_by_id = {row["document_identifier"]: row for row in template_rows}
    updates = 0
    invalid_labels: list[dict[str, str]] = []
    unknown_ids: list[str] = []

    for raw_row in response_rows:
        if not isinstance(raw_row, dict):
            continue
        document_identifier = str(raw_row.get("document_identifier") or "").strip()
        if not document_identifier:
            continue
        template_row = template_by_id.get(document_identifier)
        if template_row is None:
            unknown_ids.append(document_identifier)
            continue
        label = str(raw_row.get("label") or "").strip()
        confidence = str(raw_row.get("confidence") or "").strip()
        notes = str(raw_row.get("notes") or "").strip()
        if not label and not confidence and not notes:
            continue
        if label and label not in VALID_LABELS:
            invalid_labels.append({"document_identifier": document_identifier, "label": label})
            continue
        template_row["label"] = label
        template_row["confidence"] = confidence
        template_row["notes"] = notes
        updates += 1

    if invalid_labels:
        raise SystemExit("Invalid labels found: " + json.dumps(invalid_labels[:10], ensure_ascii=True))
    if args.strict and unknown_ids:
        raise SystemExit("Unknown document identifiers found: " + json.dumps(unknown_ids[:10], ensure_ascii=True))

    template_json.write_text(json.dumps(template_rows, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "template_json": str(template_json),
                "response_file": str(response_file),
                "updated_rows": updates,
                "unknown_ids": len(unknown_ids),
                "strict": args.strict,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
