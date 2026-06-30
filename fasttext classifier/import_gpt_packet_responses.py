#!/usr/bin/env python3
"""Bulk import GPT/manual response files into packet templates."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
FEEDBACK_DIR = WORK_DIR / "feedback"
DEFAULT_PACKETS_DIR = FEEDBACK_DIR / "labeling_packets"
DEFAULT_RESPONSES_DIR = FEEDBACK_DIR / "labeling_responses"
DEFAULT_SUMMARY_JSON = WORK_DIR / "results" / "import_gpt_packet_responses_summary.json"


def load_single_importer():
    importer_path = WORK_DIR / "import_gpt_packet_response.py"
    spec = importlib.util.spec_from_file_location("import_gpt_packet_response", importer_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load importer from {importer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets-dir", default=str(DEFAULT_PACKETS_DIR))
    parser.add_argument("--responses-dir", default=str(DEFAULT_RESPONSES_DIR))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--include-stubs", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def find_response_file(responses_dir: Path, stem: str, include_stubs: bool) -> Path | None:
    candidates = [
        responses_dir / f"{stem}.response.json",
        responses_dir / f"{stem}.response.md",
        responses_dir / f"{stem}.response.txt",
    ]
    if include_stubs:
        candidates.append(responses_dir / f"{stem}.response.stub.json")
    for path in candidates:
        if path.exists():
            return path
    return None


def main() -> None:
    args = parse_args()
    packets_dir = Path(args.packets_dir)
    responses_dir = Path(args.responses_dir)
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    importer = load_single_importer()
    batches = sorted(packets_dir.glob("batch_*.template.json"))
    if not batches:
        raise SystemExit(f"No packet templates found in {packets_dir}")

    processed = 0
    updated_rows = 0
    missing_responses = 0
    per_batch: list[dict[str, object]] = []

    for template_json in batches:
        stem = template_json.name.removesuffix(".template.json")
        response_file = find_response_file(responses_dir, stem, args.include_stubs)
        if response_file is None:
            missing_responses += 1
            per_batch.append(
                {
                    "batch": stem,
                    "template_json": str(template_json),
                    "response_file": None,
                    "updated_rows": 0,
                    "status": "missing_response",
                }
            )
            continue

        template_rows = json.loads(template_json.read_text(encoding="utf-8"))
        response_rows = importer.extract_json_payload(response_file.read_text(encoding="utf-8"))
        template_by_id = {row["document_identifier"]: row for row in template_rows}

        batch_updates = 0
        unknown_ids: list[str] = []
        invalid_labels: list[dict[str, str]] = []

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
            if label and label not in importer.VALID_LABELS:
                invalid_labels.append({"document_identifier": document_identifier, "label": label})
                continue
            template_row["label"] = label
            template_row["confidence"] = confidence
            template_row["notes"] = notes
            batch_updates += 1

        if invalid_labels:
            raise SystemExit("Invalid labels found: " + json.dumps(invalid_labels[:10], ensure_ascii=True))
        if args.strict and unknown_ids:
            raise SystemExit("Unknown document identifiers found: " + json.dumps(unknown_ids[:10], ensure_ascii=True))

        template_json.write_text(json.dumps(template_rows, indent=2), encoding="utf-8")
        processed += 1
        updated_rows += batch_updates
        per_batch.append(
            {
                "batch": stem,
                "template_json": str(template_json),
                "response_file": str(response_file),
                "updated_rows": batch_updates,
                "unknown_ids": len(unknown_ids),
                "status": "imported",
            }
        )

    summary = {
        "packets_dir": str(packets_dir),
        "responses_dir": str(responses_dir),
        "processed_batches": processed,
        "missing_response_batches": missing_responses,
        "updated_rows": updated_rows,
        "include_stubs": args.include_stubs,
        "strict": args.strict,
        "batches": per_batch,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
