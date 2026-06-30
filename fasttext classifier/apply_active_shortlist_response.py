#!/usr/bin/env python3
"""Apply a shortlist response payload into the current active packet template."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
RESULTS_DIR = WORK_DIR / "results"
STATUS_JSON = RESULTS_DIR / "labeling_status.json"
SHORTLIST_RESPONSE_JSON = RESULTS_DIR / "active_labeling_shortlist.response.json"
SUMMARY_JSON = RESULTS_DIR / "apply_active_shortlist_response_summary.json"


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
    parser.add_argument("--status-json", default=str(STATUS_JSON))
    parser.add_argument("--shortlist-response-json", default=str(SHORTLIST_RESPONSE_JSON))
    parser.add_argument("--summary-json", default=str(SUMMARY_JSON))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--sync-response-file", action="store_true")
    return parser.parse_args()


def load_status(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_active_batch(status: dict[str, object]) -> dict[str, object]:
    packet_batches = status.get("packet_batches", [])
    with_real_response = [batch for batch in packet_batches if batch.get("response_present")]
    if with_real_response:
        return sorted(with_real_response, key=lambda batch: str(batch.get("batch") or ""))[0]
    next_batch_name = status.get("next_batch")
    for batch in packet_batches:
        if batch.get("batch") == next_batch_name:
            return batch
    raise SystemExit("No active or next batch found in labeling_status.json")


def merge_payload(existing_rows: list[dict[str, object]], new_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_id = {}
    order: list[str] = []
    for row in existing_rows + new_rows:
        if not isinstance(row, dict):
            continue
        document_identifier = str(row.get("document_identifier") or "").strip()
        if not document_identifier:
            continue
        if document_identifier not in by_id:
            order.append(document_identifier)
        by_id[document_identifier] = {
            "document_identifier": document_identifier,
            "label": str(row.get("label") or "").strip(),
            "confidence": str(row.get("confidence") or "").strip(),
            "notes": str(row.get("notes") or "").strip(),
        }
    return [by_id[document_identifier] for document_identifier in order]


def main() -> None:
    args = parse_args()
    status = load_status(Path(args.status_json))
    shortlist_response_json = Path(args.shortlist_response_json)
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    importer = load_single_importer()
    batch = choose_active_batch(status)
    template_json = Path(batch["template_path"])
    response_path = Path(batch["response_path"]) if batch.get("response_path") else Path(batch["expected_response_json"])

    template_rows = json.loads(template_json.read_text(encoding="utf-8"))
    response_rows = importer.extract_json_payload(shortlist_response_json.read_text(encoding="utf-8"))
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
        if label and label not in importer.VALID_LABELS:
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

    synced_response_rows = 0
    if args.sync_response_file:
        existing_response_rows: list[dict[str, object]] = []
        if response_path.exists():
            existing_response_rows = importer.extract_json_payload(response_path.read_text(encoding="utf-8"))
        merged = merge_payload(existing_response_rows, response_rows)
        response_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        synced_response_rows = len(merged)

    summary = {
        "active_batch": batch["batch"],
        "template_json": str(template_json),
        "response_path": str(response_path),
        "shortlist_response_json": str(shortlist_response_json),
        "updated_rows": updates,
        "unknown_ids": len(unknown_ids),
        "strict": args.strict,
        "sync_response_file": args.sync_response_file,
        "synced_response_rows": synced_response_rows,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
