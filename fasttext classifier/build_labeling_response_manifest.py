#!/usr/bin/env python3
"""Build a manifest for expected packet response files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
FEEDBACK_DIR = WORK_DIR / "feedback"
PACKETS_DIR = FEEDBACK_DIR / "labeling_packets"
RESPONSES_DIR = FEEDBACK_DIR / "labeling_responses"
MANIFEST_JSON = RESPONSES_DIR / "manifest.json"
MANIFEST_MD = RESPONSES_DIR / "manifest.md"
MANIFEST_CSV = RESPONSES_DIR / "manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets-dir", default=str(PACKETS_DIR))
    parser.add_argument("--responses-dir", default=str(RESPONSES_DIR))
    parser.add_argument("--manifest-json", default=str(MANIFEST_JSON))
    parser.add_argument("--manifest-md", default=str(MANIFEST_MD))
    parser.add_argument("--manifest-csv", default=str(MANIFEST_CSV))
    return parser.parse_args()


def read_packet_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_template_rows(path: Path) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    packets_dir = Path(args.packets_dir)
    responses_dir = Path(args.responses_dir)
    manifest_json = Path(args.manifest_json)
    manifest_md = Path(args.manifest_md)
    manifest_csv = Path(args.manifest_csv)

    responses_dir.mkdir(parents=True, exist_ok=True)
    batches: list[dict[str, object]] = []

    for rows_path in sorted(packets_dir.glob("batch_*.rows.csv")):
        stem = rows_path.name.removesuffix(".rows.csv")
        template_path = packets_dir / f"{stem}.template.json"
        prompt_path = packets_dir / f"{stem}.prompt.md"
        response_json = responses_dir / f"{stem}.response.json"
        response_md = responses_dir / f"{stem}.response.md"
        response_txt = responses_dir / f"{stem}.response.txt"
        stub_json = responses_dir / f"{stem}.response.stub.json"

        rows = read_packet_rows(rows_path)
        template_rows = read_template_rows(template_path) if template_path.exists() else []
        completed = sum(1 for row in template_rows if (row.get("label") or "").strip())

        if not stub_json.exists():
            stub_payload = [
                {
                    "document_identifier": row["document_identifier"],
                    "label": "",
                    "confidence": "",
                    "notes": "",
                }
                for row in rows
            ]
            stub_json.write_text(json.dumps(stub_payload, indent=2), encoding="utf-8")

        batches.append(
            {
                "batch": stem,
                "rows": len(rows),
                "completed_template_rows": completed,
                "remaining_template_rows": max(len(template_rows) - completed, 0),
                "strata_counts": dict(sorted(Counter(row["stratum"] for row in rows).items())),
                "prompt_path": str(prompt_path),
                "template_path": str(template_path),
                "rows_path": str(rows_path),
                "expected_response_json": str(response_json),
                "expected_response_md": str(response_md),
                "expected_response_txt": str(response_txt),
                "stub_response_json": str(stub_json),
            }
        )

    summary = {
        "responses_dir": str(responses_dir),
        "batch_count": len(batches),
        "batches": batches,
    }
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_md.parent.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import csv

    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "batch",
            "rows",
            "completed_template_rows",
            "remaining_template_rows",
            "prompt_path",
            "template_path",
            "rows_path",
            "expected_response_json",
            "expected_response_md",
            "expected_response_txt",
            "stub_response_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for batch in batches:
            writer.writerow({key: batch[key] for key in fieldnames})

    lines = [
        "# Labeling Response Manifest",
        "",
        f"Responses directory: `{responses_dir}`",
        "",
        f"Manifest CSV: `{manifest_csv}`",
        "",
    ]
    for batch in batches:
        lines.extend(
            [
                f"## {batch['batch']}",
                "",
                f"- Rows: `{batch['rows']}`",
                f"- Remaining template rows: `{batch['remaining_template_rows']}`",
                f"- Prompt: `{batch['prompt_path']}`",
                f"- Template: `{batch['template_path']}`",
                f"- Rows CSV: `{batch['rows_path']}`",
                f"- Expected response file: `{batch['expected_response_json']}`",
                f"- Alternate response files: `{batch['expected_response_md']}`, `{batch['expected_response_txt']}`",
                f"- Stub response file: `{batch['stub_response_json']}`",
                f"- Strata: `{json.dumps(batch['strata_counts'], ensure_ascii=True)}`",
                "",
            ]
        )
    manifest_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
