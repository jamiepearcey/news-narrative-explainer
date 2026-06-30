#!/usr/bin/env python3
"""Build a concise operator brief for the current active labeling batch."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
RESULTS_DIR = WORK_DIR / "results"
OUTPUT_MD = RESULTS_DIR / "active_labeling_brief.md"
OUTPUT_JSON = RESULTS_DIR / "active_labeling_brief.json"
STATUS_JSON = RESULTS_DIR / "labeling_status.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-json", default=str(STATUS_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    return parser.parse_args()


def load_status(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def choose_active_batch(status: dict[str, object]) -> dict[str, object]:
    packet_batches = status.get("packet_batches", [])
    with_real_response = [batch for batch in packet_batches if batch.get("response_present")]
    if with_real_response:
        return sorted(
            with_real_response,
            key=lambda batch: (str(batch.get("batch") or ""),),
        )[0]
    next_batch_name = status.get("next_batch")
    for batch in packet_batches:
        if batch.get("batch") == next_batch_name:
            return batch
    raise SystemExit("No active or next batch found in labeling_status.json")


def main() -> None:
    args = parse_args()
    status = load_status(Path(args.status_json))
    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    batch = choose_active_batch(status)
    rows_path = Path(batch["rows_path"])
    template_path = Path(batch["template_path"])
    response_path = Path(batch["response_path"]) if batch.get("response_path") else Path(batch["expected_response_json"])

    rows = read_rows(rows_path)
    template_rows = json.loads(template_path.read_text(encoding="utf-8"))
    template_by_id = {row["document_identifier"]: row for row in template_rows}

    completed = 0
    preview_rows: list[dict[str, str]] = []
    for row in rows:
        template_row = template_by_id.get(row["document_identifier"], {})
        label = (template_row.get("label") or "").strip()
        if label:
            completed += 1
        if len(preview_rows) < 8:
            preview_rows.append(
                {
                    "document_identifier": row["document_identifier"],
                    "source_domain": row.get("source_domain", ""),
                    "stratum": row.get("stratum", ""),
                    "weak_label": row.get("weak_label", ""),
                    "predicted_label": row.get("predicted_label", ""),
                    "predicted_score": row.get("predicted_score", ""),
                    "title": row.get("title", ""),
                    "current_label": label,
                }
            )

    summary = {
        "active_batch": batch["batch"],
        "rows": len(rows),
        "completed_rows": completed,
        "remaining_rows": len(rows) - completed,
        "strata_counts": dict(sorted(Counter(row["stratum"] for row in rows).items())),
        "prompt_path": batch["prompt_path"],
        "template_path": batch["template_path"],
        "rows_path": batch["rows_path"],
        "response_path": str(response_path),
        "response_present": batch.get("response_present", False),
        "preview_rows": preview_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# Active Labeling Brief: {summary['active_batch']}",
        "",
        f"- Rows: `{summary['rows']}`",
        f"- Completed: `{summary['completed_rows']}`",
        f"- Remaining: `{summary['remaining_rows']}`",
        f"- Prompt: `{summary['prompt_path']}`",
        f"- Template: `{summary['template_path']}`",
        f"- Rows CSV: `{summary['rows_path']}`",
        f"- Response target: `{summary['response_path']}`",
        f"- Response present: `{summary['response_present']}`",
        f"- Strata: `{json.dumps(summary['strata_counts'], ensure_ascii=True)}`",
        "",
        "## Preview",
        "",
    ]
    for row in preview_rows:
        lines.extend(
            [
                f"### {row['source_domain']} | {row['stratum']}",
                "",
                f"- Weak label: `{row['weak_label']}`",
                f"- Predicted label: `{row['predicted_label']}` at `{row['predicted_score']}`",
                f"- Current label: `{row['current_label'] or 'unlabeled'}`",
                f"- Title: {row['title']}",
                f"- Document: `{row['document_identifier']}`",
                "",
            ]
        )
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
