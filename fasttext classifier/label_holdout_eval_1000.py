#!/usr/bin/env python3
"""Label the 1000-row classifier holdout using a local Ollama model."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
FEEDBACK_DIR = WORK_DIR / "feedback"
RESULTS_DIR = WORK_DIR / "results"
INPUT_CSV = FEEDBACK_DIR / "gpt_eval_holdout_1000.csv"
SUMMARY_JSON = RESULTS_DIR / "gpt_eval_holdout_1000_labeling_summary.json"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"
VALID_LABELS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=str(INPUT_CSV))
    parser.add_argument("--output-csv", default=str(INPUT_CSV))
    parser.add_argument("--summary-json", default=str(SUMMARY_JSON))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def build_prompt(batch: list[dict[str, str]]) -> str:
    instructions = """
You are labeling news article records for a finance and macro narrative system.

Choose exactly one label per record from:
- keep_finance: finance, markets, earnings, deals, equities, credit, commodities, company valuation, analyst/investor relevance
- keep_macro: macroeconomics, inflation, rates, central banks, labor, trade, fiscal policy, energy supply with broad market impact
- keep_geopolitics: war, sanctions, elections, diplomacy, shipping chokepoints, regulatory state action with market relevance
- keep_company_event: specific company event relevant to markets but narrower than broad finance coverage
- drop_sports
- drop_entertainment
- drop_lifestyle
- drop_local_crime
- drop_low_quality: low-signal junk, scraper, generic clickbait, low-information rewrite, SEO sludge, thin article
- drop_press_release: press release, PR mirror, sponsored corporate announcement, syndication of issuer/company publicity

Important constraints:
- Be conservative about dropping. If an industry or niche story could plausibly matter to markets, macro, geopolitics, supply chains, regulation, or a tradable company, prefer a keep label.
- Do not use source reputation alone. Judge the actual content first.
- If the item is mainly a copied press-release style corporate announcement, use drop_press_release even if it mentions finance terms.
- Output JSON only: a top-level array of objects with keys document_identifier, label, confidence, notes.
- confidence must be one of: high, medium, low.
- notes must be brief.
""".strip()

    batch_payload = []
    for row in batch:
        batch_payload.append(
            {
                "document_identifier": row["document_identifier"],
                "source_domain": row.get("source_domain", ""),
                "partition_date": row.get("partition_date", ""),
                "title": row.get("title", "")[:300],
                "summary": row.get("summary", "")[:700],
                "text": row.get("text", "")[:1200],
            }
        )
    return f"{instructions}\n\nRecords:\n{json.dumps(batch_payload, ensure_ascii=True, indent=2)}"


def ollama_label_batch(batch: list[dict[str, str]], model: str) -> list[dict[str, str]]:
    prompt = build_prompt(batch)
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.loads(response.read().decode("utf-8"))
    parsed = json.loads(payload["response"])
    if not isinstance(parsed, list):
        raise ValueError("Expected top-level JSON list from model")
    return parsed


def apply_labels(rows: list[dict[str, str]], labels: list[dict[str, str]]) -> int:
    by_id = {row["document_identifier"]: row for row in rows}
    updated = 0
    for item in labels:
        if not isinstance(item, dict):
            continue
        document_identifier = str(item.get("document_identifier") or "").strip()
        label = str(item.get("label") or "").strip()
        confidence = str(item.get("confidence") or "").strip().lower()
        notes = str(item.get("notes") or "").strip()
        if not document_identifier or document_identifier not in by_id:
            continue
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid label {label!r} for {document_identifier}")
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        row = by_id[document_identifier]
        row["gpt_label"] = label
        row["gpt_confidence"] = confidence
        row["gpt_notes"] = notes
        updated += 1
    return updated


def label_single_row(row: dict[str, str], model: str) -> dict[str, str]:
    response = ollama_label_batch([row], model)
    if not response:
        raise ValueError(f"No response for {row['document_identifier']}")
    item = response[0]
    return {
        "document_identifier": str(item.get("document_identifier") or row["document_identifier"]),
        "label": str(item.get("label") or ""),
        "confidence": str(item.get("confidence") or "low"),
        "notes": str(item.get("notes") or ""),
    }


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_csv)
    unlabeled = [row for row in rows if not row.get("gpt_label")]
    if args.max_rows is not None:
        unlabeled = unlabeled[: args.max_rows]

    processed_rows = 0
    updated_rows = 0
    batches = 0
    failures: list[dict[str, str]] = []
    started_at = time.time()

    for offset in range(0, len(unlabeled), args.batch_size):
        batch = unlabeled[offset : offset + args.batch_size]
        batches += 1
        try:
            labels = ollama_label_batch(batch, args.model)
            updated_rows += apply_labels(rows, labels)
            processed_rows += len(batch)
        except Exception as exc:
            for row in batch:
                try:
                    label = label_single_row(row, args.model)
                    updated_rows += apply_labels(rows, [label])
                    processed_rows += 1
                except Exception as row_exc:
                    failures.append(
                        {
                            "document_identifier": row["document_identifier"],
                            "error": str(row_exc),
                        }
                    )
            if not batch and str(exc):
                failures.append({"document_identifier": "", "error": str(exc)})
        write_rows(output_csv, rows)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "model": args.model,
        "batch_size": args.batch_size,
        "requested_rows": len(unlabeled),
        "processed_rows": processed_rows,
        "updated_rows": updated_rows,
        "remaining_unlabeled_rows": sum(1 for row in rows if not row.get("gpt_label")),
        "batches": batches,
        "failures": failures[:50],
        "failure_count": len(failures),
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
