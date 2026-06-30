#!/usr/bin/env python3
"""Merge newly labeled GPT/manual rows into the canonical eval set."""

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
DATA_DIR = WORK_DIR / "data"
DEFAULT_INPUT_CSV = FEEDBACK_DIR / "gpt_eval_hard_cases.csv"
DEFAULT_PACKET_DIR = FEEDBACK_DIR / "labeling_packets"
DEFAULT_HARD_CASES_CSV = FEEDBACK_DIR / "gpt_eval_hard_cases.csv"
DEFAULT_OUTPUT_CSV = FEEDBACK_DIR / "gpt_labeled_eval_set.csv"
DEFAULT_SCORED_CSV = RESULTS_DIR / "scored_weak_labels.csv"
DEFAULT_REVIEW_CSV = FEEDBACK_DIR / "review_queue.csv"
DEFAULT_CORPUS_CSV = DATA_DIR / "corpus_projection.csv"
DEFAULT_SUMMARY_JSON = RESULTS_DIR / "merge_gpt_eval_labels_summary.json"

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

PREFERRED_FIELD_ORDER = [
    "document_identifier",
    "label",
    "stratum",
    "source_domain",
    "partition_date",
    "title",
    "summary_text",
    "market_context_text",
    "summary",
    "text",
    "weak_label",
    "predicted_label",
    "predicted_score",
    "decision_band",
    "label_source",
    "finance_cluster_score",
    "source_profile",
    "market_relevance_rate",
    "industry_signal_rate",
    "junk_rate",
    "reasons",
    "why_selected",
    "confidence",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--packet-dir", default=str(DEFAULT_PACKET_DIR))
    parser.add_argument("--hard-cases-csv", default=str(DEFAULT_HARD_CASES_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--scored-csv", default=str(DEFAULT_SCORED_CSV))
    parser.add_argument("--review-csv", default=str(DEFAULT_REVIEW_CSV))
    parser.add_argument("--corpus-csv", default=str(DEFAULT_CORPUS_CSV))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_packet_templates(packet_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not packet_dir.exists():
        return rows
    for path in sorted(packet_dir.glob("batch_*.template.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit(f"Expected list payload in {path}")
        for row in payload:
            if not isinstance(row, dict):
                raise SystemExit(f"Expected object rows in {path}")
            rows.append({key: str(value) if value is not None else "" for key, value in row.items()})
    return rows


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        normalized[key] = (value or "").strip()
    normalized["document_identifier"] = normalized.get("document_identifier", "")
    normalized["label"] = normalized.get("label", "")
    normalized["stratum"] = normalized.get("stratum", "")
    normalized["source_domain"] = normalized.get("source_domain", "")
    normalized["confidence"] = normalized.get("confidence", "")
    normalized["notes"] = normalized.get("notes", "")
    return normalized


def preferred_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    ordered = [field for field in PREFERRED_FIELD_ORDER if field in keys]
    remainder = sorted(keys - set(ordered))
    return ordered + remainder


def merge_prefer_existing(base: dict[str, str], incoming: dict[str, str]) -> dict[str, str]:
    merged = dict(base)
    for key, value in incoming.items():
        if key not in merged or not merged[key]:
            merged[key] = value
    return merged


def enrich_row(
    row: dict[str, str],
    hard_cases_by_id: dict[str, dict[str, str]],
    scored_by_id: dict[str, dict[str, str]],
    review_by_id: dict[str, dict[str, str]],
    corpus_by_id: dict[str, dict[str, str]],
) -> dict[str, str]:
    document_identifier = row.get("document_identifier", "")
    enriched = dict(row)
    if document_identifier in hard_cases_by_id:
        enriched = merge_prefer_existing(enriched, normalize_row(hard_cases_by_id[document_identifier]))
    scored = scored_by_id.get(document_identifier)
    if scored:
        scored_projection = {
            "partition_date": scored.get("partition_date", ""),
            "source_domain": scored.get("source_domain", ""),
            "title": scored.get("title", ""),
            "summary": scored.get("summary", ""),
            "text": scored.get("text", ""),
            "weak_label": scored.get("label", ""),
            "predicted_label": scored.get("predicted_label", ""),
            "predicted_score": scored.get("predicted_score", ""),
            "decision_band": scored.get("decision_band", ""),
            "label_source": scored.get("label_source", ""),
            "finance_cluster_score": scored.get("finance_cluster_score", ""),
            "source_profile": scored.get("source_profile", ""),
            "market_relevance_rate": scored.get("market_relevance_rate", ""),
            "industry_signal_rate": scored.get("industry_signal_rate", ""),
            "junk_rate": scored.get("junk_rate", ""),
            "reasons": scored.get("reasons", ""),
        }
        enriched = merge_prefer_existing(enriched, normalize_row(scored_projection))
    review = review_by_id.get(document_identifier)
    if review:
        review_projection = {
            "partition_date": review.get("partition_date", ""),
            "source_domain": review.get("source_domain", ""),
            "title": review.get("title", ""),
            "summary": review.get("summary", ""),
            "text": review.get("text", ""),
            "decision_band": "review",
            "label_source": review.get("label_source", ""),
            "finance_cluster_score": review.get("finance_cluster_score", ""),
            "source_profile": review.get("source_profile", ""),
            "market_relevance_rate": review.get("market_relevance_rate", ""),
            "industry_signal_rate": review.get("industry_signal_rate", ""),
            "junk_rate": review.get("junk_rate", ""),
            "reasons": review.get("reasons", ""),
        }
        enriched = merge_prefer_existing(enriched, normalize_row(review_projection))
    corpus = corpus_by_id.get(document_identifier)
    if corpus:
        corpus_projection = {
            "partition_date": corpus.get("partition_date", ""),
            "source_domain": corpus.get("source_domain", ""),
            "title": corpus.get("resolved_title", ""),
            "summary": corpus.get("summary", ""),
            "text": corpus.get("text_excerpt", ""),
        }
        enriched = merge_prefer_existing(enriched, normalize_row(corpus_projection))
    return enriched


def merge_rows(
    source_rows: list[dict[str, str]],
    existing_by_id: dict[str, dict[str, str]],
) -> tuple[int, int, int, list[dict[str, str]]]:
    inserted = 0
    updated = 0
    skipped_unlabeled = 0
    invalid_labels: list[dict[str, str]] = []

    for raw_row in source_rows:
        label = (raw_row.get("label") or "").strip()
        if not label:
            skipped_unlabeled += 1
            continue
        if label not in VALID_LABELS:
            invalid_labels.append(
                {
                    "document_identifier": raw_row.get("document_identifier", ""),
                    "label": label,
                }
            )
            continue
        row = normalize_row(raw_row)
        document_identifier = row["document_identifier"]
        if document_identifier in existing_by_id:
            existing_by_id[document_identifier] = row
            updated += 1
        else:
            existing_by_id[document_identifier] = row
            inserted += 1

    return inserted, updated, skipped_unlabeled, invalid_labels


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    packet_dir = Path(args.packet_dir)
    hard_cases_csv = Path(args.hard_cases_csv)
    output_csv = Path(args.output_csv)
    scored_csv = Path(args.scored_csv)
    review_csv = Path(args.review_csv)
    corpus_csv = Path(args.corpus_csv)
    summary_json = Path(args.summary_json)

    incoming_rows = read_csv(input_csv)
    packet_rows = read_packet_templates(packet_dir)
    existing_rows = read_csv(output_csv)
    hard_case_rows = read_csv(hard_cases_csv)
    scored_rows = read_csv(scored_csv)
    review_rows = read_csv(review_csv)
    corpus_rows = read_csv(corpus_csv)
    if not incoming_rows and not packet_rows:
        raise SystemExit(f"No rows found in {input_csv}")

    existing_by_id = {
        row["document_identifier"]: normalize_row(row)
        for row in existing_rows
        if row.get("document_identifier")
    }
    hard_cases_by_id = {
        row["document_identifier"]: dict(row)
        for row in hard_case_rows
        if row.get("document_identifier")
    }
    scored_by_id = {
        row["document_identifier"]: dict(row)
        for row in scored_rows
        if row.get("document_identifier")
    }
    review_by_id = {
        row["document_identifier"]: dict(row)
        for row in review_rows
        if row.get("document_identifier")
    }
    corpus_by_id = {
        row["document_identifier"]: dict(row)
        for row in corpus_rows
        if row.get("document_identifier")
    }

    packet_inserted, packet_updated, packet_skipped, packet_invalid = merge_rows(packet_rows, existing_by_id)
    csv_inserted, csv_updated, csv_skipped, csv_invalid = merge_rows(incoming_rows, existing_by_id)

    invalid_labels = packet_invalid + csv_invalid
    if invalid_labels:
        raise SystemExit(
            "Invalid labels found: "
            + json.dumps(invalid_labels[:10], ensure_ascii=True)
        )

    merged_rows = [
        enrich_row(row, hard_cases_by_id, scored_by_id, review_by_id, corpus_by_id)
        for row in existing_by_id.values()
    ]
    merged_rows.sort(key=lambda row: (row.get("stratum", ""), row["document_identifier"]))
    fieldnames = preferred_fieldnames(merged_rows)
    write_csv(output_csv, merged_rows, fieldnames)

    hard_case_updates = 0
    if hard_case_rows:
        for row in merged_rows:
            existing = hard_cases_by_id.get(row["document_identifier"])
            if not existing:
                continue
            if (existing.get("label") or "").strip() != row["label"]:
                hard_case_updates += 1
            existing["label"] = row["label"]
            existing["confidence"] = row["confidence"]
            existing["notes"] = row["notes"]
            existing["stratum"] = existing.get("stratum") or row["stratum"]
            existing["source_domain"] = existing.get("source_domain") or row["source_domain"]
        hard_case_fieldnames = list(hard_case_rows[0].keys())
        updated_hard_case_rows = [
            hard_cases_by_id[row["document_identifier"]] if row.get("document_identifier") in hard_cases_by_id else row
            for row in hard_case_rows
        ]
        write_csv(hard_cases_csv, updated_hard_case_rows, hard_case_fieldnames)

    summary = {
        "input_csv": str(input_csv),
        "packet_dir": str(packet_dir),
        "hard_cases_csv": str(hard_cases_csv),
        "output_csv": str(output_csv),
        "existing_rows": len(existing_rows),
        "incoming_rows": len(incoming_rows),
        "packet_rows": len(packet_rows),
        "packet_inserted_rows": packet_inserted,
        "packet_updated_rows": packet_updated,
        "packet_skipped_unlabeled_rows": packet_skipped,
        "csv_inserted_rows": csv_inserted,
        "csv_updated_rows": csv_updated,
        "csv_skipped_unlabeled_rows": csv_skipped,
        "inserted_rows": packet_inserted + csv_inserted,
        "updated_rows": packet_updated + csv_updated,
        "skipped_unlabeled_rows": packet_skipped + csv_skipped,
        "hard_case_updates": hard_case_updates,
        "merged_rows": len(merged_rows),
        "strata_counts": dict(sorted(Counter(row["stratum"] for row in merged_rows).items())),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
