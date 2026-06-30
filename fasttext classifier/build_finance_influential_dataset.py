#!/usr/bin/env python3
"""Build a binary finance-influential dataset from the existing weak labels."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
DATA_DIR = WORK_DIR / "data"
RESULTS_DIR = WORK_DIR / "results"

WEAK_LABEL_CSV = DATA_DIR / "weak_labels.csv"
BINARY_LABEL_CSV = DATA_DIR / "finance_influential_labels.csv"
TRAIN_TXT = DATA_DIR / "finance_influential_train.txt"
VALID_TXT = DATA_DIR / "finance_influential_valid.txt"
SUMMARY_JSON = RESULTS_DIR / "finance_influential_dataset_summary.json"


def binary_label(label: str) -> str | None:
    if label.startswith("keep_"):
        return "finance_influential"
    if label.startswith("drop_"):
        return "not_finance_influential"
    return None


def split_bucket(document_identifier: str) -> str:
    digest = hashlib.sha1(document_identifier.encode("utf-8")).hexdigest()
    return "valid" if int(digest[:8], 16) % 10 == 0 else "train"


def main() -> None:
    if not WEAK_LABEL_CSV.exists():
        raise SystemExit("Missing weak_labels.csv. Run build_weak_labels.py first.")

    rows_out: list[dict[str, str]] = []
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    with WEAK_LABEL_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = binary_label(row.get("label", ""))
            if not label:
                continue
            split = split_bucket(row["document_identifier"])
            text = row.get("text", "").strip()
            if not text:
                continue

            out = dict(row)
            out["original_label"] = row["label"]
            out["binary_label"] = label
            out["dataset_split"] = split
            rows_out.append(out)

            split_counts[split] += 1
            label_counts[label] += 1
            source_counts[row.get("label_source") or ""] += 1

    fieldnames = list(rows_out[0].keys()) if rows_out else []
    BINARY_LABEL_CSV.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with BINARY_LABEL_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    with TRAIN_TXT.open("w", encoding="utf-8") as train_handle, VALID_TXT.open("w", encoding="utf-8") as valid_handle:
        for row in rows_out:
            line = f"__label__{row['binary_label']} {row['text'].strip()}\n"
            if row["dataset_split"] == "valid":
                valid_handle.write(line)
            else:
                train_handle.write(line)

    summary = {
        "rows": len(rows_out),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "label_source_counts": dict(sorted(source_counts.items())),
        "train_txt": str(TRAIN_TXT),
        "valid_txt": str(VALID_TXT),
        "binary_label_csv": str(BINARY_LABEL_CSV),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
