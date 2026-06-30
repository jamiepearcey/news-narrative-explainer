#!/usr/bin/env python3
"""Build a diverse 1000-row classifier holdout set from the scored corpus."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
RESULTS_DIR = WORK_DIR / "results"
FEEDBACK_DIR = WORK_DIR / "feedback"
SCORED_CSV = RESULTS_DIR / "scored_weak_labels.csv"
OUTPUT_CSV = FEEDBACK_DIR / "gpt_eval_holdout_1000.csv"
SUMMARY_JSON = RESULTS_DIR / "gpt_eval_holdout_1000_summary.json"

LABEL_TARGETS = {
    "keep_macro": 300,
    "keep_finance": 260,
    "keep_geopolitics": 120,
    "drop_low_quality": 110,
    "drop_press_release": 110,
    "keep_company_event": 50,
    "drop_local_crime": 50,
}
DECISION_BAND_ORDER = ["review", "auto_keep", "band_keep", "auto_drop"]
MAX_PER_DOMAIN = 3
RELAXED_MAX_PER_DOMAIN = 6
FINAL_MAX_PER_DOMAIN = 12


def stable_score(document_identifier: str) -> int:
    digest = hashlib.sha256(document_identifier.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prepare_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "partition_date": row.get("partition_date", ""),
        "source_domain": row.get("source_domain", ""),
        "document_identifier": row.get("document_identifier", ""),
        "predicted_label": row.get("predicted_label", ""),
        "predicted_score": row.get("predicted_score", ""),
        "decision_band": row.get("decision_band", ""),
        "weak_label": row.get("label", ""),
        "title": row.get("title", ""),
        "summary": row.get("summary", "")[:800],
        "text": row.get("text", "")[:1200],
        "reasons": row.get("reasons", ""),
        "finance_cluster_score": row.get("finance_cluster_score", ""),
        "finance_hits": row.get("finance_hits", ""),
        "macro_hits": row.get("macro_hits", ""),
        "geo_hits": row.get("geo_hits", ""),
        "company_hits": row.get("company_hits", ""),
        "equity_hits": row.get("equity_hits", ""),
        "press_hits": row.get("press_hits", ""),
        "keep_theme_hits": row.get("keep_theme_hits", ""),
        "macro_theme_hits": row.get("macro_theme_hits", ""),
        "geo_theme_hits": row.get("geo_theme_hits", ""),
        "gpt_label": "",
        "gpt_confidence": "",
        "gpt_notes": "",
    }


def rank_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    def sort_key(row: dict[str, str]) -> tuple[int, int, int, str]:
        title = row.get("title", "")
        text = row.get("text", "")
        natural_length = len(title) + len(text)
        review_bonus = 1 if row.get("decision_band") == "review" else 0
        context_bonus = 1 if row.get("summary") or row.get("text") else 0
        hash_score = stable_score(row.get("document_identifier", ""))
        return (-review_bonus, -context_bonus, -natural_length, f"{hash_score:020d}")

    return sorted(rows, key=sort_key)


def build_sample(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        predicted_label = row.get("predicted_label", "")
        if predicted_label in LABEL_TARGETS:
            by_label[predicted_label].append(row)

    for label in by_label:
        by_label[label] = rank_rows(by_label[label])

    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()
    domain_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    for label, target in LABEL_TARGETS.items():
        label_rows = by_label.get(label, [])
        band_buckets: dict[str, list[dict[str, str]]] = {band: [] for band in DECISION_BAND_ORDER}
        other_rows: list[dict[str, str]] = []
        for row in label_rows:
            band = row.get("decision_band", "")
            if band in band_buckets:
                band_buckets[band].append(row)
            else:
                other_rows.append(row)

        band_targets: dict[str, int] = {}
        remaining = target
        available_total = sum(len(bucket) for bucket in band_buckets.values()) + len(other_rows)
        if available_total < target:
            target = available_total
            remaining = target

        for band in DECISION_BAND_ORDER:
            bucket = band_buckets[band]
            if not bucket or remaining <= 0:
                band_targets[band] = 0
                continue
            share = round(target * len(bucket) / max(available_total, 1))
            share = max(1, min(len(bucket), share))
            share = min(share, remaining)
            band_targets[band] = share
            remaining -= share

        for band in DECISION_BAND_ORDER:
            bucket = band_buckets[band]
            idx = 0
            while band_targets[band] > 0 and idx < len(bucket):
                row = bucket[idx]
                idx += 1
                document_identifier = row["document_identifier"]
                source_domain = row["source_domain"]
                if document_identifier in selected_ids or domain_counts[source_domain] >= MAX_PER_DOMAIN:
                    continue
                selected.append(prepare_row(row))
                selected_ids.add(document_identifier)
                domain_counts[source_domain] += 1
                label_counts[label] += 1
                band_targets[band] -= 1

        leftovers = []
        for band in DECISION_BAND_ORDER:
            leftovers.extend(band_buckets[band])
        leftovers.extend(other_rows)
        leftovers = rank_rows(leftovers)
        for row in leftovers:
            if label_counts[label] >= target:
                break
            document_identifier = row["document_identifier"]
            source_domain = row["source_domain"]
            if document_identifier in selected_ids or domain_counts[source_domain] >= MAX_PER_DOMAIN:
                continue
            selected.append(prepare_row(row))
            selected_ids.add(document_identifier)
            domain_counts[source_domain] += 1
            label_counts[label] += 1

        if label_counts[label] < target:
            for row in leftovers:
                if label_counts[label] >= target:
                    break
                document_identifier = row["document_identifier"]
                source_domain = row["source_domain"]
                if document_identifier in selected_ids or domain_counts[source_domain] >= RELAXED_MAX_PER_DOMAIN:
                    continue
                selected.append(prepare_row(row))
                selected_ids.add(document_identifier)
                domain_counts[source_domain] += 1
                label_counts[label] += 1
                if label_counts[label] >= target:
                    break

        if label_counts[label] < target:
            for row in leftovers:
                if label_counts[label] >= target:
                    break
                document_identifier = row["document_identifier"]
                source_domain = row["source_domain"]
                if document_identifier in selected_ids or domain_counts[source_domain] >= FINAL_MAX_PER_DOMAIN:
                    continue
                selected.append(prepare_row(row))
                selected_ids.add(document_identifier)
                domain_counts[source_domain] += 1
                label_counts[label] += 1
                if label_counts[label] >= target:
                    break

    return selected


def main() -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows(SCORED_CSV)
    sample = build_sample(rows)

    fieldnames = list(sample[0].keys()) if sample else []
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(sample)

    summary = {
        "source_csv": str(SCORED_CSV),
        "output_csv": str(OUTPUT_CSV),
        "rows": len(sample),
        "label_targets": LABEL_TARGETS,
        "predicted_label_counts": dict(sorted(Counter(row["predicted_label"] for row in sample).items())),
        "decision_band_counts": dict(sorted(Counter(row["decision_band"] for row in sample).items())),
        "unique_domains": len({row["source_domain"] for row in sample}),
        "max_rows_per_domain": max(Counter(row["source_domain"] for row in sample).values()) if sample else 0,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
