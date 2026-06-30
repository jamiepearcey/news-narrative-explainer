#!/usr/bin/env python3
"""Summarize teacher-vs-weak-label disagreements by domain and label."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
RESULTS_DIR = WORK_DIR / "results"
TEACHER_LABELS_CSV = RESULTS_DIR / "vector_teacher_labels.csv"
DOMAIN_SCORES_CSV = RESULTS_DIR / "effective_domain_scores.csv"
OUTPUT_CSV = RESULTS_DIR / "vector_teacher_domain_disagreements.csv"
OUTPUT_JSON = RESULTS_DIR / "vector_teacher_domain_disagreements_summary.json"


def to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except ValueError:
        return default


def load_domain_scores() -> dict[str, dict[str, str]]:
    if not DOMAIN_SCORES_CSV.exists():
        return {}
    with DOMAIN_SCORES_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["source_domain"]: row for row in reader}


def main() -> None:
    domain_scores = load_domain_scores()
    aggregates: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "rows": 0,
            "trainable_rows": 0,
            "review_rows": 0,
            "teacher_drop_rows": 0,
            "teacher_keep_rows": 0,
            "weak_drop_rows": 0,
            "weak_keep_rows": 0,
            "weak_teacher_disagreements": 0,
            "predicted_teacher_disagreements": 0,
            "teacher_labels": Counter(),
            "weak_labels": Counter(),
            "reasons": Counter(),
            "examples": [],
        }
    )

    if not TEACHER_LABELS_CSV.exists():
        raise SystemExit("Missing vector_teacher_labels.csv. Run build_vector_teacher_labels.py first.")

    with TEACHER_LABELS_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            domain = row["source_domain"]
            bucket = aggregates[domain]
            teacher_label = row["teacher_label"]
            weak_label = row["weak_label"]
            predicted_label = row["predicted_label"]
            bucket["rows"] += 1
            if row["teacher_trainable"] == "1":
                bucket["trainable_rows"] += 1
            if teacher_label == "review":
                bucket["review_rows"] += 1
            elif teacher_label.startswith("drop_"):
                bucket["teacher_drop_rows"] += 1
            elif teacher_label.startswith("keep_"):
                bucket["teacher_keep_rows"] += 1
            if weak_label.startswith("drop_"):
                bucket["weak_drop_rows"] += 1
            elif weak_label.startswith("keep_"):
                bucket["weak_keep_rows"] += 1
            if teacher_label != "review" and teacher_label != weak_label:
                bucket["weak_teacher_disagreements"] += 1
            if teacher_label != "review" and teacher_label != predicted_label:
                bucket["predicted_teacher_disagreements"] += 1
            bucket["teacher_labels"][teacher_label] += 1
            bucket["weak_labels"][weak_label] += 1
            bucket["reasons"][row["teacher_reason"]] += 1
            if (
                teacher_label != "review"
                and teacher_label != weak_label
                and len(bucket["examples"]) < 5
            ):
                bucket["examples"].append(
                    {
                        "weak_label": weak_label,
                        "predicted_label": predicted_label,
                        "teacher_label": teacher_label,
                        "teacher_reason": row["teacher_reason"],
                        "title": row["title"],
                    }
                )

    rows_out: list[dict[str, str]] = []
    for domain, stats in aggregates.items():
        score_meta = domain_scores.get(domain, {})
        rows = int(stats["rows"])
        trainable_rows = int(stats["trainable_rows"])
        review_rows = int(stats["review_rows"])
        weak_teacher_disagreements = int(stats["weak_teacher_disagreements"])
        teacher_drop_rows = int(stats["teacher_drop_rows"])
        teacher_keep_rows = int(stats["teacher_keep_rows"])
        weak_teacher_disagreement_rate = (weak_teacher_disagreements / rows) if rows else 0.0
        predicted_teacher_disagreement_rate = (int(stats["predicted_teacher_disagreements"]) / rows) if rows else 0.0
        trainable_rate = (trainable_rows / rows) if rows else 0.0
        volume_weighted_disagreement = weak_teacher_disagreement_rate * math.log2(rows + 1) if rows else 0.0
        drop_promotion_rows = sum(
            count for label, count in stats["teacher_labels"].items() if label.startswith("drop_")
        )
        weak_keep_to_teacher_drop = max(0, drop_promotion_rows - int(stats["weak_drop_rows"]))
        rows_out.append(
            {
                "source_domain": domain,
                "rows": str(rows),
                "trainable_rows": str(trainable_rows),
                "review_rows": str(review_rows),
                "teacher_drop_rows": str(teacher_drop_rows),
                "teacher_keep_rows": str(teacher_keep_rows),
                "weak_drop_rows": str(stats["weak_drop_rows"]),
                "weak_keep_rows": str(stats["weak_keep_rows"]),
                "weak_teacher_disagreements": str(weak_teacher_disagreements),
                "weak_teacher_disagreement_rate": f"{weak_teacher_disagreement_rate:.4f}",
                "predicted_teacher_disagreements": str(stats["predicted_teacher_disagreements"]),
                "predicted_teacher_disagreement_rate": f"{predicted_teacher_disagreement_rate:.4f}",
                "trainable_rate": f"{trainable_rate:.4f}",
                "volume_weighted_disagreement": f"{volume_weighted_disagreement:.4f}",
                "weak_keep_to_teacher_drop": str(weak_keep_to_teacher_drop),
                "effective_archetype": score_meta.get("effective_archetype", ""),
                "effective_score_0_10": score_meta.get("effective_score_0_10", ""),
                "market_relevance_rate": score_meta.get("market_relevance_rate", ""),
                "junk_rate": score_meta.get("junk_rate", ""),
                "top_teacher_labels_json": json.dumps(stats["teacher_labels"].most_common(4)),
                "top_weak_labels_json": json.dumps(stats["weak_labels"].most_common(4)),
                "top_reasons_json": json.dumps(stats["reasons"].most_common(4)),
                "examples_json": json.dumps(stats["examples"], ensure_ascii=False),
            }
        )

    rows_out.sort(key=lambda row: (-to_float(row["volume_weighted_disagreement"]), -int(row["rows"])))

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()) if rows_out else [])
        if rows_out:
            writer.writeheader()
            writer.writerows(rows_out)

    summary = {
        "rows": len(rows_out),
        "top_domains_by_volume_weighted_disagreement": rows_out[:25],
        "top_domains_by_raw_disagreement_count": sorted(
            rows_out,
            key=lambda row: (-int(row["weak_teacher_disagreements"]), -int(row["rows"])),
        )[:25],
        "top_domains_by_teacher_drop_promotion": sorted(
            rows_out,
            key=lambda row: (-int(row["weak_keep_to_teacher_drop"]), -int(row["teacher_drop_rows"]), -int(row["rows"])),
        )[:25],
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
