#!/usr/bin/env python3
"""Evaluate the binary finance-influential objective on the manually judged holdout."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
FEEDBACK_DIR = WORK_DIR / "feedback"
RESULTS_DIR = WORK_DIR / "results"

HOLDOUT_CSV = FEEDBACK_DIR / "gpt_eval_holdout_1000.csv"
SCORED_BINARY_CSV = RESULTS_DIR / "finance_influential_scored.csv"
OUTPUT_JSON = RESULTS_DIR / "finance_influential_holdout_metrics.json"
OUTPUT_MD = RESULTS_DIR / "finance_influential_holdout_report.md"


def collapse_multiclass(label: str) -> str | None:
    if label.startswith("keep_"):
        return "finance_influential"
    if label.startswith("drop_"):
        return "not_finance_influential"
    return None


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    with SCORED_BINARY_CSV.open("r", encoding="utf-8") as handle:
        scored_by_id = {row["document_identifier"]: row for row in csv.DictReader(handle)}

    rows = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    score_bands: Counter[str] = Counter()
    score_band_correct: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    truth_counts: Counter[str] = Counter()

    with HOLDOUT_CSV.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gpt_label = (row.get("gpt_label") or "").strip()
            if not gpt_label:
                continue
            truth = collapse_multiclass(gpt_label)
            scored = scored_by_id.get(row["document_identifier"])
            if not truth or not scored:
                continue

            predicted = scored["predicted_binary_label"]
            score = float(scored["predicted_binary_score"])
            band = scored["predicted_binary_band"]
            confusion[truth][predicted] += 1
            truth_counts[truth] += 1
            predicted_counts[predicted] += 1
            score_bands[band] += 1
            if truth == predicted:
                score_band_correct[band] += 1
            rows.append(
                {
                    "document_identifier": row["document_identifier"],
                    "truth": truth,
                    "predicted": predicted,
                    "score": score,
                    "band": band,
                }
            )

    correct = sum(1 for row in rows if row["truth"] == row["predicted"])
    finance_tp = confusion["finance_influential"]["finance_influential"]
    finance_fp = confusion["not_finance_influential"]["finance_influential"]
    finance_fn = confusion["finance_influential"]["not_finance_influential"]
    precision = safe_div(finance_tp, finance_tp + finance_fp)
    recall = safe_div(finance_tp, finance_tp + finance_fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    metrics = {
        "rows_labeled_and_scored": len(rows),
        "accuracy": round(safe_div(correct, len(rows)), 4),
        "finance_influential_precision": round(precision, 4),
        "finance_influential_recall": round(recall, 4),
        "finance_influential_f1": round(f1, 4),
        "confusion_matrix": {
            truth: dict(sorted(counter.items()))
            for truth, counter in sorted(confusion.items())
        },
        "truth_counts": dict(sorted(truth_counts.items())),
        "predicted_counts": dict(sorted(predicted_counts.items())),
        "band_quality": {
            band: {
                "rows": rows_count,
                "accuracy": round(safe_div(score_band_correct[band], rows_count), 4),
            }
            for band, rows_count in sorted(score_bands.items())
        },
    }

    OUTPUT_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(
        "\n".join(
            [
                "# Finance Influential Holdout",
                "",
                f"- rows: {metrics['rows_labeled_and_scored']}",
                f"- accuracy: {metrics['accuracy']}",
                f"- finance influential precision: {metrics['finance_influential_precision']}",
                f"- finance influential recall: {metrics['finance_influential_recall']}",
                f"- finance influential f1: {metrics['finance_influential_f1']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
