#!/usr/bin/env python3
"""Evaluate classifier predictions against the LLM-labeled holdout set."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
FEEDBACK_DIR = WORK_DIR / "feedback"
RESULTS_DIR = WORK_DIR / "results"
INPUT_CSV = FEEDBACK_DIR / "gpt_eval_holdout_1000.csv"
OUTPUT_JSON = RESULTS_DIR / "gpt_eval_holdout_1000_metrics.json"
OUTPUT_MD = RESULTS_DIR / "gpt_eval_holdout_1000_report.md"

KEEP_LABELS = {"keep_finance", "keep_macro", "keep_geopolitics", "keep_company_event"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred) / len(y_true)


def label_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        support = sum(1 for truth in y_true if truth == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return metrics


def macro_f1(metrics: dict[str, dict[str, float]]) -> float:
    if not metrics:
        return 0.0
    return round(sum(item["f1"] for item in metrics.values()) / len(metrics), 4)


def weighted_f1(metrics: dict[str, dict[str, float]]) -> float:
    total = sum(item["support"] for item in metrics.values())
    if not total:
        return 0.0
    return round(sum(item["f1"] * item["support"] for item in metrics.values()) / total, 4)


def coarse(label: str) -> str:
    return "keep" if label in KEEP_LABELS else "drop"


def main() -> None:
    rows = [row for row in read_rows(INPUT_CSV) if row.get("gpt_label")]
    labels = sorted({row["gpt_label"] for row in rows} | {row["predicted_label"] for row in rows})
    y_true = [row["gpt_label"] for row in rows]
    y_pred = [row["predicted_label"] for row in rows]

    fine = label_metrics(y_true, y_pred, labels)
    coarse_true = [coarse(label) for label in y_true]
    coarse_pred = [coarse(label) for label in y_pred]
    coarse_labels = ["drop", "keep"]
    coarse_metrics = label_metrics(coarse_true, coarse_pred, coarse_labels)

    confusion: dict[str, dict[str, int]] = defaultdict(dict)
    for truth in labels:
        for pred in labels:
            confusion[truth][pred] = 0
    for truth, pred in zip(y_true, y_pred):
        confusion[truth][pred] += 1

    band_accuracy: dict[str, float] = {}
    confidence_accuracy: dict[str, float] = {}
    for band in sorted({row["decision_band"] for row in rows}):
        band_rows = [row for row in rows if row["decision_band"] == band]
        band_accuracy[band] = round(
            accuracy([row["gpt_label"] for row in band_rows], [row["predicted_label"] for row in band_rows]),
            4,
        )
    for confidence in sorted({row["gpt_confidence"] for row in rows}):
        conf_rows = [row for row in rows if row["gpt_confidence"] == confidence]
        confidence_accuracy[confidence] = round(
            accuracy([row["gpt_label"] for row in conf_rows], [row["predicted_label"] for row in conf_rows]),
            4,
        )

    summary = {
        "input_csv": str(INPUT_CSV),
        "rows": len(rows),
        "gpt_label_counts": dict(sorted(Counter(y_true).items())),
        "predicted_label_counts": dict(sorted(Counter(y_pred).items())),
        "fine_accuracy": round(accuracy(y_true, y_pred), 4),
        "fine_macro_f1": macro_f1(fine),
        "fine_weighted_f1": weighted_f1(fine),
        "coarse_accuracy": round(accuracy(coarse_true, coarse_pred), 4),
        "coarse_macro_f1": macro_f1(coarse_metrics),
        "coarse_weighted_f1": weighted_f1(coarse_metrics),
        "decision_band_accuracy": band_accuracy,
        "gpt_confidence_accuracy": confidence_accuracy,
        "per_label_metrics": fine,
        "coarse_metrics": coarse_metrics,
        "confusion_matrix": confusion,
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Holdout Eval 1000",
        "",
        f"- Rows labeled: {summary['rows']}",
        f"- Fine accuracy: {summary['fine_accuracy']}",
        f"- Fine macro F1: {summary['fine_macro_f1']}",
        f"- Fine weighted F1: {summary['fine_weighted_f1']}",
        f"- Coarse keep/drop accuracy: {summary['coarse_accuracy']}",
        f"- Coarse macro F1: {summary['coarse_macro_f1']}",
        "",
        "## Decision Band Accuracy",
        "",
    ]
    for key, value in summary["decision_band_accuracy"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## GPT Confidence Accuracy", ""])
    for key, value in summary["gpt_confidence_accuracy"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Per-Label Metrics", ""])
    for label in labels:
        item = fine[label]
        lines.append(
            f"- {label}: precision={item['precision']} recall={item['recall']} f1={item['f1']} support={item['support']}"
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
