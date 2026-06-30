#!/usr/bin/env python3
"""Evaluate current weak and model predictions against the GPT-labeled eval set."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
FEEDBACK_DIR = WORK_DIR / "feedback"
RESULTS_DIR = WORK_DIR / "results"
EVAL_CSV = FEEDBACK_DIR / "gpt_labeled_eval_set.csv"
SCORED_CSV = RESULTS_DIR / "scored_weak_labels.csv"
REVIEW_CSV = FEEDBACK_DIR / "review_queue.csv"
CORPUS_CSV = WORK_DIR / "data" / "corpus_projection.csv"
OUTPUT_JSON = RESULTS_DIR / "gpt_eval_set_metrics.json"
DETAIL_CSV = RESULTS_DIR / "gpt_eval_set_detailed.csv"
MISSING_CSV = RESULTS_DIR / "gpt_eval_set_missing.csv"


def coarse(label: str) -> str:
    if label.startswith("keep_"):
        return "keep"
    if label.startswith("drop_"):
        return "drop"
    return "other"


def metric_summary(truths: list[str], preds: list[str]) -> dict[str, object]:
    correct = sum(1 for t, p in zip(truths, preds, strict=True) if t == p)
    coarse_correct = sum(1 for t, p in zip(truths, preds, strict=True) if coarse(t) == coarse(p))
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for truth, pred in zip(truths, preds, strict=True):
        confusion[truth][pred] += 1
    return {
        "rows": len(truths),
        "fine_accuracy": round(correct / len(truths), 4) if truths else 0.0,
        "coarse_accuracy": round(coarse_correct / len(truths), 4) if truths else 0.0,
        "confusion_matrix": {
            truth: dict(sorted(counter.items()))
            for truth, counter in sorted(confusion.items())
        },
    }


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def main() -> None:
    with EVAL_CSV.open("r", encoding="utf-8") as handle:
        eval_rows = list(csv.DictReader(handle))
    with SCORED_CSV.open("r", encoding="utf-8") as handle:
        scored_by_id = {row["document_identifier"]: row for row in csv.DictReader(handle)}
    with REVIEW_CSV.open("r", encoding="utf-8") as handle:
        review_by_id = {row["document_identifier"]: row for row in csv.DictReader(handle)}
    with CORPUS_CSV.open("r", encoding="utf-8") as handle:
        corpus_by_id = {row["document_identifier"]: row for row in csv.DictReader(handle)}

    truths: list[str] = []
    weak_preds: list[str] = []
    model_preds: list[str] = []
    by_stratum: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"truths": [], "weak": [], "model": []})
    stratum_coverage: dict[str, Counter[str]] = defaultdict(Counter)
    missing_scored: list[str] = []
    review_only: list[str] = []
    missing_from_corpus: list[str] = []
    detailed_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []

    for row in eval_rows:
        document_identifier = row["document_identifier"]
        stratum = row.get("stratum", "")
        scored = scored_by_id.get(document_identifier)
        if not scored:
            review = review_by_id.get(document_identifier)
            corpus = corpus_by_id.get(document_identifier)
            coverage_status = (
                "present_review_only" if review
                else "present_projection_only" if corpus
                else "missing_from_corpus"
            )
            stratum_coverage[stratum][coverage_status] += 1
            missing_scored.append(document_identifier)
            if review:
                review_only.append(document_identifier)
            if not review and not corpus:
                missing_from_corpus.append(document_identifier)
            detail_row = {
                "document_identifier": document_identifier,
                "stratum": stratum,
                "source_domain": row.get("source_domain", "") or (review or corpus or {}).get("source_domain", ""),
                "title": row.get("title", "") or (review or {}).get("title", "") or (corpus or {}).get("resolved_title", ""),
                "truth_label": row.get("label", ""),
                "weak_label": "",
                "model_label": "",
                "weak_correct": "",
                "model_correct": "",
                "weak_coarse_correct": "",
                "model_coarse_correct": "",
                "predicted_score": "",
                "decision_band": (review or {}).get("label", ""),
                "finance_cluster_score": (review or {}).get("finance_cluster_score", ""),
                "reasons": (review or {}).get("reasons", ""),
                "summary": row.get("summary_text", "") or row.get("summary", "") or (review or {}).get("summary", "") or (corpus or {}).get("summary", ""),
                "market_context_text": row.get("market_context_text", "") or (review or {}).get("text", "") or (corpus or {}).get("text_excerpt", ""),
                "coverage_status": coverage_status,
                "review_label_source": (review or {}).get("label_source", ""),
                "review_reasons": (review or {}).get("reasons", ""),
                "notes": row.get("notes", ""),
            }
            detailed_rows.append(detail_row)
            missing_rows.append(
                {
                    "document_identifier": document_identifier,
                    "label": row.get("label", ""),
                    "stratum": stratum,
                    "source_domain": detail_row["source_domain"],
                    "title": detail_row["title"],
                    "coverage_status": coverage_status,
                    "review_label_source": detail_row["review_label_source"],
                    "review_reasons": detail_row["review_reasons"],
                    "notes": row.get("notes", ""),
                }
            )
            continue
        truth = row["label"]
        weak = scored["label"]
        model = scored["predicted_label"]
        stratum_coverage[stratum]["matched_scored"] += 1
        truths.append(truth)
        weak_preds.append(weak)
        model_preds.append(model)
        by_stratum[stratum]["truths"].append(truth)
        by_stratum[stratum]["weak"].append(weak)
        by_stratum[stratum]["model"].append(model)
        detailed_rows.append(
            {
                "document_identifier": document_identifier,
                "stratum": stratum,
                "source_domain": row.get("source_domain", "") or scored.get("source_domain", ""),
                "title": row.get("title", "") or scored.get("title", ""),
                "truth_label": truth,
                "weak_label": weak,
                "model_label": model,
                "weak_correct": "1" if truth == weak else "0",
                "model_correct": "1" if truth == model else "0",
                "weak_coarse_correct": "1" if coarse(truth) == coarse(weak) else "0",
                "model_coarse_correct": "1" if coarse(truth) == coarse(model) else "0",
                "predicted_score": scored.get("predicted_score", ""),
                "decision_band": scored.get("decision_band", ""),
                "finance_cluster_score": scored.get("finance_cluster_score", ""),
                "reasons": scored.get("reasons", ""),
                "summary": row.get("summary_text", "") or row.get("summary", "") or scored.get("summary", ""),
                "market_context_text": row.get("market_context_text", ""),
                "coverage_status": "matched_scored",
                "review_label_source": "",
                "review_reasons": "",
                "notes": row.get("notes", ""),
            }
        )

    all_strata = sorted(set(by_stratum) | set(stratum_coverage))

    summary = {
        "rows": len(eval_rows),
        "matched_rows": len(truths),
        "missing_scored_rows": len(missing_scored),
        "review_only_rows": len(review_only),
        "missing_from_corpus_rows": len(missing_from_corpus),
        "scored_coverage": ratio(len(truths), len(eval_rows)),
        "review_only_coverage": ratio(len(review_only), len(eval_rows)),
        "corpus_coverage": ratio(len(eval_rows) - len(missing_from_corpus), len(eval_rows)),
        "missing_document_identifiers": missing_scored,
        "review_only_document_identifiers": review_only,
        "missing_from_corpus_document_identifiers": missing_from_corpus,
        "weak_label_metrics": metric_summary(truths, weak_preds),
        "model_metrics": metric_summary(truths, model_preds),
        "by_stratum": {
            stratum: {
                "coverage": {
                    "rows": sum(stratum_coverage[stratum].values()),
                    "matched_scored_rows": stratum_coverage[stratum]["matched_scored"],
                    "review_only_rows": stratum_coverage[stratum]["present_review_only"],
                    "projection_only_rows": stratum_coverage[stratum]["present_projection_only"],
                    "missing_from_corpus_rows": stratum_coverage[stratum]["missing_from_corpus"],
                    "scored_coverage": ratio(
                        stratum_coverage[stratum]["matched_scored"],
                        sum(stratum_coverage[stratum].values()),
                    ),
                    "corpus_coverage": ratio(
                        stratum_coverage[stratum]["matched_scored"]
                        + stratum_coverage[stratum]["present_review_only"]
                        + stratum_coverage[stratum]["present_projection_only"],
                        sum(stratum_coverage[stratum].values()),
                    ),
                },
                "weak_label_metrics": metric_summary(
                    by_stratum[stratum]["truths"],
                    by_stratum[stratum]["weak"],
                ),
                "model_metrics": metric_summary(
                    by_stratum[stratum]["truths"],
                    by_stratum[stratum]["model"],
                ),
            }
            for stratum in all_strata
        },
    }

    detailed_rows.sort(
        key=lambda row: (
            row["coverage_status"],
            row["model_correct"],
            row["weak_correct"],
            row["stratum"],
            row["document_identifier"],
        )
    )

    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with DETAIL_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "document_identifier",
            "stratum",
            "source_domain",
            "title",
            "truth_label",
            "weak_label",
            "model_label",
            "weak_correct",
            "model_correct",
            "weak_coarse_correct",
            "model_coarse_correct",
            "predicted_score",
            "decision_band",
            "finance_cluster_score",
            "reasons",
            "summary",
            "market_context_text",
            "coverage_status",
            "review_label_source",
            "review_reasons",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)
    with MISSING_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "document_identifier",
            "label",
            "stratum",
            "source_domain",
            "title",
            "coverage_status",
            "review_label_source",
            "review_reasons",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(missing_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
