#!/usr/bin/env python3
"""Build a consolidated quality report for the fastText classifier and domain scores."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
RESULTS_DIR = WORK_DIR / "results"
FEEDBACK_DIR = WORK_DIR / "feedback"

TRAINING_SUMMARY_JSON = RESULTS_DIR / "training_summary.json"
SCORED_WEAK_LABELS_CSV = RESULTS_DIR / "scored_weak_labels.csv"
AUDIT_CSV = RESULTS_DIR / "domain_score_audit.csv"
EFFECTIVE_CSV = RESULTS_DIR / "effective_domain_scores.csv"
OVERRIDES_CSV = RESULTS_DIR / "reviewed_domain_overrides.csv"
LABELED_FEEDBACK_CSV = FEEDBACK_DIR / "labeled_feedback.csv"

OUTPUT_JSON = RESULTS_DIR / "quality_evaluation.json"
OUTPUT_MD = RESULTS_DIR / "quality_evaluation.md"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def label_polarity(label: str) -> str:
    if label.startswith("keep_"):
        return "keep"
    if label.startswith("drop_"):
        return "drop"
    return "other"


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def round4(value: float) -> float:
    return round(value, 4)


def compute_per_label_metrics(confusion_matrix: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    labels = sorted(confusion_matrix)
    predicted_totals: Counter[str] = Counter()
    truth_totals: Counter[str] = Counter()
    for truth, predictions in confusion_matrix.items():
        for predicted, count in predictions.items():
            predicted_totals[predicted] += count
            truth_totals[truth] += count

    metrics: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = confusion_matrix.get(label, {}).get(label, 0)
        predicted_total = predicted_totals[label]
        truth_total = truth_totals[label]
        precision = safe_div(tp, predicted_total)
        recall = safe_div(tp, truth_total)
        f1 = safe_div(2 * precision * recall, precision + recall)
        metrics[label] = {
            "support": truth_total,
            "precision": round4(precision),
            "recall": round4(recall),
            "f1": round4(f1),
        }
    return metrics


def compute_classifier_quality() -> dict[str, object]:
    training_summary = json.loads(TRAINING_SUMMARY_JSON.read_text(encoding="utf-8"))
    scored_rows = load_csv(SCORED_WEAK_LABELS_CSV)
    labeled_feedback_rows = load_csv(LABELED_FEEDBACK_CSV) if LABELED_FEEDBACK_CSV.exists() else []

    exact_matches = 0
    polarity_matches = 0
    auto_rows = 0
    auto_exact_matches = 0
    auto_polarity_matches = 0
    band_counts: Counter[str] = Counter()
    exact_by_band: Counter[str] = Counter()
    polarity_by_band: Counter[str] = Counter()
    truth_counter: Counter[str] = Counter()
    predicted_counter: Counter[str] = Counter()
    exact_confusion: Counter[tuple[str, str]] = Counter()
    polarity_confusion: Counter[tuple[str, str]] = Counter()
    hard_negative_rows = 0
    hard_negative_auto_rows = 0
    false_drop_rows = 0
    false_drop_auto_rows = 0

    for row in scored_rows:
        truth = row["label"]
        predicted = row["predicted_label"]
        band = row["decision_band"]
        truth_polarity = label_polarity(truth)
        predicted_polarity = label_polarity(predicted)

        truth_counter[truth] += 1
        predicted_counter[predicted] += 1
        band_counts[band] += 1
        exact_confusion[(truth, predicted)] += 1
        polarity_confusion[(truth_polarity, predicted_polarity)] += 1

        if truth == predicted:
            exact_matches += 1
            exact_by_band[band] += 1
        if truth_polarity == predicted_polarity:
            polarity_matches += 1
            polarity_by_band[band] += 1

        if band != "review":
            auto_rows += 1
            if truth == predicted:
                auto_exact_matches += 1
            if truth_polarity == predicted_polarity:
                auto_polarity_matches += 1

        if truth_polarity == "drop" and predicted_polarity == "keep":
            hard_negative_rows += 1
            if band != "review":
                hard_negative_auto_rows += 1
        if truth_polarity == "keep" and predicted_polarity == "drop":
            false_drop_rows += 1
            if band != "review":
                false_drop_auto_rows += 1

    per_label_metrics = compute_per_label_metrics(training_summary["confusion_matrix"])
    supports = [metric["support"] for metric in per_label_metrics.values()]
    macro_f1 = safe_div(sum(metric["f1"] for metric in per_label_metrics.values()), len(per_label_metrics))
    weighted_f1 = safe_div(
        sum(metric["f1"] * metric["support"] for metric in per_label_metrics.values()),
        sum(supports),
    )

    top_exact_confusions = [
        {"truth": truth, "predicted": predicted, "rows": rows}
        for (truth, predicted), rows in exact_confusion.most_common(20)
        if truth != predicted
    ]

    band_quality = {}
    for band, rows in sorted(band_counts.items()):
        band_quality[band] = {
            "rows": rows,
            "exact_match_rate": round4(safe_div(exact_by_band[band], rows)),
            "coarse_keep_drop_match_rate": round4(safe_div(polarity_by_band[band], rows)),
        }

    feedback_summary = {
        "rows": len(labeled_feedback_rows),
        "available": bool(labeled_feedback_rows),
    }

    return {
        "evidence": {
            "weak_label_validation_available": True,
            "human_labeled_row_feedback_available": bool(labeled_feedback_rows),
            "human_labeled_row_feedback_rows": len(labeled_feedback_rows),
            "confidence_note": (
                "Classifier quality is currently estimated from weak-label validation and corpus-behavior proxies; "
                "there is no labeled row-level feedback set yet."
            ),
        },
        "training_validation": {
            "train_rows": training_summary["train_rows"],
            "valid_rows": training_summary["valid_rows"],
            "validation_accuracy": training_summary["validation_accuracy"],
            "macro_f1": round4(macro_f1),
            "weighted_f1": round4(weighted_f1),
            "per_label_metrics": per_label_metrics,
            "precision_by_cutoff": training_summary["precision_by_cutoff"],
        },
        "full_corpus_proxy": {
            "rows": len(scored_rows),
            "weak_label_exact_match_rate": round4(safe_div(exact_matches, len(scored_rows))),
            "weak_label_coarse_keep_drop_match_rate": round4(safe_div(polarity_matches, len(scored_rows))),
            "non_review_rows": auto_rows,
            "non_review_exact_match_rate": round4(safe_div(auto_exact_matches, auto_rows)),
            "non_review_coarse_keep_drop_match_rate": round4(safe_div(auto_polarity_matches, auto_rows)),
            "decision_band_counts": dict(sorted(band_counts.items())),
            "decision_band_quality": band_quality,
            "hard_negative_rows": hard_negative_rows,
            "hard_negative_rate": round4(safe_div(hard_negative_rows, len(scored_rows))),
            "hard_negative_non_review_rows": hard_negative_auto_rows,
            "hard_negative_non_review_rate": round4(safe_div(hard_negative_auto_rows, auto_rows)),
            "false_drop_rows": false_drop_rows,
            "false_drop_rate": round4(safe_div(false_drop_rows, len(scored_rows))),
            "false_drop_non_review_rows": false_drop_auto_rows,
            "false_drop_non_review_rate": round4(safe_div(false_drop_auto_rows, auto_rows)),
            "top_exact_confusions": top_exact_confusions,
            "truth_label_distribution": dict(sorted(truth_counter.items())),
            "predicted_label_distribution": dict(sorted(predicted_counter.items())),
        },
        "feedback": feedback_summary,
    }


def compute_domain_quality() -> dict[str, object]:
    audit_rows = {row["source_domain"]: row for row in load_csv(AUDIT_CSV)}
    effective_rows = load_csv(EFFECTIVE_CSV)
    overrides = {row["source_domain"]: row for row in load_csv(OVERRIDES_CSV)}

    manual_rows = [row for row in effective_rows if row["review_status"] == "reviewed_manual"]
    bulk_rows = [row for row in effective_rows if row["review_status"] == "reviewed_bulk_finalized"]

    total_domains = len(effective_rows)
    total_source_rows = sum(int(row["rows"] or 0) for row in effective_rows)
    manual_source_rows = sum(int(row["rows"] or 0) for row in manual_rows)

    exact_archetype_matches = 0
    score_abs_errors: list[float] = []
    changed_archetypes: Counter[tuple[str, str]] = Counter()
    biggest_manual_overrides: list[dict[str, object]] = []

    for row in manual_rows:
        domain = row["source_domain"]
        audit = audit_rows.get(domain)
        if not audit:
            continue
        heuristic_archetype = audit["proposed_archetype"]
        heuristic_score = float(audit["proposed_score_0_10"])
        final_archetype = row["effective_archetype"]
        final_score = float(row["effective_score_0_10"])
        abs_error = abs(final_score - heuristic_score)
        score_abs_errors.append(abs_error)
        if heuristic_archetype == final_archetype:
            exact_archetype_matches += 1
        else:
            changed_archetypes[(heuristic_archetype, final_archetype)] += 1
        biggest_manual_overrides.append(
            {
                "source_domain": domain,
                "heuristic_archetype": heuristic_archetype,
                "heuristic_score_0_10": round4(heuristic_score),
                "final_archetype": final_archetype,
                "final_score_0_10": round4(final_score),
                "score_abs_delta": round4(abs_error),
                "rows": int(row["rows"] or 0),
                "proposal_basis": row.get("proposal_basis", ""),
                "manual_rationale": overrides.get(domain, {}).get("rationale", ""),
            }
        )

    biggest_manual_overrides.sort(
        key=lambda item: (-float(item["score_abs_delta"]), -int(item["rows"]), item["source_domain"])
    )

    archetype_distribution = Counter(row["effective_archetype"] for row in effective_rows)
    manual_archetype_distribution = Counter(row["effective_archetype"] for row in manual_rows)
    bulk_archetype_distribution = Counter(row["effective_archetype"] for row in bulk_rows)

    return {
        "evidence": {
            "manual_domain_rows": len(manual_rows),
            "bulk_domain_rows": len(bulk_rows),
            "manual_domain_coverage_rate": round4(safe_div(len(manual_rows), total_domains)),
            "manual_source_row_coverage_rate": round4(safe_div(manual_source_rows, total_source_rows)),
            "confidence_note": (
                "Domain-score quality has direct human evidence only where a manual override exists; "
                "all other domains remain bulk-finalized from the sampled-story audit."
            ),
        },
        "manual_vs_heuristic": {
            "manual_rows": len(manual_rows),
            "heuristic_archetype_match_rate_on_manual_rows": round4(
                safe_div(exact_archetype_matches, len(manual_rows))
            ),
            "score_mae_on_manual_rows": round4(
                safe_div(sum(score_abs_errors), len(score_abs_errors))
            ),
            "score_rmse_on_manual_rows": round4(
                math.sqrt(safe_div(sum(error * error for error in score_abs_errors), len(score_abs_errors)))
            )
            if score_abs_errors
            else 0.0,
            "top_archetype_changes": [
                {"heuristic": heuristic, "final": final, "rows": rows}
                for (heuristic, final), rows in changed_archetypes.most_common(20)
            ],
            "largest_manual_overrides": biggest_manual_overrides[:20],
        },
        "final_distribution": {
            "domains": total_domains,
            "source_rows": total_source_rows,
            "archetype_distribution": dict(sorted(archetype_distribution.items())),
            "manual_archetype_distribution": dict(sorted(manual_archetype_distribution.items())),
            "bulk_archetype_distribution": dict(sorted(bulk_archetype_distribution.items())),
        },
    }


def build_markdown(report: dict[str, object]) -> str:
    classifier = report["classifier_quality"]
    domain = report["domain_score_quality"]

    top_confusions = classifier["full_corpus_proxy"]["top_exact_confusions"][:10]
    top_overrides = domain["manual_vs_heuristic"]["largest_manual_overrides"][:10]

    lines = [
        "# Quality Evaluation",
        "",
        "## Summary",
        "",
        f"- Classifier weak-label validation accuracy: `{classifier['training_validation']['validation_accuracy']:.4f}`",
        f"- Classifier weak-label macro F1: `{classifier['training_validation']['macro_f1']:.4f}`",
        f"- Classifier full-corpus coarse keep/drop match rate: `{classifier['full_corpus_proxy']['weak_label_coarse_keep_drop_match_rate']:.4f}`",
        f"- Classifier hard-negative non-review rate: `{classifier['full_corpus_proxy']['hard_negative_non_review_rate']:.4f}`",
        f"- Domain manual coverage: `{domain['evidence']['manual_domain_rows']}` / `{domain['final_distribution']['domains']}` domains",
        f"- Domain manual source-row coverage: `{domain['evidence']['manual_source_row_coverage_rate']:.4f}`",
        f"- Domain heuristic archetype match rate on manual rows: `{domain['manual_vs_heuristic']['heuristic_archetype_match_rate_on_manual_rows']:.4f}`",
        f"- Domain score MAE on manual rows: `{domain['manual_vs_heuristic']['score_mae_on_manual_rows']:.4f}`",
        "",
        "## Classifier",
        "",
        f"- Evidence: {classifier['evidence']['confidence_note']}",
        f"- Human labeled feedback rows: `{classifier['evidence']['human_labeled_row_feedback_rows']}`",
        f"- Non-review rows: `{classifier['full_corpus_proxy']['non_review_rows']}` of `{classifier['full_corpus_proxy']['rows']}`",
        "",
        "### Decision Bands",
        "",
    ]

    for band, metrics in classifier["full_corpus_proxy"]["decision_band_quality"].items():
        lines.append(
            f"- `{band}`: rows `{metrics['rows']}`, exact `{metrics['exact_match_rate']:.4f}`, coarse keep/drop `{metrics['coarse_keep_drop_match_rate']:.4f}`"
        )

    lines.extend(
        [
            "",
            "### Top Exact Confusions",
            "",
        ]
    )
    for item in top_confusions:
        lines.append(f"- `{item['truth']}` -> `{item['predicted']}`: `{item['rows']}` rows")

    lines.extend(
        [
            "",
            "## Domain Score",
            "",
            f"- Evidence: {domain['evidence']['confidence_note']}",
            f"- Manual rows: `{domain['evidence']['manual_domain_rows']}`",
            f"- Bulk rows: `{domain['evidence']['bulk_domain_rows']}`",
            "",
            "### Largest Manual Overrides",
            "",
        ]
    )
    for item in top_overrides:
        lines.append(
            f"- `{item['source_domain']}`: `{item['heuristic_archetype']}` `{item['heuristic_score_0_10']}` -> `{item['final_archetype']}` `{item['final_score_0_10']}`"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    report = {
        "classifier_quality": compute_classifier_quality(),
        "domain_score_quality": compute_domain_quality(),
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(build_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "quality_json": str(OUTPUT_JSON),
                "quality_md": str(OUTPUT_MD),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
