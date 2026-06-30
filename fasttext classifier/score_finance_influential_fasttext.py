#!/usr/bin/env python3
"""Score the corpus with the binary finance-influential fastText classifier."""

from __future__ import annotations

import csv
import json
from pathlib import Path


try:
    import fasttext
except ModuleNotFoundError as error:  # pragma: no cover
    raise SystemExit(
        "fasttext is required. Run with: uv run --with fasttext python3 'fasttext classifier/score_finance_influential_fasttext.py'"
    ) from error


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
DATA_DIR = WORK_DIR / "data"
MODELS_DIR = WORK_DIR / "models"
RESULTS_DIR = WORK_DIR / "results"

MODEL_BIN = MODELS_DIR / "finance_influential.bin"
INPUT_CSV = DATA_DIR / "finance_influential_labels.csv"
OUTPUT_CSV = RESULTS_DIR / "finance_influential_scored.csv"
DOMAIN_SCORES_CSV = RESULTS_DIR / "effective_domain_scores.csv"


def decision_band(predicted_label: str, score: float) -> str:
    if score >= 0.9:
        return "auto_keep" if predicted_label == "finance_influential" else "auto_drop"
    if score >= 0.7:
        return "likely_keep" if predicted_label == "finance_influential" else "likely_drop"
    return "review"


def load_domain_scores() -> dict[str, dict[str, str]]:
    if not DOMAIN_SCORES_CSV.exists():
        return {}
    with DOMAIN_SCORES_CSV.open("r", encoding="utf-8") as handle:
        return {row["source_domain"]: row for row in csv.DictReader(handle)}


def apply_veto(predicted_label: str, row: dict[str, str], domain_score: dict[str, str] | None) -> tuple[str, str]:
    if predicted_label != "finance_influential":
        return predicted_label, ""

    original_label = row.get("original_label", "")
    if original_label.startswith("drop_"):
        return "not_finance_influential", "original_drop_label"
    press_hits = int(row.get("press_hits") or 0)

    if press_hits >= 1:
        return "not_finance_influential", "press_release_veto"

    return predicted_label, ""


def main() -> None:
    if not MODEL_BIN.exists():
        raise SystemExit("Missing finance_influential model. Run train_finance_influential_fasttext.py first.")
    if not INPUT_CSV.exists():
        raise SystemExit("Missing finance_influential labels CSV. Run build_finance_influential_dataset.py first.")

    model = fasttext.load_model(str(MODEL_BIN))
    domain_scores = load_domain_scores()
    rows_out: list[dict[str, str]] = []

    with INPUT_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels, scores = model.predict(row["text"], k=1)
            predicted_label = labels[0].removeprefix("__label__")
            score = float(scores[0])
            adjusted_label, veto_reason = apply_veto(
                predicted_label,
                row,
                domain_scores.get(row.get("source_domain", "")),
            )
            out = dict(row)
            out["predicted_binary_label"] = adjusted_label
            out["predicted_binary_score"] = f"{score:.4f}"
            out["predicted_binary_band"] = decision_band(adjusted_label, score)
            out["predicted_binary_veto_reason"] = veto_reason
            rows_out.append(out)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()) if rows_out else [])
        if rows_out:
            writer.writeheader()
            writer.writerows(rows_out)

    print(json.dumps({"rows": len(rows_out), "output_csv": str(OUTPUT_CSV)}, indent=2))


if __name__ == "__main__":
    main()
