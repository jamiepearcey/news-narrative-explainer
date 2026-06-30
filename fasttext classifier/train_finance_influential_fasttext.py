#!/usr/bin/env python3
"""Train a binary finance-influential fastText classifier."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path


try:
    import fasttext
except ModuleNotFoundError as error:  # pragma: no cover
    raise SystemExit(
        "fasttext is required. Run with: uv run --with fasttext python3 'fasttext classifier/train_finance_influential_fasttext.py'"
    ) from error


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
DATA_DIR = WORK_DIR / "data"
MODELS_DIR = WORK_DIR / "models"
RESULTS_DIR = WORK_DIR / "results"

DEFAULT_TRAIN_TXT = DATA_DIR / "finance_influential_train.txt"
DEFAULT_VALID_TXT = DATA_DIR / "finance_influential_valid.txt"
DEFAULT_MODEL_BIN = MODELS_DIR / "finance_influential.bin"
DEFAULT_MODEL_FTZ = MODELS_DIR / "finance_influential.ftz"
DEFAULT_SUMMARY_JSON = RESULTS_DIR / "finance_influential_training_summary.json"
DEFAULT_BENCHMARK_JSON = RESULTS_DIR / "finance_influential_training_benchmark.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-txt", default=str(DEFAULT_TRAIN_TXT))
    parser.add_argument("--valid-txt", default=str(DEFAULT_VALID_TXT))
    parser.add_argument("--model-bin", default=str(DEFAULT_MODEL_BIN))
    parser.add_argument("--model-ftz", default=str(DEFAULT_MODEL_FTZ))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--benchmark-json", default=str(DEFAULT_BENCHMARK_JSON))
    return parser.parse_args()


def read_labeled_lines(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        label, _, text = raw.partition(" ")
        rows.append((label.removeprefix("__label__"), text))
    return rows


def precision_by_cutoff(rows: list[tuple[str, str]], predictions: list[tuple[str, float]]) -> dict[str, dict[str, float]]:
    cutoffs = [0.5, 0.7, 0.85, 0.9]
    out: dict[str, dict[str, float]] = {}
    for cutoff in cutoffs:
        kept = 0
        correct = 0
        for (truth, _text), (predicted, score) in zip(rows, predictions, strict=True):
            if score < cutoff:
                continue
            kept += 1
            if truth == predicted:
                correct += 1
        out[str(cutoff)] = {
            "rows": kept,
            "precision": round(correct / kept, 4) if kept else 0.0,
        }
    return out


def main() -> None:
    args = parse_args()
    train_txt = Path(args.train_txt)
    valid_txt = Path(args.valid_txt)
    model_bin = Path(args.model_bin)
    model_ftz = Path(args.model_ftz)
    summary_json = Path(args.summary_json)
    benchmark_json = Path(args.benchmark_json)

    if not train_txt.exists() or not valid_txt.exists():
        raise SystemExit("Binary training data missing. Run build_finance_influential_dataset.py first.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = read_labeled_lines(train_txt)
    valid_rows = read_labeled_lines(valid_txt)

    train_start = time.perf_counter()
    model = fasttext.train_supervised(
        input=str(train_txt),
        lr=0.3,
        epoch=18,
        wordNgrams=2,
        dim=64,
        loss="ova",
        minn=2,
        maxn=5,
        thread=8,
        bucket=500000,
    )
    train_seconds = time.perf_counter() - train_start
    model.save_model(str(model_bin))
    model.quantize(input=str(train_txt), retrain=True, cutoff=100000)
    model.save_model(str(model_ftz))

    valid_texts = [text for _label, text in valid_rows]
    predict_start = time.perf_counter()
    labels, scores = model.predict(valid_texts, k=1)
    predict_seconds = time.perf_counter() - predict_start
    predictions = [
        (label_list[0].removeprefix("__label__"), float(score_list[0]))
        for label_list, score_list in zip(labels, scores, strict=True)
    ]

    confusion: dict[str, Counter[str]] = {
        "finance_influential": Counter(),
        "not_finance_influential": Counter(),
    }
    correct = 0
    for (truth, _text), (predicted, _score) in zip(valid_rows, predictions, strict=True):
        confusion.setdefault(truth, Counter())[predicted] += 1
        if truth == predicted:
            correct += 1

    summary = {
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "train_seconds": round(train_seconds, 4),
        "validation_accuracy": round(correct / len(valid_rows), 4) if valid_rows else 0.0,
        "precision_by_cutoff": precision_by_cutoff(valid_rows, predictions),
        "confusion_matrix": {
            truth: dict(sorted(counter.items()))
            for truth, counter in sorted(confusion.items())
        },
        "labels": model.get_labels(),
        "model_bin": str(model_bin),
        "model_ftz": str(model_ftz),
        "train_txt": str(train_txt),
        "valid_txt": str(valid_txt),
    }

    benchmark = {
        "train_seconds": round(train_seconds, 4),
        "predict_seconds": round(predict_seconds, 4),
        "prediction_rows": len(valid_rows),
        "predictions_per_second": round(len(valid_rows) / predict_seconds, 2) if predict_seconds else None,
        "train_rows_per_second": round(len(train_rows) / train_seconds, 2) if train_seconds else None,
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    benchmark_json.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "benchmark": benchmark}, indent=2))


if __name__ == "__main__":
    main()
