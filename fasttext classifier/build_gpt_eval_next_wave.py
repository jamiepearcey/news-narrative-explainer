#!/usr/bin/env python3
"""Build the next eval-labeling wave from current weak/model failure patterns."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
RESULTS_DIR = WORK_DIR / "results"
FEEDBACK_DIR = WORK_DIR / "feedback"

SCORED_CSV = RESULTS_DIR / "scored_weak_labels.csv"
LABELED_EVAL_CSV = FEEDBACK_DIR / "gpt_labeled_eval_set.csv"
EVAL_DETAIL_CSV = RESULTS_DIR / "gpt_eval_set_detailed.csv"
OUTPUT_CSV = FEEDBACK_DIR / "gpt_eval_next_wave.csv"
SUMMARY_JSON = RESULTS_DIR / "gpt_eval_next_wave_summary.json"

DEFAULT_QUOTAS = {
    "macro_geo_boundary": 24,
    "finance_company_boundary": 24,
    "press_release_boundary": 16,
    "finance_macro_boundary": 16,
}

MACRO_GEO_TERMS = {
    "hormuz",
    "sanctions",
    "shipping",
    "tanker",
    "oil flows",
    "maritime",
    "central bank",
    "crude exports",
    "global economy",
}

COMPANY_CATALYST_TERMS = {
    "earnings",
    "guidance",
    "fda",
    "approval",
    "feedback",
    "fundraise",
    "equity fundraise",
    "convertible bond",
    "acquisition",
    "merger",
}

PRESS_RELEASE_HINTS = {
    "eqs-news",
    "company announcement",
    "announces closing of",
    "official marketing partner",
    "news release",
}

FINANCE_MARKET_TERMS = {
    "shares",
    "stock",
    "nasdaq",
    "nyse",
    "analyst",
    "investors",
    "downgrade",
    "upgraded",
    "what investors need to know",
}


def normalize_text(*parts: str) -> str:
    return " ".join(part.strip().lower() for part in parts if part).strip()


def contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def to_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def load_labeled_ids() -> set[str]:
    with LABELED_EVAL_CSV.open("r", encoding="utf-8") as handle:
        return {row["document_identifier"] for row in csv.DictReader(handle)}


def load_eval_error_patterns() -> dict[str, Counter[tuple[str, str]]]:
    by_stratum: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    with EVAL_DETAIL_CSV.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("coverage_status") != "matched_scored":
                continue
            if row.get("weak_correct") != "0":
                continue
            truth = row.get("truth_label", "")
            weak = row.get("weak_label", "")
            if truth and weak:
                by_stratum[row.get("stratum", "")][(truth, weak)] += 1
    return by_stratum


def build_row(stratum: str, row: dict[str, str], why: str) -> dict[str, str]:
    return {
        "stratum": stratum,
        "document_identifier": row["document_identifier"],
        "source_domain": row.get("source_domain", ""),
        "title": row.get("title", ""),
        "weak_label": row.get("label", ""),
        "predicted_label": row.get("predicted_label", ""),
        "predicted_score": row.get("predicted_score", ""),
        "decision_band": row.get("decision_band", ""),
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
        "summary_text": row.get("summary", "")[:600],
        "market_context_text": row.get("text", "")[:900],
        "why_selected": why,
        "label": "",
        "confidence": "",
        "notes": "",
    }


def priority(row: dict[str, str]) -> tuple[float, float, int, int]:
    return (
        to_float(row.get("predicted_score", "")),
        to_float(row.get("finance_cluster_score", "")),
        to_int(row.get("keep_theme_hits", "")),
        len(row.get("title", "")),
    )


def main() -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labeled_ids = load_labeled_ids()
    eval_error_patterns = load_eval_error_patterns()

    with SCORED_CSV.open("r", encoding="utf-8") as handle:
        scored_rows = [row for row in csv.DictReader(handle) if row["document_identifier"] not in labeled_ids]

    buckets: dict[str, list[dict[str, str]]] = {key: [] for key in DEFAULT_QUOTAS}

    for row in scored_rows:
        text = normalize_text(row.get("title", ""), row.get("summary", ""), row.get("text", ""))
        weak_label = row.get("label", "")
        predicted_label = row.get("predicted_label", "")
        finance_hits = to_int(row.get("finance_hits", ""))
        macro_hits = to_int(row.get("macro_hits", ""))
        geo_hits = to_int(row.get("geo_hits", ""))
        company_hits = to_int(row.get("company_hits", ""))
        equity_hits = to_int(row.get("equity_hits", ""))
        press_hits = to_int(row.get("press_hits", ""))
        keep_theme_hits = to_int(row.get("keep_theme_hits", ""))
        macro_theme_hits = to_int(row.get("macro_theme_hits", ""))
        geo_theme_hits = to_int(row.get("geo_theme_hits", ""))

        if (
            {weak_label, predicted_label} & {"keep_macro", "keep_geopolitics"}
            and (
                contains_any(text, MACRO_GEO_TERMS)
                or macro_theme_hits >= 1
                or geo_theme_hits >= 1
            )
        ):
            buckets["macro_geo_boundary"].append(
                build_row("macro_geo_boundary", row, "next_wave_macro_geo")
            )

        if (
            {weak_label, predicted_label} & {"keep_finance", "keep_company_event", "drop_press_release"}
            and (
                contains_any(text, COMPANY_CATALYST_TERMS)
                or contains_any(text, PRESS_RELEASE_HINTS)
                or company_hits >= 1
                or press_hits >= 1
            )
        ):
            buckets["finance_company_boundary"].append(
                build_row("finance_company_boundary", row, "next_wave_finance_company")
            )

        if (
            weak_label == "drop_press_release"
            or predicted_label == "drop_press_release"
            or press_hits >= 1
            or contains_any(text, PRESS_RELEASE_HINTS)
        ):
            buckets["press_release_boundary"].append(
                build_row("press_release_boundary", row, "next_wave_press_release")
            )

        if (
            {weak_label, predicted_label} & {"keep_finance", "keep_macro"}
            and (
                finance_hits >= 1
                or equity_hits >= 1
                or macro_hits >= 1
                or keep_theme_hits >= 4
                or contains_any(text, FINANCE_MARKET_TERMS)
            )
        ):
            buckets["finance_macro_boundary"].append(
                build_row("finance_macro_boundary", row, "next_wave_finance_macro")
            )

    selected_rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    strata_counts: dict[str, int] = {}

    for stratum, rows in buckets.items():
        rows.sort(key=priority, reverse=True)
        picked = 0
        for row in rows:
            document_identifier = row["document_identifier"]
            if document_identifier in seen_ids:
                continue
            seen_ids.add(document_identifier)
            selected_rows.append(row)
            picked += 1
            if picked >= DEFAULT_QUOTAS[stratum]:
                break
        strata_counts[stratum] = picked

    fieldnames = list(selected_rows[0].keys()) if selected_rows else []
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(selected_rows)

    summary = {
        "output_csv": str(OUTPUT_CSV),
        "rows": len(selected_rows),
        "strata_counts": strata_counts,
        "eval_error_patterns": {
            stratum: {
                f"{truth}->{weak}": count
                for (truth, weak), count in patterns.items()
            }
            for stratum, patterns in sorted(eval_error_patterns.items())
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
