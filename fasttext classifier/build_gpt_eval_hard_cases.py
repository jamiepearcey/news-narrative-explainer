#!/usr/bin/env python3
"""Build a second-stage hard-case eval batch from the scored corpus."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
FEEDBACK_DIR = WORK_DIR / "feedback"
RESULTS_DIR = WORK_DIR / "results"
SCORED_CSV = RESULTS_DIR / "scored_weak_labels.csv"
LABELED_EVAL_CSV = FEEDBACK_DIR / "gpt_labeled_eval_set.csv"
OUTPUT_CSV = FEEDBACK_DIR / "gpt_eval_hard_cases.csv"
SUMMARY_JSON = RESULTS_DIR / "gpt_eval_hard_cases_summary.json"

PRESS_RELEASE_HINT_DOMAINS = {
    "itnewsonline.com",
    "manilatimes.net",
    "searchlight.vc",
    "interfax.com.ua",
    "en.acnnewswire.com",
    "pr.com",
}

LOW_QUALITY_DOMAINS = {
    "zazoom.it",
    "river949.com.au",
    "drudge.com",
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
}

COMPANY_FINANCE_TERMS = {
    "earnings",
    "guidance",
    "fda",
    "acquisition",
    "merger",
    "fundraise",
    "ipo",
    "shares",
    "stock",
    "stake",
}


def contains_any(text: str, terms: set[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


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
    if not LABELED_EVAL_CSV.exists():
        return set()
    with LABELED_EVAL_CSV.open("r", encoding="utf-8") as handle:
        return {row["document_identifier"] for row in csv.DictReader(handle)}


def build_row(stratum: str, row: dict[str, str], why: str) -> dict[str, str]:
    return {
        "stratum": stratum,
        "document_identifier": row["document_identifier"],
        "source_domain": row["source_domain"],
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
        "summary_text": row.get("summary", "")[:600],
        "market_context_text": row.get("text", "")[:900],
        "why_selected": why,
        "label": "",
        "confidence": "",
        "notes": "",
    }


def priority(row: dict[str, str]) -> tuple[float, float, float, int]:
    predicted_score = to_float(row.get("predicted_score", ""))
    cluster_score = to_float(row.get("finance_cluster_score", ""))
    signal_hits = (
        to_int(row.get("finance_hits", ""))
        + to_int(row.get("macro_hits", ""))
        + to_int(row.get("geo_hits", ""))
        + to_int(row.get("company_hits", ""))
        + to_int(row.get("equity_hits", ""))
    )
    title_length = len(row.get("title", ""))
    return (predicted_score, cluster_score, float(signal_hits), title_length)


def main() -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labeled_ids = load_labeled_ids()
    with SCORED_CSV.open("r", encoding="utf-8") as handle:
        scored_rows = [row for row in csv.DictReader(handle) if row["document_identifier"] not in labeled_ids]

    buckets: dict[str, list[dict[str, str]]] = {
        "weak_model_disagreement": [],
        "macro_geo_boundary": [],
        "finance_company_boundary": [],
        "press_release_boundary": [],
        "low_quality_boundary": [],
        "uncertain_mid_band": [],
    }

    for row in scored_rows:
        title = row.get("title", "")
        text = f"{title} {row.get('summary', '')} {row.get('text', '')}"
        weak_label = row.get("label", "")
        predicted_label = row.get("predicted_label", "")
        predicted_score = to_float(row.get("predicted_score", ""))
        source_domain = row.get("source_domain", "")
        press_hits = to_int(row.get("press_hits", ""))

        if weak_label != predicted_label:
            buckets["weak_model_disagreement"].append(build_row("weak_model_disagreement", row, "weak_vs_model_disagree"))

        if (
            {weak_label, predicted_label} & {"keep_macro", "keep_geopolitics"}
            and contains_any(text, MACRO_GEO_TERMS)
        ):
            buckets["macro_geo_boundary"].append(build_row("macro_geo_boundary", row, "macro_vs_geopolitics_boundary"))

        if (
            {weak_label, predicted_label} & {"keep_finance", "keep_company_event"}
            and (contains_any(text, COMPANY_FINANCE_TERMS) or "/earnings/" in row.get("document_identifier", "").lower())
        ):
            buckets["finance_company_boundary"].append(build_row("finance_company_boundary", row, "finance_vs_company_boundary"))

        if (
            source_domain in PRESS_RELEASE_HINT_DOMAINS
            or press_hits >= 1
            or weak_label == "drop_press_release"
            or predicted_label == "drop_press_release"
        ):
            buckets["press_release_boundary"].append(build_row("press_release_boundary", row, "press_release_edge_case"))

        if (
            source_domain in LOW_QUALITY_DOMAINS
            or weak_label == "drop_low_quality"
            or predicted_label == "drop_low_quality"
        ):
            buckets["low_quality_boundary"].append(build_row("low_quality_boundary", row, "low_quality_edge_case"))

        if 0.55 <= predicted_score <= 0.85 or row.get("decision_band", "") in {"review", "band_keep"}:
            buckets["uncertain_mid_band"].append(build_row("uncertain_mid_band", row, "mid_confidence_or_review_band"))

    quotas = {
        "weak_model_disagreement": 36,
        "macro_geo_boundary": 24,
        "finance_company_boundary": 24,
        "press_release_boundary": 20,
        "low_quality_boundary": 20,
        "uncertain_mid_band": 20,
    }

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
            if picked >= quotas[stratum]:
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
        "labeled_eval_rows_excluded": len(labeled_ids),
        "strata_counts": strata_counts,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
