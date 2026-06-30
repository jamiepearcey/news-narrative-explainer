#!/usr/bin/env python3
"""Score weak-label rows with fastText confidence bands."""

from __future__ import annotations

import csv
import json
from pathlib import Path


try:
    import fasttext
except ModuleNotFoundError as error:  # pragma: no cover
    raise SystemExit(
        "fasttext is required. Run with: uv run --with fasttext python3 'fasttext classifier/score_fasttext.py'"
    ) from error


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
DATA_DIR = WORK_DIR / "data"
MODELS_DIR = WORK_DIR / "models"
RESULTS_DIR = WORK_DIR / "results"
MODEL_BIN = MODELS_DIR / "news_filter.bin"
WEAK_LABEL_CSV = DATA_DIR / "weak_labels.csv"
SCORED_CSV = RESULTS_DIR / "scored_weak_labels.csv"

LOW_QUALITY_DOMAINS = {
    "zazoom.it",
    "river949.com.au",
    "drudge.com",
}

PRESS_RELEASE_DOMAINS = {
    "prnewswire.com",
    "businesswire.com",
    "globenewswire.com",
    "newsfilecorp.com",
    "accessnewswire.com",
    "openpr.com",
}

PRESS_RELEASE_HINT_DOMAINS = {
    "itnewsonline.com",
    "manilatimes.net",
    "searchlight.vc",
    "interfax.com.ua",
    "en.acnnewswire.com",
    "pr.com",
}

STRONG_COMPANY_EVENT_PHRASES = {
    "earnings guidance",
    "releases earnings results",
    "beats estimates",
    "fda feedback",
    "fda approval",
    "fundraise",
    "equity fundraise",
    "announces closing of",
    "make-or-break moment",
}

IPO_MARKET_STORY_PHRASES = {
    "stock at ipo",
    "before the ipo",
    "ipo valuations",
    "ipo playbook",
    "shares available",
    "etf makes it easier",
}

STOCK_MOVE_FINANCE_PHRASES = {
    "shares are sliding",
    "shares are soaring",
    "stock soaring",
    "stock jumps",
    "stock falls",
    "what investors need to know",
    "trading ideas",
    "movers",
}

MACRO_POLICY_PHRASES = {
    "central bank",
    "deputy chief",
    "interest rates",
    "inflation",
    "policy",
    "economy",
    "treasury",
    "bond yields",
}


def count_hits(text: str, phrases: set[str]) -> int:
    return sum(1 for phrase in phrases if phrase in text)


def refine_predicted_label(
    predicted_label: str,
    source_domain: str,
    document_identifier: str,
    title: str,
    summary: str,
    text: str,
    finance_cluster_score: float,
    finance_hits: int,
    macro_hits: int,
    geo_hits: int,
    company_hits: int,
    equity_hits: int,
    keep_theme_hits: int,
    macro_theme_hits: int,
    geo_theme_hits: int,
    press_hits: int,
) -> str:
    url = document_identifier.lower()
    natural_text = f"{title} {summary}".strip().lower()
    full_text = f"{natural_text} {text}".strip().lower()
    strong_company_hits = count_hits(full_text, STRONG_COMPANY_EVENT_PHRASES)
    ipo_market_story_hits = count_hits(full_text, IPO_MARKET_STORY_PHRASES)
    stock_move_finance_hits = count_hits(full_text, STOCK_MOVE_FINANCE_PHRASES)
    macro_policy_hits = count_hits(full_text, MACRO_POLICY_PHRASES)
    shipping_geo_story = any(
        phrase in natural_text
        for phrase in (
            "sanctioned tanker",
            "maritime blockade",
            "oil tanker",
            "shipping insurance",
            "crude exports",
            "pentagon says",
            "boards sanctioned tanker",
        )
    ) or (
        "hormuz" in natural_text
        and any(term in natural_text for term in ("tanker", "blockade", "shipping", "maritime", "strait"))
    )

    if source_domain in LOW_QUALITY_DOMAINS and finance_hits == 0 and macro_hits == 0 and geo_hits == 0 and company_hits == 0:
        return "drop_low_quality"

    if (
        source_domain in PRESS_RELEASE_DOMAINS
        or (source_domain in PRESS_RELEASE_HINT_DOMAINS and press_hits >= 1)
        or "announces closing of" in full_text
        or "eqs-news" in full_text
    ) and press_hits >= 1:
        return "drop_press_release"

    if (
        predicted_label in {"keep_macro", "keep_finance"}
        and shipping_geo_story
        and (geo_hits >= 1 or geo_theme_hits >= 1 or "pentagon says" in natural_text)
        and macro_hits == 0
        and macro_policy_hits == 0
    ):
        return "keep_geopolitics"

    if (
        predicted_label == "keep_geopolitics"
        and (
            (macro_hits >= 2 and macro_policy_hits >= 1)
            or (not shipping_geo_story and geo_hits <= 1 and geo_theme_hits == 0 and finance_hits >= 1)
        )
    ):
        return "keep_macro"

    if (
        predicted_label in {"keep_macro", "keep_finance"}
        and strong_company_hits >= 1
        and (company_hits + finance_hits >= 2 or finance_cluster_score >= 0.65)
        and ipo_market_story_hits == 0
        and stock_move_finance_hits == 0
    ):
        return "keep_company_event"

    if (
        predicted_label == "keep_macro"
        and finance_cluster_score >= 0.45
        and finance_hits >= 1
        and macro_hits == 0
        and geo_hits == 0
        and company_hits == 0
        and "/earnings/" not in url
    ):
        return "keep_finance"

    if (
        predicted_label == "keep_macro"
        and finance_cluster_score >= 0.65
        and (
            (
                (company_hits >= 1 or strong_company_hits >= 1)
                and strong_company_hits >= 1
                and stock_move_finance_hits == 0
            )
            or (
                "/earnings/" in url
                and equity_hits >= 1
            )
        )
    ):
        return "keep_company_event"

    if (
        predicted_label == "keep_finance"
        and (
            (
                (company_hits >= 1 or strong_company_hits >= 1)
                and strong_company_hits >= 1
                and stock_move_finance_hits == 0
            )
            or (
                "/earnings/" in url
                and finance_cluster_score >= 0.6
                and equity_hits >= 1
            )
        )
        and ipo_market_story_hits == 0
    ):
        return "keep_company_event"

    if (
        predicted_label == "keep_finance"
        and geo_hits >= 1
        and shipping_geo_story
    ):
        return "keep_geopolitics"

    if (
        predicted_label == "keep_finance"
        and finance_cluster_score <= 0.25
        and finance_hits == 0
        and macro_hits == 0
        and geo_hits == 0
        and company_hits == 0
        and keep_theme_hits >= 2
        and macro_theme_hits == 0
        and geo_theme_hits == 0
    ):
        return "keep_macro"

    if (
        predicted_label == "keep_macro"
        and keep_theme_hits >= 6
        and macro_theme_hits >= 4
        and finance_cluster_score < 0.55
        and finance_hits == 0
        and company_hits == 0
        and geo_hits == 0
    ):
        return "drop_low_quality" if source_domain in LOW_QUALITY_DOMAINS else predicted_label

    return predicted_label


def band_decision(
    predicted_label: str,
    score: float,
    market_relevance_rate: float,
    industry_signal_rate: float,
    finance_cluster_score: float,
    source_profile: str,
    source_domain: str,
    finance_hits: int,
    macro_hits: int,
    geo_hits: int,
    company_hits: int,
    keep_theme_hits: int,
    macro_theme_hits: int,
    geo_theme_hits: int,
    press_hits: int,
) -> str:
    natural_signal_hits = finance_hits + macro_hits + geo_hits + company_hits
    theme_signal_hits = keep_theme_hits + macro_theme_hits + geo_theme_hits
    pathological_theme_keep = (
        natural_signal_hits == 0
        and theme_signal_hits >= 8
        and predicted_label.startswith("keep_")
    )
    low_quality_pathology = (
        source_domain in LOW_QUALITY_DOMAINS
        and natural_signal_hits == 0
        and predicted_label.startswith("keep_")
    )
    press_release_pathology = (
        predicted_label.startswith("keep_")
        and (
            source_domain in PRESS_RELEASE_DOMAINS
            or (source_domain in PRESS_RELEASE_HINT_DOMAINS and press_hits >= 1)
        )
    )
    weak_keep_pathology = (
        predicted_label.startswith("keep_")
        and natural_signal_hits <= 1
        and theme_signal_hits >= 4
        and finance_cluster_score < 0.35
        and source_profile not in {"industry_useful", "market_relevant"}
    )

    if score > 0.85 and not (
        pathological_theme_keep
        or low_quality_pathology
        or press_release_pathology
        or weak_keep_pathology
    ):
        return "auto_keep" if predicted_label.startswith("keep_") else "auto_drop"
    if score >= 0.55:
        if (
            (
                not pathological_theme_keep
                and not low_quality_pathology
                and not press_release_pathology
            )
            and (
                market_relevance_rate >= 0.45
                or industry_signal_rate >= 0.2
                or finance_cluster_score >= 0.35
                or source_profile == "industry_useful"
            )
        ) and not weak_keep_pathology:
            return "band_keep"
        return "review"
    return "review"


def main() -> None:
    if not MODEL_BIN.exists():
        raise SystemExit("Missing model. Run train_fasttext.py first.")
    model = fasttext.load_model(str(MODEL_BIN))

    rows_out: list[dict[str, str]] = []
    with WEAK_LABEL_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = row["text"]
            labels, scores = model.predict(text, k=1)
            predicted_label = labels[0].removeprefix("__label__")
            score = float(scores[0])
            market_relevance_rate = float(row.get("market_relevance_rate") or 0.0)
            industry_signal_rate = float(row.get("industry_signal_rate") or 0.0)
            cluster_score = float(row.get("finance_cluster_score") or 0.0)
            source_profile = row.get("source_profile") or ""
            source_domain = (row.get("source_domain") or "").split(":", 1)[0]
            finance_hits = int(row.get("finance_hits") or 0)
            macro_hits = int(row.get("macro_hits") or 0)
            geo_hits = int(row.get("geo_hits") or 0)
            company_hits = int(row.get("company_hits") or 0)
            equity_hits = int(row.get("equity_hits") or 0)
            keep_theme_hits = int(row.get("keep_theme_hits") or 0)
            macro_theme_hits = int(row.get("macro_theme_hits") or 0)
            geo_theme_hits = int(row.get("geo_theme_hits") or 0)
            press_hits = int(row.get("press_hits") or 0)
            refined_label = refine_predicted_label(
                predicted_label,
                source_domain,
                row.get("document_identifier") or "",
                row.get("title") or "",
                row.get("summary") or "",
                text,
                cluster_score,
                finance_hits,
                macro_hits,
                geo_hits,
                company_hits,
                equity_hits,
                keep_theme_hits,
                macro_theme_hits,
                geo_theme_hits,
                press_hits,
            )
            row["predicted_label"] = refined_label
            row["predicted_score"] = f"{score:.4f}"
            row["decision_band"] = band_decision(
                refined_label,
                score,
                market_relevance_rate,
                industry_signal_rate,
                cluster_score,
                source_profile,
                source_domain,
                finance_hits,
                macro_hits,
                geo_hits,
                company_hits,
                keep_theme_hits,
                macro_theme_hits,
                geo_theme_hits,
                press_hits,
            )
            rows_out.append(row)

    with SCORED_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()) if rows_out else [])
        if rows_out:
            writer.writeheader()
            writer.writerows(rows_out)
    print(json.dumps({"scored_csv": str(SCORED_CSV), "rows": len(rows_out)}, indent=2))


if __name__ == "__main__":
    main()
