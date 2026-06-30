#!/usr/bin/env python3
"""Build conservative per-source probe profiles from the local corpus."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
DATA_DIR = WORK_DIR / "data"
RESULTS_DIR = WORK_DIR / "results"
RAW_EXPORT_CSV = DATA_DIR / "corpus_projection.csv"
PROFILES_CSV = RESULTS_DIR / "source_probe_profiles.csv"
SUMMARY_JSON = RESULTS_DIR / "source_probe_profiles_summary.json"
MAX_DOCS_PER_SOURCE = 500
TOP_K_PER_PROBE = 6

PROBES: dict[str, tuple[str, ...]] = {
    "finance": ("oil", "crude", "wti", "brent", "yield", "treasury", "stocks", "equity", "bank", "earnings"),
    "macro": ("fed", "central bank", "inflation", "gdp", "payrolls", "recession", "rate cut", "rate hike"),
    "geopolitics": ("israel", "iran", "ukraine", "russia", "taiwan", "china", "sanctions", "hormuz", "drone", "missile"),
    "company_event": ("earnings", "guidance", "merger", "acquisition", "ipo", "layoffs", "plant", "bankruptcy"),
    "industry_signal": ("shipping", "freight", "refinery", "pipeline", "lng", "mining", "smelter", "utility", "grid", "crop"),
    "junk": ("celebrity", "striker", "premier league", "recipe", "wedding", "cake", "local police", "horoscope"),
}

KEEP_LABEL_TERMS = {
    "finance": ("oil", "crude", "wti", "brent", "yield", "treasury", "stocks", "equity", "bank", "earnings"),
    "macro": ("fed", "central bank", "inflation", "gdp", "payrolls", "recession", "rate cut", "rate hike"),
    "geopolitics": ("israel", "iran", "ukraine", "russia", "taiwan", "china", "sanctions", "hormuz", "drone", "missile"),
    "company_event": ("earnings", "guidance", "merger", "acquisition", "ipo", "layoffs", "plant", "bankruptcy"),
    "industry_signal": ("shipping", "freight", "refinery", "pipeline", "lng", "mining", "smelter", "utility", "grid", "crop"),
}

JUNK_TERMS = ("celebrity", "recipe", "fashion", "lifestyle", "horoscope", "sports", "striker", "cake", "police", "arrested")
GENERIC_THEME_NOISE = (
    "education", "general_health", "medical", "leader", "legislation",
    "science", "media_msm", "uspec_policy1", "uspec_politics_general1",
    "wb_470_education", "wb_621_health_nutrition_and_population",
)


def normalize_text(*parts: str) -> str:
    text = " ".join(part for part in parts if part).lower()
    return re.sub(r"\s+", " ", text).strip()


def theme_tokens(raw: str) -> list[str]:
    tokens: list[str] = []
    for item in raw.split(";"):
        token = item.split(",", 1)[0].strip().lower()
        if token:
            tokens.append(token)
    return tokens


def score_probe(text: str, probe_terms: tuple[str, ...]) -> int:
    return sum(text.count(term) for term in probe_terms)


def main() -> None:
    if not RAW_EXPORT_CSV.exists():
        raise SystemExit("Missing corpus projection. Run build_weak_labels.py once first.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    source_docs: dict[str, list[dict[str, object]]] = defaultdict(list)
    with RAW_EXPORT_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            docs = source_docs[row["source_domain"]]
            if len(docs) >= MAX_DOCS_PER_SOURCE:
                continue
            natural_text = normalize_text(
                row.get("resolved_title", ""),
                row.get("summary", ""),
                row.get("text_excerpt", ""),
            )
            if len(natural_text) < 40:
                continue
            themes = theme_tokens(row.get("v2_themes", ""))
            theme_count = len(themes)
            generic_theme_noise = sum(1 for token in themes if token in GENERIC_THEME_NOISE)
            docs.append(
                {
                    "document_identifier": row["document_identifier"],
                    "text": natural_text,
                    "theme_count": theme_count,
                    "text_length": len(natural_text),
                    "generic_theme_noise": generic_theme_noise,
                }
            )

    rows: list[dict[str, object]] = []
    profile_counts: Counter[str] = Counter()
    for source_domain, docs in source_docs.items():
        if not docs:
            continue
        sampled_docs: list[dict[str, object]] = []
        probe_hits: dict[str, int] = {}
        probe_top_doc_ids: dict[str, set[str]] = {}
        for probe_name, terms in PROBES.items():
            scored = []
            for doc in docs:
                score = score_probe(str(doc["text"]), terms)
                if score > 0:
                    scored.append((score, doc))
            scored.sort(key=lambda item: (-item[0], -int(item[1]["theme_count"]), -int(item[1]["text_length"])))
            top_docs = [doc for _score, doc in scored[:TOP_K_PER_PROBE]]
            probe_hits[probe_name] = len(top_docs)
            probe_top_doc_ids[probe_name] = {str(doc["document_identifier"]) for doc in top_docs}
            sampled_docs.extend(top_docs)

        deduped: dict[str, dict[str, object]] = {}
        for doc in sampled_docs:
            deduped[str(doc["document_identifier"])] = doc
        sampled = list(deduped.values())
        if not sampled:
            continue

        sample_size = len(sampled)
        industry_hits = 0
        finance_hits = 0
        macro_hits = 0
        geo_hits = 0
        company_hits = 0
        junk_hits = 0
        evidence_density_values = []
        for doc in sampled:
            text = str(doc["text"])
            generic_theme_noise = int(doc.get("generic_theme_noise", 0))
            evidence_density_values.append(
                ((int(doc["theme_count"]) - generic_theme_noise) + math.log1p(int(doc["text_length"]))) / 10.0
            )
            if score_probe(text, KEEP_LABEL_TERMS["industry_signal"]) > 0:
                industry_hits += 1
            if score_probe(text, KEEP_LABEL_TERMS["finance"]) > 0:
                finance_hits += 1
            if score_probe(text, KEEP_LABEL_TERMS["macro"]) > 0:
                macro_hits += 1
            if score_probe(text, KEEP_LABEL_TERMS["geopolitics"]) > 0:
                geo_hits += 1
            if score_probe(text, KEEP_LABEL_TERMS["company_event"]) > 0:
                company_hits += 1
            if score_probe(text, JUNK_TERMS) > 0:
                junk_hits += 1

        enough_sample = sample_size >= 4
        market_relevance_rate = (finance_hits + macro_hits + geo_hits + company_hits + industry_hits) / (sample_size * 2.0)
        novelty_rate = len([name for name, ids in probe_top_doc_ids.items() if ids]) / max(1, len(PROBES))
        cross_signal_hits = finance_hits + macro_hits + geo_hits + company_hits
        row = {
            "source_domain": source_domain,
            "corpus_docs_sampled": len(docs),
            "sample_size": sample_size,
            "finance_probe_hits": probe_hits["finance"],
            "macro_probe_hits": probe_hits["macro"],
            "geopolitics_probe_hits": probe_hits["geopolitics"],
            "company_event_probe_hits": probe_hits["company_event"],
            "industry_signal_probe_hits": probe_hits["industry_signal"],
            "junk_probe_hits": probe_hits["junk"],
            "market_relevance_rate": round(min(1.0, market_relevance_rate), 4),
            "industry_signal_rate": round(industry_hits / sample_size, 4),
            "macro_signal_rate": round(macro_hits / sample_size, 4),
            "geopolitics_signal_rate": round(geo_hits / sample_size, 4),
            "company_event_signal_rate": round(company_hits / sample_size, 4),
            "junk_rate": round(junk_hits / sample_size, 4),
            "novelty_rate": round(novelty_rate, 4),
            "evidence_density": round(sum(evidence_density_values) / sample_size, 4),
            "enough_sample": int(enough_sample),
            "source_profile": (
                "industry_useful"
                if enough_sample
                and industry_hits >= 2
                and cross_signal_hits >= 2
                and novelty_rate >= 0.3
                and junk_hits <= max(1, sample_size // 5)
                else "junk_heavy"
                if enough_sample and junk_hits >= max(3, math.ceil(sample_size * 0.5))
                else "market_relevant"
                if enough_sample and cross_signal_hits >= 3 and novelty_rate >= 0.25
                else "mixed_or_sparse"
            ),
        }
        profile_counts[str(row["source_profile"])] += 1
        rows.append(row)

    rows.sort(key=lambda row: (-float(row["market_relevance_rate"]), -float(row["industry_signal_rate"]), row["source_domain"]))
    with PROFILES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "rows": len(rows),
        "profiles_csv": str(PROFILES_CSV),
        "profile_counts": dict(sorted(profile_counts.items())),
        "top_20": rows[:20],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
