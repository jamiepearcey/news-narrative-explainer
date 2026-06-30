#!/usr/bin/env python3
"""Build a stratified candidate set for GPT-assisted relevance labeling."""

from __future__ import annotations

import csv
import json
import urllib.request
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
RESULTS_DIR = WORK_DIR / "results"
FEEDBACK_DIR = WORK_DIR / "feedback"
SCORED_CSV = RESULTS_DIR / "scored_weak_labels.csv"
DISAGREEMENTS_CSV = RESULTS_DIR / "vector_teacher_domain_disagreements.csv"
CANDIDATES_CSV = FEEDBACK_DIR / "gpt_eval_candidates.csv"
SUMMARY_JSON = RESULTS_DIR / "gpt_eval_candidates_summary.json"
QDRANT_URL = "http://127.0.0.1:6333"
QDRANT_COLLECTION = "news_narrative_v3_20260605_allminilm"
OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "all-minilm:latest"
TRUNCATE_DIM = 256

SEARCH_QUERIES = [
    ("search_macro_energy", "WTI oil prices sanctions OPEC Hormuz shipping Strait of Hormuz tanker flows"),
    ("search_macro_rates", "Federal Reserve Treasury yields inflation payrolls jobs report rates bond market"),
    ("search_finance_equities", "earnings guidance analyst downgrade insider selling AI chips Nasdaq stocks"),
    ("search_geopolitics_market", "Ukraine sanctions Russia crude exports shipping insurance commodities"),
    ("search_ai_ipo", "OpenAI Anthropic SpaceX IPO valuation investors stock market"),
    ("search_company_event", "quarterly earnings guidance acquisition merger stake listing bond sale"),
]

DISAGREEMENT_DOMAINS = [
    "dailypolitical.com",
    "themarketsdaily.com",
    "tickerreport.com",
    "finance.yahoo.com",
    "insidermonkey.com",
    "fool.com",
    "finanznachrichten.de",
    "investegate.co.uk",
    "sharemanthan.in",
]

NEGATIVE_DOMAINS = [
    "prnewswire.com",
    "manilatimes.net",
    "openpr.com",
    "zazoom.it",
    "newswire.ca",
    "itnewsonline.com",
]


def ollama_embed(text: str) -> list[float]:
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=json.dumps({"model": EMBED_MODEL, "input": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["embeddings"][0][:TRUNCATE_DIM]


def qdrant_search(query: str, limit: int = 12) -> list[dict[str, object]]:
    vector = ollama_embed(query)
    request = urllib.request.Request(
        f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
        data=json.dumps(
            {
                "vector": vector,
                "limit": limit,
                "with_payload": [
                    "document_identifier",
                    "title",
                    "source_domain",
                    "summary_text",
                    "market_context_text",
                    "partition_date",
                ],
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["result"]


def load_scored_rows() -> list[dict[str, str]]:
    with SCORED_CSV.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_disagreement_domains() -> set[str]:
    domains: set[str] = set(DISAGREEMENT_DOMAINS)
    if DISAGREEMENTS_CSV.exists():
        with DISAGREEMENTS_CSV.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if len(domains) >= 20:
                    break
                domains.add(row["source_domain"])
    return domains


def main() -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scored_rows = load_scored_rows()
    scored_by_id = {row["document_identifier"]: row for row in scored_rows}
    disagreement_domains = load_disagreement_domains()

    candidate_rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for stratum, query in SEARCH_QUERIES:
        for hit in qdrant_search(query):
            payload = hit["payload"]
            document_identifier = payload["document_identifier"]
            if document_identifier in seen_ids:
                continue
            seen_ids.add(document_identifier)
            scored = scored_by_id.get(document_identifier, {})
            candidate_rows.append(
                {
                    "stratum": stratum,
                    "document_identifier": document_identifier,
                    "source_domain": payload.get("source_domain", ""),
                    "title": payload.get("title", ""),
                    "search_query": query,
                    "search_score": f"{float(hit.get('score', 0.0)):.4f}",
                    "weak_label": scored.get("label", ""),
                    "predicted_label": scored.get("predicted_label", ""),
                    "predicted_score": scored.get("predicted_score", ""),
                    "decision_band": scored.get("decision_band", ""),
                    "finance_cluster_score": scored.get("finance_cluster_score", ""),
                    "summary_text": (payload.get("summary_text") or "")[:600],
                    "market_context_text": (payload.get("market_context_text") or "")[:900],
                    "notes": "",
                }
            )

    for row in scored_rows:
        if row["source_domain"] in disagreement_domains:
            document_identifier = row["document_identifier"]
            if document_identifier in seen_ids:
                continue
            if row["predicted_label"] not in {"keep_finance", "keep_macro", "drop_press_release", "drop_low_quality"}:
                continue
            seen_ids.add(document_identifier)
            candidate_rows.append(
                {
                    "stratum": "domain_disagreement",
                    "document_identifier": document_identifier,
                    "source_domain": row["source_domain"],
                    "title": row["title"],
                    "search_query": "",
                    "search_score": "",
                    "weak_label": row.get("label", ""),
                    "predicted_label": row.get("predicted_label", ""),
                    "predicted_score": row.get("predicted_score", ""),
                    "decision_band": row.get("decision_band", ""),
                    "finance_cluster_score": row.get("finance_cluster_score", ""),
                    "summary_text": row.get("summary", "")[:600],
                    "market_context_text": row.get("text", "")[:900],
                    "notes": "",
                }
            )
            if sum(1 for item in candidate_rows if item["stratum"] == "domain_disagreement") >= 36:
                break

    for row in scored_rows:
        if row["source_domain"] in NEGATIVE_DOMAINS and float(row.get("predicted_score") or 0.0) >= 0.9:
            document_identifier = row["document_identifier"]
            if document_identifier in seen_ids:
                continue
            seen_ids.add(document_identifier)
            candidate_rows.append(
                {
                    "stratum": "likely_negative",
                    "document_identifier": document_identifier,
                    "source_domain": row["source_domain"],
                    "title": row["title"],
                    "search_query": "",
                    "search_score": "",
                    "weak_label": row.get("label", ""),
                    "predicted_label": row.get("predicted_label", ""),
                    "predicted_score": row.get("predicted_score", ""),
                    "decision_band": row.get("decision_band", ""),
                    "finance_cluster_score": row.get("finance_cluster_score", ""),
                    "summary_text": row.get("summary", "")[:600],
                    "market_context_text": row.get("text", "")[:900],
                    "notes": "",
                }
            )
            if sum(1 for item in candidate_rows if item["stratum"] == "likely_negative") >= 24:
                break

    fieldnames = list(candidate_rows[0].keys()) if candidate_rows else []
    with CANDIDATES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(candidate_rows)

    summary = {
        "candidate_rows": len(candidate_rows),
        "strata_counts": {
            key: sum(1 for row in candidate_rows if row["stratum"] == key)
            for key in sorted({row["stratum"] for row in candidate_rows})
        },
        "output_csv": str(CANDIDATES_CSV),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
