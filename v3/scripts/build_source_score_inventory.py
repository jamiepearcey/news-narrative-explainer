#!/usr/bin/env python3
"""Build a source domain/type scoring inventory from v3 graph artifacts."""

from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path


GRAPH_ROOT = Path(
    "/Users/jamiepearcey/projects/research/news-narrative-explainer/data/narrative_graph_parquet_v3/graph_doc_nodes_daily"
)
RESULTS_DIR = Path(
    "/Users/jamiepearcey/projects/research/news-narrative-explainer/v3/results"
)
CSV_OUT = RESULTS_DIR / "source_score_inventory.csv"
SUMMARY_OUT = RESULTS_DIR / "source_score_inventory_summary.json"

TOP_TIER = {
    "bloomberg.com",
    "cnbc.com",
    "nikkei.com",
    "ft.com",
    "wsj.com",
    "federalreserve.gov",
    "treasury.gov",
    "ecb.europa.eu",
    "bankofengland.co.uk",
    "opec.org",
    "iea.org",
    "imf.org",
    "worldbank.org",
    "reuters.com",
}

STRONG_SECONDARY = {
    "moneycontrol.com",
    "livemint.com",
    "business-standard.com",
    "benzinga.com",
    "seekingalpha.com",
    "kitco.com",
    "oilprice.com",
    "hellenicshippingnews.com",
    "shipandbunker.com",
    "gcaptain.com",
    "rigzone.com",
    "worldoil.com",
    "bullionvault.com",
    "argusmedia.com",
    "financialpost.com",
    "afr.com",
    "borsaitaliana.it",
    "theglobeandmail.com",
    "channelnewsasia.com",
    "nasdaq.com",
    "morningstar.com",
    "investors.com",
    "apnews.com",
    "marketwatch.com",
}

LOW_TRUST = {
    "prnewswire.com",
    "openpr.com",
    "financialcontent.com",
    "tickerreport.com",
}


def domain_type_base_weight(source_domain: str, source_type: str) -> tuple[float, str]:
    if source_domain in TOP_TIER:
        return 1.0, "explicit_top_tier"
    if source_domain in STRONG_SECONDARY:
        return 0.9, "explicit_strong_secondary"
    if source_domain in LOW_TRUST:
        return 0.15, "explicit_low_trust"
    if source_type == "market_wrap":
        return 0.85, "type_fallback_market_wrap"
    if source_type == "commodity_specialist":
        return 0.8, "type_fallback_commodity_specialist"
    if source_type == "company_specific":
        return 0.45, "type_fallback_company_specific"
    return 0.55, "type_fallback_general_news"


def current_actual_score(source_domain: str, source_type: str, source_priority: int) -> tuple[float, str]:
    base_weight, basis = domain_type_base_weight(source_domain, source_type)
    priority_adjustment = 0.03 * max(source_priority - 1, 0)
    return max(0.1, min(1.1, base_weight + priority_adjustment)), basis


def build_inventory() -> list[dict[str, object]]:
    query = f"""
        with src as (
            select
                source_domain,
                source_type,
                source_priority,
                partition_date
            from read_parquet('{GRAPH_ROOT.as_posix()}/**/*.parquet')
        )
        select
            source_domain,
            source_type,
            min(source_priority) as min_source_priority,
            max(source_priority) as max_source_priority,
            count(*) as row_count,
            count(distinct partition_date) as day_count
        from src
        group by 1, 2
        order by row_count desc, source_domain asc
    """
    result = subprocess.run(
        ["duckdb", "-csv", "-c", query],
        check=True,
        capture_output=True,
        text=True,
    )
    reader = csv.DictReader(result.stdout.splitlines())
    inventory: list[dict[str, object]] = []
    for raw in reader:
        source_domain = raw["source_domain"]
        source_type = raw["source_type"]
        min_priority = int(raw["min_source_priority"])
        max_priority = int(raw["max_source_priority"])
        row_count = int(raw["row_count"])
        day_count = int(raw["day_count"])
        suggested_score, mapping_basis = domain_type_base_weight(source_domain, source_type)
        actual_score, _ = current_actual_score(source_domain, source_type, max_priority)
        inventory.append(
            {
                "source_domain": source_domain,
                "source_type": source_type,
                "min_source_priority": min_priority,
                "max_source_priority": max_priority,
                "row_count": row_count,
                "day_count": day_count,
                "mapping_basis": mapping_basis,
                "suggested_score": round(suggested_score, 4),
                "current_actual_score": round(actual_score, 4),
                "priority_adjustment": round(actual_score - suggested_score, 4),
            }
        )
    return inventory


def write_outputs(inventory: list[dict[str, object]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_domain",
                "source_type",
                "min_source_priority",
                "max_source_priority",
                "row_count",
                "day_count",
                "mapping_basis",
                "suggested_score",
                "current_actual_score",
                "priority_adjustment",
            ],
        )
        writer.writeheader()
        writer.writerows(inventory)

    type_counter = Counter(row["source_type"] for row in inventory)
    basis_counter = Counter(row["mapping_basis"] for row in inventory)
    summary = {
        "rows": len(inventory),
        "distinct_domains": len({row["source_domain"] for row in inventory}),
        "source_types": dict(sorted(type_counter.items())),
        "mapping_basis": dict(sorted(basis_counter.items())),
        "top_20_by_row_count": inventory[:20],
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    inventory = build_inventory()
    write_outputs(inventory)
    print(json.dumps({"csv": str(CSV_OUT), "summary": str(SUMMARY_OUT), "rows": len(inventory)}, indent=2))


if __name__ == "__main__":
    main()
