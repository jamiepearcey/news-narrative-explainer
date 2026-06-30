#!/usr/bin/env python3
"""Enrich staged GDELT parquet files with article-facing text fields."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "gdelt_candidates"


def ensure_dependencies() -> None:
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except ModuleNotFoundError:
        uv = shutil.which("uv") or "/opt/homebrew/bin/uv"
        if not Path(uv).exists():
            raise RuntimeError("pyarrow is required, and `uv` was not found to bootstrap it") from None
        if os.environ.get("NEWS_NARRATIVE_ENRICH_UV_BOOTSTRAPPED") == "1":
            raise
        env = os.environ.copy()
        env["NEWS_NARRATIVE_ENRICH_UV_BOOTSTRAPPED"] = "1"
        os.execvpe(
            uv,
            [uv, "run", "--with", "pyarrow>=16", "--with", "beautifulsoup4>=4.12", str(Path(__file__).resolve()), *sys.argv[1:]],
            env,
        )


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fetch_gdelt_bigquery_candidates import DEFAULT_HTTP_TIMEOUT, DEFAULT_USER_AGENT, enrich_rows
from narrative_text_matching import asset_cues, factor_cues, match_count

PRIORITY_MARKET_SOURCES = {
    "reuters.com",
    "wsj.com",
    "barrons.com",
    "marketwatch.com",
    "apnews.com",
    "finance.yahoo.com",
    "investopedia.com",
    "business-standard.com",
    "cnbcafrica.com",
    "moneycontrol.com",
    "bloomberg.com",
    "cnbc.com",
    "ft.com",
    "kitco.com",
    "businesstimes.com.sg",
    "fxstreet.com",
    "investing.com",
}
MACRO_ASSETS = {"US2Y", "US10Y", "DXY", "NDX"}
INDEX_ASSETS = {"NDX", "SPX"}
PRIORITY_CROSS_ASSET_ASSETS = {"DXY", "Gold"}
ASSET_MARKET_CUES = {
    "NDX": ["NASDAQ 100", "NASDAQ-100", "NASDAQ FALLS", "TECH SELLOFF", "CHIP STOCKS", "QQQ", "WALL STREET"],
    "SPX": ["S&P 500", "SP 500", "SPX", "WALL STREET", "US STOCKS", "STOCKS SLUMP", "STOCKS FALL"],
    "US2Y": ["2-YEAR", "TWO-YEAR", "TREASURY", "FRONT-END YIELD", "FED", "RATE CUT", "RATE HIKE"],
    "US10Y": ["10-YEAR", "TEN-YEAR", "TREASURY", "YIELD", "TERM PREMIUM", "DURATION"],
    "DXY": ["DOLLAR INDEX", "DOLLAR", "GREENBACK", "USD", "CURRENCY", "FED"],
    "Gold": ["GOLD", "BULLION", "XAU", "PRECIOUS METAL", "FED", "REAL YIELD", "DOLLAR", "ETF OUTFLOWS"],
}
COMPANY_SPECIFIC_TITLE_TERMS = [
    "FINANCIAL COMPARISON",
    " VS. ",
    " VS ",
    "(NASDAQ:",
    "(NYSE:",
    "QUARTER ENDED",
    "EBIT MARGIN",
    "REVENUE TRENDS",
    "SHARES OF",
]


def parquet_files(root: Path, include_glob: str | None = None) -> list[Path]:
    if include_glob:
        return sorted(path for path in root.glob(include_glob) if path.is_file() and path.suffix == ".parquet")
    return sorted(path for path in root.rglob("*.parquet") if path.is_file())


def _text_len(value: Any) -> int:
    return len(value.strip()) if isinstance(value, str) else 0


def _row_has_sufficient_article_text(row: dict[str, Any]) -> bool:
    title_len = _text_len(row.get("title"))
    summary_len = _text_len(row.get("summary"))
    text_len = _text_len(row.get("text"))
    if text_len >= 500:
        return True
    if summary_len >= 160 and title_len >= 40:
        return True
    return False


def _row_needs_enrichment(row: dict[str, Any]) -> bool:
    return not _row_has_sufficient_article_text(row)


def _row_relevance_text(row: dict[str, Any]) -> str:
    parts = [
        row.get("title"),
        row.get("summary"),
        row.get("text"),
        row.get("source_common_name"),
        row.get("all_names"),
        row.get("v2_themes"),
        row.get("v2_persons"),
        row.get("v2_organizations"),
        row.get("v2_locations"),
        row.get("counts"),
        row.get("v2_counts"),
        row.get("dates"),
        row.get("quotations"),
        row.get("amounts"),
        row.get("gcam"),
        row.get("gkg_extras"),
        row.get("document_identifier"),
    ]
    return " ".join(str(part) for part in parts if part)


def _source_bonus(row: dict[str, Any], requested_asset_label: str | None) -> float:
    source_common_name = str(row.get("source_common_name") or "").lower()
    document_identifier = str(row.get("document_identifier") or "").lower()
    if any(source in source_common_name or source in document_identifier for source in PRIORITY_MARKET_SOURCES):
        if requested_asset_label in INDEX_ASSETS:
            return 8.0
        if requested_asset_label in MACRO_ASSETS or requested_asset_label in PRIORITY_CROSS_ASSET_ASSETS:
            return 6.0
        return 3.0
    return 0.0


def _asset_market_bonus(text: str, requested_asset_label: str | None) -> float:
    if not requested_asset_label:
        return 0.0
    cues = ASSET_MARKET_CUES.get(requested_asset_label, [])
    if requested_asset_label in INDEX_ASSETS:
        weight = 5.0
    elif requested_asset_label in PRIORITY_CROSS_ASSET_ASSETS:
        weight = 4.0
    else:
        weight = 3.0
    return match_count(text, cues) * weight


def _company_specific_penalty(row: dict[str, Any], requested_asset_label: str | None) -> float:
    if requested_asset_label not in INDEX_ASSETS:
        return 0.0
    title = str(row.get("title") or "")
    url = str(row.get("document_identifier") or "")
    upper_title = title.upper()
    penalty = 0.0
    if any(token in upper_title for token in COMPANY_SPECIFIC_TITLE_TERMS):
        penalty += 16.0
    if "/markets/stocks/" in url:
        penalty += 10.0
    return penalty


def _incomplete_text_bonus(row: dict[str, Any]) -> float:
    title_len = _text_len(row.get("title"))
    summary_len = _text_len(row.get("summary"))
    text_len = _text_len(row.get("text"))
    if title_len > 0 and summary_len == 0 and text_len == 0:
        return 5.0
    if title_len > 0 and summary_len > 0 and text_len < 200:
        return 3.0
    return 0.0


def _candidate_relevance_score(
    row: dict[str, Any],
    requested_asset_label: str | None,
    requested_factor_label: str | None,
) -> float:
    text = _row_relevance_text(row)
    asset_hits = match_count(text, asset_cues(requested_asset_label))
    factor_hits = match_count(text, factor_cues(requested_factor_label))
    source_bonus = _source_bonus(row, requested_asset_label)
    asset_market_bonus = _asset_market_bonus(text, requested_asset_label)
    company_specific_penalty = _company_specific_penalty(row, requested_asset_label)
    incomplete_bonus = _incomplete_text_bonus(row)
    title_bonus = 1.0 if isinstance(row.get("title"), str) and row.get("title", "").strip() else 0.0
    summary_bonus = 0.5 if isinstance(row.get("summary"), str) and row.get("summary", "").strip() else 0.0
    text_bonus = 0.5 if isinstance(row.get("text"), str) and row.get("text", "").strip() else 0.0
    return (
        (asset_hits * 6.0)
        + (factor_hits * 8.0)
        + source_bonus
        + asset_market_bonus
        + incomplete_bonus
        + title_bonus
        + summary_bonus
        + text_bonus
        - company_specific_penalty
    )


def _select_target_rows(
    rows: list[dict[str, Any]],
    enrich_max_docs: int,
    requested_asset_label: str | None,
    requested_factor_label: str | None,
) -> tuple[list[dict[str, Any]], int]:
    missing_rows = [row for row in rows if _row_needs_enrichment(row)]
    ranked_rows = [
        (
            _candidate_relevance_score(
                row,
                requested_asset_label=requested_asset_label,
                requested_factor_label=requested_factor_label,
            ),
            index,
            row,
        )
        for index, row in enumerate(missing_rows)
    ]
    if requested_asset_label or requested_factor_label:
        ranked_rows.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        positive_rows = [row for score, _, row in ranked_rows if score > 0.0]
        selected = positive_rows if positive_rows else [row for _, _, row in ranked_rows]
    else:
        selected = [row for _, _, row in ranked_rows]
    if enrich_max_docs >= 0:
        selected = selected[:enrich_max_docs]
    return selected, len(missing_rows)


def enrich_parquet_file(
    parquet_path: Path,
    enrich_max_docs: int,
    timeout: float,
    user_agent: str,
    overwrite: bool,
    requested_asset_label: str | None,
    requested_factor_label: str | None,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.ParquetFile(parquet_path).read()
    rows = table.to_pylist()
    target_rows, missing_count = _select_target_rows(
        rows,
        enrich_max_docs=enrich_max_docs,
        requested_asset_label=requested_asset_label,
        requested_factor_label=requested_factor_label,
    )
    stats = enrich_rows(
        target_rows,
        enrich_max_docs=enrich_max_docs if enrich_max_docs >= 0 else len(target_rows),
        timeout=timeout,
        user_agent=user_agent,
    )

    if stats["rows_enriched"] > 0:
        enriched_table = pa.Table.from_pylist(rows)
        target_path = parquet_path if overwrite else parquet_path.with_name(f"{parquet_path.stem}-enriched{parquet_path.suffix}")
        pq.write_table(enriched_table, target_path)
    else:
        target_path = parquet_path

    return {
        "path": str(parquet_path),
        "output_path": str(target_path),
        "rows_total": len(rows),
        "rows_missing_text_before": missing_count,
        "rows_selected_for_enrichment": len(target_rows),
        "rows_enriched": stats["rows_enriched"],
        "attempted_fetches": stats["attempted_fetches"],
        "unique_urls_seen": stats["unique_urls_seen"],
        "overwrote_source": overwrite and stats["rows_enriched"] > 0,
    }


def enrich_staged_parquet(
    input_root: Path,
    enrich_max_docs_per_file: int,
    timeout: float,
    user_agent: str,
    overwrite: bool,
    include_glob: str | None,
    requested_asset_label: str | None,
    requested_factor_label: str | None,
) -> dict[str, Any]:
    files = parquet_files(input_root, include_glob=include_glob)
    if not files:
        raise FileNotFoundError(f"no parquet files found under {input_root}")

    file_payloads = [
        enrich_parquet_file(
            parquet_path=path,
            enrich_max_docs=enrich_max_docs_per_file,
            timeout=timeout,
            user_agent=user_agent,
            overwrite=overwrite,
            requested_asset_label=requested_asset_label,
            requested_factor_label=requested_factor_label,
        )
        for path in files
    ]
    summary = {
        "input_root": str(input_root),
        "file_count": len(file_payloads),
        "overwrite": overwrite,
        "include_glob": include_glob,
        "requested_asset_label": requested_asset_label,
        "requested_factor_label": requested_factor_label,
        "rows_total": sum(int(item["rows_total"]) for item in file_payloads),
        "rows_missing_text_before": sum(int(item["rows_missing_text_before"]) for item in file_payloads),
        "rows_selected_for_enrichment": sum(int(item["rows_selected_for_enrichment"]) for item in file_payloads),
        "rows_enriched": sum(int(item["rows_enriched"]) for item in file_payloads),
        "attempted_fetches": sum(int(item["attempted_fetches"]) for item in file_payloads),
        "files": file_payloads,
    }
    manifest_path = input_root / "enrichment-manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2))
    summary["manifest_path"] = str(manifest_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--enrich-max-docs-per-file", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=DEFAULT_HTTP_TIMEOUT)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-glob")
    parser.add_argument("--asset-label")
    parser.add_argument("--factor-label")
    return parser.parse_args()


def main() -> int:
    ensure_dependencies()
    args = parse_args()
    payload = enrich_staged_parquet(
        input_root=Path(args.input_root),
        enrich_max_docs_per_file=args.enrich_max_docs_per_file,
        timeout=args.timeout,
        user_agent=args.user_agent,
        overwrite=args.overwrite,
        include_glob=args.include_glob,
        requested_asset_label=args.asset_label,
        requested_factor_label=args.factor_label,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
