#!/usr/bin/env python3
"""Build batched GPT/manual labeling packets from candidate CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
FEEDBACK_DIR = WORK_DIR / "feedback"
RESULTS_DIR = WORK_DIR / "results"
DEFAULT_INPUT_CSV = FEEDBACK_DIR / "gpt_eval_hard_cases.csv"
DEFAULT_OUTPUT_DIR = FEEDBACK_DIR / "labeling_packets"
DEFAULT_SUMMARY_JSON = RESULTS_DIR / "gpt_labeling_packets_summary.json"
EVAL_DETAIL_CSV = RESULTS_DIR / "gpt_eval_set_detailed.csv"

LABELS = [
    "keep_finance",
    "keep_macro",
    "keep_geopolitics",
    "keep_company_event",
    "drop_sports",
    "drop_entertainment",
    "drop_lifestyle",
    "drop_local_crime",
    "drop_low_quality",
    "drop_press_release",
]

EXEMPLAR_STRATUM_MAP = {
    "macro_geo_boundary": ["search_geopolitics_market", "search_macro_energy"],
    "finance_company_boundary": ["search_company_event", "domain_disagreement"],
    "press_release_boundary": ["search_company_event", "likely_negative", "domain_disagreement"],
    "finance_macro_boundary": ["domain_disagreement", "search_macro_rates", "search_macro_energy"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if not (row.get("label") or "").strip()]


def interleave_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    buckets: dict[str, deque[dict[str, str]]] = defaultdict(deque)
    for row in rows:
        buckets[row["stratum"]].append(row)

    ordered_strata = sorted(
        buckets,
        key=lambda key: (-len(buckets[key]), key),
    )
    interleaved: list[dict[str, str]] = []
    while ordered_strata:
        next_round: list[str] = []
        for stratum in ordered_strata:
            bucket = buckets[stratum]
            if not bucket:
                continue
            interleaved.append(bucket.popleft())
            if bucket:
                next_round.append(stratum)
        ordered_strata = next_round
    return interleaved


def load_eval_exemplars() -> dict[str, list[dict[str, str]]]:
    if not EVAL_DETAIL_CSV.exists():
        return {}
    by_stratum: dict[str, list[dict[str, str]]] = defaultdict(list)
    with EVAL_DETAIL_CSV.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("coverage_status") != "matched_scored":
                continue
            by_stratum[row.get("stratum", "")].append(row)
    return by_stratum


def boundary_exemplars(strata: set[str], exemplar_pool: dict[str, list[dict[str, str]]]) -> list[str]:
    lines: list[str] = []
    for boundary in sorted(strata):
        source_strata = EXEMPLAR_STRATUM_MAP.get(boundary, [])
        examples: list[dict[str, str]] = []
        for source_stratum in source_strata:
            for row in exemplar_pool.get(source_stratum, []):
                if row.get("weak_correct") == "1" and row.get("model_correct") == "1":
                    continue
                examples.append(row)
        if not examples:
            continue
        lines.append(f"- `{boundary}` exemplars:")
        seen: set[tuple[str, str, str]] = set()
        added = 0
        for row in examples:
            key = (
                row.get("truth_label", ""),
                row.get("weak_label", ""),
                row.get("title", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            title = row.get("title", "").strip() or row.get("document_identifier", "")
            truth = row.get("truth_label", "")
            weak = row.get("weak_label", "")
            note = row.get("notes", "").strip()
            if note:
                lines.append(f"  - `{truth}` not `{weak}`: {title}. Note: {note}")
            else:
                lines.append(f"  - `{truth}` not `{weak}`: {title}")
            added += 1
            if added >= 3:
                break
    return lines


def packet_prompt(rows: list[dict[str, str]], exemplar_pool: dict[str, list[dict[str, str]]]) -> str:
    strata = {row["stratum"] for row in rows}
    records = []
    for row in rows:
        records.append(
            {
                "document_identifier": row["document_identifier"],
                "stratum": row["stratum"],
                "source_domain": row["source_domain"],
                "title": row.get("title", ""),
                "weak_label": row.get("weak_label", ""),
                "predicted_label": row.get("predicted_label", ""),
                "predicted_score": row.get("predicted_score", ""),
                "finance_cluster_score": row.get("finance_cluster_score", ""),
                "why_selected": row.get("why_selected", ""),
                "summary_text": row.get("summary_text", ""),
                "market_context_text": row.get("market_context_text", ""),
            }
        )
    boundary_rules: list[str] = []
    if "macro_geo_boundary" in strata:
        boundary_rules.extend(
            [
                "- `macro_geo_boundary`: use `keep_macro` when the main point is market/policy/economic transmission, even if sanctions, conflict, or state actors are mentioned.",
                "- `macro_geo_boundary`: use `keep_geopolitics` when the geopolitical event itself is the core event and markets are secondary.",
            ]
        )
    if "finance_company_boundary" in strata:
        boundary_rules.extend(
            [
                "- `finance_company_boundary`: use `keep_company_event` for firm-specific catalysts like earnings, guidance, FDA, fundraise, M&A, listings, or operational announcements.",
                "- `finance_company_boundary`: use `keep_finance` for stock-move explainers, analyst commentary, valuations, positioning, or broader security-market framing where the company event is not the central catalyst.",
            ]
        )
    if "press_release_boundary" in strata:
        boundary_rules.extend(
            [
                "- `press_release_boundary`: use `drop_press_release` for mirrored releases, promotional company announcements, wire-style notice language, or thin announcement-only items.",
                "- `press_release_boundary`: keep the row only if it contains substantive market-relevant reporting beyond the announcement shell.",
            ]
        )
    if "finance_macro_boundary" in strata:
        boundary_rules.extend(
            [
                "- `finance_macro_boundary`: use `keep_macro` for rates, inflation, central-bank policy, fiscal, labor, broad economy, capital flows, or sovereign/market-structure policy.",
                "- `finance_macro_boundary`: use `keep_finance` when the focus is security selection, company shares, investor positioning, or direct market moves rather than macro transmission.",
            ]
        )
    exemplar_lines = boundary_exemplars(strata, exemplar_pool)
    return (
        "Label each record with exactly one label from this closed set:\n"
        + ", ".join(LABELS)
        + "\n\nUse these rules:\n"
        + "- keep_finance: market moves, securities, yields, commodities, market positioning.\n"
        + "- keep_macro: macro policy, rates, inflation, fiscal, labor, broad economy.\n"
        + "- keep_geopolitics: conflict, sanctions, state action, supply-route risk when the geopolitical event itself is central.\n"
        + "- keep_company_event: earnings, guidance, M&A, FDA, listings, operational company catalysts.\n"
        + "- drop_press_release: company/wire announcement or mirrored release.\n"
        + "- drop_low_quality: scraper, junk, fake-local or thin low-value item.\n"
        + "- Other drop labels only for clear sports/entertainment/lifestyle/local-crime content.\n"
        + (
            "\nBoundary-specific adjudication for this batch:\n"
            + "\n".join(boundary_rules)
            + "\n"
            if boundary_rules
            else ""
        )
        + (
            "\nCalibration examples from the current labeled eval set:\n"
            + "\n".join(exemplar_lines)
            + "\n"
            if exemplar_lines
            else ""
        )
        + "\n"
        + "Return only JSON as a list of objects with keys:\n"
        + "document_identifier, label, confidence, notes\n\n"
        + "Records:\n"
        + json.dumps(records, indent=2, ensure_ascii=True)
    )


def packet_template(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "document_identifier": row["document_identifier"],
            "label": "",
            "stratum": row["stratum"],
            "source_domain": row["source_domain"],
            "confidence": "",
            "notes": "",
        }
        for row in rows
    ]


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    summary_json = Path(args.summary_json)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_csv)
    if not rows:
        raise SystemExit(f"No unlabeled rows found in {input_csv}")
    exemplar_pool = load_eval_exemplars()

    batch_size = max(1, args.batch_size)
    packet_rows = interleave_rows(rows)
    packet_count = math.ceil(len(packet_rows) / batch_size)
    batch_summaries: list[dict[str, object]] = []

    for batch_index in range(packet_count):
        start = batch_index * batch_size
        end = start + batch_size
        batch_rows = packet_rows[start:end]
        stem = f"batch_{batch_index + 1:03d}"
        prompt_path = output_dir / f"{stem}.prompt.md"
        template_path = output_dir / f"{stem}.template.json"
        csv_path = output_dir / f"{stem}.rows.csv"

        prompt_path.write_text(packet_prompt(batch_rows, exemplar_pool), encoding="utf-8")
        template_path.write_text(json.dumps(packet_template(batch_rows), indent=2), encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(batch_rows[0].keys()))
            writer.writeheader()
            writer.writerows(batch_rows)

        batch_summaries.append(
            {
                "batch": batch_index + 1,
                "rows": len(batch_rows),
                "prompt_path": str(prompt_path),
                "template_path": str(template_path),
                "csv_path": str(csv_path),
                "strata_counts": dict(sorted(Counter(row["stratum"] for row in batch_rows).items())),
            }
        )

    summary = {
        "input_csv": str(input_csv),
        "rows": len(rows),
        "batch_size": batch_size,
        "packet_count": packet_count,
        "output_dir": str(output_dir),
        "strata_counts": dict(sorted(Counter(row["stratum"] for row in rows).items())),
        "batches": batch_summaries,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
