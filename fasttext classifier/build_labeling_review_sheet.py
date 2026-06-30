#!/usr/bin/env python3
"""Build a compact operator review sheet for the current active labeling batch."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
RESULTS_DIR = WORK_DIR / "results"
STATUS_JSON = RESULTS_DIR / "labeling_status.json"
OUTPUT_JSON = RESULTS_DIR / "active_labeling_review_sheet.json"
OUTPUT_MD = RESULTS_DIR / "active_labeling_review_sheet.md"
OUTPUT_CSV = RESULTS_DIR / "active_labeling_review_sheet.csv"
OUTPUT_SHORTLIST_JSON = RESULTS_DIR / "active_labeling_shortlist.json"
OUTPUT_SHORTLIST_MD = RESULTS_DIR / "active_labeling_shortlist.md"
OUTPUT_SHORTLIST_CSV = RESULTS_DIR / "active_labeling_shortlist.csv"
OUTPUT_SHORTLIST_RESPONSE_JSON = RESULTS_DIR / "active_labeling_shortlist.response.json"
OUTPUT_SHORTLIST_RESPONSE_MD = RESULTS_DIR / "active_labeling_shortlist.response.md"

SIGNAL_COLUMNS = [
    "finance_hits",
    "macro_hits",
    "geo_hits",
    "company_hits",
    "equity_hits",
    "press_hits",
    "keep_theme_hits",
    "macro_theme_hits",
    "geo_theme_hits",
]

TEXT_COLUMNS = ["title", "summary_text", "market_context_text"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-json", default=str(STATUS_JSON))
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--output-csv", default=str(OUTPUT_CSV))
    parser.add_argument("--output-shortlist-json", default=str(OUTPUT_SHORTLIST_JSON))
    parser.add_argument("--output-shortlist-md", default=str(OUTPUT_SHORTLIST_MD))
    parser.add_argument("--output-shortlist-csv", default=str(OUTPUT_SHORTLIST_CSV))
    parser.add_argument("--output-shortlist-response-json", default=str(OUTPUT_SHORTLIST_RESPONSE_JSON))
    parser.add_argument("--output-shortlist-response-md", default=str(OUTPUT_SHORTLIST_RESPONSE_MD))
    parser.add_argument("--shortlist-limit", type=int, default=12)
    return parser.parse_args()


def load_status(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def choose_active_batch(status: dict[str, object]) -> dict[str, object]:
    packet_batches = status.get("packet_batches", [])
    with_real_response = [batch for batch in packet_batches if batch.get("response_present")]
    if with_real_response:
        return sorted(with_real_response, key=lambda batch: str(batch.get("batch") or ""))[0]
    next_batch_name = status.get("next_batch")
    for batch in packet_batches:
        if batch.get("batch") == next_batch_name:
            return batch
    raise SystemExit("No active or next batch found in labeling_status.json")


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def summarize_text(row: dict[str, str]) -> str:
    parts: list[str] = []
    for column in TEXT_COLUMNS:
        value = (row.get(column) or "").strip()
        if value and value not in parts:
            parts.append(value)
    if not parts:
        return ""
    return " | ".join(parts)


def summarize_signals(row: dict[str, str]) -> dict[str, int]:
    return {column: parse_int(row.get(column, "")) for column in SIGNAL_COLUMNS}


def build_cues(row: dict[str, str], signals: dict[str, int]) -> list[str]:
    cues: list[str] = []
    weak_label = row.get("weak_label", "")
    predicted_label = row.get("predicted_label", "")
    if weak_label and predicted_label and weak_label != predicted_label:
        cues.append(f"disagreement: weak `{weak_label}` vs model `{predicted_label}`")
    decision_band = row.get("decision_band", "")
    if decision_band:
        cues.append(f"decision band `{decision_band}`")
    if signals.get("press_hits", 0) > 0:
        cues.append(f"press signal {signals['press_hits']}")
    if signals.get("company_hits", 0) > 0:
        cues.append(f"company signal {signals['company_hits']}")
    if signals.get("finance_hits", 0) > 0:
        cues.append(f"finance signal {signals['finance_hits']}")
    if signals.get("macro_hits", 0) > 0:
        cues.append(f"macro signal {signals['macro_hits']}")
    if signals.get("geo_hits", 0) > 0:
        cues.append(f"geo signal {signals['geo_hits']}")
    if signals.get("equity_hits", 0) > 0:
        cues.append(f"equity signal {signals['equity_hits']}")
    if signals.get("keep_theme_hits", 0) > 0:
        cues.append(f"theme hits {signals['keep_theme_hits']}")
    return cues


def priority_score(row: dict[str, object]) -> int:
    score = 0
    weak_label = str(row.get("weak_label", ""))
    predicted_label = str(row.get("predicted_label", ""))
    decision_band = str(row.get("decision_band", ""))
    stratum = str(row.get("stratum", ""))
    text = str(row.get("text", ""))
    signals = row.get("signals", {})
    if weak_label and predicted_label and weak_label != predicted_label:
        score += 100
    if decision_band == "review":
        score += 60
    if "press_release" in stratum:
        score += 40
    if not str(row.get("title", "")).strip():
        score += 35
    if len(text) < 80:
        score += 20
    if isinstance(signals, dict):
        if int(signals.get("press_hits", 0)) > 0:
            score += 12
        if int(signals.get("company_hits", 0)) > 0:
            score += 8
        if int(signals.get("finance_hits", 0)) > 0 and int(signals.get("macro_hits", 0)) == 0:
            score += 6
        if int(signals.get("keep_theme_hits", 0)) >= 20:
            score += 5
    return score


def recommended_focus_bucket(row: dict[str, object]) -> str:
    weak_label = str(row.get("weak_label", ""))
    predicted_label = str(row.get("predicted_label", ""))
    decision_band = str(row.get("decision_band", ""))
    stratum = str(row.get("stratum", ""))
    if weak_label and predicted_label and weak_label != predicted_label:
        return "model_vs_weak_disagreement"
    if decision_band == "review":
        return "manual_review_band"
    if "press_release" in stratum:
        return "press_release_boundary"
    return "lower_priority_confirmation"


def render_row_md(index: int, row: dict[str, object]) -> list[str]:
    lines = [
        f"## {index}. {row['source_domain']} | {row['stratum']}",
        "",
        f"- Focus: `{row['focus_bucket']}` at priority `{row['priority_score']}`",
        f"- Title: {row['title'] or '(missing title)'}",
        f"- Current label: `{row['current_label'] or 'unlabeled'}`",
        f"- Weak vs model: `{row['weak_label']}` -> `{row['predicted_label']}` at `{row['predicted_score'] or 'n/a'}`",
        f"- Decision band: `{row['decision_band'] or 'n/a'}`",
        f"- Signals: `finance={row['signals']['finance_hits']}` `macro={row['signals']['macro_hits']}` `geo={row['signals']['geo_hits']}` `company={row['signals']['company_hits']}` `equity={row['signals']['equity_hits']}` `press={row['signals']['press_hits']}` `themes={row['signals']['keep_theme_hits']}/{row['signals']['macro_theme_hits']}/{row['signals']['geo_theme_hits']}`",
    ]
    if row["cues"]:
        lines.append(f"- Cues: {'; '.join(row['cues'])}")
    if row["text"]:
        lines.append(f"- Text: {row['text']}")
    lines.append(f"- Document: `{row['document_identifier']}`")
    lines.append("")
    return lines


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "priority_rank",
        "priority_score",
        "focus_bucket",
        "stratum",
        "source_domain",
        "title",
        "weak_label",
        "predicted_label",
        "predicted_score",
        "decision_band",
        "current_label",
        "finance_hits",
        "macro_hits",
        "geo_hits",
        "company_hits",
        "equity_hits",
        "press_hits",
        "keep_theme_hits",
        "macro_theme_hits",
        "geo_theme_hits",
        "document_identifier",
        "text",
        "cues",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            signals = row["signals"]
            writer.writerow(
                {
                    "priority_rank": index,
                    "priority_score": row["priority_score"],
                    "focus_bucket": row["focus_bucket"],
                    "stratum": row["stratum"],
                    "source_domain": row["source_domain"],
                    "title": row["title"],
                    "weak_label": row["weak_label"],
                    "predicted_label": row["predicted_label"],
                    "predicted_score": row["predicted_score"],
                    "decision_band": row["decision_band"],
                    "current_label": row["current_label"],
                    "finance_hits": signals["finance_hits"],
                    "macro_hits": signals["macro_hits"],
                    "geo_hits": signals["geo_hits"],
                    "company_hits": signals["company_hits"],
                    "equity_hits": signals["equity_hits"],
                    "press_hits": signals["press_hits"],
                    "keep_theme_hits": signals["keep_theme_hits"],
                    "macro_theme_hits": signals["macro_theme_hits"],
                    "geo_theme_hits": signals["geo_theme_hits"],
                    "document_identifier": row["document_identifier"],
                    "text": row["text"],
                    "cues": " | ".join(row["cues"]),
                }
            )


def build_focus_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    focus_counts: dict[str, int] = {}
    for row in rows:
        focus_bucket = str(row["focus_bucket"])
        focus_counts[focus_bucket] = focus_counts.get(focus_bucket, 0) + 1
    return focus_counts


def render_sheet_markdown(
    title: str,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    csv_path: Path,
    full_rows: int,
    is_shortlist: bool = False,
) -> str:
    lines = [
        f"# {title}: {summary['active_batch']}",
        "",
        f"- Rows: `{len(rows)}`",
        f"- Completed in batch: `{summary['completed_rows']}`",
        f"- Remaining in batch: `{summary['remaining_rows']}`",
        f"- Prompt: `{summary['prompt_path']}`",
        f"- Template: `{summary['template_path']}`",
        f"- Response target: `{summary['response_path']}`",
        f"- Compact CSV: `{csv_path}`",
        f"- Focus buckets: `{json.dumps(build_focus_counts(rows), ensure_ascii=True)}`",
    ]
    if is_shortlist:
        lines.append(f"- Shortlist limit: `{summary['shortlist_limit']}`")
        lines.append(f"- Full batch rows: `{full_rows}`")
    lines.extend(
        [
            "",
            "## Priority Order",
            "",
            "Work from the top down. Rows are sorted to put disagreements and explicit `review` cases first.",
            "",
            "## Top Priority Preview",
            "",
        ]
    )
    for index, row in enumerate(rows[:8], start=1):
        lines.append(
            f"- `{index}` `{row['focus_bucket']}` `{row['source_domain']}` `{row['weak_label']}` -> `{row['predicted_label']}` `{row['decision_band']}`"
        )
    lines.extend(["", "## Full List", ""])
    for index, row in enumerate(rows, start=1):
        lines.extend(render_row_md(index, row))
    return "\n".join(lines)


def build_response_stub(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    return [
        {
            "document_identifier": str(row["document_identifier"]),
            "label": "",
            "confidence": "",
            "notes": "",
        }
        for row in rows
    ]


def render_response_stub_markdown(
    rows: list[dict[str, object]],
    response_json_path: Path,
    response_stub: list[dict[str, str]],
) -> str:
    lines = [
        "# Active Labeling Shortlist Response Stub",
        "",
        f"- Rows: `{len(rows)}`",
        f"- JSON target: `{response_json_path}`",
        "",
        "Fill the `label`, `confidence`, and `notes` fields in the JSON payload below.",
        "",
        "```json",
        json.dumps(response_stub, indent=2),
        "```",
        "",
        "## Row Guide",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"- `{index}` `{row['source_domain']}` `{row['weak_label']}` -> `{row['predicted_label']}` `{row['decision_band']}` `{row['document_identifier']}`",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    status = load_status(Path(args.status_json))
    batch = choose_active_batch(status)
    rows_path = Path(batch["rows_path"])
    template_path = Path(batch["template_path"])
    response_path = Path(batch["response_path"]) if batch.get("response_path") else Path(batch["expected_response_json"])
    rows = read_rows(rows_path)
    template_rows = json.loads(template_path.read_text(encoding="utf-8"))
    template_by_id = {row["document_identifier"]: row for row in template_rows}

    review_rows: list[dict[str, object]] = []
    completed_rows = 0
    for row in rows:
        template_row = template_by_id.get(row["document_identifier"], {})
        current_label = (template_row.get("label") or "").strip()
        if current_label:
            completed_rows += 1
        signals = summarize_signals(row)
        review_rows.append(
            {
                "document_identifier": row["document_identifier"],
                "source_domain": row.get("source_domain", ""),
                "stratum": row.get("stratum", ""),
                "title": row.get("title", ""),
                "weak_label": row.get("weak_label", ""),
                "predicted_label": row.get("predicted_label", ""),
                "predicted_score": parse_float(row.get("predicted_score", "")),
                "decision_band": row.get("decision_band", ""),
                "current_label": current_label,
                "confidence": template_row.get("confidence", ""),
                "notes": template_row.get("notes", ""),
                "signals": signals,
                "text": summarize_text(row),
                "cues": build_cues(row, signals),
            }
        )

    for row in review_rows:
        row["priority_score"] = priority_score(row)
        row["focus_bucket"] = recommended_focus_bucket(row)
    review_rows.sort(
        key=lambda row: (
            -int(row["priority_score"]),
            str(row["focus_bucket"]),
            str(row["source_domain"]),
            str(row["document_identifier"]),
        )
    )

    focus_counts = build_focus_counts(review_rows)
    shortlist_rows = [row for row in review_rows if not str(row["current_label"]).strip()][
        : max(args.shortlist_limit, 0)
    ]
    shortlist_focus_counts = build_focus_counts(shortlist_rows)

    summary = {
        "active_batch": batch["batch"],
        "rows": len(review_rows),
        "completed_rows": completed_rows,
        "remaining_rows": len(review_rows) - completed_rows,
        "prompt_path": batch["prompt_path"],
        "template_path": batch["template_path"],
        "rows_path": batch["rows_path"],
        "response_path": str(response_path),
        "focus_counts": focus_counts,
        "shortlist_limit": args.shortlist_limit,
        "shortlist_rows": len(shortlist_rows),
        "shortlist_focus_counts": shortlist_focus_counts,
        "review_rows": review_rows,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_csv = Path(args.output_csv)
    output_shortlist_json = Path(args.output_shortlist_json)
    output_shortlist_md = Path(args.output_shortlist_md)
    output_shortlist_csv = Path(args.output_shortlist_csv)
    output_shortlist_response_json = Path(args.output_shortlist_response_json)
    output_shortlist_response_md = Path(args.output_shortlist_response_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_shortlist_json.parent.mkdir(parents=True, exist_ok=True)
    output_shortlist_md.parent.mkdir(parents=True, exist_ok=True)
    output_shortlist_csv.parent.mkdir(parents=True, exist_ok=True)
    output_shortlist_response_json.parent.mkdir(parents=True, exist_ok=True)
    output_shortlist_response_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(output_csv, review_rows)
    shortlist_summary = {
        "active_batch": summary["active_batch"],
        "completed_rows": summary["completed_rows"],
        "remaining_rows": summary["remaining_rows"],
        "prompt_path": summary["prompt_path"],
        "template_path": summary["template_path"],
        "response_path": summary["response_path"],
        "shortlist_limit": summary["shortlist_limit"],
        "focus_counts": shortlist_focus_counts,
        "review_rows": shortlist_rows,
    }
    output_shortlist_json.write_text(json.dumps(shortlist_summary, indent=2), encoding="utf-8")
    write_csv(output_shortlist_csv, shortlist_rows)
    response_stub = build_response_stub(shortlist_rows)
    output_shortlist_response_json.write_text(json.dumps(response_stub, indent=2), encoding="utf-8")
    output_shortlist_response_md.write_text(
        render_response_stub_markdown(shortlist_rows, output_shortlist_response_json, response_stub),
        encoding="utf-8",
    )
    output_md.write_text(
        render_sheet_markdown(
            "Active Labeling Review Sheet",
            review_rows,
            summary,
            output_csv,
            full_rows=len(review_rows),
        ),
        encoding="utf-8",
    )
    output_shortlist_md.write_text(
        render_sheet_markdown(
            "Active Labeling Shortlist",
            shortlist_rows,
            shortlist_summary,
            output_shortlist_csv,
            full_rows=len(review_rows),
            is_shortlist=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
