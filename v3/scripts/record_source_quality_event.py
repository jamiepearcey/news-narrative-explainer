#!/usr/bin/env python3
"""Append source-quality feedback events to the JSONL ledger."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path("/Users/jamiepearcey/projects")
LEDGER_PATH = PROJECT_ROOT / "research/news-narrative-explainer/data/source_quality_events.jsonl"
VALID_SIGNALS = {
    "llm_judged_article_useful",
    "frequently_cited_by_high_quality_sources",
    "often_contradicted_by_later_reporting",
    "user_selected_as_evidence",
    "user_ignored_or_dismissed",
    "original_reporting",
    "duplicate_of_another_article",
    "produced_hallucination_in_summary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-domain", required=True)
    parser.add_argument("--signal", required=True, choices=sorted(VALID_SIGNALS))
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--article-id")
    parser.add_argument("--note")
    parser.add_argument("--actor", default="codex")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "ts_utc": datetime.now(UTC).isoformat(),
        "source_domain": args.source_domain.lower().removeprefix("www."),
        "signal": args.signal,
        "count": args.count,
        "actor": args.actor,
    }
    if args.article_id:
        event["article_id"] = args.article_id
    if args.note:
        event["note"] = args.note
    with LEDGER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    print(json.dumps({"ledger": str(LEDGER_PATH), "event": event}, indent=2))


if __name__ == "__main__":
    main()
