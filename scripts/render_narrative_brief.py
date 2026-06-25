#!/usr/bin/env python3
"""Render a markdown narrative brief from an explain-move JSON payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to explain-move JSON")
    parser.add_argument("--output", required=True, help="Path to markdown output")
    return parser.parse_args()


def render(payload: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Narrative Brief: {payload['asset_label']}")
    lines.append("")
    window = payload["window"]
    lines.append(f"- `start_date`: `{window['start_date']}`")
    lines.append(f"- `end_date`: `{window['end_date']}`")
    lines.append("")

    lines.append("## Top Narratives")
    lines.append("")
    for row in payload["top_narratives"]:
        lines.append(
            f"- `{row['factor_label']}` | news `{row['news_count']}` | "
            f"avg tone `{row['avg_tone_mean']}` | intensity `{row['avg_event_intensity']}`"
        )
    lines.append("")

    lines.append("## Timeline")
    lines.append("")
    for row in payload["timeline"]:
        lines.append(
            f"- `{row['bucket_time']}` | `{row['factor_label']}` | news `{row['news_count']}` | "
            f"tone `{row['tone_mean']}` | dispersion `{row['source_dispersion']}`"
        )
    lines.append("")

    lines.append("## Supporting Documents")
    lines.append("")
    for row in payload["supporting_docs"]:
        lines.append(
            f"- `{row['event_time']}` | `{row['factor_label']}` | `{row['source_domain']}` | {row['document_identifier']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text())
    Path(args.output).write_text(render(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
