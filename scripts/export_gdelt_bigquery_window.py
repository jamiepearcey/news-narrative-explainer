#!/usr/bin/env python3
"""Export a multi-day BigQuery GDELT candidate window to local parquet."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from fetch_gdelt_bigquery_candidates import (
    DEFAULT_THEME_PATTERN,
    bq_timestamp,
    ensure_dependencies,
    parse_datetime,
    resolve_project,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "gdelt_candidates_10d"
DEFAULT_ROWS_PER_DAY = 2_000


@dataclass(frozen=True)
class WindowDay:
    day: date

    @property
    def start(self) -> datetime:
        return datetime(self.day.year, self.day.month, self.day.day, tzinfo=UTC)

    @property
    def end(self) -> datetime:
        return self.start + timedelta(days=1)

    @property
    def partition(self) -> str:
        return self.day.isoformat()


def iter_window_days(start: datetime, end: datetime) -> list[WindowDay]:
    start_day = start.astimezone(UTC).date()
    end_day = end.astimezone(UTC).date()
    if start >= end:
        raise ValueError("start must be before end")
    days: list[WindowDay] = []
    cursor = start_day
    while cursor < end_day:
        days.append(WindowDay(cursor))
        cursor += timedelta(days=1)
    return days


def build_day_query(window_day: WindowDay, theme_pattern: str, rows_per_day: int | None) -> str:
    limit_clause = "" if rows_per_day is None else f"\nLIMIT {int(rows_per_day)}"
    return f"""
SELECT
  CAST(DATE AS STRING) AS record_datetime,
  CAST(DATE(_PARTITIONTIME) AS STRING) AS partition_date,
  SourceCommonName AS source_common_name,
  DocumentIdentifier AS document_identifier,
  CAST(NULL AS STRING) AS title,
  CAST(NULL AS STRING) AS summary,
  CAST(NULL AS STRING) AS text,
  V2Themes AS v2_themes,
  V2Tone AS v2_tone,
  V2Locations AS v2_locations,
  V2Persons AS v2_persons,
  V2Organizations AS v2_organizations,
  AllNames AS all_names,
  TO_JSON_STRING(STRUCT(
    SourceCollectionIdentifier AS source_collection_identifier,
    Counts AS counts,
    V2Counts AS v2_counts,
    Dates AS dates,
    GCAM AS gcam,
    SharingImage AS sharing_image,
    RelatedImages AS related_images,
    SocialImageEmbeds AS social_image_embeds,
    SocialVideoEmbeds AS social_video_embeds,
    Quotations AS quotations,
    Amounts AS amounts,
    TranslationInfo AS translation_info,
    Extras AS extras,
    CURRENT_TIMESTAMP() AS fetched_at,
    'gdelt-bq.gdeltv2.gkg_partitioned' AS source_table
  )) AS metadata_json
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= TIMESTAMP('{bq_timestamp(window_day.start)}')
  AND _PARTITIONTIME < TIMESTAMP('{bq_timestamp(window_day.end)}')
  AND REGEXP_CONTAINS(IFNULL(V2Themes, ''), r'{theme_pattern}'){limit_clause}
""".strip()


def build_day_count_query(window_day: WindowDay, theme_pattern: str) -> str:
    return f"""
SELECT COUNT(*) AS total_rows
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= TIMESTAMP('{bq_timestamp(window_day.start)}')
  AND _PARTITIONTIME < TIMESTAMP('{bq_timestamp(window_day.end)}')
  AND REGEXP_CONTAINS(IFNULL(V2Themes, ''), r'{theme_pattern}')
""".strip()


def output_path(output_root: Path, window_day: WindowDay, run_time: datetime) -> Path:
    stamp = run_time.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return output_root / f"dt={window_day.partition}" / f"part-{stamp}-bigquery-window.parquet"


def write_day_parquet_from_batches(batches: object, path: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    row_count = 0
    try:
        for batch in batches:
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            row_count += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    return row_count


def export_window(args: argparse.Namespace) -> dict[str, object]:
    ensure_dependencies()
    from google.cloud import bigquery

    start = parse_datetime(args.start)
    end = parse_datetime(args.end)
    days = iter_window_days(start, end)
    if not days:
        raise ValueError("window produced no days")

    service_account_json = args.service_account_json
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=True) if service_account_json else nullcontext() as cred_file:
        if service_account_json:
            cred_file.write(service_account_json)
            cred_file.flush()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_file.name

        client = bigquery.Client(project=args.project, location=args.location)
        day_payloads: list[dict[str, object]] = []
        total_bytes_processed = 0
        total_rows_exported = 0
        total_matching_rows = 0

        for window_day in days:
            query = build_day_query(window_day, args.theme_pattern, args.rows_per_day)
            job_config = bigquery.QueryJobConfig(
                dry_run=args.dry_run,
                use_query_cache=not args.dry_run,
            )
            query_job = client.query(query, job_config=job_config)
            count_job = client.query(
                build_day_count_query(window_day, args.theme_pattern),
                job_config=job_config,
            )
            query_bytes = int(query_job.total_bytes_processed or 0)
            count_bytes = int(count_job.total_bytes_processed or 0)
            total_bytes_processed += query_bytes + count_bytes

            payload: dict[str, object] = {
                "date": window_day.partition,
                "query_bytes_processed": query_bytes,
                "count_bytes_processed": count_bytes,
                "rows_cap": args.rows_per_day,
                "query": query if args.include_queries else None,
            }
            if args.dry_run:
                day_payloads.append(payload)
                continue

            total_rows = int(next(count_job.result())["total_rows"])
            out = output_path(Path(args.output_root), window_day, datetime.now(UTC))
            row_count = write_day_parquet_from_batches(
                query_job.result(page_size=args.page_size).to_arrow_iterable(),
                out,
            )
            file_bytes = out.stat().st_size
            total_rows_exported += row_count
            total_matching_rows += total_rows
            payload.update(
                {
                    "total_matching_rows": total_rows,
                    "rows_exported": row_count,
                    "output_path": str(out),
                    "output_bytes": file_bytes,
                    "output_mb": round(file_bytes / 1_048_576, 2),
                }
            )
            day_payloads.append(payload)

    result: dict[str, object] = {
        "project": args.project,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": len(days),
        "rows_per_day": args.rows_per_day,
        "dry_run": args.dry_run,
        "day_results": day_payloads,
        "total_bytes_processed": total_bytes_processed,
    }
    if not args.dry_run:
        result["total_rows_exported"] = total_rows_exported
        result["total_matching_rows"] = total_matching_rows
        result["output_root"] = str(args.output_root)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("SOURCE_BIGQUERY_IPC__PROJECT_ID"))
    parser.add_argument("--location", default="US")
    parser.add_argument(
        "--service-account-json",
        default=os.environ.get("SOURCE_BIGQUERY_IPC__SERVICE_ACCOUNT_JSON"),
        help="Service account JSON string. Defaults to SOURCE_BIGQUERY_IPC__SERVICE_ACCOUNT_JSON.",
    )
    parser.add_argument("--start", required=True, help="UTC ISO timestamp, inclusive.")
    parser.add_argument("--end", required=True, help="UTC ISO timestamp, exclusive.")
    parser.add_argument("--theme-pattern", default=DEFAULT_THEME_PATTERN)
    parser.add_argument("--rows-per-day", type=int, default=DEFAULT_ROWS_PER_DAY)
    parser.add_argument("--page-size", type=int, default=10_000)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-queries", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.project = resolve_project(args.project, args.service_account_json)
    if not args.project:
        raise SystemExit("--project or GOOGLE_CLOUD_PROJECT is required")
    if args.rows_per_day is not None and args.rows_per_day <= 0:
        raise SystemExit("--rows-per-day must be positive")
    if args.page_size <= 0:
        raise SystemExit("--page-size must be positive")
    if parse_datetime(args.start) >= parse_datetime(args.end):
        raise SystemExit("--start must be before --end")
    payload = export_window(args)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
