#!/usr/bin/env python3
"""Fetch GDELT BigQuery candidate rows to local parquet."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "gdelt_candidates"
DEFAULT_LOOKBACK_HOURS = 24
DEFAULT_MAX_RESULTS = 50_000
DEFAULT_THEME_PATTERN = (
    r"ECON_|BANKING|BANK_|CENTRAL_BANK|INFLATION|INTEREST_RATE|RATE_|GDP|"
    r"UNEMPLOYMENT|LABOR|LABOUR|RECESSION|DEBT|DEFAULT|LIQUIDITY|CREDIT|"
    r"WAR|SANCTION|TARIFF|TRADE|ELECTION|PROTEST|UNREST|TERROR|"
    r"SUPPLY_CHAIN|SHIPPING|OIL|GAS|LNG|POWER|METAL|COPPER|GOLD|"
    r"AGRICULTURE|EARNINGS|MERGER|ACQUISITION|LAYOFF|FRAUD|BITCOIN|"
    r"CRYPTO|STABLECOIN|ETF"
)


def ensure_dependencies() -> None:
    try:
        import google.cloud.bigquery  # noqa: F401
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except ModuleNotFoundError:
        uv = shutil.which("uv") or "/opt/homebrew/bin/uv"
        if not Path(uv).exists():
            raise RuntimeError(
                "google-cloud-bigquery and pyarrow are required, and `uv` was not found "
                "to bootstrap them"
            ) from None
        if os.environ.get("NEWS_NARRATIVE_FETCH_UV_BOOTSTRAPPED") == "1":
            raise
        env = os.environ.copy()
        env["NEWS_NARRATIVE_FETCH_UV_BOOTSTRAPPED"] = "1"
        os.execvpe(
            uv,
            [
                uv,
                "run",
                "--with",
                "google-cloud-bigquery>=3.25",
                "--with",
                "pyarrow>=16",
                str(Path(__file__).resolve()),
                *sys.argv[1:],
            ],
            env,
        )


def parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def default_window(lookback_hours: int) -> tuple[datetime, datetime]:
    end = datetime.now(UTC).replace(microsecond=0)
    return end - timedelta(hours=lookback_hours), end


def bq_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def output_path(output_root: Path, partition_date: str, run_time: datetime, suffix: str = "bigquery") -> Path:
    stamp = run_time.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return output_root / f"dt={partition_date}" / f"part-{stamp}-{suffix}.parquet"


def build_query(
    start: datetime,
    end: datetime,
    theme_pattern: str,
    max_results: int | None,
) -> str:
    limit_clause = "" if max_results is None else f"\nLIMIT {int(max_results)}"
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
  Extras AS gkg_extras,
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
WHERE _PARTITIONTIME >= TIMESTAMP('{bq_timestamp(start)}')
  AND _PARTITIONTIME < TIMESTAMP('{bq_timestamp(end)}')
  AND REGEXP_CONTAINS(IFNULL(V2Themes, ''), r'{theme_pattern}')
ORDER BY DATE DESC, DocumentIdentifier ASC{limit_clause}
""".strip()


def estimate_parquet_size(rows: int, bytes_processed: int | None) -> dict[str, Any]:
    # Observed local quant-algos finance-candidate parquet is about 1.1 KiB per
    # row with the narrower legacy schema. The richer standalone fetch preserves
    # quotations/media/extras, so use a wider 2-10 KiB planning band.
    low_bytes_per_row = 2_000
    high_bytes_per_row = 10_000
    return {
        "rows": rows,
        "estimated_parquet_mb_low": round(rows * low_bytes_per_row / 1_048_576, 1),
        "estimated_parquet_mb_high": round(rows * high_bytes_per_row / 1_048_576, 1),
        "bigquery_bytes_processed": bytes_processed,
    }


def fetch_to_parquet(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dependencies()
    from google.cloud import bigquery
    import pyarrow as pa
    import pyarrow.parquet as pq

    if args.start:
        start = parse_datetime(args.start)
        end = parse_datetime(args.end) if args.end else datetime.now(UTC).replace(microsecond=0)
    else:
        start, end = default_window(args.lookback_hours)

    query = build_query(
        start=start,
        end=end,
        theme_pattern=args.theme_pattern,
        max_results=args.max_results,
    )
    service_account_json = args.service_account_json
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=True) if service_account_json else nullcontext() as cred_file:
        if service_account_json:
            cred_file.write(service_account_json)
            cred_file.flush()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_file.name

        client = bigquery.Client(project=args.project, location=args.location)
        job_config = bigquery.QueryJobConfig(
            dry_run=args.dry_run,
            use_query_cache=not args.dry_run,
        )
        job = client.query(query, job_config=job_config)
        if args.dry_run:
            return {
                "dry_run": True,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "max_results": args.max_results,
                "total_bytes_processed": job.total_bytes_processed,
                "query": query,
            }

        rows_iter = job.result(max_results=args.max_results)
        rows = [dict(row.items()) for row in rows_iter]
    if args.dry_run:
        raise AssertionError("unreachable dry-run branch")
    schema = [
        ("record_datetime", pa.string()),
        ("partition_date", pa.string()),
        ("source_common_name", pa.string()),
        ("document_identifier", pa.string()),
        ("title", pa.string()),
        ("summary", pa.string()),
        ("text", pa.string()),
        ("v2_themes", pa.string()),
        ("v2_tone", pa.string()),
        ("v2_locations", pa.string()),
        ("v2_persons", pa.string()),
        ("v2_organizations", pa.string()),
        ("all_names", pa.string()),
        ("counts", pa.string()),
        ("v2_counts", pa.string()),
        ("dates", pa.string()),
        ("gcam", pa.string()),
        ("sharing_image", pa.string()),
        ("related_images", pa.string()),
        ("social_image_embeds", pa.string()),
        ("social_video_embeds", pa.string()),
        ("quotations", pa.string()),
        ("amounts", pa.string()),
        ("translation_info", pa.string()),
        ("gkg_extras", pa.string()),
        ("metadata_json", pa.string()),
    ]
    arrays = [
        pa.array([None if row.get(name) is None else str(row.get(name)) for row in rows], type=arrow_type)
        for name, arrow_type in schema
    ]
    table = pa.Table.from_arrays(arrays, names=[name for name, _ in schema])
    partition_date = end.date().isoformat()
    path = output_path(Path(args.output_root), partition_date, datetime.now(UTC))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return {
        "dry_run": False,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "max_results": args.max_results,
        "rows": len(rows),
        "output_path": str(path),
        "total_bytes_processed": job.total_bytes_processed,
        "size_estimate": estimate_parquet_size(len(rows), job.total_bytes_processed),
    }


def resolve_project(project: str | None, service_account_json: str | None) -> str | None:
    if project:
        return project
    if not service_account_json:
        return None
    try:
        payload = json.loads(service_account_json)
    except json.JSONDecodeError:
        return None
    value = payload.get("project_id")
    return value if isinstance(value, str) and value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("SOURCE_BIGQUERY_IPC__PROJECT_ID"),
    )
    parser.add_argument("--location", default="US")
    parser.add_argument(
        "--service-account-json",
        default=os.environ.get("SOURCE_BIGQUERY_IPC__SERVICE_ACCOUNT_JSON"),
        help="Service account JSON string. Defaults to SOURCE_BIGQUERY_IPC__SERVICE_ACCOUNT_JSON.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--lookback-hours", type=int, default=DEFAULT_LOOKBACK_HOURS)
    parser.add_argument("--start", help="UTC ISO timestamp, e.g. 2026-06-25T00:00:00Z")
    parser.add_argument("--end", help="UTC ISO timestamp. Defaults to now when --start is set.")
    parser.add_argument("--theme-pattern", default=DEFAULT_THEME_PATTERN)
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-query", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.project = resolve_project(args.project, args.service_account_json)
    if args.max_results is not None and args.max_results <= 0:
        raise SystemExit("--max-results must be positive")
    if args.start and args.end and parse_datetime(args.start) >= parse_datetime(args.end):
        raise SystemExit("--start must be before --end")

    if args.print_query:
        if args.start:
            start = parse_datetime(args.start)
            end = parse_datetime(args.end) if args.end else datetime.now(UTC).replace(microsecond=0)
        else:
            start, end = default_window(args.lookback_hours)
        print(build_query(start, end, args.theme_pattern, args.max_results))
        return 0

    if not args.project:
        raise SystemExit("--project or GOOGLE_CLOUD_PROJECT is required")

    payload = fetch_to_parquet(args)
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
