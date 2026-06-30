#!/usr/bin/env python3
"""Daily parquet materialization and day-scoped DuckDB adapter."""

from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _duckdb_bootstrap import ensure_duckdb

ensure_duckdb(__file__)

import duckdb

from build_narrative_graph import build_narrative_graph, initialize_schema


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET_ROOT = ROOT / "data" / "narrative_graph_parquet"
MANIFEST_PATH = DEFAULT_PARQUET_ROOT / "manifest.json"


@dataclass(frozen=True)
class TableLayout:
    name: str
    partition_column: str
    scope: str


TABLE_LAYOUTS = (
    TableLayout("bronze_candidates", "partition_date", "window"),
    TableLayout("silver_event_graph", "event_date", "window"),
    TableLayout("silver_factor_mentions", "bucket_time", "window"),
    TableLayout("silver_asset_factor_mentions", "bucket_time", "window"),
    TableLayout("silver_market_context_mentions", "bucket_time", "window"),
    TableLayout("gold_factor_buckets_daily", "bucket_time", "all_gold"),
    TableLayout("gold_asset_factor_panel_daily", "bucket_time", "all_gold"),
    TableLayout("gold_factor_crossover_links_daily", "bucket_time", "all_gold"),
    TableLayout("gold_asset_factor_crossover_links_daily", "bucket_time", "all_gold"),
)

TABLE_LAYOUT_BY_NAME = {layout.name: layout for layout in TABLE_LAYOUTS}
DATE_FORMAT = "%Y-%m-%d"


def _date_text(value: date | datetime | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.date().isoformat()
    return value.isoformat()


def _parse_date(value: str) -> date:
    return datetime.strptime(value, DATE_FORMAT).date()


def _iter_dates(start_date: str, end_date: str) -> list[str]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    cursor = start
    days: list[str] = []
    while cursor <= end:
        days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return days


def _table_dir(root: Path, table_name: str) -> Path:
    return root / table_name


def _partition_dir(root: Path, table_name: str, partition_column: str, partition_value: str) -> Path:
    return _table_dir(root, table_name) / f"{partition_column}={partition_value}"


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def read_manifest(root: Path) -> dict[str, object]:
    path = _manifest_path(root)
    if not path.exists():
        return {"materialized_dates": []}
    return json.loads(path.read_text())


def write_manifest(root: Path, payload: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _manifest_path(root).write_text(json.dumps(payload, indent=2, sort_keys=True))


def materialized_dates(root: Path) -> set[str]:
    payload = read_manifest(root)
    return {str(value) for value in payload.get("materialized_dates", [])}


def _discover_date_from_path(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("dt="):
            try:
                return _parse_date(part.split("=", 1)[1]).isoformat()
            except ValueError:
                continue
    match = re.search(r"gdelt_candidates_etl_day_(\d{4})_(\d{2})_(\d{2})", str(path))
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None


def group_input_files_by_date(input_globs: list[str]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for pattern in input_globs:
        matches = sorted(Path(match) for match in glob.glob(pattern))
        for match in matches:
            if match.is_dir():
                continue
            day = _discover_date_from_path(match)
            if day is None:
                continue
            grouped.setdefault(day, []).append(match)
    return grouped


def _copy_table_to_parquet(con: duckdb.DuckDBPyConnection, sql: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    con.execute(f"COPY ({sql}) TO {json.dumps(str(output_path))} (FORMAT PARQUET)")


def _export_day_tables(day_db: Path, output_root: Path, day: str) -> None:
    con = duckdb.connect(str(day_db), read_only=True)
    try:
        exports = {
            "bronze_candidates": f"SELECT * FROM bronze_candidates WHERE partition_date = DATE '{day}'",
            "silver_event_graph": f"SELECT *, CAST(event_time AS DATE) AS event_date FROM silver_event_graph WHERE CAST(event_time AS DATE) = DATE '{day}'",
            "silver_factor_mentions": f"SELECT * FROM silver_factor_mentions WHERE bucket_time = DATE '{day}'",
            "silver_asset_factor_mentions": f"SELECT * FROM silver_asset_factor_mentions WHERE bucket_time = DATE '{day}'",
            "silver_market_context_mentions": f"SELECT * FROM silver_market_context_mentions WHERE bucket_time = DATE '{day}'",
            "gold_factor_buckets_daily": f"SELECT * FROM gold_factor_buckets_daily WHERE bucket_time = DATE '{day}'",
            "gold_asset_factor_panel_daily": f"SELECT * FROM gold_asset_factor_panel_daily WHERE bucket_time = DATE '{day}'",
        }
        for table_name, sql in exports.items():
            layout = TABLE_LAYOUT_BY_NAME[table_name]
            partition_path = _partition_dir(output_root, table_name, layout.partition_column, day)
            if partition_path.exists():
                shutil.rmtree(partition_path)
            output_path = partition_path / "part-000.parquet"
            _copy_table_to_parquet(con, sql, output_path)
    finally:
        con.close()


def _build_daily_crossover_parquet(day_db: Path, output_root: Path, day: str) -> None:
    previous_day = (_parse_date(day) - timedelta(days=1)).isoformat()
    prev_factor = _partition_dir(output_root, "gold_factor_buckets_daily", "bucket_time", previous_day) / "part-000.parquet"
    prev_asset = _partition_dir(output_root, "gold_asset_factor_panel_daily", "bucket_time", previous_day) / "part-000.parquet"
    curr_factor = _partition_dir(output_root, "gold_factor_buckets_daily", "bucket_time", day) / "part-000.parquet"
    curr_asset = _partition_dir(output_root, "gold_asset_factor_panel_daily", "bucket_time", day) / "part-000.parquet"
    factor_out_dir = _partition_dir(output_root, "gold_factor_crossover_links_daily", "bucket_time", day)
    asset_out_dir = _partition_dir(output_root, "gold_asset_factor_crossover_links_daily", "bucket_time", day)
    for directory in (factor_out_dir, asset_out_dir):
        if directory.exists():
            shutil.rmtree(directory)
    if not (prev_factor.exists() and curr_factor.exists() and prev_asset.exists() and curr_asset.exists()):
        return
    con = duckdb.connect(str(day_db))
    try:
        factor_sql = f"""
            SELECT
                prev.bucket_time AS prior_bucket_time,
                curr.bucket_time,
                curr.factor_id,
                curr.factor_label,
                curr.geo_id,
                curr.geo_label,
                prev.doc_count AS prior_doc_count,
                curr.doc_count,
                prev.narrative_score AS prior_narrative_score,
                curr.narrative_score,
                curr.doc_count - prev.doc_count AS doc_count_delta,
                curr.narrative_score - prev.narrative_score AS narrative_score_delta
            FROM read_parquet({json.dumps(str(curr_factor))}, union_by_name=true) AS curr
            JOIN read_parquet({json.dumps(str(prev_factor))}, union_by_name=true) AS prev
              ON prev.factor_id = curr.factor_id
             AND prev.geo_id = curr.geo_id
        """
        asset_sql = f"""
            SELECT
                prev.bucket_time AS prior_bucket_time,
                curr.bucket_time,
                curr.asset_id,
                curr.asset_label,
                curr.factor_id,
                curr.factor_label,
                curr.geo_id,
                curr.geo_label,
                prev.doc_count AS prior_doc_count,
                curr.doc_count,
                prev.narrative_score AS prior_narrative_score,
                curr.narrative_score,
                curr.doc_count - prev.doc_count AS doc_count_delta,
                curr.narrative_score - prev.narrative_score AS narrative_score_delta
            FROM read_parquet({json.dumps(str(curr_asset))}, union_by_name=true) AS curr
            JOIN read_parquet({json.dumps(str(prev_asset))}, union_by_name=true) AS prev
              ON prev.asset_id = curr.asset_id
             AND prev.factor_id = curr.factor_id
             AND prev.geo_id = curr.geo_id
        """
        _copy_table_to_parquet(con, factor_sql, factor_out_dir / "part-000.parquet")
        _copy_table_to_parquet(con, asset_sql, asset_out_dir / "part-000.parquet")
    finally:
        con.close()


def build_parquet_graph(input_globs: list[str], output_root: Path, overwrite: bool = False) -> list[str]:
    grouped = group_input_files_by_date(input_globs)
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    done = materialized_dates(output_root)
    built: list[str] = []
    for day in sorted(grouped):
        if day in done:
            continue
        files = grouped[day]
        with tempfile.TemporaryDirectory(prefix=f"narrative-day-{day}-") as tmp_dir:
            day_db = Path(tmp_dir) / "day.duckdb"
            if len(files) == 1:
                day_glob = str(files[0])
            else:
                day_glob = str(files[0].parent / "*.parquet")
            build_narrative_graph(
                input_glob=day_glob,
                output_db=day_db,
                taxonomy_path=ROOT / "config" / "news_narrative_taxonomy.json",
                overwrite=True,
            )
            _export_day_tables(day_db, output_root, day)
            _build_daily_crossover_parquet(day_db, output_root, day)
        done.add(day)
        built.append(day)
    manifest = {
        "materialized_dates": sorted(done),
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "input_globs": input_globs,
    }
    write_manifest(output_root, manifest)
    return built


def _load_files_into_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    paths: list[Path],
    drop_columns: tuple[str, ...] = (),
) -> None:
    if not paths:
        return
    sql = f"SELECT * FROM read_parquet({json.dumps([str(path) for path in paths])}, union_by_name=true)"
    if drop_columns:
        drop_set = {column.lower() for column in drop_columns}
        columns = [row[1] for row in con.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
        keep = [column for column in columns if column.lower() not in drop_set]
        sql = f"SELECT {', '.join(keep)} FROM ({sql})"
    con.execute(f"INSERT INTO {table_name} {sql}")


def _paths_for_dates(root: Path, table_name: str, dates: list[str]) -> list[Path]:
    layout = TABLE_LAYOUT_BY_NAME[table_name]
    paths: list[Path] = []
    for day in dates:
        path = _partition_dir(root, table_name, layout.partition_column, day) / "part-000.parquet"
        if path.exists():
            paths.append(path)
    return paths


def _all_partition_paths(root: Path, table_name: str) -> list[Path]:
    return sorted(_table_dir(root, table_name).glob(f"{TABLE_LAYOUT_BY_NAME[table_name].partition_column}=*/part-000.parquet"))


def _load_graph_build_partitions(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    dates = sorted(materialized_dates(root))
    if not dates:
        return
    con.executemany(
        "INSERT INTO graph_build_partitions VALUES (?, ?, current_timestamp::TIMESTAMP)",
        [(day, str(root)) for day in dates],
    )


@contextmanager
def parquet_graph_day_db(root: Path, start_date: str | None = None, end_date: str | None = None) -> Iterator[Path]:
    requested_dates = _iter_dates(start_date, end_date or start_date) if start_date else []
    with tempfile.TemporaryDirectory(prefix=f"narrative-parquet-{start_date}-") as tmp_dir:
        db_path = Path(tmp_dir) / "query.duckdb"
        con = duckdb.connect(str(db_path))
        try:
            initialize_schema(con)
            for layout in TABLE_LAYOUTS:
                if layout.scope == "window":
                    paths = _paths_for_dates(root, layout.name, requested_dates)
                else:
                    paths = _all_partition_paths(root, layout.name)
                drop_columns = ("event_date",) if layout.name == "silver_event_graph" else ()
                _load_files_into_table(con, layout.name, paths, drop_columns=drop_columns)
            _load_graph_build_partitions(con, root)
        finally:
            con.close()
        yield db_path


@contextmanager
def resolve_query_db(store_path: Path, start_date: str | None = None, end_date: str | None = None) -> Iterator[Path]:
    if store_path.suffix == ".duckdb":
        yield store_path
        return
    with parquet_graph_day_db(store_path, start_date, end_date) as db_path:
        yield db_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_PARQUET_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--input-glob",
        action="append",
        required=True,
        help="Repeat for each raw parquet corpus glob to materialize.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    built = build_parquet_graph(args.input_glob, Path(args.output_root), overwrite=args.overwrite)
    print(json.dumps({"output_root": args.output_root, "built_dates": built}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
