#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _duckdb_bootstrap import ensure_duckdb

ensure_duckdb(__file__)

import duckdb

from narrative_explainer_mcp import call_tool
from parquet_narrative_store import build_parquet_graph, parquet_graph_day_db, read_manifest
from query_narrative_graph import query_summary


class ParquetNarrativeStoreTests(unittest.TestCase):
    def _write_day_file(self, path: Path, rows: list[tuple[str, ...]]) -> None:
        values_sql = ",\n                            ".join(
            "(" + ", ".join(repr(value) if value is not None else "CAST(NULL AS VARCHAR)" for value in row) + ")"
            for row in rows
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        con.execute(
            f"""
            COPY (
                SELECT * FROM (
                    VALUES
                        {values_sql}
                ) AS t(
                    record_datetime,
                    partition_date,
                    source_common_name,
                    document_identifier,
                    title,
                    summary,
                    text,
                    v2_themes,
                    v2_tone,
                    v2_locations,
                    v2_persons,
                    v2_organizations,
                    all_names
                )
            ) TO ?
            (FORMAT PARQUET)
            """,
            [str(path)],
        )
        con.close()

    def test_build_parquet_graph_and_query_day_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            raw_root = tmp / "raw"
            output_root = tmp / "graph_parquet"
            self._write_day_file(
                raw_root / "dt=2025-01-02" / "part-000.parquet",
                [
                    (
                        "20250102124500",
                        "2025-01-02",
                        "ft.com",
                        "https://example.com/red-sea-oil-day1",
                        "Red Sea disruption lifts oil risk premium",
                        "Shipping interruptions pushed oil-linked narratives higher.",
                        "Tanker disruptions near the Red Sea raised concern about crude flows.",
                        "SHIPPING,30;OIL,30;SANCTIONS,10",
                        "-4.0,0,0,0,0,0",
                        "1#Yemen#YM#YM#15.5#47.5#0",
                        "",
                        "OPEC,20",
                        "Red Sea,10;OPEC,20",
                    ),
                ],
            )
            self._write_day_file(
                raw_root / "dt=2025-01-03" / "part-000.parquet",
                [
                    (
                        "20250103101500",
                        "2025-01-03",
                        "ft.com",
                        "https://example.com/red-sea-oil-day2",
                        "Oil risk premium builds for a second day",
                        "Fresh shipping disruption kept oil risk elevated.",
                        "Crude markets stayed focused on tanker security and supply routes.",
                        "SHIPPING,25;OIL,35;SANCTIONS,10",
                        "-3.0,0,0,0,0,0",
                        "1#Yemen#YM#YM#15.5#47.5#0",
                        "",
                        "OPEC,20",
                        "Red Sea,10;OPEC,20",
                    ),
                ],
            )

            built = build_parquet_graph([str(raw_root / "dt=*" / "part-*.parquet")], output_root, overwrite=True)
            self.assertEqual(built, ["2025-01-02", "2025-01-03"])
            manifest = read_manifest(output_root)
            self.assertEqual(manifest["materialized_dates"], ["2025-01-02", "2025-01-03"])
            self.assertTrue(
                (output_root / "gold_asset_factor_crossover_links_daily" / "bucket_time=2025-01-03" / "part-000.parquet").exists()
            )

            with parquet_graph_day_db(output_root, "2025-01-03", "2025-01-03") as db_path:
                summary = query_summary(db_path)
            self.assertEqual(summary["table_counts"]["bronze_candidates"], 1)
            self.assertGreater(summary["table_counts"]["gold_asset_factor_crossover_links_daily"], 0)
            self.assertEqual(summary["build_partitions"]["count"], 2)

            explain_day = call_tool(
                "explain_day",
                {
                    "db": str(output_root),
                    "date": "2025-01-03",
                    "universe": ["WTI"],
                    "limit": 5,
                },
            )
            self.assertIn("Top narratives for 2025-01-03", explain_day["content"][0]["text"])

            similar_days = call_tool(
                "similar_days",
                {
                    "db": str(output_root),
                    "date": "2025-01-03",
                    "limit": 5,
                },
            )
            self.assertIn("Similar days for 2025-01-03", similar_days["content"][0]["text"])
            self.assertIn("2025-01-02", similar_days["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
