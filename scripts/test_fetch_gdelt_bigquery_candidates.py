#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fetch_gdelt_bigquery_candidates import build_query, estimate_parquet_size, output_path


class FetchGdeltBigQueryCandidatesTests(unittest.TestCase):
    def test_query_contains_original_and_richer_columns(self) -> None:
        start = datetime(2026, 6, 24, 0, 0, tzinfo=UTC)
        end = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)
        sql = build_query(start, end, "OIL|INFLATION", 50_000)

        self.assertIn("FROM `gdelt-bq.gdeltv2.gkg_partitioned`", sql)
        self.assertIn("SourceCommonName AS source_common_name", sql)
        self.assertIn("DocumentIdentifier AS document_identifier", sql)
        self.assertIn("CAST(NULL AS STRING) AS title", sql)
        self.assertIn("CAST(NULL AS STRING) AS summary", sql)
        self.assertIn("CAST(NULL AS STRING) AS text", sql)
        self.assertIn("Extras AS gkg_extras", sql)
        self.assertIn("Quotations AS quotations", sql)
        self.assertIn("TO_JSON_STRING(STRUCT", sql)
        self.assertIn("LIMIT 50000", sql)

    def test_output_path_is_hive_partitioned(self) -> None:
        path = output_path(
            Path("data/gdelt_candidates"),
            "2026-06-25",
            datetime(2026, 6, 25, 12, 30, 0, tzinfo=UTC),
        )
        self.assertEqual(
            str(path),
            "data/gdelt_candidates/dt=2026-06-25/part-20260625T123000Z-bigquery.parquet",
        )

    def test_size_estimate_scales_with_rows(self) -> None:
        estimate = estimate_parquet_size(50_000, 123)
        self.assertEqual(estimate["rows"], 50_000)
        self.assertEqual(estimate["bigquery_bytes_processed"], 123)
        self.assertGreater(estimate["estimated_parquet_mb_high"], estimate["estimated_parquet_mb_low"])


if __name__ == "__main__":
    unittest.main()
