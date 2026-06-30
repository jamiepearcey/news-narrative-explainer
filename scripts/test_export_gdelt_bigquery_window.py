#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_gdelt_bigquery_window import (
    build_day_count_query,
    build_day_query,
    export_window,
    iter_window_days,
)


class ExportGdeltBigqueryWindowTests(unittest.TestCase):
    def test_iter_window_days_yields_each_day(self) -> None:
        from fetch_gdelt_bigquery_candidates import parse_datetime

        days = iter_window_days(
            parse_datetime("2026-06-15T00:00:00Z"),
            parse_datetime("2026-06-18T00:00:00Z"),
        )
        self.assertEqual([day.partition for day in days], ["2026-06-15", "2026-06-16", "2026-06-17"])

    def test_build_day_query_uses_single_day_partition_window(self) -> None:
        from fetch_gdelt_bigquery_candidates import parse_datetime

        day = iter_window_days(
            parse_datetime("2026-06-15T00:00:00Z"),
            parse_datetime("2026-06-16T00:00:00Z"),
        )[0]
        query = build_day_query(day, "OIL|GOLD", 1234)
        self.assertIn("2026-06-15 00:00:00 UTC", query)
        self.assertIn("2026-06-16 00:00:00 UTC", query)
        self.assertIn("LIMIT 1234", query)
        self.assertIn("metadata_json", query)

    def test_build_day_count_query_has_no_limit(self) -> None:
        from fetch_gdelt_bigquery_candidates import parse_datetime

        day = iter_window_days(
            parse_datetime("2026-06-15T00:00:00Z"),
            parse_datetime("2026-06-16T00:00:00Z"),
        )[0]
        query = build_day_count_query(day, "OIL")
        self.assertIn("COUNT(*) AS total_rows", query)
        self.assertNotIn("LIMIT", query)

    def test_export_window_writes_partitioned_parquet(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        class FakeCountResult:
            def __iter__(self):
                return self

            def __next__(self):
                if getattr(self, "_done", False):
                    raise StopIteration
                self._done = True
                return {"total_rows": 3}

        class FakeQueryResult:
            def __init__(self, table):
                self._table = table

            def to_arrow_iterable(self):
                return self._table.to_batches()

        class FakeQueryJob:
            def __init__(self, table=None, bytes_processed=111):
                self._table = table
                self.total_bytes_processed = bytes_processed

            def result(self, page_size=None):
                return FakeQueryResult(self._table)

        class FakeCountJob:
            def __init__(self, bytes_processed=111):
                self.total_bytes_processed = bytes_processed

            def result(self, page_size=None):
                return FakeCountResult()

        class FakeClient:
            def __init__(self, jobs):
                self._jobs = list(jobs)

            def query(self, query, job_config=None):
                return self._jobs.pop(0)

        table = pa.table(
            {
                "record_datetime": ["20260615000000"],
                "partition_date": ["2026-06-15"],
                "source_common_name": ["reuters.com"],
                "document_identifier": ["https://example.com/story"],
                "title": [None],
                "summary": [None],
                "text": [None],
                "v2_themes": ["OIL"],
                "v2_tone": ["-1.2,0,0,0,0,0"],
                "v2_locations": ["US"],
                "v2_persons": [None],
                "v2_organizations": [None],
                "all_names": [None],
                "metadata_json": [json.dumps({"source_table": "gdelt"})],
            }
        )

        args = SimpleNamespace(
            project="demo-project",
            location="US",
            service_account_json=None,
            start="2026-06-15T00:00:00Z",
            end="2026-06-16T00:00:00Z",
            theme_pattern="OIL",
            rows_per_day=2000,
            page_size=1000,
            output_root=None,
            dry_run=False,
            include_queries=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            args.output_root = tmpdir
            fake_client = FakeClient(
                [FakeQueryJob(table=table, bytes_processed=100), FakeCountJob(bytes_processed=25)]
            )
            with patch("export_gdelt_bigquery_window.ensure_dependencies"), patch(
                "google.cloud.bigquery.Client",
                return_value=fake_client,
            ):
                payload = export_window(args)

            self.assertEqual(payload["total_rows_exported"], 1)
            out = Path(payload["day_results"][0]["output_path"])
            self.assertTrue(out.exists())
            written = pq.read_table(out)
            self.assertEqual(written.num_rows, 1)
            self.assertEqual(written.column("partition_date").to_pylist(), ["2026-06-15"])


if __name__ == "__main__":
    unittest.main()
