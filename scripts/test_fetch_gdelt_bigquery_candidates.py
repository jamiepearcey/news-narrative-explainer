#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fetch_gdelt_bigquery_candidates import (
    build_count_query,
    build_query,
    clean_text,
    enrich_rows,
    estimate_parquet_size,
    extract_article_fields,
    output_path,
    project_output_size,
    resolve_project,
)


class FetchGdeltBigQueryCandidatesTests(unittest.TestCase):
    def test_query_contains_original_and_richer_columns(self) -> None:
        start = datetime(2026, 6, 24, 0, 0, tzinfo=UTC)
        end = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)
        sql = build_query(start, end, "OIL|INFLATION", 50_000)
        count_sql = build_count_query(start, end, "OIL|INFLATION")

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
        self.assertNotIn("ORDER BY DATE DESC", sql)
        self.assertIn("COUNT(*) AS total_rows", count_sql)

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

    def test_project_can_come_from_service_account_json(self) -> None:
        self.assertEqual(resolve_project(None, '{"project_id":"demo-project"}'), "demo-project")
        self.assertEqual(resolve_project("explicit-project", '{"project_id":"demo-project"}'), "explicit-project")

    def test_extract_article_fields_prefers_title_description_and_article_text(self) -> None:
        html = """
        <html>
          <head>
            <title>Fallback title</title>
            <meta property="og:title" content="Actual article title" />
            <meta name="description" content="Short summary for the article." />
          </head>
          <body>
            <article>
              <p>This is the first paragraph with enough substance to be useful.</p>
              <p>This is the second paragraph adding more context for the explainer.</p>
            </article>
          </body>
        </html>
        """
        fields = extract_article_fields(html)
        self.assertEqual(fields["title"], "Actual article title")
        self.assertEqual(fields["summary"], "Short summary for the article.")
        self.assertIn("first paragraph", fields["text"])

    def test_enrich_rows_populates_text_fields(self) -> None:
        rows = [
            {"document_identifier": "https://example.com/story", "title": None, "summary": None, "text": None}
        ]
        with mock.patch(
            "fetch_gdelt_bigquery_candidates.fetch_html",
            return_value="""
            <html><head><title>Example story</title></head>
            <body><article><p>Useful article body text for enrichment.</p></article></body></html>
            """,
        ):
            stats = enrich_rows(rows, enrich_max_docs=5, timeout=5.0, user_agent="ua")
        self.assertEqual(stats["rows_enriched"], 1)
        self.assertEqual(rows[0]["title"], "Example story")
        self.assertIn("Useful article body text", rows[0]["text"])

    def test_clean_text_normalizes_whitespace(self) -> None:
        self.assertEqual(clean_text("  one \n two\tthree "), "one two three")

    def test_project_output_size_scales_to_total_rows(self) -> None:
        with mock.patch.object(Path, "stat", return_value=mock.Mock(st_size=3000)):
            projection = project_output_size(
                Path("/tmp/example.parquet"),
                observed_rows=3,
                total_rows=30,
                enriched_rows=2,
            )
        self.assertEqual(projection["output_bytes"], 3000)
        self.assertEqual(projection["observed_bytes_per_row"], 1000.0)
        self.assertEqual(projection["projected_full_window_bytes"], 30000)
        self.assertEqual(projection["projected_full_window_rows"], 30)
        self.assertEqual(projection["enriched_row_ratio"], 0.6667)


if __name__ == "__main__":
    unittest.main()
