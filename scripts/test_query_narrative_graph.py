#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _duckdb_bootstrap import ensure_duckdb

ensure_duckdb(__file__)

import tempfile
import unittest

import duckdb

from build_narrative_graph import build_narrative_graph
from query_narrative_graph import (
    query_asset_narratives,
    query_explain_move,
    query_factor_daily,
    query_summary,
    query_supporting_docs,
    query_top_factors,
)


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "config" / "news_narrative_taxonomy.json"


class QueryNarrativeGraphTests(unittest.TestCase):
    def _build_fixture(self, db_path: Path, parquet_path: Path) -> None:
        con = duckdb.connect()
        con.execute(
            """
            COPY (
                SELECT * FROM (
                    VALUES
                        (
                            '20250102123000',
                            '2025-01-02',
                            'reuters.com',
                            'https://example.com/fed-inflation',
                            'ECON_INFLATION,50;CENTRAL_BANK,50;INTEREST_RATE,30',
                            '-2.5,0,0,0,0,0',
                            '1#United States#US#US#38.0#-77.0#0',
                            'Jerome Powell,20',
                            'Federal Reserve,20',
                            'Jerome Powell,20;United States,10'
                        ),
                        (
                            '20250102124500',
                            '2025-01-02',
                            'ft.com',
                            'https://example.com/red-sea-oil',
                            'SHIPPING,30;OIL,30;SANCTIONS,10',
                            '-4.0,0,0,0,0,0',
                            '1#Yemen#YM#YM#15.5#47.5#0;1#Egypt#EG#EG#26.0#30.0#0',
                            '',
                            'OPEC,20',
                            'Red Sea,10;OPEC,20'
                        )
                ) AS t(
                    record_datetime,
                    partition_date,
                    source_common_name,
                    document_identifier,
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
            [str(parquet_path)],
        )
        con.close()

        build_narrative_graph(
            input_glob=str(parquet_path),
            output_db=db_path,
            taxonomy_path=TAXONOMY,
            overwrite=True,
        )

    def test_summary_and_factor_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            parquet_path = tmp / "candidates.parquet"
            db_path = tmp / "narrative_graph.duckdb"
            self._build_fixture(db_path, parquet_path)

            summary = query_summary(db_path)
            self.assertEqual(summary["table_counts"]["bronze_candidates"], 2)
            self.assertEqual(summary["bucket_span"]["bucket_dates"], 1)

            top_factors = query_top_factors(db_path, limit=6)
            factor_labels = [row["factor_label"] for row in top_factors]
            self.assertIn("inflation", factor_labels)
            self.assertIn("shipping_disruption", factor_labels)

            factor_daily = query_factor_daily(db_path, factor_label="inflation", limit=5)
            self.assertEqual(str(factor_daily[0]["bucket_time"]), "2025-01-02")

    def test_asset_narratives_and_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            parquet_path = tmp / "candidates.parquet"
            db_path = tmp / "narrative_graph.duckdb"
            self._build_fixture(db_path, parquet_path)

            narratives = query_asset_narratives(
                db_path,
                asset_label="WTI",
                start_date="2025-01-02",
                end_date="2025-01-02",
                limit=10,
            )
            self.assertGreaterEqual(len(narratives), 1)
            self.assertEqual(narratives[0]["asset_label"], "WTI")

            docs = query_supporting_docs(
                db_path,
                asset_label="WTI",
                factor_label="oil",
                start_date="2025-01-02",
                end_date="2025-01-02",
                limit=10,
            )
            self.assertEqual(docs[0]["asset_label"], "WTI")
            self.assertIn("example.com/red-sea-oil", docs[0]["document_identifier"])

            explain = query_explain_move(
                db_path,
                asset_label="WTI",
                start_date="2025-01-02",
                end_date="2025-01-02",
                limit=10,
            )
            self.assertEqual(explain["asset_label"], "WTI")
            self.assertGreaterEqual(len(explain["top_narratives"]), 1)
            self.assertGreaterEqual(len(explain["supporting_docs"]), 1)


if __name__ == "__main__":
    unittest.main()
