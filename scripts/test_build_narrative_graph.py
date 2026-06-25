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

from build_narrative_graph import (
    build_narrative_graph,
    classify_factors,
    load_taxonomy,
    normalized_match_text,
)


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "config" / "news_narrative_taxonomy.json"


class NarrativeGraphTests(unittest.TestCase):
    def test_taxonomy_matches_core_rules(self) -> None:
        rules = load_taxonomy(TAXONOMY)
        match_text = normalized_match_text(
            ["ECON_INFLATION", "CENTRAL_BANK", "SHIPPING", "OIL"],
            [],
            ["Federal Reserve"],
            ["Jerome Powell"],
            ["US"],
        )
        labels = [rule.label for rule in classify_factors(match_text, rules)]
        self.assertIn("inflation", labels)
        self.assertIn("central_bank_policy", labels)
        self.assertIn("shipping_disruption", labels)
        self.assertIn("oil", labels)

    def test_build_narrative_graph_creates_lookup_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            parquet_path = tmp / "candidates.parquet"
            db_path = tmp / "narrative_graph.duckdb"
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
                                'Fed officials keep inflation focus',
                                'Policy signals stayed hawkish after the latest inflation data.',
                                'Federal Reserve officials said inflation remains too high and rate cuts are not imminent.',
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
                                'Red Sea disruption lifts oil risk premium',
                                'Shipping interruptions and sanctions concerns pushed oil-linked narratives higher.',
                                'Tanker disruptions near the Red Sea raised concern about supply routes and near-term crude flows.',
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
                [str(parquet_path)],
            )
            con.close()

            build_narrative_graph(
                input_glob=str(parquet_path),
                output_db=db_path,
                taxonomy_path=TAXONOMY,
                overwrite=True,
            )

            out = duckdb.connect(str(db_path))
            self.assertEqual(out.execute("SELECT COUNT(*) FROM bronze_candidates").fetchone()[0], 2)
            self.assertEqual(out.execute("SELECT COUNT(*) FROM silver_event_graph").fetchone()[0], 2)
            factor_labels = {
                row[0]
                for row in out.execute(
                    "SELECT DISTINCT factor_label FROM gold_factor_buckets_daily"
                ).fetchall()
            }
            self.assertIn("inflation", factor_labels)
            self.assertIn("shipping_disruption", factor_labels)
            asset_labels = {
                row[0]
                for row in out.execute(
                    "SELECT DISTINCT asset_label FROM gold_asset_factor_panel_daily"
                ).fetchall()
            }
            self.assertIn("US2Y", asset_labels)
            self.assertIn("WTI", asset_labels)
            rich_text = out.execute(
                """
                SELECT title, summary_text, body_text, relevant_text
                FROM bronze_candidates
                WHERE document_identifier = 'https://example.com/red-sea-oil'
                """
            ).fetchone()
            self.assertEqual(rich_text[0], "Red Sea disruption lifts oil risk premium")
            self.assertIn("Shipping interruptions", rich_text[1])
            self.assertIn("Tanker disruptions", rich_text[2])
            self.assertIn("OPEC", rich_text[3])
            out.close()


if __name__ == "__main__":
    unittest.main()
