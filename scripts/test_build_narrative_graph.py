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
    classify_source_type,
    load_taxonomy,
    normalized_match_text,
)


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "config" / "news_narrative_taxonomy.json"


class NarrativeGraphTests(unittest.TestCase):
    def _write_candidate_parquet(self, parquet_path: Path, rows: list[tuple[str, ...]]) -> None:
        values_sql = ",\n                            ".join(
            "(" + ", ".join(repr(value) if value is not None else "CAST(NULL AS VARCHAR)" for value in row) + ")"
            for row in rows
        )
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
                    metadata_json,
                    gkg_extras,
                    quotations,
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
                                '{"source_table":"gdelt-bq.gdeltv2.gkg_partitioned"}',
                                'fed extras payload',
                                'Powell said rates remain restrictive.',
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
                                '{"source_table":"gdelt-bq.gdeltv2.gkg_partitioned"}',
                                'red sea extras payload',
                                'Traders quoted a higher risk premium.',
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
                        metadata_json,
                        gkg_extras,
                        quotations,
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
                SELECT title, summary_text, body_text, relevant_text, metadata_json, gkg_extras, quotations, source_type, market_context_text
                FROM bronze_candidates
                WHERE document_identifier = 'https://example.com/red-sea-oil'
                """
            ).fetchone()
            self.assertEqual(rich_text[0], "Red Sea disruption lifts oil risk premium")
            self.assertIn("Shipping interruptions", rich_text[1])
            self.assertIn("Tanker disruptions", rich_text[2])
            self.assertIn("OPEC", rich_text[3])
            self.assertIn("gdelt-bq", rich_text[4])
            self.assertEqual(rich_text[5], "red sea extras payload")
            self.assertIn("risk premium", rich_text[6])
            self.assertEqual(rich_text[7], "market_wrap")
            self.assertIn("Shipping interruptions", rich_text[8])
            out.close()

    def test_build_narrative_graph_falls_back_to_gkg_page_title(self) -> None:
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
                                '20250102153000',
                                '2025-01-02',
                                'example.com',
                                'https://example.com/shipping-story',
                                CAST(NULL AS VARCHAR),
                                CAST(NULL AS VARCHAR),
                                CAST(NULL AS VARCHAR),
                                '{"source_table":"gdelt-bq.gdeltv2.gkg_partitioned"}',
                                '<PAGE_TITLE>Shipping shock &amp; oil risks rise</PAGE_TITLE>',
                                CAST(NULL AS VARCHAR),
                                'SHIPPING,30;OIL,30',
                                '-1.0,0,0,0,0,0',
                                '1#Egypt#EG#EG#26.0#30.0#0',
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
                        metadata_json,
                        gkg_extras,
                        quotations,
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
            title, relevant_text = out.execute(
                """
                SELECT title, relevant_text
                FROM bronze_candidates
                WHERE document_identifier = 'https://example.com/shipping-story'
                """
            ).fetchone()
            self.assertEqual(title, "Shipping shock & oil risks rise")
            self.assertIn("Shipping shock & oil risks rise", relevant_text)
            out.close()

    def test_build_narrative_graph_requires_asset_context_for_asset_mentions(self) -> None:
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
                                '20250102124500',
                                '2025-01-02',
                                'ft.com',
                                'https://example.com/red-sea-oil',
                                'Red Sea disruption lifts oil risk premium',
                                'Shipping interruptions and sanctions concerns pushed oil-linked narratives higher.',
                                'Tanker disruptions near the Red Sea raised concern about supply routes and near-term crude flows.',
                                '{"source_table":"gdelt-bq.gdeltv2.gkg_partitioned"}',
                                'red sea extras payload',
                                'Traders quoted a higher risk premium.',
                                'SHIPPING,30;OIL,30;SANCTIONS,10',
                                '-4.0,0,0,0,0,0',
                                '1#Yemen#YM#YM#15.5#47.5#0;1#Egypt#EG#EG#26.0#30.0#0',
                                '',
                                'OPEC,20',
                                'Red Sea,10;OPEC,20'
                            ),
                            (
                                '20250102125500',
                                '2025-01-02',
                                'example.net',
                                'https://example.com/generic-war',
                                'Regional leaders discuss security tensions',
                                'Officials met to discuss regional tensions and trade coordination.',
                                'Diplomats held talks on policy and regional stability.',
                                '{"source_table":"gdelt-bq.gdeltv2.gkg_partitioned"}',
                                'generic extras payload',
                                'Officials declined to comment on market impact.',
                                'WAR,30',
                                '-1.0,0,0,0,0,0',
                                '1#Yemen#YM#YM#15.5#47.5#0',
                                '',
                                '',
                                'Regional leaders,10'
                            )
                    ) AS t(
                        record_datetime,
                        partition_date,
                        source_common_name,
                        document_identifier,
                        title,
                        summary,
                        text,
                        metadata_json,
                        gkg_extras,
                        quotations,
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
            wti_docs = out.execute(
                """
                SELECT DISTINCT document_identifier
                FROM silver_asset_factor_mentions m
                JOIN bronze_candidates b USING (doc_id)
                WHERE asset_label = 'WTI'
                ORDER BY document_identifier
                """
            ).fetchall()
            self.assertEqual(
                [row[0] for row in wti_docs],
                ['https://example.com/red-sea-oil'],
            )
            out.close()

    def test_build_narrative_graph_requires_index_context_for_ndx(self) -> None:
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
                                '20250102124500',
                                '2025-01-02',
                                'finance.yahoo.com',
                                'https://example.com/nasdaq-100-yields',
                                'Nasdaq 100 slides as higher Treasury yields pressure tech stocks',
                                'The Nasdaq 100 fell as softer growth activity and higher Treasury yields hit long-duration technology shares.',
                                'Wall Street marked down the Nasdaq 100 and other tech stocks as activity fears grew and yields rose.',
                                '{"source_table":"gdelt-bq.gdeltv2.gkg_partitioned"}',
                                'market wrap extras',
                                'Traders cited higher yields.',
                                'ECON_GROWTH,30;ACTIVITY,20;EARNINGS,10',
                                '-2.0,0,0,0,0,0',
                                '1#United States#US#US#38.0#-77.0#0',
                                '',
                                'Federal Reserve,20',
                                'Nasdaq 100,20;Wall Street,10'
                            ),
                            (
                                '20250102130000',
                                '2025-01-02',
                                'tickerreport.com',
                                'https://example.com/root-comparison',
                                'Financial Comparison: Root (NASDAQ:ROOT) and Stewart Information Services (NYSE:STC)',
                                'A single-name comparison of two insurance stocks.',
                                'Mastercard style company comparison article with quarterly metrics and margin discussion.',
                                '{"source_table":"gdelt-bq.gdeltv2.gkg_partitioned"}',
                                'company extras',
                                'Investors compared EBIT margins.',
                                'EARNINGS,30;ECON_STOCKMARKET,20',
                                '-0.5,0,0,0,0,0',
                                '1#United States#US#US#38.0#-77.0#0',
                                '',
                                '',
                                'Root,20;Stewart Information Services,10'
                            )
                    ) AS t(
                        record_datetime,
                        partition_date,
                        source_common_name,
                        document_identifier,
                        title,
                        summary,
                        text,
                        metadata_json,
                        gkg_extras,
                        quotations,
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
            ndx_urls = {
                row[0]
                for row in out.execute(
                    """
                    SELECT DISTINCT document_identifier
                    FROM bronze_candidates b
                    JOIN silver_asset_factor_mentions m USING (doc_id)
                    WHERE m.asset_label = 'NDX'
                    """
                ).fetchall()
            }
            self.assertIn("https://example.com/nasdaq-100-yields", ndx_urls)
            self.assertNotIn("https://example.com/root-comparison", ndx_urls)
            source_type = out.execute(
                """
                SELECT source_type
                FROM bronze_candidates
                WHERE document_identifier = 'https://example.com/root-comparison'
                """
            ).fetchone()[0]
            self.assertEqual(source_type, "company_specific")
            out.close()

    def test_classify_source_type_prefers_company_specific_over_ticker_titles(self) -> None:
        source_type, priority = classify_source_type(
            "tickerreport.com",
            "Financial Comparison: Root (NASDAQ:ROOT) and Stewart Information Services (NYSE:STC)",
            "https://example.com/root-comparison",
        )
        self.assertEqual(source_type, "company_specific")
        self.assertEqual(priority, 2)

    def test_build_narrative_graph_appends_new_dates_and_materializes_crossovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            first_parquet = tmp / "day1.parquet"
            second_parquet = tmp / "day2.parquet"
            db_path = tmp / "narrative_graph.duckdb"
            self._write_candidate_parquet(
                first_parquet,
                [
                    (
                        "20250102124500",
                        "2025-01-02",
                        "ft.com",
                        "https://example.com/red-sea-oil-day1",
                        "Red Sea disruption lifts oil risk premium",
                        "Shipping interruptions pushed oil-linked narratives higher.",
                        "Tanker disruptions near the Red Sea raised concern about crude flows.",
                        '{"source_table":"gdelt-bq.gdeltv2.gkg_partitioned"}',
                        "red sea extras payload",
                        "Traders quoted a higher risk premium.",
                        "SHIPPING,30;OIL,30;SANCTIONS,10",
                        "-4.0,0,0,0,0,0",
                        "1#Yemen#YM#YM#15.5#47.5#0",
                        "",
                        "OPEC,20",
                        "Red Sea,10;OPEC,20",
                    ),
                ],
            )
            self._write_candidate_parquet(
                second_parquet,
                [
                    (
                        "20250103101500",
                        "2025-01-03",
                        "ft.com",
                        "https://example.com/red-sea-oil-day2",
                        "Oil risk premium builds for a second day",
                        "Fresh shipping disruption kept oil risk elevated.",
                        "Crude markets stayed focused on tanker security and supply routes.",
                        '{"source_table":"gdelt-bq.gdeltv2.gkg_partitioned"}',
                        "day two extras payload",
                        "Desk participants kept a higher risk premium.",
                        "SHIPPING,25;OIL,35;SANCTIONS,10",
                        "-3.0,0,0,0,0,0",
                        "1#Yemen#YM#YM#15.5#47.5#0",
                        "",
                        "OPEC,20",
                        "Red Sea,10;OPEC,20",
                    ),
                ],
            )

            build_narrative_graph(
                input_glob=str(first_parquet),
                output_db=db_path,
                taxonomy_path=TAXONOMY,
                overwrite=True,
            )
            build_narrative_graph(
                input_glob=str(second_parquet),
                output_db=db_path,
                taxonomy_path=TAXONOMY,
                overwrite=False,
            )

            out = duckdb.connect(str(db_path))
            self.assertEqual(out.execute("SELECT COUNT(*) FROM bronze_candidates").fetchone()[0], 2)
            self.assertEqual(
                out.execute("SELECT COUNT(DISTINCT partition_date) FROM graph_build_partitions").fetchone()[0],
                2,
            )
            crossover = out.execute(
                """
                SELECT prior_bucket_time, bucket_time, asset_label, factor_label
                FROM gold_asset_factor_crossover_links_daily
                WHERE asset_label = 'WTI' AND factor_label = 'oil'
                """
            ).fetchone()
            self.assertEqual(str(crossover[0]), "2025-01-02")
            self.assertEqual(str(crossover[1]), "2025-01-03")
            self.assertEqual(crossover[2], "WTI")
            self.assertEqual(crossover[3], "oil")
            factor_crossover = out.execute(
                """
                SELECT COUNT(*)
                FROM gold_factor_crossover_links_daily
                WHERE factor_label = 'oil'
                  AND prior_bucket_time = DATE '2025-01-02'
                  AND bucket_time = DATE '2025-01-03'
                """
            ).fetchone()[0]
            self.assertGreaterEqual(factor_crossover, 1)
            out.close()


if __name__ == "__main__":
    unittest.main()
