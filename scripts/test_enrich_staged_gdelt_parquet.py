#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from enrich_staged_gdelt_parquet import enrich_staged_parquet, ensure_dependencies

ensure_dependencies()


class EnrichStagedGdeltParquetTests(unittest.TestCase):
    def _write_parquet(self, path: Path, rows: list[dict[str, object]]) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(rows), path)

    def test_enrich_staged_parquet_writes_enriched_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "part-000000.parquet"
            self._write_parquet(
                source,
                [
                    {
                        "partition_date": "2026-06-24",
                        "document_identifier": "https://example.com/story",
                        "title": None,
                        "summary": None,
                        "text": None,
                    }
                ],
            )

            with mock.patch(
                "enrich_staged_gdelt_parquet.enrich_rows",
                side_effect=lambda rows, **_: rows[0].update(
                    {"title": "Example story", "summary": "Example summary", "text": "Example body text"}
                )
                or {
                    "requested_docs": 1,
                    "attempted_fetches": 1,
                    "rows_enriched": 1,
                    "unique_urls_seen": 1,
                },
            ):
                payload = enrich_staged_parquet(
                    input_root=root,
                    enrich_max_docs_per_file=10,
                    timeout=5.0,
                    user_agent="ua",
                    overwrite=False,
                    include_glob=None,
                    requested_asset_label=None,
                    requested_factor_label=None,
                )

            target = root / "part-000000-enriched.parquet"
            self.assertTrue(target.exists())
            self.assertEqual(payload["rows_enriched"], 1)

    def test_enrich_staged_parquet_can_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "part-000000.parquet"
            self._write_parquet(
                source,
                [
                    {
                        "partition_date": "2026-06-24",
                        "document_identifier": "https://example.com/story",
                        "title": None,
                        "summary": None,
                        "text": None,
                    }
                ],
            )

            with mock.patch(
                "enrich_staged_gdelt_parquet.enrich_rows",
                side_effect=lambda rows, **_: rows[0].update(
                    {"title": "Example story", "summary": "Example summary", "text": "Example body text"}
                )
                or {
                    "requested_docs": 1,
                    "attempted_fetches": 1,
                    "rows_enriched": 1,
                    "unique_urls_seen": 1,
                },
            ):
                payload = enrich_staged_parquet(
                    input_root=root,
                    enrich_max_docs_per_file=10,
                    timeout=5.0,
                    user_agent="ua",
                    overwrite=True,
                    include_glob=None,
                    requested_asset_label=None,
                    requested_factor_label=None,
                )

            self.assertTrue(source.exists())
            self.assertEqual(payload["rows_enriched"], 1)
            self.assertTrue((root / "enrichment-manifest.json").exists())

    def test_enrich_staged_parquet_prioritizes_requested_asset_and_factor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "part-000000.parquet"
            self._write_parquet(
                source,
                [
                    {
                        "partition_date": "2026-06-24",
                        "document_identifier": "https://example.com/unrelated",
                        "title": None,
                        "summary": None,
                        "text": None,
                        "v2_themes": "SPORTS;WEATHER",
                        "all_names": "Football",
                    },
                    {
                        "partition_date": "2026-06-24",
                        "document_identifier": "https://example.com/oil-war",
                        "title": None,
                        "summary": None,
                        "text": None,
                        "v2_themes": "OIL;WAR;SHIPPING",
                        "all_names": "Red Sea crude tanker",
                    },
                ],
            )

            seen_urls: list[str] = []

            def fake_enrich(rows: list[dict[str, object]], **_: object) -> dict[str, int]:
                seen_urls.extend(str(row["document_identifier"]) for row in rows)
                rows[0].update(
                    {"title": "Oil story", "summary": "Oil summary", "text": "Oil body"}
                )
                return {
                    "requested_docs": len(rows),
                    "attempted_fetches": len(rows),
                    "rows_enriched": 1,
                    "unique_urls_seen": len(rows),
                }

            with mock.patch("enrich_staged_gdelt_parquet.enrich_rows", side_effect=fake_enrich):
                payload = enrich_staged_parquet(
                    input_root=root,
                    enrich_max_docs_per_file=1,
                    timeout=5.0,
                    user_agent="ua",
                    overwrite=False,
                    include_glob="part-*.parquet",
                    requested_asset_label="WTI",
                    requested_factor_label="war_conflict",
                )

            self.assertEqual(seen_urls, ["https://example.com/oil-war"])
            self.assertEqual(payload["rows_selected_for_enrichment"], 1)

    def test_enrich_staged_parquet_prefers_priority_market_sources_for_macro_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "part-000000.parquet"
            self._write_parquet(
                source,
                [
                    {
                        "partition_date": "2026-06-24",
                        "source_common_name": "local.example",
                        "document_identifier": "https://example.com/local-inflation",
                        "title": None,
                        "summary": None,
                        "text": None,
                        "v2_themes": "INFLATION;INTEREST_RATES",
                        "all_names": "Treasury yields inflation rates",
                    },
                    {
                        "partition_date": "2026-06-24",
                        "source_common_name": "reuters.com",
                        "document_identifier": "https://www.reuters.com/markets/us/yields-inflation-story",
                        "title": None,
                        "summary": None,
                        "text": None,
                        "v2_themes": "INFLATION;INTEREST_RATES",
                        "all_names": "Treasury yields inflation rates",
                    },
                ],
            )

            seen_urls: list[str] = []

            def fake_enrich(rows: list[dict[str, object]], **_: object) -> dict[str, int]:
                seen_urls.extend(str(row["document_identifier"]) for row in rows)
                rows[0].update({"title": "Macro story", "summary": "Macro summary", "text": "Macro body"})
                return {
                    "requested_docs": len(rows),
                    "attempted_fetches": len(rows),
                    "rows_enriched": 1,
                    "unique_urls_seen": len(rows),
                }

            with mock.patch("enrich_staged_gdelt_parquet.enrich_rows", side_effect=fake_enrich):
                enrich_staged_parquet(
                    input_root=root,
                    enrich_max_docs_per_file=1,
                    timeout=5.0,
                    user_agent="ua",
                    overwrite=False,
                    include_glob="part-*.parquet",
                    requested_asset_label="US2Y",
                    requested_factor_label="inflation",
                )

            self.assertEqual(seen_urls, ["https://www.reuters.com/markets/us/yields-inflation-story"])

    def test_enrich_staged_parquet_reenriches_partial_text_for_index_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "part-000000.parquet"
            self._write_parquet(
                source,
                [
                    {
                        "partition_date": "2026-06-24",
                        "source_common_name": "local.example",
                        "document_identifier": "https://example.com/company-story",
                        "title": "Financial Comparison: Root (NASDAQ:ROOT) and Stewart Information Services (NYSE:STC)",
                        "summary": None,
                        "text": None,
                        "v2_themes": "ECON_GROWTH;EARNINGS",
                        "all_names": "Root;Stewart Information Services",
                    },
                    {
                        "partition_date": "2026-06-24",
                        "source_common_name": "reuters.com",
                        "document_identifier": "https://www.reuters.com/markets/us/nasdaq-falls-tech-selloff",
                        "title": "Nasdaq falls as tech selloff deepens",
                        "summary": None,
                        "text": None,
                        "v2_themes": "ECON_GROWTH;ACTIVITY",
                        "all_names": "Nasdaq 100;Wall Street;chip stocks",
                    },
                ],
            )

            seen_urls: list[str] = []

            def fake_enrich(rows: list[dict[str, object]], **_: object) -> dict[str, int]:
                seen_urls.extend(str(row["document_identifier"]) for row in rows)
                rows[0].update({"summary": "Fetched summary", "text": "Fetched body text"})
                return {
                    "requested_docs": len(rows),
                    "attempted_fetches": len(rows),
                    "rows_enriched": 1,
                    "unique_urls_seen": len(rows),
                }

            with mock.patch("enrich_staged_gdelt_parquet.enrich_rows", side_effect=fake_enrich):
                payload = enrich_staged_parquet(
                    input_root=root,
                    enrich_max_docs_per_file=1,
                    timeout=5.0,
                    user_agent="ua",
                    overwrite=False,
                    include_glob="part-*.parquet",
                    requested_asset_label="NDX",
                    requested_factor_label="growth_activity",
                )

            self.assertEqual(seen_urls, ["https://www.reuters.com/markets/us/nasdaq-falls-tech-selloff"])
            self.assertEqual(payload["rows_selected_for_enrichment"], 1)

    def test_enrich_staged_parquet_prefers_fx_and_gold_market_wraps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "part-000000.parquet"
            self._write_parquet(
                source,
                [
                    {
                        "partition_date": "2026-06-24",
                        "source_common_name": "local.example",
                        "document_identifier": "https://example.com/gold-story",
                        "title": None,
                        "summary": None,
                        "text": None,
                        "v2_themes": "CENTRAL_BANK;INFLATION",
                        "all_names": "gold prices bullion dollar",
                    },
                    {
                        "partition_date": "2026-06-24",
                        "source_common_name": "fxstreet.com",
                        "document_identifier": "https://www.fxstreet.com/news/dollar-index-rises-on-hawkish-fed",
                        "title": None,
                        "summary": None,
                        "text": None,
                        "v2_themes": "CENTRAL_BANK;INTEREST_RATES",
                        "all_names": "dollar index greenback fed currency",
                    },
                    {
                        "partition_date": "2026-06-24",
                        "source_common_name": "kitco.com",
                        "document_identifier": "https://www.kitco.com/news/gold-falls-as-dollar-rises",
                        "title": None,
                        "summary": None,
                        "text": None,
                        "v2_themes": "CENTRAL_BANK;INFLATION",
                        "all_names": "gold bullion dollar real yield",
                    },
                ],
            )

            seen_urls: list[str] = []

            def fake_enrich(rows: list[dict[str, object]], **_: object) -> dict[str, int]:
                seen_urls.extend(str(row["document_identifier"]) for row in rows)
                rows[0].update({"summary": "Fetched summary", "text": "Fetched body text"})
                return {
                    "requested_docs": len(rows),
                    "attempted_fetches": len(rows),
                    "rows_enriched": 1,
                    "unique_urls_seen": len(rows),
                }

            with mock.patch("enrich_staged_gdelt_parquet.enrich_rows", side_effect=fake_enrich):
                enrich_staged_parquet(
                    input_root=root,
                    enrich_max_docs_per_file=1,
                    timeout=5.0,
                    user_agent="ua",
                    overwrite=False,
                    include_glob="part-*.parquet",
                    requested_asset_label="DXY",
                    requested_factor_label="central_bank_policy",
                )
                enrich_staged_parquet(
                    input_root=root,
                    enrich_max_docs_per_file=1,
                    timeout=5.0,
                    user_agent="ua",
                    overwrite=False,
                    include_glob="part-*.parquet",
                    requested_asset_label="Gold",
                    requested_factor_label="central_bank_policy",
                )

            self.assertEqual(seen_urls[0], "https://www.fxstreet.com/news/dollar-index-rises-on-hawkish-fed")
            self.assertEqual(seen_urls[-1], "https://www.kitco.com/news/gold-falls-as-dollar-rises")
            self.assertIn("https://www.kitco.com/news/gold-falls-as-dollar-rises", seen_urls)


if __name__ == "__main__":
    unittest.main()
