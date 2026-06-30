#!/usr/bin/env python3
from __future__ import annotations

import json
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

from build_narrative_graph import build_narrative_graph
from narrative_explainer_mcp import (
    _dedupe_factor_docs,
    _preferred_narrative_for_state,
    _supported_block_rank,
    _supported_factor_blocks,
    call_tool,
    handle_request,
)


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "config" / "news_narrative_taxonomy.json"


class NarrativeExplainerMcpTests(unittest.TestCase):
    def _build_fixture(self, db_path: Path, parquet_path: Path) -> None:
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
                            'https://example.com/generic-risk',
                            'Regional leaders discuss security tensions',
                            'Officials met to discuss regional tensions and trade coordination.',
                            'Diplomats held talks on policy and regional stability.',
                            'WAR,30;OIL,10',
                            '-1.0,0,0,0,0,0',
                            '1#Yemen#YM#YM#15.5#47.5#0',
                            '',
                            '',
                            'Regional leaders,10'
                        ),
                        (
                            '20250102130500',
                            '2025-01-02',
                            'markets.example',
                            'https://example.com/gold-dollar-fed',
                            'Gold slips as dollar firms and Fed outlook stays tight',
                            'Gold fell as the dollar strengthened and traders repriced a tighter Fed path.',
                            'Bullion prices eased as a firmer U.S. dollar and steady real-rate pressure weighed on non-yielding metals.',
                            'GOLD,30;INFLATION,15;CENTRAL_BANK,15',
                            '-2.0,0,0,0,0,0',
                            '1#United States#US#US#38.0#-97.0#0',
                            'Federal Reserve,10',
                            'Federal Reserve,10',
                            'Gold,20;Dollar,10;Federal Reserve,10'
                        ),
                        (
                            '20250102131500',
                            '2025-01-02',
                            'macro.example',
                            'https://example.com/oil-inflation-fed',
                            'Oil slump cools inflation outlook as Fed watch continues',
                            'Oil prices fell, inflation concerns eased, and markets kept focus on the Fed policy path.',
                            'Falling energy prices softened inflation pressure while traders assessed the Federal Reserve outlook and front-end rate sensitivity.',
                            'INFLATION,30;CENTRAL_BANK,20;INTEREST_RATES,20;OIL,15;LABOR,5',
                            '-1.5,0,0,0,0,0',
                            '1#United States#US#US#38.0#-97.0#0',
                            'Federal Reserve,10',
                            'Federal Reserve,10',
                            'Oil,15;Inflation,10;Federal Reserve,10'
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

    def test_initialize_and_list_tools(self) -> None:
        init_response = handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        self.assertEqual(init_response["result"]["serverInfo"]["name"], "news-narrative-explainer")
        list_response = handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = [tool["name"] for tool in list_response["result"]["tools"]]
        self.assertEqual(
            names,
            [
                "explain_move",
                "summarize_narrative",
                "supporting_docs",
                "explain_day",
                "explain_cross_asset_move",
                "build_narrative_frame",
                "find_contradictory_assets",
                "explain_asset_via_day_context",
                "query_duckdb",
                "similar_days",
                "intraday_evolution",
            ],
        )

    def test_call_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            parquet_path = tmp / "candidates.parquet"
            db_path = tmp / "narrative_graph.duckdb"
            self._build_fixture(db_path, parquet_path)

            explain = call_tool(
                "explain_move",
                {
                    "db": str(db_path),
                    "asset_label": "WTI",
                    "start_date": "2025-01-02",
                    "end_date": "2025-01-02",
                    "limit": 5,
                },
            )
            explain_payload = json.loads(explain["content"][0]["text"])
            self.assertEqual(explain_payload["asset_label"], "WTI")
            self.assertGreaterEqual(len(explain_payload["top_narratives"]), 1)

            summary = call_tool(
                "summarize_narrative",
                {
                    "db": str(db_path),
                    "asset_label": "WTI",
                    "start_date": "2025-01-02",
                    "end_date": "2025-01-02",
                    "limit": 5,
                },
            )
            self.assertIn("Graph-ranked factors for WTI", summary["content"][0]["text"])
            self.assertIn("Primary regime:", summary["content"][0]["text"])
            self.assertIn("Asset overlay:", summary["content"][0]["text"])
            self.assertIn("Contradiction:", summary["content"][0]["text"])
            self.assertIn("Confidence:", summary["content"][0]["text"])
            self.assertIn("Most defensible text-backed explanations for WTI", summary["content"][0]["text"])
            self.assertIn("Impact paths:", summary["content"][0]["text"])
            self.assertIn("source_confidence=", summary["content"][0]["text"])
            self.assertIn("provenance=", summary["content"][0]["text"])
            self.assertIn("Red Sea oil-risk repricing", summary["content"][0]["text"])
            self.assertIn("shipping_disruption", summary["content"][0]["text"])
            self.assertIn("Red Sea disruption lifts oil risk premium", summary["content"][0]["text"])
            self.assertIn("Shipping interruptions and sanctions concerns", summary["content"][0]["text"])
            self.assertIn("source=ft.com | https://example.com/red-sea-oil", summary["content"][0]["text"])
            self.assertIn("Graph-ranked but weakly evidenced in stored text", summary["content"][0]["text"])

            us2y_summary = call_tool(
                "summarize_narrative",
                {
                    "db": str(db_path),
                    "asset_label": "US2Y",
                    "start_date": "2025-01-02",
                    "end_date": "2025-01-02",
                    "limit": 5,
                },
            )
            self.assertIn("Primary regime:", us2y_summary["content"][0]["text"])
            self.assertIn("proxy/channel=", us2y_summary["content"][0]["text"])
            self.assertIn("direct=low", us2y_summary["content"][0]["text"])
            self.assertIn("source=macro.example | https://example.com/oil-inflation-fed", us2y_summary["content"][0]["text"])

            docs = call_tool(
                "supporting_docs",
                {
                    "db": str(db_path),
                    "asset_label": "WTI",
                    "factor_label": "oil",
                    "start_date": "2025-01-02",
                    "end_date": "2025-01-02",
                    "limit": 5,
                },
            )
            docs_payload = json.loads(docs["content"][0]["text"])
            self.assertIn("example.com/red-sea-oil", docs_payload[0]["document_identifier"])
            self.assertGreaterEqual(len(docs_payload), 1)
            self.assertGreater(docs_payload[0]["relevance_score"], 0.0)

            explain_day = call_tool(
                "explain_day",
                {
                    "db": str(db_path),
                    "date": "2025-01-02",
                    "universe": ["WTI", "Gold"],
                    "limit": 5,
                },
            )
            self.assertIn("Top narratives for 2025-01-02", explain_day["content"][0]["text"])
            self.assertIn("Evidence strength ranking:", explain_day["content"][0]["text"])
            self.assertIn("Market impact ranking:", explain_day["content"][0]["text"])
            self.assertIn("Transmission ranking:", explain_day["content"][0]["text"])
            self.assertIn("source_confidence=", explain_day["content"][0]["text"])
            self.assertIn("provenance=", explain_day["content"][0]["text"])
            self.assertIn("Red Sea oil-risk repricing", explain_day["content"][0]["text"])
            self.assertIn("affected_assets=WTI", explain_day["content"][0]["text"])
            self.assertIn("explained_assets=Gold, WTI", explain_day["content"][0]["text"])
            self.assertIn("supporting_sources=[S1]", explain_day["content"][0]["text"])
            self.assertIn("References:", explain_day["content"][0]["text"])
            self.assertIn("[S1] https://example.com/red-sea-oil", explain_day["content"][0]["text"])
            self.assertIn("Trust summary:", explain_day["content"][0]["text"])
            self.assertIn("fit_confidence=", explain_day["content"][0]["text"])
            self.assertIn("contradiction_score=", explain_day["content"][0]["text"])
            self.assertIn("unsupported_assets=", explain_day["content"][0]["text"])
            self.assertIn("Unsupported / cannot answer:", explain_day["content"][0]["text"])
            self.assertIn("Failure mode:", explain_day["content"][0]["text"])

            cross_asset = call_tool(
                "explain_cross_asset_move",
                {
                    "db": str(db_path),
                    "date": "2025-01-02",
                    "assets": ["WTI", "Gold"],
                    "limit": 5,
                },
            )
            self.assertIn("Cross-asset move for 2025-01-02", cross_asset["content"][0]["text"])
            self.assertIn("Trust summary:", cross_asset["content"][0]["text"])
            self.assertIn("fit_confidence=", cross_asset["content"][0]["text"])
            self.assertIn("contradiction_score=", cross_asset["content"][0]["text"])
            self.assertIn("unsupported_assets=", cross_asset["content"][0]["text"])
            self.assertIn("Unsupported / cannot answer:", cross_asset["content"][0]["text"])
            self.assertIn("Confidence by asset:", cross_asset["content"][0]["text"])
            self.assertIn("evidence=", cross_asset["content"][0]["text"])
            self.assertIn("direct=", cross_asset["content"][0]["text"])
            self.assertIn("provenance=", cross_asset["content"][0]["text"])
            self.assertIn("Failure mode:", cross_asset["content"][0]["text"])

            frame = json.loads(
                call_tool(
                    "build_narrative_frame",
                    {
                        "db": str(db_path),
                        "date": "2025-01-02",
                        "universe": ["WTI", "Gold", "US2Y"],
                        "limit": 5,
                    },
                )["content"][0]["text"]
            )
            self.assertIn("primary_regime", frame)
            self.assertIn("dominant_narrative", frame)
            self.assertIn("best_explanation", frame)
            self.assertIn("top_competing_hypotheses", frame)
            self.assertIn("rankings", frame)
            self.assertIn("first_link", frame)
            self.assertIn("transmission_chain", frame)
            self.assertIn("blocking_overlay", frame)
            self.assertIn("consistency_warnings", frame)
            self.assertIn("confidence_summary", frame)
            self.assertTrue(frame["primary_regime"])
            self.assertIsInstance(frame["market_impact_rows"], list)
            self.assertEqual(frame["dominant_narrative"]["question"], "What was the dominant market narrative today?")
            self.assertIn("source_weight_breakdown", frame["dominant_narrative"])
            self.assertIn("parallel_channels", frame["best_explanation"])
            self.assertTrue(frame["top_competing_hypotheses"])
            self.assertIn("hypothesis", frame["top_competing_hypotheses"][0])
            self.assertIn("confidence", frame["top_competing_hypotheses"][0])
            self.assertIn("explains", frame["top_competing_hypotheses"][0])
            self.assertIn("weakness", frame["top_competing_hypotheses"][0])
            self.assertIn("model_composition", frame["top_competing_hypotheses"][0])
            self.assertIn("evidence_strength", frame["rankings"])

            contradictory = call_tool(
                "find_contradictory_assets",
                {
                    "db": str(db_path),
                    "date": "2025-01-02",
                    "universe": ["WTI", "Gold"],
                    "limit": 5,
                },
            )
            self.assertIn("Dominant narrative for 2025-01-02", contradictory["content"][0]["text"])
            self.assertIn("source_confidence=", contradictory["content"][0]["text"])
            self.assertIn("Fed-path repricing", contradictory["content"][0]["text"])
            self.assertIn("Gold", contradictory["content"][0]["text"])
            self.assertIn("No strong contradictions were found relative to the dominant narrative.", contradictory["content"][0]["text"])
            self.assertIn("Failure mode:", contradictory["content"][0]["text"])

            via_day = call_tool(
                "explain_asset_via_day_context",
                {
                    "db": str(db_path),
                    "date": "2025-01-02",
                    "asset_label": "Gold",
                    "universe": ["WTI", "Gold"],
                    "limit": 5,
                },
            )
            self.assertIn("Asset day-context explanation for Gold on 2025-01-02", via_day["content"][0]["text"])
            self.assertIn("Direct evidence:", via_day["content"][0]["text"])
            self.assertIn("Indirect day context:", via_day["content"][0]["text"])
            self.assertIn("source_confidence=", via_day["content"][0]["text"])
            self.assertIn("evidence=", via_day["content"][0]["text"])
            self.assertIn("provenance=", via_day["content"][0]["text"])
            self.assertIn("context_substitution=", via_day["content"][0]["text"])
            self.assertIn("substitution_reason=", via_day["content"][0]["text"])
            self.assertIn("replaced=", via_day["content"][0]["text"])
            self.assertIn("substitution_confidence=", via_day["content"][0]["text"])
            self.assertIn("replaced=none", via_day["content"][0]["text"])
            self.assertIn("Fed-path repricing", via_day["content"][0]["text"])
            self.assertIn("Failure mode:", via_day["content"][0]["text"])

            query_payload = json.loads(
                call_tool(
                    "query_duckdb",
                    {
                        "db": str(db_path),
                        "sql": "select asset_label, factor_label from gold_asset_factor_panel_daily order by asset_label, factor_label",
                        "limit": 3,
                    },
                )["content"][0]["text"]
            )
            self.assertEqual(query_payload["row_count"], 3)
            self.assertEqual(len(query_payload["rows"]), 3)
            self.assertIn("asset_label", query_payload["rows"][0])
            self.assertIn("factor_label", query_payload["rows"][0])
            with self.assertRaisesRegex(ValueError, "only SELECT or WITH queries are allowed"):
                call_tool(
                    "query_duckdb",
                    {
                        "db": str(db_path),
                        "sql": "delete from gold_asset_factor_panel_daily",
                    },
                )

            similar_days = call_tool(
                "similar_days",
                {
                    "db": str(db_path),
                    "date": "2025-01-02",
                    "limit": 3,
                },
            )
            self.assertIn("No prior local days were available", similar_days["content"][0]["text"])

            intraday = call_tool(
                "intraday_evolution",
                {
                    "db": str(db_path),
                    "date": "2025-01-02",
                    "limit": 3,
                },
            )
            self.assertIn("Intraday evolution for 2025-01-02", intraday["content"][0]["text"])
            self.assertIn("2025-01-02 12:00:00", intraday["content"][0]["text"])
            self.assertIn("2025-01-02 13:00:00", intraday["content"][0]["text"])

    def test_macro_supported_blocks_prefer_text_backed_policy_factor(self) -> None:
        factors = [
            {
                "factor_label": "elections_policy",
                "adjusted_narrative_score": 900.0,
                "avg_narrative_score": 900.0,
            },
            {
                "factor_label": "central_bank_policy",
                "adjusted_narrative_score": 200.0,
                "avg_narrative_score": 200.0,
            },
        ]
        shared_policy_doc = {
            "title": "Dollar climbs as hawkish Fed signals support rate differentials",
            "summary_text": "The dollar index rose as traders priced a firmer Fed path and wider policy-rate differentials.",
            "evidence_text": "Hawkish Federal Reserve guidance lifted the dollar and widened rate differentials across FX markets.",
            "relevant_text": "Hawkish Federal Reserve guidance lifted the dollar and widened rate differentials across FX markets.",
            "source_type": "market_wrap",
            "market_context_score": 12.0,
            "relevance_score": 30.0,
            "classification_confidence": 0.95,
            "document_identifier": "https://example.com/fed-dollar",
            "event_time": "2025-01-02 11:00:00",
            "source_domain": "example.com",
        }
        factor_docs = {
            "elections_policy": [
                {
                    "title": "Election uncertainty keeps markets cautious as dollar edges up",
                    "summary_text": "Campaign headlines kept investors cautious while the dollar edged firmer.",
                    "evidence_text": "Election headlines dominated the session while the dollar firmed modestly against peers.",
                    "relevant_text": "Election headlines dominated the session while the dollar firmed modestly against peers.",
                    "source_type": "market_wrap",
                    "market_context_score": 6.0,
                    "relevance_score": 18.0,
                    "classification_confidence": 0.8,
                    "document_identifier": "https://example.com/election-dollar",
                    "event_time": "2025-01-02 10:00:00",
                    "source_domain": "example.com",
                },
                shared_policy_doc,
            ],
            "central_bank_policy": [
                shared_policy_doc
            ],
        }

        deduped_docs = _dedupe_factor_docs("DXY", factor_docs)
        supported_blocks, weak_labels = _supported_factor_blocks("DXY", factors, deduped_docs)

        self.assertEqual(supported_blocks[0]["factor"]["factor_label"], "central_bank_policy")
        self.assertEqual(len(deduped_docs["central_bank_policy"]), 1)
        self.assertIn("elections_policy", {block["factor"]["factor_label"] for block in supported_blocks} | set(weak_labels))

    def test_dxy_conflict_doc_does_not_beat_clear_policy_doc(self) -> None:
        factors = [
            {
                "factor_label": "war_conflict",
                "adjusted_narrative_score": 500.0,
                "avg_narrative_score": 500.0,
            },
            {
                "factor_label": "central_bank_policy",
                "adjusted_narrative_score": 300.0,
                "avg_narrative_score": 300.0,
            },
        ]
        shared_wrap = {
            "title": "Dollar climbs as stocks slump",
            "summary_text": "The dollar rose as traders priced a firmer Fed path and sought safety during a tech selloff.",
            "evidence_text": "The dollar index climbed as investors positioned for possible Federal Reserve increases to rates and sought shelter from a selloff in technology shares.",
            "relevant_text": "The dollar index climbed as investors positioned for possible Federal Reserve increases to rates and sought shelter from a selloff in technology shares.",
            "source_type": "market_wrap",
            "market_context_score": 12.0,
            "relevance_score": 28.0,
            "classification_confidence": 0.95,
            "document_identifier": "https://example.com/dollar-climbs",
            "event_time": "2025-01-02 11:00:00",
            "source_domain": "example.com",
        }
        factor_docs = {
            "war_conflict": [shared_wrap],
            "central_bank_policy": [shared_wrap],
        }

        deduped_docs = _dedupe_factor_docs("DXY", factor_docs)
        supported_blocks, _ = _supported_factor_blocks("DXY", factors, deduped_docs)

        self.assertEqual(supported_blocks[0]["factor"]["factor_label"], "central_bank_policy")

    def test_specific_text_backed_label_breaks_generic_tie(self) -> None:
        generic_specific = {
            "factor": {"factor_label": "growth_activity", "adjusted_narrative_score": 300.0},
            "docs": [
                {
                    "source_type": "market_wrap",
                    "market_context_score": 8.0,
                    "impact_strength": {"evidence_mode": "proxy"},
                }
            ],
            "best_score": 30.0,
            "narrative_label": "tech-led growth scare",
        }
        generic_fallback = {
            "factor": {"factor_label": "growth_activity", "adjusted_narrative_score": 300.0},
            "docs": [
                {
                    "source_type": "market_wrap",
                    "market_context_score": 8.0,
                    "impact_strength": {"evidence_mode": "proxy"},
                }
            ],
            "best_score": 30.0,
            "narrative_label": "growth-sensitive de-rating",
        }

        specific_rank = _supported_block_rank("NDX", generic_specific)
        fallback_rank = _supported_block_rank("NDX", generic_fallback)

        self.assertGreater(specific_rank, fallback_rank)

    def test_day_context_proxy_can_override_taxonomy_fallback(self) -> None:
        state = {
            "asset_label": "NDX",
            "primary_factor": "growth_activity",
            "supported_blocks": [
                {
                    "factor": {"factor_label": "growth_activity"},
                    "narrative_label": "growth-sensitive de-rating",
                    "narrative_provenance": "taxonomy-fallback",
                    "docs": [
                        {"impact_strength": {"evidence_mode": "proxy"}},
                    ],
                }
            ],
        }
        impact_rows = [
            {
                "factor_label": "central_bank_policy",
                "narrative_label": "Fed-path repricing",
                "narrative_provenance": "proxy-text",
                "explained_assets": ["NDX", "US2Y"],
                "source_confidence": "medium",
                "supporting_urls": ["https://example.com/fed-dollar"],
            }
        ]

        preferred = _preferred_narrative_for_state(state, impact_rows)

        self.assertEqual(preferred["label"], "Fed-path repricing")
        self.assertEqual(preferred["provenance"], "proxy-text")
        self.assertEqual(preferred["source"], "day-context")
        self.assertEqual(preferred["context_substitution"], "yes")
        self.assertEqual(preferred["substitution_reason"], "taxonomy-fallback replaced by proxy-text day narrative")
        self.assertEqual(preferred["replaced"], "growth-sensitive de-rating -> Fed-path repricing")
        self.assertEqual(preferred["substitution_confidence"], "medium")
        self.assertEqual(preferred["supporting_urls"], ["https://example.com/fed-dollar"])


if __name__ == "__main__":
    unittest.main()
