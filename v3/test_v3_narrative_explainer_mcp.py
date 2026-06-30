#!/usr/bin/env python3
"""Tests for the v3 hosted-parquet MCP wrapper."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v3_narrative_explainer_mcp as mcp


class AssessEventImpactTests(unittest.TestCase):
    def test_assess_event_impact_requires_date_or_collection(self) -> None:
        with self.assertRaisesRegex(ValueError, "either date or collection is required"):
            mcp._tool_assess_event_impact({"question": "Will this matter for WTI?"})

    def test_assess_event_impact_ranks_wti_for_oil_supply_shock(self) -> None:
        docs = [
            {
                "title": "Hormuz disruption fears lift crude supply risk",
                "summary_text": "Tanker traffic through Hormuz faces disruption, raising crude supply concerns and oil price risk.",
                "market_context_text": "Crude traders warn that Hormuz disruption could tighten physical oil supply and lift WTI.",
                "evidence_text": "Crude traders warn that Hormuz disruption could tighten physical oil supply and lift WTI.",
                "relevant_text": "Hormuz disruption crude oil WTI tanker shipping supply risk",
                "document_identifier": "https://example.com/hormuz",
                "source_domain": "example.com",
                "source_type": "commodity_specialist",
                "partition_date": "2026-06-05",
                "event_time": "2026-06-05",
                "market_context_score": 6.0,
                "search_score": 4.0,
                "relevance_score": 4.0,
                "classification_confidence": 0.0,
            },
            {
                "title": "Safe haven demand edges higher",
                "summary_text": "Gold firms as investors seek havens after regional tensions.",
                "market_context_text": "Investors bought bullion and the dollar as tensions rose.",
                "evidence_text": "Investors bought bullion and the dollar as tensions rose.",
                "relevant_text": "gold dollar safe haven tensions",
                "document_identifier": "https://example.com/gold",
                "source_domain": "example.com",
                "source_type": "market_wrap",
                "partition_date": "2026-06-05",
                "event_time": "2026-06-05",
                "market_context_score": 4.0,
                "search_score": 2.0,
                "relevance_score": 2.0,
                "classification_confidence": 0.0,
            },
        ]
        with (
            patch.object(mcp, "_search_event_corpus", return_value=docs),
            patch.object(mcp, "_graph_factor_context_by_asset", return_value={}),
        ):
            result = mcp._tool_assess_event_impact(
                {
                    "question": "Could a Hormuz disruption impact WTI, Gold, or DXY?",
                    "collection": "test-collection",
                    "candidate_assets": ["WTI", "Gold", "DXY"],
                    "limit": 3,
                    "hybrid_search_url": "",
                }
            )
        payload = json.loads(result["content"][0]["text"])
        self.assertEqual(payload["collection"], "test-collection")
        self.assertGreaterEqual(payload["docs_considered"], 2)
        self.assertGreaterEqual(len(payload["ranked_impacts"]), 1)
        top = payload["ranked_impacts"][0]
        self.assertEqual(top["asset_label"], "WTI")
        self.assertIn(top["primary_factor"], {"oil", "war_conflict", "shipping_disruption"})
        evidence_urls = [row["document_identifier"] for row in top["supporting_evidence"]]
        self.assertIn("https://example.com/hormuz", evidence_urls)

    def test_source_confidence_requires_top_tier_for_high(self) -> None:
        docs = [
            {
                "source_domain": "moneycontrol.com",
                "source_type": "market_wrap",
                "impact_strength": {"evidence_mode": "direct"},
            },
            {
                "source_domain": "kitco.com",
                "source_type": "commodity_specialist",
                "impact_strength": {"evidence_mode": "direct"},
            },
        ]
        self.assertEqual(mcp._source_confidence_for_docs(docs), "medium")

    def test_factor_doc_strength_penalizes_tier4_sources(self) -> None:
        common = {
            "title": "Oil prices fall as de-escalation hopes rise",
            "summary_text": "Oil prices and inflation pressures eased on de-escalation hopes.",
            "market_context_text": "WTI and crude fell as oil traders priced less supply risk.",
            "evidence_text": "WTI and crude fell as oil traders priced less supply risk.",
            "relevant_text": "WTI crude oil inflation supply risk",
            "market_context_score": 5.0,
            "relevance_score": 3.0,
            "classification_confidence": 1.0,
        }
        tier2 = {
            **common,
            "document_identifier": "https://www.moneycontrol.com/a",
            "source_domain": "moneycontrol.com",
            "source_type": "market_wrap",
        }
        tier4 = {
            **common,
            "document_identifier": "https://www.openpr.com/b",
            "source_domain": "openpr.com",
            "source_type": "general_news",
        }
        strong = mcp._factor_doc_strength(tier2, "WTI", "oil")
        weak = mcp._factor_doc_strength(tier4, "WTI", "oil")
        self.assertGreater(strong["score"], weak["score"])


if __name__ == "__main__":
    unittest.main()
