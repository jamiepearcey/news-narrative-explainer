#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v3_validate_mcp_narrative import (
    SearchResult,
    _clean_duckduckgo_url,
    _shortlist_evidence_for_claim,
    _extract_reference_urls,
    _extract_stage1_claims,
    _extract_trust_summary,
    _query_hints_for_label,
    _render_validation_summary,
    assess_claim,
    phased_hybrid_search,
)


class V3ValidateMcpNarrativeTests(unittest.TestCase):
    def test_extract_reference_urls(self) -> None:
        text = "References:\n[S1] https://example.com/a\n[S2] https://example.com/b\n"
        refs = _extract_reference_urls(text)
        self.assertEqual(refs["S1"], "https://example.com/a")
        self.assertEqual(refs["S2"], "https://example.com/b")

    def test_extract_stage1_claims(self) -> None:
        explain_day = (
            "Top narratives for 2026-06-24 across WTI, Gold:\n"
            "Explanatory fit ranking:\n"
            "1. Fed policy repricing + Middle East geopolitical repricing: fit_score=123.0, provenance=proxy-text, direct-text, "
            "source_confidence=high, high, matched_assets=Gold, WTI, contradictions=none, unresolved=none, supporting_sources=[S1] [S2]\n"
            "Evidence strength ranking:\n"
            "1. Fed policy repricing (central_bank_policy): score=99.0, provenance=proxy-text, source_diversity=3, source_confidence=high, affected_assets=Gold, supporting_sources=[S1]\n"
            "Trust summary:\n"
            "fit_confidence=0.84\n"
            "contradiction_score=0.18\n"
            "unsupported_assets=NDX\n"
            "weakest_core_asset=NDX\n"
            "Unsupported / cannot answer: NDX (direct=low, overall=low).\n"
            "References:\n"
            "[S1] https://example.com/fed\n"
            "[S2] https://example.com/oil\n"
        )
        cross = (
            "Cross-asset move for 2026-06-24 across WTI, Gold:\n"
            "Trust summary:\n"
            "fit_confidence=0.79\n"
            "contradiction_score=0.22\n"
            "unsupported_assets=NDX\n"
            "weakest_core_asset=NDX\n"
            "Unsupported / cannot answer: NDX (direct=low, overall=low).\n"
        )
        claims = _extract_stage1_claims(explain_day, cross, "2026-06-24")
        self.assertEqual(claims[0]["claim_type"], "best_fit_combination")
        self.assertEqual(claims[1]["claim_type"], "weakest_core_asset")

    def test_extract_trust_summary(self) -> None:
        text = (
            "Trust summary:\n"
            "fit_confidence=0.91\n"
            "contradiction_score=0.12\n"
            "unsupported_assets=NDX, DXY\n"
            "weakest_core_asset=NDX\n"
            "Unsupported / cannot answer: NDX (direct=low); DXY (combination unresolved).\n"
        )
        trust = _extract_trust_summary(text)
        self.assertEqual(trust["fit_confidence"], 0.91)
        self.assertEqual(trust["unsupported_assets"], ["NDX", "DXY"])

    def test_clean_duckduckgo_url(self) -> None:
        url = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fstory"
        self.assertEqual(_clean_duckduckgo_url(url), "https://example.com/story")

    def test_query_hints_for_label(self) -> None:
        hints = _query_hints_for_label("Fed policy repricing + Middle East geopolitical repricing")
        self.assertIn("Fed", hints)
        self.assertIn("Middle East", hints)

    def test_assess_claim(self) -> None:
        claim = {"claim_type": "best_fit_combination", "label": "Fed policy repricing", "date": "2026-06-24"}
        results = [
            SearchResult(
                title="Fed policy repricing hits stocks and gold",
                url="https://example.com/1",
                snippet="Investors repriced the Fed path and the dollar strengthened.",
            ),
            SearchResult(
                title="Oil markets settle lower",
                url="https://example.com/2",
                snippet="Crude fell as geopolitical tensions cooled.",
            ),
        ]
        page_texts = {
            "https://example.com/1": "The Fed policy repricing drove gold lower and lifted the dollar.",
            "https://example.com/2": "Oil prices fell.",
        }
        assessed = assess_claim(claim, results, page_texts, model=None)
        self.assertIn(assessed["status"], {"confirmed", "refined"})
        self.assertGreaterEqual(len(assessed["supporting_results"]), 1)
        self.assertEqual(assessed["judge"], "heuristic_fallback")

    def test_assess_claim_uses_llm_judge_when_available(self) -> None:
        claim = {"claim_type": "best_fit_combination", "label": "Fed policy repricing", "date": "2026-06-24"}
        results = [
            SearchResult(
                title="Fed policy repricing hits stocks and gold",
                url="https://example.com/1",
                snippet="Investors repriced the Fed path and the dollar strengthened.",
                citation="https://example.com/1",
            ),
            SearchResult(
                title="Oil markets settle lower",
                url="https://example.com/2",
                snippet="Crude fell as geopolitical tensions cooled.",
                citation="https://example.com/2",
            ),
        ]
        with patch(
            "v3_validate_mcp_narrative._run_codex_json",
            return_value={
                "status": "refined",
                "reason": "Evidence partly supports a narrower inflation-and-rates interpretation.",
                "supporting_evidence_ids": ["E1"],
            },
        ):
            assessed = assess_claim(claim, results, {}, model="gpt-5.4-mini")
        self.assertEqual(assessed["status"], "refined")
        self.assertEqual(assessed["judge"], "codex")
        self.assertEqual(len(assessed["supporting_results"]), 1)
        self.assertEqual(assessed["supporting_results"][0]["citation"], "https://example.com/1")

    def test_assess_claim_marks_codex_failure_explicitly(self) -> None:
        claim = {"claim_type": "best_fit_combination", "label": "Fed policy repricing", "date": "2026-06-24"}
        results = [
            SearchResult(
                title="Fed policy repricing hits stocks and gold",
                url="https://example.com/1",
                snippet="Investors repriced the Fed path and the dollar strengthened.",
                citation="https://example.com/1",
            )
        ]
        with patch("v3_validate_mcp_narrative._run_codex_json", side_effect=RuntimeError("boom")):
            assessed = assess_claim(claim, results, {}, model="gpt-5.4-mini")
        self.assertEqual(assessed["judge"], "codex_failed_heuristic_fallback")
        self.assertIn("codex judge failed", assessed["reason"])

    def test_shortlist_evidence_prefers_specific_finance_candidates(self) -> None:
        claim = {"claim_type": "evidence_ranked_narrative", "label": "inflation-relief repricing", "factor": "inflation"}
        results = [
            SearchResult(
                title="General update",
                url="https://example.com/general",
                snippet="A broad markets wrap without direct inflation detail.",
                score=10.0,
                source_type="general_news",
                source_domain="example.com",
            ),
            SearchResult(
                title="Reuters: Oil prices ease, easing inflation pressure",
                url="https://example.com/reuters-oil",
                snippet="Oil and inflation relief both featured in the move.",
                score=8.0,
                source_type="market_wrap",
                source_domain="reuters.com",
            ),
        ]
        shortlist = _shortlist_evidence_for_claim(claim, results, limit=1)
        self.assertEqual(len(shortlist), 1)
        self.assertEqual(shortlist[0].url, "https://example.com/reuters-oil")

    def test_shortlist_penalizes_tier4_press_release_domains(self) -> None:
        claim = {"claim_type": "evidence_ranked_narrative", "label": "inflation-relief repricing", "factor": "inflation"}
        results = [
            SearchResult(
                title="Inflation relief from lower oil prices",
                url="https://example.com/openpr",
                snippet="Oil prices eased and inflation pressure softened.",
                score=12.0,
                source_type="general_news",
                source_domain="openpr.com",
            ),
            SearchResult(
                title="Oil prices ease, reducing inflation pressure",
                url="https://example.com/moneycontrol",
                snippet="Market participants tied lower crude to easing inflation pressure.",
                score=8.0,
                source_type="market_wrap",
                source_domain="moneycontrol.com",
            ),
        ]
        shortlist = _shortlist_evidence_for_claim(claim, results, limit=1)
        self.assertEqual(shortlist[0].url, "https://example.com/moneycontrol")

    def test_phased_hybrid_search_widens_when_strict_phase_is_empty(self) -> None:
        def fake_search(query: str, min_source_score: float) -> list[SearchResult]:
            if min_source_score >= 0.95:
                return []
            if min_source_score >= 0.85:
                return [
                    SearchResult(
                        title="Useful finance hit",
                        url="https://example.com/usable",
                        snippet="Direct macro evidence.",
                        score=1.0,
                        source_score=0.9,
                    )
                ]
            return [
                SearchResult(
                    title="Noisy tail",
                    url="https://example.com/noisy",
                    snippet="Weak tail evidence.",
                    score=0.5,
                    source_score=0.4,
                )
            ]

        results = phased_hybrid_search(fake_search, "test query", top_k=2)
        self.assertEqual([row.url for row in results], ["https://example.com/usable", "https://example.com/noisy"])

    def test_render_validation_summary(self) -> None:
        validations = [
            {
                "claim_type": "best_fit_combination",
                "label": "Fed policy repricing",
                "validation": {
                    "status": "refined",
                    "supporting_results": [{"title": "Dollar firms", "hits": None, "citation": "https://example.com"}],
                },
            }
        ]
        trust = {"explain_day": {"fit_confidence": 0.88, "contradiction_score": 0.12, "unsupported_assets": ["NDX"]}}
        lines = _render_validation_summary("2026-06-24", validations, trust)
        rendered = "\n".join(lines)
        self.assertIn("Validation summary for 2026-06-24", rendered)
        self.assertIn("citation=https://example.com", rendered)


if __name__ == "__main__":
    unittest.main()
