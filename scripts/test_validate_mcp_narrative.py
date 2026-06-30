#!/usr/bin/env python3
from __future__ import annotations

import unittest

from validate_mcp_narrative import (
    SearchResult,
    _answer_question,
    _build_evidence_object,
    _classify_question,
    _clean_duckduckgo_url,
    _extract_reference_urls,
    _extract_stage1_claims,
    _extract_trust_summary,
    _query_hints_for_label,
    _render_validation_summary,
    assess_claim,
)


class ValidateMcpNarrativeTests(unittest.TestCase):
    def _minimal_evidence(self, payload: dict) -> dict:
        return {
            "best_fit_combination": {"label": payload["stage2"][0]["label"]} if payload.get("stage2") else None,
            "transmission_chains": [],
            "market_impact_rows": [],
            "weakest_asset": None,
            "unsupported_assets": payload["stage1"]["trust"]["explain_day"].get("unsupported_assets", []),
            "cannot_answer": payload["stage1"]["trust"]["explain_day"].get("cannot_answer", []),
            "references": {},
            "validation_refinements": payload["stage2"],
        }

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
            "2. Middle East geopolitical repricing (war_conflict): score=88.0, provenance=direct-text, source_diversity=2, source_confidence=high, affected_assets=WTI, supporting_sources=[S2]\n"
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
            "Weakest-fitting core asset: NDX.\n"
            "References:\n"
            "[S1] https://example.com/fed\n"
        )
        claims = _extract_stage1_claims(explain_day, cross, "2026-06-24")
        self.assertEqual(claims[0]["claim_type"], "best_fit_combination")
        self.assertIn("Fed policy repricing", claims[0]["label"])
        self.assertEqual(claims[0]["fit_confidence"], 0.84)
        self.assertEqual(claims[0]["contradiction_score"], 0.18)
        self.assertEqual(claims[0]["unsupported_assets"], ["NDX"])
        weakest = [claim for claim in claims if claim["claim_type"] == "weakest_core_asset"][0]
        self.assertEqual(weakest["label"], "NDX")
        self.assertEqual(weakest["fit_confidence"], 0.79)

    def test_extract_stage1_claims_uses_trust_weakest_core_asset(self) -> None:
        explain_day = (
            "Top narratives for 2026-06-24 across WTI, NDX:\n"
            "Explanatory fit ranking:\n"
            "1. Fed policy repricing: fit_score=99.0, provenance=proxy-text, source_confidence=high, matched_assets=WTI, contradictions=NDX, unresolved=NDX, supporting_sources=[S1]\n"
            "Trust summary:\n"
            "fit_confidence=0.44\n"
            "contradiction_score=0.51\n"
            "unsupported_assets=NDX\n"
            "Unsupported / cannot answer: NDX (unresolved).\n"
            "References:\n"
            "[S1] https://example.com/fed\n"
        )
        cross = (
            "Cross-asset move for 2026-06-24 across WTI, NDX:\n"
            "Trust summary:\n"
            "fit_confidence=0.44\n"
            "contradiction_score=0.51\n"
            "unsupported_assets=NDX\n"
            "weakest_core_asset=NDX\n"
            "Unsupported / cannot answer: NDX (unresolved).\n"
        )
        claims = _extract_stage1_claims(explain_day, cross, "2026-06-24")
        weakest = [claim for claim in claims if claim["claim_type"] == "weakest_core_asset"][0]
        self.assertEqual(weakest["label"], "NDX")

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
        self.assertEqual(trust["contradiction_score"], 0.12)
        self.assertEqual(trust["unsupported_assets"], ["NDX", "DXY"])
        self.assertEqual(trust["weakest_core_asset"], "NDX")
        self.assertEqual(
            trust["cannot_answer"],
            ["NDX (direct=low)", "DXY (combination unresolved)"],
        )

    def test_clean_duckduckgo_url(self) -> None:
        url = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fstory"
        self.assertEqual(_clean_duckduckgo_url(url), "https://example.com/story")

    def test_query_hints_for_label(self) -> None:
        hints = _query_hints_for_label("Fed policy repricing + Middle East geopolitical repricing")
        self.assertIn("Fed", hints)
        self.assertIn("Middle East", hints)
        self.assertIn("oil", hints)

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
        assessed = assess_claim(claim, results, page_texts)
        self.assertIn(assessed["status"], {"confirmed", "refined"})
        self.assertGreaterEqual(len(assessed["supporting_results"]), 1)

    def test_assess_claim_honors_mcp_unresolved(self) -> None:
        claim = {
            "claim_type": "weakest_core_asset",
            "label": "NDX",
            "date": "2026-06-24",
            "fit_confidence": 0.81,
            "contradiction_score": 0.24,
            "unsupported_assets": ["NDX"],
        }
        assessed = assess_claim(claim, [], {})
        self.assertEqual(assessed["status"], "mcp_unresolved")
        self.assertIn("explicitly unresolved", assessed["reason"])

    def test_assess_claim_honors_low_mcp_fit_confidence(self) -> None:
        claim = {
            "claim_type": "best_fit_combination",
            "label": "Fed policy repricing",
            "date": "2026-06-24",
            "fit_confidence": 0.2,
            "contradiction_score": 0.1,
            "unsupported_assets": [],
        }
        assessed = assess_claim(claim, [], {})
        self.assertEqual(assessed["status"], "mcp_low_confidence")
        self.assertIn("fit_confidence", assessed["reason"])

    def test_render_validation_summary(self) -> None:
        validations = [
            {
                "claim_type": "best_fit_combination",
                "label": "Fed policy repricing + oil/geopolitical premium repricing",
                "web_validation": {
                    "status": "refined",
                    "supporting_results": [{"title": "Dollar firms as oil falls", "hits": 2}],
                },
            },
            {
                "claim_type": "weakest_core_asset",
                "label": "NDX",
                "web_validation": {
                    "status": "mcp_unresolved",
                    "reason": "NDX is explicitly unresolved in the MCP trust block",
                    "supporting_results": [],
                },
            },
            {
                "claim_type": "evidence_ranked_narrative",
                "label": "Fed policy repricing",
                "web_validation": {
                    "status": "confirmed",
                    "supporting_results": [{"title": "Fed path repricing lifts dollar", "hits": 3}],
                },
            },
        ]
        trust = {
            "explain_day": {
                "fit_confidence": 0.88,
                "contradiction_score": 0.12,
                "unsupported_assets": ["NDX"],
            }
        }
        lines = _render_validation_summary("2026-06-24", validations, trust)
        rendered = "\n".join(lines)
        self.assertIn("Validation summary for 2026-06-24", rendered)
        self.assertIn("Best-fit explanation: Fed policy repricing + oil/geopolitical premium repricing [refined]", rendered)
        self.assertIn("Validated claims:", rendered)
        self.assertIn("Fed policy repricing: Fed path repricing lifts dollar", rendered)
        self.assertIn("MCP unresolved:", rendered)
        self.assertIn("NDX is explicitly unresolved in the MCP trust block", rendered)

    def test_build_evidence_object(self) -> None:
        explain_day = (
            "Top narratives for 2026-06-24 across WTI, NDX:\n"
            "Explanatory fit ranking:\n"
            "1. Fed policy repricing + oil/geopolitical premium repricing: fit_score=120.0, provenance=proxy-text, taxonomy-fallback, source_confidence=high, high, matched_assets=WTI, contradictions=NDX, unresolved=NDX, supporting_sources=[S1]\n"
            "Market impact ranking:\n"
            "1. Fed policy repricing (central_bank_policy): impact_score=44.0, provenance=proxy-text, source_confidence=high, explained_assets=NDX, WTI, direct_support=WTI\n"
            "Transmission ranking:\n"
            "1. oil/geopolitical premium repricing (oil): provenance=taxonomy-fallback; oil/geopolitical unwind -> lower inflation pressure -> lower yields -> stronger dollar / weaker gold / incomplete equity relief\n"
            "References:\n[S1] https://example.com/a\n"
        )
        trust = {"explain_day": {"unsupported_assets": ["NDX"], "cannot_answer": ["NDX (unresolved)"], "weakest_core_asset": "NDX"}}
        validations = [{"claim_type": "weakest_core_asset", "label": "NDX", "web_validation": {"status": "mcp_unresolved"}}]
        evidence = _build_evidence_object("2026-06-24", explain_day, "", "", None, trust, validations)
        self.assertEqual(evidence["best_fit_combination"]["label"], "Fed policy repricing + oil/geopolitical premium repricing")
        self.assertEqual(evidence["transmission_chains"][0]["factor"], "oil")
        self.assertEqual(evidence["market_impact_rows"][0]["factor"], "central_bank_policy")
        self.assertEqual(evidence["weakest_asset"]["label"], "NDX")

    def test_answer_question_best_explanation(self) -> None:
        payload = {
            "stage1": {
                "trust": {
                    "explain_day": {
                        "fit_confidence": 0.88,
                        "contradiction_score": 0.12,
                        "unsupported_assets": ["NDX"],
                        "cannot_answer": ["NDX (unresolved)"],
                    }
                }
            },
            "stage2": [
                {
                    "claim_type": "best_fit_combination",
                    "label": "Fed policy repricing + oil/geopolitical premium repricing",
                    "web_validation": {
                        "status": "refined",
                        "supporting_results": [{"title": "Dollar firms as oil falls", "hits": 2}],
                    },
                }
            ],
        }
        evidence = self._minimal_evidence(payload)
        evidence["transmission_chains"] = [
            {"factor": "central_bank_policy", "chain": "hawkish Fed repricing / relative policy support -> stronger dollar and pressure on duration and gold"},
            {"factor": "oil", "chain": "oil/geopolitical unwind -> lower oil -> lower inflation pressure -> lower yields; in parallel, stronger dollar / Fed restraint can weigh on gold and limit equity relief"},
        ]
        answer = _answer_question(
            "What is the single best explanation for today’s cross-asset market behaviour?",
            "2026-06-24",
            payload,
            evidence,
        )
        self.assertIn("Fed/dollar restraint + oil/geopolitical premium unwind", answer)
        self.assertIn("lower oil", answer)
        self.assertIn("lower inflation pressure", answer)
        self.assertIn("prevented that from becoming a clean risk-on day", answer)
        self.assertIn("fit metric 0.88", answer)

    def test_answer_question_contradictory_asset(self) -> None:
        payload = {
            "stage1": {
                "trust": {
                    "explain_day": {
                        "fit_confidence": 1.0,
                        "contradiction_score": 0.06,
                        "unsupported_assets": ["NDX"],
                        "cannot_answer": ["NDX (unresolved)"],
                    }
                }
            },
            "stage2": [
                {
                    "claim_type": "weakest_core_asset",
                    "label": "NDX",
                    "web_validation": {
                        "status": "mcp_unresolved",
                        "reason": "NDX is explicitly unresolved in the MCP trust block",
                    },
                }
            ],
        }
        evidence = self._minimal_evidence(payload)
        evidence["weakest_asset"] = {
            "label": "NDX",
            "validation": {
                "status": "mcp_unresolved",
                "reason": "NDX is explicitly unresolved in the MCP trust block",
            },
        }
        answer = _answer_question(
            "Which asset most contradicts the dominant narrative?",
            "2026-06-24",
            payload,
            evidence,
        )
        self.assertIn("NDX", answer)
        self.assertIn("mcp_unresolved", answer)

    def test_classify_question(self) -> None:
        self.assertEqual(
            _classify_question("What is the single best explanation for today’s cross-asset market behaviour?"),
            "best_explanation",
        )
        self.assertEqual(
            _classify_question("Which asset behaved most inconsistently with the prevailing narrative?"),
            "contradictory_asset",
        )
        self.assertEqual(
            _classify_question("What remains unexplained?"),
            "unexplained",
        )

    def test_answer_question_paraphrase_uses_taxonomy(self) -> None:
        payload = {
            "stage1": {
                "trust": {
                    "explain_day": {
                        "fit_confidence": 0.91,
                        "contradiction_score": 0.09,
                        "unsupported_assets": ["Gold"],
                        "cannot_answer": ["Gold (unresolved)"],
                    }
                }
            },
            "stage2": [
                {
                    "claim_type": "weakest_core_asset",
                    "label": "NDX",
                    "web_validation": {
                        "status": "mcp_unresolved",
                        "reason": "NDX is explicitly unresolved in the MCP trust block",
                    },
                }
            ],
        }
        answer = _answer_question(
            "Which asset behaved most inconsistently with the prevailing narrative?",
            "2026-06-24",
            payload,
            {
                **self._minimal_evidence(payload),
                "weakest_asset": {
                    "label": "NDX",
                    "validation": {
                        "status": "mcp_unresolved",
                        "reason": "NDX is explicitly unresolved in the MCP trust block",
                    },
                },
            },
        )
        self.assertIn("NDX", answer)


if __name__ == "__main__":
    unittest.main()
