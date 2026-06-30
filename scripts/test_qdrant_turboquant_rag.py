#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qdrant_turboquant_rag import (
    _build_filter,
    _codex_rerank_prompt,
    _default_embedding_model,
    _date_to_ordinal,
    _embedding_text,
    _maybe_rerank_hits,
    _parse_theme_tags,
    _qdrant_client,
    _resolve_embedding_model,
    _row_payload,
    _stable_point_id,
    _truncate_vector,
)


class QdrantTurboQuantRagTests(unittest.TestCase):
    def test_parse_theme_tags_and_payload(self) -> None:
        row = {
            "doc_id": 42,
            "partition_date": "2026-06-24",
            "document_identifier": "https://example.com/fed-oil",
            "source_domain": "example.com",
            "source_type": "market_wrap",
            "title": "Fed and oil drive cross-asset repricing",
            "summary_text": "Treasury yields and crude both moved after a hawkish policy read.",
            "body_text": "Longer body text",
            "relevant_text": "Fed officials stayed hawkish while oil remained bid.",
            "market_context_text": "Treasury yields rose while crude held firm.",
            "market_context_score": 4.0,
            "v2_themes": "ECON_INFLATION,50;OIL,30;CENTRAL_BANK,20",
            "v2_persons": "Jerome Powell,20",
            "v2_organizations": "Federal Reserve,20",
        }
        payload = _row_payload(row, max_embed_chars=400)
        self.assertEqual(payload["doc_id"], 42)
        self.assertEqual(payload["partition_ordinal"], _date_to_ordinal("2026-06-24"))
        self.assertEqual(payload["theme_tags"], ["ECON_INFLATION", "OIL", "CENTRAL_BANK"])
        self.assertIn("Market context:", payload["embedding_text"])
        self.assertIn("Relevant text:", payload["embedding_text"])

    def test_stable_point_id_is_deterministic(self) -> None:
        row = {
            "doc_id": 7,
            "partition_date": "2026-06-25",
            "document_identifier": "https://example.com/doc",
        }
        self.assertEqual(_stable_point_id(row), _stable_point_id(dict(row)))

    def test_embedding_text_falls_back_when_relevant_text_missing(self) -> None:
        row = {
            "title": "Red Sea disruption lifts oil premium",
            "summary_text": "Shipping risk rose again.",
            "body_text": "Tanker routing changed overnight.",
            "relevant_text": None,
            "market_context_text": None,
            "v2_persons": None,
            "v2_organizations": None,
        }
        text = _embedding_text(row, max_chars=200)
        self.assertIn("Title:", text)
        self.assertIn("Body:", text)

    def test_build_filter_includes_expected_conditions(self) -> None:
        query_filter = _build_filter(
            start_date="2026-06-20",
            end_date="2026-06-24",
            source_domains=["reuters.com", "ft.com"],
            source_types=["market_wrap"],
            theme_tags=["oil"],
        )
        self.assertIsNotNone(query_filter)
        dumped = query_filter.model_dump()
        self.assertEqual(len(dumped["must"]), 4)
        range_condition = dumped["must"][0]
        self.assertEqual(range_condition["key"], "partition_ordinal")
        self.assertEqual(range_condition["range"]["gte"], _date_to_ordinal("2026-06-20"))
        self.assertEqual(range_condition["range"]["lte"], _date_to_ordinal("2026-06-24"))

    def test_parse_theme_tags_empty(self) -> None:
        self.assertEqual(_parse_theme_tags(None), [])
        self.assertEqual(_parse_theme_tags(""), [])

    def test_embedding_model_resolution_prefers_explicit(self) -> None:
        self.assertEqual(_default_embedding_model("ollama"), "embeddinggemma")
        self.assertEqual(
            _resolve_embedding_model("ollama", "nomic-embed-text"),
            "nomic-embed-text",
        )
        self.assertEqual(
            _resolve_embedding_model("openai", None),
            "text-embedding-3-large",
        )

    def test_truncate_vector_optional(self) -> None:
        vector = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(_truncate_vector(vector, None), vector)
        self.assertEqual(_truncate_vector(vector, 2), [1.0, 2.0])
        with self.assertRaises(ValueError):
            _truncate_vector(vector, 5)

    def test_codex_rerank_prompt_contains_query_and_ids(self) -> None:
        prompt = _codex_rerank_prompt(
            "hawkish Fed and stronger dollar",
            [
                {"id": "a", "title": "Fed shocks rates", "source_domain": "example.com"},
                {"id": "b", "title": "Oil falls", "source_domain": "example.net"},
            ],
        )
        self.assertIn("hawkish Fed and stronger dollar", prompt)
        self.assertIn('"id": "a"', prompt)
        self.assertIn('"id": "b"', prompt)

    def test_qdrant_client_supports_local_path(self) -> None:
        client = _qdrant_client("local:/tmp/test-qdrant-client", None)
        self.assertIsNotNone(client)

    def test_maybe_rerank_none_preserves_order(self) -> None:
        hits = [{"id": "a"}, {"id": "b"}]
        reranked = _maybe_rerank_hits(
            query="test",
            hits=hits,
            provider="none",
            rerank_limit=2,
            codex_model="gpt-5.4-mini",
        )
        self.assertEqual(reranked, hits)


if __name__ == "__main__":
    unittest.main()
