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
from narrative_explainer_mcp import call_tool, handle_request


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
        self.assertEqual(names, ["explain_move", "summarize_narrative", "supporting_docs"])

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
            self.assertIn("WTI was most associated", summary["content"][0]["text"])
            self.assertIn("Red Sea disruption lifts oil risk premium", summary["content"][0]["text"])
            self.assertIn("Shipping interruptions and sanctions concerns", summary["content"][0]["text"])

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


if __name__ == "__main__":
    unittest.main()
