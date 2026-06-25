#!/usr/bin/env python3
"""Minimal stdio MCP wrapper for the standalone narrative explainer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from query_narrative_graph import query_explain_move, query_supporting_docs


SERVER_NAME = "news-narrative-explainer"
SERVER_VERSION = "0.1.0"
DEFAULT_DB = str(Path(__file__).resolve().parents[1] / "data" / "narrative_graph.duckdb")


def _json_dumps(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def _write_message(payload: dict[str, Any]) -> None:
    body = _json_dumps(payload)
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        key, _, value = line.decode("ascii").partition(":")
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = sys.stdin.buffer.read(length)
    return json.loads(body.decode("utf-8"))


def _tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "explain_move",
            "description": "Explain which news factors were most active for an asset in a date window.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "asset_label": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                    "db": {"type": "string", "default": DEFAULT_DB},
                },
                "required": ["asset_label"],
            },
        },
        {
            "name": "summarize_narrative",
            "description": "Produce a concise deterministic text summary for an asset move from local news factors.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "asset_label": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                    "db": {"type": "string", "default": DEFAULT_DB},
                },
                "required": ["asset_label"],
            },
        },
        {
            "name": "supporting_docs",
            "description": "Return supporting document URLs for an asset and optional factor in a date window.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "asset_label": {"type": "string"},
                    "factor_label": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                    "db": {"type": "string", "default": DEFAULT_DB},
                },
                "required": ["asset_label"],
            },
        },
    ]


def _text_content(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _tool_explain_move(arguments: dict[str, Any]) -> dict[str, Any]:
    db = Path(arguments.get("db", DEFAULT_DB))
    payload = query_explain_move(
        db,
        asset_label=arguments["asset_label"],
        start_date=arguments.get("start_date"),
        end_date=arguments.get("end_date"),
        limit=int(arguments.get("limit", 10)),
    )
    return _text_content(json.dumps(payload, indent=2, default=str))


def _tool_supporting_docs(arguments: dict[str, Any]) -> dict[str, Any]:
    db = Path(arguments.get("db", DEFAULT_DB))
    payload = query_supporting_docs(
        db,
        asset_label=arguments["asset_label"],
        factor_label=arguments.get("factor_label"),
        start_date=arguments.get("start_date"),
        end_date=arguments.get("end_date"),
        limit=int(arguments.get("limit", 10)),
    )
    return _text_content(json.dumps(payload, indent=2, default=str))


def _tool_summarize_narrative(arguments: dict[str, Any]) -> dict[str, Any]:
    db = Path(arguments.get("db", DEFAULT_DB))
    payload = query_explain_move(
        db,
        asset_label=arguments["asset_label"],
        start_date=arguments.get("start_date"),
        end_date=arguments.get("end_date"),
        limit=int(arguments.get("limit", 5)),
    )
    factors = payload["top_narratives"][:3]
    docs = payload["supporting_docs"][:3]
    if not factors:
        summary = f"No narratives found for {payload['asset_label']} in the requested window."
        return _text_content(summary)

    top_factor_labels = ", ".join(f["factor_label"] for f in factors)
    summary_lines = [
        f"{payload['asset_label']} was most associated with {top_factor_labels} in the requested window.",
        "Top factors:",
    ]
    for factor in factors:
        summary_lines.append(
            f"- {factor['factor_label']}: docs={factor.get('doc_count', factor.get('news_count'))}, "
            f"sources≈{factor.get('avg_unique_sources')}, tone={factor.get('avg_tone_mean')}, "
            f"score={factor.get('avg_narrative_score', factor.get('avg_event_intensity'))}"
        )
    if docs:
        summary_lines.append("Supporting documents:")
        for doc in docs:
            summary_lines.append(
                f"- {doc['event_time']} | {doc['factor_label']} | {doc['source_domain']} | {doc['document_identifier']}"
            )
    return _text_content("\n".join(summary_lines))


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "explain_move":
        return _tool_explain_move(arguments)
    if name == "summarize_narrative":
        return _tool_summarize_narrative(arguments)
    if name == "supporting_docs":
        return _tool_supporting_docs(arguments)
    raise ValueError(f"unknown tool: {name}")


def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    req_id = request.get("id")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2025-06-18",
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "capabilities": {"tools": {}},
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": _tool_specs()}}
    if method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        try:
            result = call_tool(tool_name, arguments)
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as exc:  # pragma: no cover
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(exc)},
            }
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def main() -> int:
    while True:
        request = _read_message()
        if request is None:
            return 0
        response = handle_request(request)
        if response is not None:
            _write_message(response)


if __name__ == "__main__":
    raise SystemExit(main())
