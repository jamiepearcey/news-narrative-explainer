#!/usr/bin/env python3
"""Two-stage narrative validation: local MCP first, constrained web second."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "narrative_graph.duckdb"
DEFAULT_MCP = Path(__file__).resolve().parent / "narrative_explainer_mcp.py"
DEFAULT_V3_DIR = ROOT / "v3"
SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/?q={query}"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}"
USER_AGENT = "news-narrative-explainer-validator/0.1"
DEFAULT_QDRANT_URL = "http://127.0.0.1:6334"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_EMBEDDING_MODEL = "all-minilm"
DEFAULT_TRUNCATE_DIM = 256
DEFAULT_LOCAL_LIMIT = 8
DEFAULT_LOCAL_CANDIDATE_LIMIT = 64
DEFAULT_CODEX_MODEL = "gpt-5.4-mini"
DEFAULT_HYBRID_SEARCH_URL = ""
NARRATIVE_QUERY_HINTS = {
    "fed policy repricing": ["Fed", "rates", "dollar", "hawkish", "Treasury yields"],
    "middle east geopolitical repricing": ["Middle East", "Iran", "oil", "geopolitical risk", "safe haven"],
    "trade-and-sanctions repricing": ["sanctions", "trade", "oil", "tariffs", "shipping"],
    "oil/geopolitical premium repricing": ["oil", "geopolitical premium", "Middle East", "crude", "shipping"],
    "rate-path repricing": ["Fed path", "rates", "Treasury", "yields"],
    "inflation-relief repricing": ["inflation relief", "oil", "yields", "disinflation"],
}
FACTOR_QUERY_HINTS = {
    "central_bank_policy": ["Fed", "rate hikes", "policy path", "hawkish", "dollar"],
    "war_conflict": ["Middle East", "Iran", "conflict", "oil", "safe haven"],
    "sanctions_trade": ["sanctions", "trade tensions", "tariffs", "oil", "shipping"],
    "oil": ["oil prices", "crude", "Middle East", "supply risk"],
    "interest_rates": ["Treasury yields", "rate path", "Fed", "bond market"],
    "inflation": ["inflation", "disinflation", "oil prices", "yields"],
}
ASSET_QUERY_HINTS = {
    "NDX": ["Nasdaq", "tech stocks", "megacap", "AI", "semiconductors"],
    "SPX": ["S&P 500", "US stocks", "Wall Street"],
    "DXY": ["dollar", "US dollar", "greenback", "FX"],
    "US2Y": ["2-year Treasury", "front-end yields", "Fed path"],
    "US10Y": ["10-year Treasury", "long-end yields", "Treasury market"],
    "WTI": ["WTI", "crude oil", "oil prices"],
    "Gold": ["gold", "bullion", "precious metals"],
}
QUESTION_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    (
        "best_explanation",
        (
            "dominant market narrative",
            "best explains",
            "cross-asset move",
            "single best explanation",
            "market behaviour",
        ),
    ),
    (
        "contradictory_asset",
        (
            "contradicts the dominant narrative",
            "contradict the dominant narrative",
            "which asset behaved most inconsistently",
            "most contradicts",
        ),
    ),
    (
        "contradiction_evidence",
        (
            "evidence contradicts",
            "what evidence contradicts",
            "where is the evidence contradictory",
        ),
    ),
    (
        "unexplained",
        (
            "remain unexplained",
            "what remains unexplained",
            "what important information is still unknown",
        ),
    ),
    (
        "reaction_balance",
        (
            "overreact",
            "underreact",
            "did markets overreact",
            "did markets underreact",
        ),
    ),
    (
        "assumptions",
        (
            "assumptions",
            "narrative rely",
            "what assumptions does today",
        ),
    ),
]


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0
    query: str = ""
    source_type: str = ""
    source_domain: str = ""
    citation: str = ""


def _default_collection_name(date: str) -> str:
    return f"news_narrative_v3_{date.replace('-', '')}_allminilm"


def _read_mcp_message(stdout: Any) -> dict[str, Any]:
    header = b""
    while b"\r\n\r\n" not in header:
        chunk = stdout.read(1)
        if not chunk:
            raise RuntimeError("unexpected EOF from MCP server")
        header += chunk
    head, _, rest = header.partition(b"\r\n\r\n")
    length = None
    for line in head.decode("ascii").split("\r\n"):
        if line.lower().startswith("content-length:"):
            length = int(line.split(":", 1)[1].strip())
            break
    if length is None:
        raise RuntimeError("missing content-length header from MCP server")
    body = rest
    while len(body) < length:
        body += stdout.read(length - len(body))
    return json.loads(body.decode("utf-8"))


def _send_mcp_request(proc: subprocess.Popen[bytes], request: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(request).encode("utf-8")
    payload = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
    assert proc.stdin is not None
    proc.stdin.write(payload)
    proc.stdin.flush()
    assert proc.stdout is not None
    return _read_mcp_message(proc.stdout)


def call_mcp_tool(tool_name: str, arguments: dict[str, Any], mcp_path: Path) -> str:
    proc = subprocess.Popen(
        ["uv", "run", "--with", "duckdb>=1.0", "python", str(mcp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _send_mcp_request(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        response = _send_mcp_request(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
        )
        if "error" in response:
            raise RuntimeError(response["error"]["message"])
        return str(response["result"]["content"][0]["text"])
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def _extract_reference_urls(text: str) -> dict[str, str]:
    refs: dict[str, str] = {}
    match = re.search(r"References:\n(?P<body>.+)$", text, re.DOTALL)
    if not match:
        return refs
    for line in match.group("body").splitlines():
        ref_match = re.match(r"\[(S\d+)\]\s+(https?://\S+)", line.strip())
        if ref_match:
            refs[ref_match.group(1)] = ref_match.group(2)
    return refs


def _extract_trust_summary(text: str) -> dict[str, Any]:
    trust: dict[str, Any] = {}
    match = re.search(
        r"Trust summary:\n"
        r"fit_confidence=(?P<fit>[0-9.]+)\n"
        r"contradiction_score=(?P<contradiction>[0-9.]+)\n"
        r"unsupported_assets=(?P<unsupported>[^\n]+)"
        r"(?:\nweakest_core_asset=(?P<weakest>[^\n]+))?",
        text,
    )
    if not match:
        return trust
    unsupported_raw = (match.group("unsupported") or "none").strip()
    unsupported_assets = [] if unsupported_raw == "none" else [part.strip() for part in unsupported_raw.split(",") if part.strip()]
    trust["fit_confidence"] = float(match.group("fit"))
    trust["contradiction_score"] = float(match.group("contradiction"))
    trust["unsupported_assets"] = unsupported_assets
    if match.group("weakest"):
        trust["weakest_core_asset"] = match.group("weakest").strip()
    unsupported_match = re.search(r"Unsupported / cannot answer: (?P<body>.+?)\.", text)
    if unsupported_match:
        body = unsupported_match.group("body").strip()
        trust["cannot_answer"] = [] if body == "none" else [part.strip() for part in body.split(";") if part.strip()]
    return trust


def _extract_stage1_claims(explain_day_text: str, cross_asset_text: str, date: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    references = _extract_reference_urls(explain_day_text)
    references.update(_extract_reference_urls(cross_asset_text))
    explain_day_trust = _extract_trust_summary(explain_day_text)
    cross_asset_trust = _extract_trust_summary(cross_asset_text)

    fit_match = re.search(
        r"1\. (?P<labels>.+?): fit_score=(?P<score>[-0-9.]+).+?matched_assets=(?P<assets>.+?), contradictions=(?P<contradictions>.+?), unresolved=",
        explain_day_text,
    )
    if fit_match:
        labels = fit_match.group("labels")
        hints = _query_hints_for_label(labels)
        claims.append(
            {
                "claim_type": "best_fit_combination",
                "date": date,
                "label": labels,
                "fit_score": fit_match.group("score"),
                "matched_assets": fit_match.group("assets"),
                "contradictions": fit_match.group("contradictions"),
                "fit_confidence": explain_day_trust.get("fit_confidence"),
                "contradiction_score": explain_day_trust.get("contradiction_score"),
                "unsupported_assets": explain_day_trust.get("unsupported_assets", []),
                "query": f"{date} {' '.join(hints)} markets oil dollar yields stocks",
                "supporting_urls": list(references.values())[:5],
            }
        )

    weakest_asset = cross_asset_trust.get("weakest_core_asset")
    if not weakest_asset:
        weakest_match = re.search(r"Weakest-fitting core asset: (?P<asset>[A-Z0-9]+)\.", cross_asset_text)
        if weakest_match:
            weakest_asset = weakest_match.group("asset")
    if weakest_asset:
        claims.append(
            {
                "claim_type": "weakest_core_asset",
                "date": date,
                "label": weakest_asset,
                "fit_confidence": cross_asset_trust.get("fit_confidence"),
                "contradiction_score": cross_asset_trust.get("contradiction_score"),
                "unsupported_assets": cross_asset_trust.get("unsupported_assets", []),
                "query": f"{date} {' '.join(ASSET_QUERY_HINTS.get(weakest_asset, [weakest_asset]))} weak despite lower yields oil lower",
                "supporting_urls": list(references.values())[:3],
            }
        )

    for match in re.finditer(
        r"^\d+\. (?P<label>.+?) \((?P<factor>[^)]+)\): score=(?P<score>[-0-9.]+).+?supporting_sources=(?P<sources>.+)$",
        explain_day_text,
        re.MULTILINE,
    ):
        label = match.group("label")
        factor = match.group("factor")
        claims.append(
            {
                "claim_type": "evidence_ranked_narrative",
                "date": date,
                "label": label,
                "factor": factor,
                "score": match.group("score"),
                "query": f"{date} {' '.join(_query_hints_for_label(label) + FACTOR_QUERY_HINTS.get(factor, []))} Reuters AP market",
                "supporting_urls": list(references.values())[:3],
            }
        )
        if len([c for c in claims if c["claim_type"] == "evidence_ranked_narrative"]) >= 3:
            break

    return claims


def _extract_best_fit_combination(explain_day_text: str) -> dict[str, Any] | None:
    match = re.search(
        r"^1\. (?P<labels>.+?): fit_score=(?P<fit_score>[-0-9.]+), .*?"
        r"matched_assets=(?P<matched>.+?), contradictions=(?P<contradictions>.+?), unresolved=(?P<unresolved>.+?), supporting_sources=(?P<sources>.+)$",
        explain_day_text,
        re.MULTILINE,
    )
    if not match:
        return None
    return {
        "label": match.group("labels").strip(),
        "fit_score": float(match.group("fit_score")),
        "matched_assets": [] if match.group("matched").strip() == "none" else [part.strip() for part in match.group("matched").split(",") if part.strip()],
        "contradictions": [] if match.group("contradictions").strip() == "none" else [part.strip() for part in match.group("contradictions").split(",") if part.strip()],
        "unresolved": [] if match.group("unresolved").strip() == "none" else [part.strip() for part in match.group("unresolved").split(",") if part.strip()],
        "supporting_sources": match.group("sources").strip(),
    }


def _extract_ranked_rows(text: str, section_name: str, pattern: str) -> list[dict[str, Any]]:
    section = re.search(rf"{re.escape(section_name)}:\n(?P<body>.+?)(?:\n[A-Z][^\n]+:|\Z)", text, re.DOTALL)
    if not section:
        return []
    rows: list[dict[str, Any]] = []
    for match in re.finditer(pattern, section.group("body"), re.MULTILINE):
        rows.append(match.groupdict())
    return rows


def _build_evidence_object(
    date: str,
    explain_day_text: str,
    cross_asset_text: str,
    contradictory_text: str,
    asset_context_text: str | None,
    trust: dict[str, Any],
    validations: list[dict[str, Any]],
) -> dict[str, Any]:
    references = _extract_reference_urls(explain_day_text)
    references.update(_extract_reference_urls(cross_asset_text))
    if contradictory_text:
        references.update(_extract_reference_urls(contradictory_text))
    if asset_context_text:
        references.update(_extract_reference_urls(asset_context_text))

    transmission_rows = _extract_ranked_rows(
        explain_day_text,
        "Transmission ranking",
        r"^\d+\. (?P<label>.+?) \((?P<factor>[^)]+)\): provenance=(?P<provenance>[^;]+); (?P<chain>.+)$",
    )
    market_impact_rows = _extract_ranked_rows(
        explain_day_text,
        "Market impact ranking",
        r"^\d+\. (?P<label>.+?) \((?P<factor>[^)]+)\): impact_score=(?P<impact_score>[-0-9.]+), provenance=(?P<provenance>[^,]+), source_confidence=(?P<source_confidence>[^,]+), explained_assets=(?P<explained_assets>.+?), direct_support=(?P<direct_support>.+)$",
    )
    best_fit = _extract_best_fit_combination(explain_day_text)
    day_trust = trust.get("explain_day") or {}
    weakest_asset = day_trust.get("weakest_core_asset")
    weakest_validation = next(
        (item["web_validation"] for item in validations if item.get("claim_type") == "weakest_core_asset" and item.get("label") == weakest_asset),
        None,
    )
    return {
        "date": date,
        "best_fit_combination": best_fit,
        "transmission_chains": transmission_rows,
        "market_impact_rows": market_impact_rows,
        "weakest_asset": {
            "label": weakest_asset,
            "validation": weakest_validation,
            "contradictory_text": contradictory_text,
            "asset_context_text": asset_context_text,
        }
        if weakest_asset
        else None,
        "unsupported_assets": list(day_trust.get("unsupported_assets") or []),
        "cannot_answer": list(day_trust.get("cannot_answer") or []),
        "references": references,
        "validation_refinements": validations,
    }


def _frame_to_evidence(frame: dict[str, Any], validations: list[dict[str, Any]], trust: dict[str, Any]) -> dict[str, Any]:
    return {
        **frame,
        "best_fit_combination": {
            "label": ((frame.get("best_explanation") or {}).get("best_fit_combination")) or " + ".join(frame.get("primary_regime_raw") or []),
            "matched_assets": frame.get("diagnostics", {}).get("matched_assets", []),
            "contradictions": frame.get("diagnostics", {}).get("contradictions", []),
            "unresolved": frame.get("diagnostics", {}).get("unresolved", []),
        },
        "unsupported_assets": frame.get("unresolved_assets", []),
        "cannot_answer": list((trust.get("explain_day") or {}).get("cannot_answer", [])),
        "validation_refinements": validations,
        "weakest_asset": {
            "label": frame.get("weakest_asset"),
            "validation": next(
                (
                    item.get("web_validation")
                    for item in validations
                    if item.get("claim_type") == "weakest_core_asset" and item.get("label") == frame.get("weakest_asset")
                ),
                None,
            ),
        }
        if frame.get("weakest_asset")
        else None,
    }


def _strip_tags(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", unescape(text))
    return text.strip()


def _clean_duckduckgo_url(url: str) -> str:
    if url.startswith("//"):
        return "https:" + url
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
        qs = parse_qs(parsed.query)
        uddg = qs.get("uddg")
        if uddg:
            return uddg[0]
    return url


def _query_hints_for_label(label: str) -> list[str]:
    label_lower = label.lower()
    hints: list[str] = []
    for key, values in NARRATIVE_QUERY_HINTS.items():
        if key in label_lower:
            hints.extend(values)
    if not hints:
        tokens = re.sub(r"[^A-Za-z0-9]+", " ", label).split()
        hints.extend(token for token in tokens if len(token) >= 4)
    return list(dict.fromkeys(hints))[:8]


def _run_codex_json(prompt: str, schema: dict[str, Any], model: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="codex-validate-") as tmpdir:
        schema_path = Path(tmpdir) / "schema.json"
        output_path = Path(tmpdir) / "output.json"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        subprocess.run(
            [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--color",
                "never",
                "--model",
                model,
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                prompt,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        return json.loads(output_path.read_text(encoding="utf-8"))


def expand_claim_queries(claim: dict[str, Any], model: str, max_queries: int = 4) -> list[str]:
    base_query = str(claim.get("query") or "").strip()
    if not base_query:
        return []
    prompt = (
        "Expand a finance/news validation query into a small set of high-recall search queries.\n"
        "Return concise search strings only.\n"
        f"Claim JSON:\n{json.dumps(claim, ensure_ascii=True)}"
    )
    schema = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["queries"],
        "additionalProperties": False,
    }
    try:
        payload = _run_codex_json(prompt, schema, model)
        queries = [base_query]
        for query in payload.get("queries", []):
            query = str(query).strip()
            if query and query not in queries:
                queries.append(query)
            if len(queries) >= max_queries:
                break
        return queries
    except Exception:
        return [base_query]


def search_duckduckgo(query: str, max_results: int) -> list[SearchResult]:
    request = Request(
        SEARCH_ENDPOINT.format(query=quote_plus(query)),
        headers={"User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=20) as response:
        html = response.read().decode("utf-8", errors="ignore")
    results: list[SearchResult] = []
    pattern = re.compile(
        r'(?s)<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
        r'<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>'
    )
    for match in pattern.finditer(html):
        results.append(
            SearchResult(
                title=_strip_tags(match.group("title")),
                url=_clean_duckduckgo_url(unescape(match.group("url"))),
                snippet=_strip_tags(match.group("snippet")),
            )
        )
        if len(results) >= max_results:
            break
    return results


def search_google_news_rss(query: str, max_results: int) -> list[SearchResult]:
    request = Request(
        GOOGLE_NEWS_RSS.format(query=quote_plus(query)),
        headers={"User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=20) as response:
        xml_text = response.read().decode("utf-8", errors="ignore")
    root = ElementTree.fromstring(xml_text)
    results: list[SearchResult] = []
    for item in root.findall(".//item"):
        title = _strip_tags(item.findtext("title") or "")
        url = (item.findtext("link") or "").strip()
        snippet = _strip_tags(item.findtext("description") or "")
        if not title or not url:
            continue
        results.append(SearchResult(title=title, url=url, snippet=snippet))
        if len(results) >= max_results:
            break
    return results


def search_results_for_query(query: str, max_results: int) -> list[SearchResult]:
    results = []
    try:
        results = search_duckduckgo(query, max_results)
    except Exception:
        results = []
    if results:
        return results
    try:
        return search_google_news_rss(query, max_results)
    except Exception:
        return []


def search_local_hybrid(
    query: str,
    collection: str,
    qdrant_url: str,
    embedding_model: str,
    truncate_dim: int,
    limit: int,
    candidate_limit: int,
    codex_model: str,
) -> list[SearchResult]:
    proc = subprocess.run(
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "qdrant_day",
            "--",
            "search",
            "--collection",
            collection,
            "--qdrant-url",
            qdrant_url,
            "--query",
            query,
            "--embedding-model",
            embedding_model,
            "--truncate-dim",
            str(truncate_dim),
            "--limit",
            str(limit),
            "--candidate-limit",
            str(candidate_limit),
            "--rerank-provider",
            "codex",
            "--rerank-limit",
            str(limit),
            "--codex-model",
            codex_model,
        ],
        cwd=DEFAULT_V3_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    results: list[SearchResult] = []
    for hit in payload.get("hits", []):
        url = str(hit.get("document_identifier") or "").strip()
        title = str(hit.get("title") or "").strip() or url
        snippet = next(
            (
                str(value).strip()
                for value in (
                    hit.get("market_context_text"),
                    hit.get("summary_text"),
                )
                if value
            ),
            "",
        )
        citation = url
        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                score=float(hit.get("score") or hit.get("fused_score") or 0.0),
                query=query,
                source_type=str(hit.get("source_type") or ""),
                source_domain=str(hit.get("source_domain") or ""),
                citation=citation,
            )
        )
    return results


def search_remote_hybrid(
    endpoint: str,
    query: str,
    collection: str,
    qdrant_url: str,
    embedding_model: str,
    truncate_dim: int,
    limit: int,
    candidate_limit: int,
    codex_model: str,
) -> list[SearchResult]:
    body = json.dumps(
        {
            "query": query,
            "collection": collection,
            "qdrant_url": qdrant_url,
            "embedding_model": embedding_model,
            "truncate_dim": truncate_dim,
            "limit": limit,
            "candidate_limit": candidate_limit,
            "rerank_provider": "codex",
            "rerank_limit": limit,
            "codex_model": codex_model,
        }
    ).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8", errors="ignore"))
    results: list[SearchResult] = []
    for hit in payload.get("hits", []):
        url = str(hit.get("document_identifier") or "").strip()
        title = str(hit.get("title") or "").strip() or url
        snippet = next(
            (
                str(value).strip()
                for value in (
                    hit.get("market_context_text"),
                    hit.get("summary_text"),
                )
                if value
            ),
            "",
        )
        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                score=float(hit.get("score") or hit.get("fused_score") or 0.0),
                query=query,
                source_type=str(hit.get("source_type") or ""),
                source_domain=str(hit.get("source_domain") or ""),
                citation=url,
            )
        )
    return results


def merge_local_results(query: str, per_query_results: list[list[SearchResult]], model: str, top_k: int) -> list[SearchResult]:
    merged: dict[str, SearchResult] = {}
    for results in per_query_results:
        for rank, result in enumerate(results, start=1):
            key = result.url or result.title
            if not key:
                continue
            score_bonus = 1.0 / (rank + 10.0)
            if key not in merged:
                merged[key] = SearchResult(
                    title=result.title,
                    url=result.url,
                    snippet=result.snippet,
                    score=result.score + score_bonus,
                    query=result.query,
                    source_type=result.source_type,
                    source_domain=result.source_domain,
                    citation=result.citation,
                )
            else:
                existing = merged[key]
                existing.score = max(existing.score, result.score + score_bonus)
                if len(result.snippet) > len(existing.snippet):
                    existing.snippet = result.snippet
    candidates = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    if len(candidates) <= top_k:
        return candidates
    schema = {
        "type": "object",
        "properties": {"ranked_urls": {"type": "array", "items": {"type": "string"}}},
        "required": ["ranked_urls"],
        "additionalProperties": False,
    }
    prompt = (
        "Rerank candidate evidence documents for a finance/news validation query.\n"
        "Prefer direct relevance, market specificity, and non-duplicate coverage.\n"
        f"Query: {query}\n"
        f"Candidates JSON:\n{json.dumps([candidate.__dict__ for candidate in candidates[: max(top_k * 2, 10)]], ensure_ascii=True)}"
    )
    try:
        payload = _run_codex_json(prompt, schema, model)
        ranked_urls = [str(url) for url in payload.get("ranked_urls", [])]
        reranked: list[SearchResult] = []
        seen: set[str] = set()
        by_url = {candidate.url: candidate for candidate in candidates}
        for url in ranked_urls:
            candidate = by_url.get(url)
            if candidate and url not in seen:
                reranked.append(candidate)
                seen.add(url)
            if len(reranked) >= top_k:
                break
        for candidate in candidates:
            if candidate.url not in seen:
                reranked.append(candidate)
                seen.add(candidate.url)
            if len(reranked) >= top_k:
                break
        return reranked
    except Exception:
        return candidates[:top_k]


def fetch_page_text(url: str, max_chars: int = 12000) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=20) as response:
        raw = response.read(max_chars * 2).decode("utf-8", errors="ignore")
    return _strip_tags(raw)[:max_chars]


def _claim_keywords(claim: dict[str, Any]) -> list[str]:
    tokens = _query_hints_for_label(str(claim.get("label") or ""))
    tokens.extend(FACTOR_QUERY_HINTS.get(str(claim.get("factor") or ""), []))
    tokens.extend(token.lower() for token in ASSET_QUERY_HINTS.get(str(claim.get("label") or ""), []))
    text = " ".join(tokens)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    tokens = [token.lower() for token in text.split() if len(token) >= 4]
    return list(dict.fromkeys(tokens))[:8]


def _trust_gate_claim(claim: dict[str, Any]) -> dict[str, Any] | None:
    fit_confidence = claim.get("fit_confidence")
    contradiction_score = claim.get("contradiction_score")
    unsupported_assets = set(claim.get("unsupported_assets") or [])
    label = str(claim.get("label") or "")
    claim_type = str(claim.get("claim_type") or "")

    if claim_type == "weakest_core_asset" and label in unsupported_assets:
        return {
            "claim_type": claim_type,
            "label": label,
            "status": "mcp_unresolved",
            "keywords": [],
            "supporting_results": [],
            "reason": f"{label} is explicitly unresolved in the MCP trust block",
            "fit_confidence": fit_confidence,
            "contradiction_score": contradiction_score,
        }

    if fit_confidence is not None and float(fit_confidence) < 0.35:
        return {
            "claim_type": claim_type,
            "label": label,
            "status": "mcp_low_confidence",
            "keywords": [],
            "supporting_results": [],
            "reason": f"stage-one fit_confidence {fit_confidence:.2f} is too low for useful web validation",
            "fit_confidence": fit_confidence,
            "contradiction_score": contradiction_score,
        }

    if contradiction_score is not None and float(contradiction_score) >= 0.75:
        return {
            "claim_type": claim_type,
            "label": label,
            "status": "mcp_high_contradiction",
            "keywords": [],
            "supporting_results": [],
            "reason": f"stage-one contradiction_score {contradiction_score:.2f} is too high for clean validation",
            "fit_confidence": fit_confidence,
            "contradiction_score": contradiction_score,
        }

    return None


def assess_claim(claim: dict[str, Any], search_results: list[SearchResult], page_texts: dict[str, str]) -> dict[str, Any]:
    gated = _trust_gate_claim(claim)
    if gated is not None:
        return gated

    keywords = _claim_keywords(claim)
    negative_terms = ["opposite", "contradict", "reversed", "despite", "ignored"]
    scored: list[dict[str, Any]] = []
    for result in search_results:
        text = " ".join([result.title, result.snippet, page_texts.get(result.url, "")]).lower()
        hits = sum(1 for keyword in keywords if keyword in text)
        negative_hits = sum(1 for term in negative_terms if term in text)
        scored.append(
            {
                "url": result.url,
                "title": result.title,
                "hits": hits,
                "negative_hits": negative_hits,
                "score": result.score,
                "query": result.query,
                "source_type": result.source_type,
                "source_domain": result.source_domain,
                "citation": result.citation or result.url,
            }
        )
    scored.sort(key=lambda row: (row["hits"], row["title"]), reverse=True)
    supporting = [row for row in scored if row["hits"] > 0][:3]

    if supporting and supporting[0]["hits"] >= max(2, min(4, len(keywords))):
        status = "confirmed" if len(supporting) >= 2 else "refined"
    elif supporting:
        status = "refined"
    elif any(row["negative_hits"] > 0 for row in scored):
        status = "contradicted"
    else:
        status = "unsupported"

    return {
        "claim_type": claim["claim_type"],
        "label": claim["label"],
        "status": status,
        "keywords": keywords,
        "supporting_results": supporting,
        "fit_confidence": claim.get("fit_confidence"),
        "contradiction_score": claim.get("contradiction_score"),
    }


def _status_bucket(status: str) -> str:
    if status == "confirmed":
        return "validated"
    if status == "refined":
        return "refined"
    if status.startswith("mcp_"):
        return "mcp_unresolved"
    if status == "contradicted":
        return "contradicted"
    return "unsupported"


def _render_validation_summary(date: str, validations: list[dict[str, Any]], trust: dict[str, Any]) -> list[str]:
    lines = [f"Validation summary for {date}:"]
    day_trust = trust.get("explain_day") or {}
    if day_trust:
        lines.append(
            "MCP trust: "
            f"fit_confidence={day_trust.get('fit_confidence')} "
            f"contradiction_score={day_trust.get('contradiction_score')} "
            f"unsupported_assets={', '.join(day_trust.get('unsupported_assets', [])) or 'none'}"
        )

    best_fit = next((item for item in validations if item.get("claim_type") == "best_fit_combination"), None)
    if best_fit:
        web = best_fit["web_validation"]
        lines.append(
            "Best-fit explanation: "
            f"{best_fit['label']} [{_status_bucket(web['status'])}]"
        )

    buckets: dict[str, list[dict[str, Any]]] = {
        "validated": [],
        "refined": [],
        "mcp_unresolved": [],
        "contradicted": [],
        "unsupported": [],
    }
    for item in validations:
        bucket = _status_bucket(item["web_validation"]["status"])
        buckets[bucket].append(item)

    for label, heading in [
        ("validated", "Validated claims"),
        ("refined", "Refined claims"),
        ("mcp_unresolved", "MCP unresolved"),
        ("contradicted", "Contradicted claims"),
        ("unsupported", "Unsupported claims"),
    ]:
        rows = buckets[label]
        if not rows:
            continue
        lines.append(f"{heading}:")
        for row in rows:
            web = row["web_validation"]
            if label == "mcp_unresolved":
                lines.append(f"- {row['label']}: {web.get('reason', 'MCP trust gate')}")
                continue
            support = web.get("supporting_results") or []
            if support:
                lead = support[0]
                lines.append(f"- {row['label']}: {lead['title']} (hits={lead['hits']})")
            else:
                lines.append(f"- {row['label']}: no supporting result")
    return lines


def _best_fit_item(validations: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((item for item in validations if item.get("claim_type") == "best_fit_combination"), None)


def _weakest_item(validations: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((item for item in validations if item.get("claim_type") == "weakest_core_asset"), None)


def _join_or_none(items: list[str]) -> str:
    return ", ".join(items) if items else "none"


def _classify_question(question: str) -> str:
    normalized = " ".join(question.lower().split())
    for question_type, patterns in QUESTION_PATTERNS:
        if any(pattern in normalized for pattern in patterns):
            return question_type
    return "fallback"


def _humanize_combination(label: str) -> str:
    replacements = {
        "Fed policy repricing": "Fed/dollar restraint",
        "Middle East geopolitical repricing": "Middle East risk unwind",
        "trade-and-sanctions repricing": "trade/sanctions pressure on energy",
        "oil/geopolitical premium repricing": "oil/geopolitical premium unwind",
        "inflation-relief repricing": "inflation-relief impulse",
        "rate-path repricing": "rate-path repricing",
    }
    parts = [part.strip() for part in label.split("+")]
    human = [replacements.get(part, part) for part in parts]
    return " + ".join(human)


def _model_fit_phrase(trust: dict[str, Any]) -> str:
    fit = trust.get("fit_confidence")
    contradiction = trust.get("contradiction_score")
    if fit is None and contradiction is None:
        return "The model fit is not available."
    if fit is not None and contradiction is not None:
        return (
            f"The model fit is strong overall, with low internal contradiction "
            f"(fit metric {fit:.2f}; contradiction metric {contradiction:.2f})."
        )
    if fit is not None:
        return f"The model fit is strong overall (fit metric {fit:.2f})."
    return f"The model shows low internal contradiction (contradiction metric {contradiction:.2f})."


def _transmission_summary(best_label: str, unsupported_assets: list[str], transmission_chains: list[dict[str, Any]] | None = None) -> str:
    if transmission_chains:
        label_lower = best_label.lower()

        def rank(row: dict[str, Any]) -> tuple[int, int]:
            factor = row.get("factor", "") or row.get("factor_label", "")
            if (
                ("middle east geopolitical repricing" in label_lower or "trade-and-sanctions repricing" in label_lower)
                and factor in {"war_conflict", "oil", "shipping_disruption", "sanctions_trade"}
            ):
                return (3, 0)
            if "oil/geopolitical premium repricing" in label_lower and factor == "oil":
                return (3, 0)
            if factor in {"war_conflict", "oil", "shipping_disruption", "sanctions_trade"}:
                return (2, 0)
            if factor == "central_bank_policy":
                return (1, 0)
            return (0, 0)

        preferred = max(transmission_chains, key=rank)
        preferred_chain = preferred["chain"].strip()
        fed_chain = next(
            (
                row["chain"].strip()
                for row in transmission_chains
                if (row.get("factor") or row.get("factor_label")) == "central_bank_policy"
            ),
            None,
        )
        if (
            preferred.get("factor") in {"war_conflict", "oil", "shipping_disruption", "sanctions_trade"}
            and fed_chain
            and fed_chain != preferred_chain
        ):
            summary = (
                f"{preferred_chain}, while stronger dollar / Fed restraint prevented that from becoming a clean risk-on day"
            )
        else:
            summary = preferred_chain
        if unsupported_assets:
            summary += f" The main caveat is that {_join_or_none(unsupported_assets)} remain unresolved in the local slice."
        return summary[:1].upper() + summary[1:]
    label_lower = best_label.lower()
    if (
        "middle east geopolitical repricing" in label_lower
        and "trade-and-sanctions repricing" in label_lower
        and "fed policy repricing" in label_lower
    ):
        summary = (
            "Middle East/oil-shipping risk premium unwound first, pulling crude lower and easing inflation pressure, "
            "while Fed/dollar restraint prevented that from becoming a clean risk-on day."
        )
    elif "oil/geopolitical premium repricing" in label_lower and "fed policy repricing" in label_lower:
        summary = (
            "Oil/geopolitical premium unwound first, easing inflation pressure and yields, "
            "but the Fed/dollar channel blocked a clean risk-on response."
        )
    elif "fed policy repricing" in label_lower:
        summary = (
            "Fed-path and dollar restraint dominated the session, shaping rates and broader cross-asset pricing."
        )
    else:
        summary = f"The dominant validated regime was {best_label}."

    if unsupported_assets:
        summary += f" The main caveat is that {_join_or_none(unsupported_assets)} remain unresolved in the local slice."
    return summary


def _answer_question(question: str, date: str, payload: dict[str, Any], evidence: dict[str, Any]) -> str:
    trust = payload["stage1"]["trust"]["explain_day"]
    validations = evidence["validation_refinements"]
    best_fit = _best_fit_item(validations)
    weakest = evidence.get("weakest_asset")
    unsupported_assets = list(evidence.get("unsupported_assets") or [])
    cannot_answer = list(evidence.get("cannot_answer") or [])
    best_fit_frame = evidence.get("best_fit_combination") or {}
    best_label = (
        (best_fit["label"] if best_fit else None)
        or best_fit_frame.get("label")
        or "No supported best-fit explanation was available"
    )
    best_status = best_fit["web_validation"]["status"] if best_fit else "unsupported"
    best_support = (best_fit["web_validation"].get("supporting_results") or []) if best_fit else []
    best_support_title = best_support[0]["title"] if best_support else "no supporting result"
    question_type = _classify_question(question)
    transmission_chains = evidence.get("transmission_chains") or evidence.get("transmission_rows") or []
    market_impact_rows = evidence.get("market_impact_rows") or []

    if question_type == "best_explanation":
        dominant = evidence.get("dominant_narrative") or {}
        best_explanation = evidence.get("best_explanation") or {}
        human_label = _humanize_combination(best_label)
        return (
            f"{(dominant.get('summary') or best_explanation.get('summary') or _transmission_summary(best_label, unsupported_assets, transmission_chains))} "
            f"In the model, this reads as a combined {human_label} story, with external validation {best_status} "
            f"rather than contradicted and lead support from {best_support_title}. "
            f"{_model_fit_phrase(trust)}"
        )

    if question_type == "contradictory_asset":
        if weakest:
            weakest_label = weakest if isinstance(weakest, str) else weakest.get("label")
            validation = {} if isinstance(weakest, str) else (weakest.get("validation") or {})
            reason = validation.get("reason", "weakest fit in the MCP trust block")
            return (
                f"{weakest_label} most contradicts the dominant narrative. "
                f"It is currently classified as {validation.get('status', 'unresolved')}: {reason}."
            )
        return "No clear contradictory asset was identified from the current MCP slice."

    if question_type == "contradiction_evidence":
        if trust.get("contradiction_score", 0.0) <= 0.1:
            weakest_label = weakest if isinstance(weakest, str) else (weakest.get("label") if weakest else None)
            return (
                f"No strong contradiction breaks the combined explanation. "
                f"The main tension is unresolved coverage in {_join_or_none(unsupported_assets)}"
                + (f", with weakest fit in {weakest_label}." if weakest_label else ".")
            )
        return (
            f"The main contradiction signal is contradiction_score={trust.get('contradiction_score')}. "
            f"Assets still weakening the story are {_join_or_none(unsupported_assets)}."
        )

    if question_type == "unexplained":
        return (
            f"Unresolved areas are {_join_or_none(unsupported_assets)}. "
            f"Current cannot-answer cases: {_join_or_none(cannot_answer)}."
        )

    if question_type == "reaction_balance":
        if trust.get("contradiction_score", 0.0) <= 0.15:
            dominant_impact = market_impact_rows[0]["label"] if market_impact_rows else best_label
            return (
                f"The current slice suggests partial underreaction rather than a clean overreaction. {dominant_impact} propagated cleanly, "
                f"The dominant narrative fits well overall, but {_join_or_none(unsupported_assets)} still lack enough direct support."
            )
        return (
            "The current slice is too contradictory to call overreaction cleanly. "
            f"Contradiction score is {trust.get('contradiction_score')}."
        )

    if question_type == "assumptions":
        return (
            f"The explanation relies on three assumptions: the best-fit combination {best_label} was the real regime driver, "
            f"the unresolved assets {_join_or_none(unsupported_assets)} do not overturn it, and the external refinement status {best_status} is directionally reliable rather than decisive."
        )

    return (
        f"No direct question template matched. Best-fit explanation is {best_label} "
        f"with status {best_status}; unresolved assets are {_join_or_none(unsupported_assets)}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--mcp-script", default=str(DEFAULT_MCP))
    parser.add_argument("--date", required=True)
    parser.add_argument(
        "--universe",
        nargs="+",
        default=["WTI", "Gold", "US2Y", "US10Y", "DXY", "NDX", "SPX"],
    )
    parser.add_argument("--search-results", type=int, default=5)
    parser.add_argument("--fetch-pages", type=int, default=3)
    parser.add_argument("--collection")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--truncate-dim", type=int, default=DEFAULT_TRUNCATE_DIM)
    parser.add_argument("--local-limit", type=int, default=DEFAULT_LOCAL_LIMIT)
    parser.add_argument("--local-candidate-limit", type=int, default=DEFAULT_LOCAL_CANDIDATE_LIMIT)
    parser.add_argument("--codex-model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--hybrid-search-url", default=DEFAULT_HYBRID_SEARCH_URL)
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db = str(Path(args.db))
    mcp_path = Path(args.mcp_script)
    collection = args.collection or _default_collection_name(args.date)

    explain_day = call_mcp_tool(
        "explain_day",
        {"db": db, "date": args.date, "universe": args.universe},
        mcp_path,
    )
    cross_asset = call_mcp_tool(
        "explain_cross_asset_move",
        {"db": db, "date": args.date, "assets": args.universe[:7]},
        mcp_path,
    )
    narrative_frame = json.loads(
        call_mcp_tool(
            "build_narrative_frame",
            {"db": db, "date": args.date, "universe": args.universe[:7]},
            mcp_path,
        )
    )
    explain_day_trust = _extract_trust_summary(explain_day)
    cross_asset_trust = _extract_trust_summary(cross_asset)
    contradictory = call_mcp_tool(
        "find_contradictory_assets",
        {"db": db, "date": args.date, "universe": args.universe[:7]},
        mcp_path,
    )
    claims = _extract_stage1_claims(explain_day, cross_asset, args.date)
    weakest_asset = cross_asset_trust.get("weakest_core_asset") or explain_day_trust.get("weakest_core_asset")
    asset_day_context = (
        call_mcp_tool(
            "explain_asset_via_day_context",
            {"db": db, "date": args.date, "asset_label": weakest_asset, "universe": args.universe[:7]},
            mcp_path,
        )
        if weakest_asset
        else None
    )

    validations: list[dict[str, Any]] = []
    for claim in claims:
        expanded_queries = expand_claim_queries(claim, args.codex_model)
        local_results: list[SearchResult] = []
        try:
            search_fn = (
                (lambda query: search_remote_hybrid(
                    endpoint=args.hybrid_search_url,
                    query=query,
                    collection=collection,
                    qdrant_url=args.qdrant_url,
                    embedding_model=args.embedding_model,
                    truncate_dim=args.truncate_dim,
                    limit=args.local_limit,
                    candidate_limit=args.local_candidate_limit,
                    codex_model=args.codex_model,
                ))
                if args.hybrid_search_url
                else
                (lambda query: search_local_hybrid(
                    query=query,
                    collection=collection,
                    qdrant_url=args.qdrant_url,
                    embedding_model=args.embedding_model,
                    truncate_dim=args.truncate_dim,
                    limit=args.local_limit,
                    candidate_limit=args.local_candidate_limit,
                    codex_model=args.codex_model,
                ))
            )
            per_query_local = [search_fn(query) for query in expanded_queries]
            local_results = merge_local_results(
                claim["query"],
                per_query_local,
                args.codex_model,
                args.local_limit,
            )
        except Exception:
            local_results = []

        results = local_results
        page_texts: dict[str, str] = {}
        if not results:
            results = search_results_for_query(claim["query"], args.search_results)
            for result in results[: args.fetch_pages]:
                try:
                    page_texts[result.url] = fetch_page_text(result.url)
                except Exception:
                    continue
        validations.append(
            {
                **claim,
                "expanded_queries": expanded_queries,
                "validation_mode": "local_hybrid_first" if local_results else "web_fallback",
                "web_validation": assess_claim(claim, results, page_texts),
            }
        )

    payload = {
        "date": args.date,
        "stage1": {
            "explain_day": explain_day,
            "explain_cross_asset_move": cross_asset,
            "narrative_frame": narrative_frame,
            "trust": {
                "explain_day": explain_day_trust,
                "explain_cross_asset_move": cross_asset_trust,
            },
            "claims": claims,
        },
        "stage2": validations,
    }
    evidence = _frame_to_evidence(narrative_frame, validations, payload["stage1"]["trust"])
    evidence["contradictory_text"] = contradictory
    evidence["asset_context_text"] = asset_day_context
    payload["stage1"]["evidence"] = evidence

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    if args.question:
        print(f"Desk answers for {args.date}")
        for question in args.question:
            print(f"Q: {question}")
            print(f"A: {_answer_question(question, args.date, payload, evidence)}")
        print()

    for line in _render_validation_summary(args.date, validations, payload["stage1"]["trust"]):
        print(line)
    print()
    print(f"Stage 1 MCP review for {args.date}")
    print(explain_day)
    print()
    print(cross_asset)
    print()
    print("Stage 2 web validation")
    for item in validations:
        validation = item["web_validation"]
        print(f"- {item['claim_type']}: {item['label']}")
        print(f"  status={validation['status']} keywords={', '.join(validation['keywords']) or 'none'}")
        for result in validation["supporting_results"]:
            print(f"  - {result['title']} | {result['url']} | hits={result['hits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
