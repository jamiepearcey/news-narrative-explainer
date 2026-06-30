#!/usr/bin/env python3
"""v3 narrative validation client using the hosted-parquet MCP and hybrid retrieval."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from v3_remote_parquet_store import DEFAULT_HTTP_ROOT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = DEFAULT_HTTP_ROOT
DEFAULT_MCP = Path(__file__).resolve().parent / "v3_narrative_explainer_mcp.py"
DEFAULT_V3_DIR = Path(__file__).resolve().parent
SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/?q={query}"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}"
USER_AGENT = "news-narrative-explainer-v3-validator/0.1"
DEFAULT_QDRANT_URL = "http://127.0.0.1:6334"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_EMBEDDING_MODEL = "all-minilm"
DEFAULT_TRUNCATE_DIM = 256
DEFAULT_LOCAL_LIMIT = 8
DEFAULT_LOCAL_CANDIDATE_LIMIT = 64
DEFAULT_CODEX_MODEL = "gpt-5.4-mini"
DEFAULT_HYBRID_SEARCH_URL = ""
DEFAULT_JUDGE_SHORTLIST = 5
PHASED_SOURCE_SCORE_THRESHOLDS = (0.95, 0.85, 0.0)
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


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0
    query: str = ""
    source_type: str = ""
    source_domain: str = ""
    source_score: float = 0.0
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
    trust["fit_confidence"] = float(match.group("fit"))
    trust["contradiction_score"] = float(match.group("contradiction"))
    trust["unsupported_assets"] = [] if unsupported_raw == "none" else [part.strip() for part in unsupported_raw.split(",") if part.strip()]
    if match.group("weakest"):
        trust["weakest_core_asset"] = match.group("weakest").strip()
    unsupported_match = re.search(r"Unsupported / cannot answer: (?P<body>.+?)\.", text)
    if unsupported_match:
        body = unsupported_match.group("body").strip()
        trust["cannot_answer"] = [] if body == "none" else [part.strip() for part in body.split(";") if part.strip()]
    return trust


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

    weakest_asset = cross_asset_trust.get("weakest_core_asset") or explain_day_trust.get("weakest_core_asset")
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


def _run_codex_json(prompt: str, schema: dict[str, Any], model: str, timeout_seconds: int = 45) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="codex-v3-validate-") as tmpdir:
        schema_path = Path(tmpdir) / "schema.json"
        output_path = Path(tmpdir) / "output.json"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        proc = subprocess.run(
            [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "--ignore-user-config",
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
                "-",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            input=prompt,
            text=True,
            timeout=timeout_seconds,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"codex exec failed with exit status {proc.returncode}: {proc.stderr.strip()}")
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
        "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
        "required": ["queries"],
        "additionalProperties": False,
    }
    try:
        payload = _run_codex_json(prompt, schema, model, timeout_seconds=20)
        queries = [base_query]
        for query in payload.get("queries", []):
            value = str(query).strip()
            if value and value not in queries:
                queries.append(value)
            if len(queries) >= max_queries:
                break
        return queries
    except Exception:
        return [base_query]


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


def search_duckduckgo(query: str, max_results: int) -> list[SearchResult]:
    request = Request(SEARCH_ENDPOINT.format(query=quote_plus(query)), headers={"User-Agent": USER_AGENT})
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
    request = Request(GOOGLE_NEWS_RSS.format(query=quote_plus(query)), headers={"User-Agent": USER_AGENT})
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
        results.append(SearchResult(title=title, url=url, snippet=snippet, citation=url))
        if len(results) >= max_results:
            break
    return results


def search_results_for_query(query: str, max_results: int) -> list[SearchResult]:
    try:
        results = search_duckduckgo(query, max_results)
        if results:
            return results
    except Exception:
        pass
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
    min_source_score: float | None = None,
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
            *(["--min-source-score", str(min_source_score)] if min_source_score is not None else []),
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
    return _payload_hits_to_results(payload, query)


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
    min_source_score: float | None = None,
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
            "min_source_score": min_source_score,
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
    return _payload_hits_to_results(payload, query)


def _payload_hits_to_results(payload: dict[str, Any], query: str) -> list[SearchResult]:
    results: list[SearchResult] = []
    for hit in payload.get("hits", []):
        url = str(hit.get("document_identifier") or "").strip()
        title = str(hit.get("title") or "").strip() or url
        snippet = next((str(value).strip() for value in (hit.get("market_context_text"), hit.get("summary_text")) if value), "")
        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                score=float(hit.get("score") or hit.get("fused_score") or 0.0),
                query=query,
                source_type=str(hit.get("source_type") or ""),
                source_domain=str(hit.get("source_domain") or ""),
                source_score=float(hit.get("source_score") or 0.0),
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
                    source_score=result.source_score,
                    citation=result.citation,
                )
            else:
                existing = merged[key]
                existing.score = max(existing.score, result.score + score_bonus)
                if len(result.snippet) > len(existing.snippet):
                    existing.snippet = result.snippet
                existing.source_score = max(existing.source_score, result.source_score)
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
    return list(dict.fromkeys(token.lower() for token in text.split() if len(token) >= 4))[:8]


def _claim_text_for_llm(claim: dict[str, Any]) -> str:
    parts = [str(claim.get("label") or "").strip()]
    factor = str(claim.get("factor") or "").strip()
    if factor:
        parts.append(f"factor={factor}")
    matched_assets = str(claim.get("matched_assets") or "").strip()
    if matched_assets:
        parts.append(f"matched_assets={matched_assets}")
    contradictions = str(claim.get("contradictions") or "").strip()
    if contradictions:
        parts.append(f"contradictions={contradictions}")
    return " | ".join(part for part in parts if part)


def _effective_source_score(result: SearchResult) -> float:
    score = float(result.source_score or 0.0)
    if score > 0.0:
        return score
    kind = str(result.source_type or "").lower().strip()
    if kind == "commodity_specialist":
        return 0.8
    if kind == "market_wrap":
        return 0.85
    if kind == "company_specific":
        return 0.45
    return 0.55


def _candidate_shortlist_score(claim: dict[str, Any], result: SearchResult) -> float:
    claim_keywords = _claim_keywords(claim)
    text = " ".join(
        [
            str(result.title or "").lower(),
            str(result.snippet or "").lower(),
            str(result.source_type or "").lower(),
            str(result.source_domain or "").lower(),
        ]
    )
    keyword_hits = sum(1 for keyword in claim_keywords if keyword in text)
    title_hits = sum(1 for keyword in claim_keywords if keyword in str(result.title or "").lower())
    finance_bonus = _effective_source_score(result) * 5.0
    if result.source_type in {"macro", "policy"}:
        finance_bonus += 1.0
    return (
        (title_hits * 5.0)
        + (keyword_hits * 2.0)
        + finance_bonus
        + float(result.score or 0.0)
    )


def phased_hybrid_search(
    search_fn: Any,
    query: str,
    top_k: int,
    thresholds: tuple[float, ...] = PHASED_SOURCE_SCORE_THRESHOLDS,
) -> list[SearchResult]:
    merged: list[SearchResult] = []
    seen: set[str] = set()
    for threshold in thresholds:
        results = search_fn(query, threshold)
        for result in results:
            key = result.url or result.title
            if not key or key in seen:
                continue
            seen.add(key)
            boosted = SearchResult(**{**result.__dict__, "score": float(result.score or 0.0) + threshold})
            merged.append(boosted)
            if len(merged) >= top_k:
                return merged
    return merged


def _shortlist_evidence_for_claim(
    claim: dict[str, Any],
    search_results: list[SearchResult],
    limit: int = DEFAULT_JUDGE_SHORTLIST,
) -> list[SearchResult]:
    ranked = sorted(
        search_results,
        key=lambda result: (
            _candidate_shortlist_score(claim, result),
            float(result.score or 0.0),
            result.title,
        ),
        reverse=True,
    )
    deduped: list[SearchResult] = []
    seen: set[str] = set()
    for result in ranked:
        key = result.url or result.title
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(result)
        if len(deduped) >= limit:
            break
    return deduped


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
            "reason": f"stage-one fit_confidence {fit_confidence:.2f} is too low for useful validation",
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


def _heuristic_assess_claim(
    claim: dict[str, Any],
    search_results: list[SearchResult],
    page_texts: dict[str, str],
) -> dict[str, Any]:
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
                "reason": "heuristic_fallback",
            }
        )
    scored.sort(key=lambda row: (row["hits"], row["score"], row["title"]), reverse=True)
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
        "judge": "heuristic_fallback",
    }


def _llm_assess_claim(
    claim: dict[str, Any],
    search_results: list[SearchResult],
    page_texts: dict[str, str],
    model: str,
) -> dict[str, Any]:
    shortlist = _shortlist_evidence_for_claim(claim, search_results, DEFAULT_JUDGE_SHORTLIST)
    candidates = []
    for idx, result in enumerate(shortlist, start=1):
        candidates.append(
            {
                "evidence_id": f"E{idx}",
                "title": result.title,
                "url": result.url,
                "citation": result.citation or result.url,
                "snippet": result.snippet,
                "source_type": result.source_type,
                "source_domain": result.source_domain,
                "retrieval_score": result.score,
                "page_text_excerpt": page_texts.get(result.url, "")[:2000],
            }
        )
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["confirmed", "refined", "contradicted", "unsupported"]},
            "reason": {"type": "string"},
            "supporting_evidence_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["status", "reason", "supporting_evidence_ids"],
        "additionalProperties": False,
    }
    prompt = (
        "You are validating a finance market-narrative claim against retrieved evidence.\n"
        "Return one of:\n"
        "- confirmed: evidence directly supports the claim across multiple strong items\n"
        "- refined: evidence partly supports the claim or supports a narrower variant\n"
        "- contradicted: evidence materially points against the claim\n"
        "- unsupported: not enough relevant evidence\n"
        "Only cite evidence ids present in the input. Prefer high-specificity finance/macroeconomic evidence.\n"
        f"Claim JSON:\n{json.dumps(claim, ensure_ascii=True)}\n"
        f"Claim text: {_claim_text_for_llm(claim)}\n"
        f"Evidence JSON:\n{json.dumps(candidates, ensure_ascii=True)}"
    )
    payload = _run_codex_json(prompt, schema, model, timeout_seconds=45)
    by_id = {candidate["evidence_id"]: candidate for candidate in candidates}
    supporting_results: list[dict[str, Any]] = []
    for evidence_id in [str(value) for value in payload.get("supporting_evidence_ids", [])]:
        candidate = by_id.get(evidence_id)
        if candidate is None:
            continue
        supporting_results.append(
            {
                "evidence_id": evidence_id,
                "url": candidate["url"],
                "title": candidate["title"],
                "citation": candidate["citation"],
                "score": candidate["retrieval_score"],
                "query": "",
                "source_type": candidate["source_type"],
                "source_domain": candidate["source_domain"],
                "hits": None,
                "negative_hits": None,
                "reason": str(payload.get("reason") or "").strip(),
            }
        )
    return {
        "claim_type": claim["claim_type"],
        "label": claim["label"],
        "status": str(payload["status"]),
        "keywords": _claim_keywords(claim),
        "supporting_results": supporting_results[:3],
        "fit_confidence": claim.get("fit_confidence"),
        "contradiction_score": claim.get("contradiction_score"),
        "reason": str(payload.get("reason") or "").strip(),
        "judge": "codex",
    }


def assess_claim(
    claim: dict[str, Any],
    search_results: list[SearchResult],
    page_texts: dict[str, str],
    model: str | None = None,
) -> dict[str, Any]:
    gated = _trust_gate_claim(claim)
    if gated is not None:
        return gated
    if model and search_results:
        try:
            return _llm_assess_claim(claim, search_results, page_texts, model)
        except Exception as exc:
            heuristic = _heuristic_assess_claim(claim, search_results, page_texts)
            heuristic["judge"] = "codex_failed_heuristic_fallback"
            heuristic["reason"] = f"codex judge failed: {exc}"
            return heuristic
    heuristic = _heuristic_assess_claim(claim, search_results, page_texts)
    if model and not search_results:
        heuristic["judge"] = "codex_skipped_no_results"
        heuristic["reason"] = "No retrieval candidates survived for LLM judging."
    return heuristic


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
    buckets: dict[str, list[dict[str, Any]]] = {
        "validated": [],
        "refined": [],
        "mcp_unresolved": [],
        "contradicted": [],
        "unsupported": [],
    }
    for item in validations:
        bucket = _status_bucket(item["validation"]["status"])
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
            validation = row["validation"]
            if label == "mcp_unresolved":
                lines.append(f"- {row['label']}: {validation.get('reason', 'MCP trust gate')}")
                continue
            support = validation.get("supporting_results") or []
            if support:
                lead = support[0]
                hits = lead.get("hits")
                hit_text = f"hits={hits}, " if hits is not None else ""
                lines.append(f"- {row['label']}: {lead['title']} ({hit_text}citation={lead['citation']})")
            else:
                lines.append(f"- {row['label']}: no supporting result")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--mcp-script", default=str(DEFAULT_MCP))
    parser.add_argument("--date", required=True)
    parser.add_argument("--universe", nargs="+", default=["WTI", "Gold", "US2Y", "US10Y", "DXY", "NDX", "SPX"])
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
    parser.add_argument("--enable-query-expansion", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mcp_path = Path(args.mcp_script)
    collection = args.collection or _default_collection_name(args.date)

    explain_day = call_mcp_tool("explain_day", {"db": args.db, "date": args.date, "universe": args.universe}, mcp_path)
    cross_asset = call_mcp_tool("explain_cross_asset_move", {"db": args.db, "date": args.date, "assets": args.universe[:7]}, mcp_path)
    narrative_frame = json.loads(call_mcp_tool("build_narrative_frame", {"db": args.db, "date": args.date, "universe": args.universe[:7]}, mcp_path))
    contradictory = call_mcp_tool("find_contradictory_assets", {"db": args.db, "date": args.date, "universe": args.universe[:7]}, mcp_path)
    explain_day_trust = _extract_trust_summary(explain_day)
    cross_asset_trust = _extract_trust_summary(cross_asset)
    claims = _extract_stage1_claims(explain_day, cross_asset, args.date)

    validations: list[dict[str, Any]] = []
    for claim in claims:
        expanded_queries = expand_claim_queries(claim, args.codex_model) if args.enable_query_expansion else [claim["query"]]
        local_results: list[SearchResult] = []
        try:
            search_fn = (
                (
                    lambda query, min_source_score: search_remote_hybrid(
                        args.hybrid_search_url,
                        query,
                        collection,
                        args.qdrant_url,
                        args.embedding_model,
                        args.truncate_dim,
                        args.local_limit,
                        args.local_candidate_limit,
                        args.codex_model,
                        min_source_score,
                    )
                )
                if args.hybrid_search_url
                else
                (
                    lambda query, min_source_score: search_local_hybrid(
                        query,
                        collection,
                        args.qdrant_url,
                        args.embedding_model,
                        args.truncate_dim,
                        args.local_limit,
                        args.local_candidate_limit,
                        args.codex_model,
                        min_source_score,
                    )
                )
            )
            per_query_local = [phased_hybrid_search(search_fn, query, args.local_limit) for query in expanded_queries]
            local_results = merge_local_results(claim["query"], per_query_local, args.codex_model, args.local_limit)
        except Exception:
            local_results = []

        results = local_results
        page_texts: dict[str, str] = {}
        validation_mode = "local_hybrid_first"
        if not results:
            validation_mode = "web_fallback"
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
                "validation_mode": validation_mode,
                "validation": assess_claim(claim, results, page_texts, args.codex_model),
            }
        )

    payload = {
        "date": args.date,
        "stage1": {
            "explain_day": explain_day,
            "explain_cross_asset_move": cross_asset,
            "narrative_frame": narrative_frame,
            "contradictory_text": contradictory,
            "trust": {"explain_day": explain_day_trust, "explain_cross_asset_move": cross_asset_trust},
            "claims": claims,
        },
        "stage2": validations,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    for line in _render_validation_summary(args.date, validations, payload["stage1"]["trust"]):
        print(line)
    print()
    print(f"Stage 1 MCP review for {args.date}")
    print(explain_day)
    print()
    print(cross_asset)
    print()
    print("Stage 2 validation")
    for item in validations:
        validation = item["validation"]
        print(f"- {item['claim_type']}: {item['label']}")
        print(f"  mode={item['validation_mode']} status={validation['status']} judge={validation.get('judge', 'unknown')} keywords={', '.join(validation['keywords']) or 'none'}")
        for result in validation["supporting_results"]:
            extra = f" | hits={result['hits']}" if result.get("hits") is not None else ""
            print(f"  - {result['title']} | {result['citation']}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
