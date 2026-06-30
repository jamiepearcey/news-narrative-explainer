#!/usr/bin/env python3
"""Experimental Qdrant retrieval over bronze narrative docs.

This path uses:

- pluggable embedding backends with a local-first default
- optional shortened embedding dimensions when the backend supports them
- Qdrant TurboQuant for compressed ANN search
- Qdrant quantization oversampling + rescoring on stored higher-precision vectors

It is intentionally additive and does not change the deterministic narrative
graph or MCP explanation path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import uuid
from datetime import date
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _duckdb_bootstrap import ensure_duckdb

ensure_duckdb(__file__)

import duckdb

from parquet_narrative_store import resolve_query_db


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "narrative_graph_parquet"
DEFAULT_COLLECTION = "news_narrative_bronze"
DEFAULT_BATCH_SIZE = 64
DEFAULT_LIMIT = 10
DEFAULT_OVERSAMPLING = 2.0
DEFAULT_HNSW_EF = 128
DEFAULT_TURBO_BITS = "bits2"
DEFAULT_MAX_EMBED_CHARS = 6000
DEFAULT_RERANK_LIMIT = 8
DEFAULT_EMBEDDING_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "embeddinggemma"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OPENAI_MODEL = "text-embedding-3-large"
DEFAULT_SENTENCE_TRANSFORMERS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CODEX_RERANK_MODEL = "gpt-5.4-mini"
SENTENCE_TRANSFORMER_CACHE: dict[str, Any] = {}


def ensure_runtime_packages(provider: str) -> None:
    try:
        import qdrant_client  # noqa: F401
    except ModuleNotFoundError:
        _reexec_with_packages(["qdrant-client>=1.15"])
    if provider == "openai":
        try:
            import openai  # noqa: F401
        except ModuleNotFoundError:
            _reexec_with_packages(["qdrant-client>=1.15", "openai>=1.35"])
    elif provider == "sentence-transformers":
        try:
            import sentence_transformers  # noqa: F401
        except ModuleNotFoundError:
            _reexec_with_packages(["qdrant-client>=1.15", "sentence-transformers>=3.0"])


def ensure_rerank_runtime(provider: str) -> None:
    if provider == "none":
        return
    if provider == "codex-sdk":
        try:
            import openai_codex  # noqa: F401
        except ModuleNotFoundError:
            _reexec_with_packages(["qdrant-client>=1.15", "openai-codex"])


def _reexec_with_packages(packages: list[str]) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "required runtime packages are missing, and `uv` was not found "
            "to bootstrap them automatically"
        ) from None
    command = [uv, "run", "--with", "duckdb>=1.0"]
    for package in packages:
        command.extend(["--with", package])
    command.extend([str(Path(__file__).resolve()), *sys.argv[1:]])
    os.execvp(uv, command)


def _default_embedding_model(provider: str) -> str:
    if provider == "ollama":
        return DEFAULT_OLLAMA_MODEL
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    if provider == "sentence-transformers":
        return DEFAULT_SENTENCE_TRANSFORMERS_MODEL
    raise ValueError(f"unsupported embedding provider: {provider}")


def _resolve_embedding_model(provider: str, explicit_model: str | None) -> str:
    return explicit_model or _default_embedding_model(provider)


def _truncate_vector(vector: list[float], dimensions: int | None) -> list[float]:
    if dimensions is None:
        return vector
    if dimensions <= 0:
        raise ValueError("dimensions must be positive when provided")
    if len(vector) < dimensions:
        raise ValueError(
            f"requested dimensions {dimensions} exceed embedding width {len(vector)}"
        )
    return vector[:dimensions]


def _date_to_ordinal(value: str) -> int:
    return date.fromisoformat(value).toordinal()


def _stable_point_id(row: dict[str, Any]) -> str:
    raw = "\0".join(
        [
            str(row.get("doc_id") or ""),
            str(row.get("partition_date") or ""),
            str(row.get("document_identifier") or ""),
        ]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def _parse_theme_tags(value: str | None) -> list[str]:
    if not value:
        return []
    tags: list[str] = []
    for part in value.split(";"):
        token = part.split(",")[0].strip().upper()
        if token:
            tags.append(token)
    return tags


def _clean_text(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = " ".join(str(value).split())
    return cleaned or None


def _embedding_text(row: dict[str, Any], max_chars: int) -> str:
    segments: list[str] = []
    title = _clean_text(row.get("title"))
    if title:
        segments.append(f"Title: {title}")
    market_context = _clean_text(row.get("market_context_text"))
    if market_context:
        segments.append(f"Market context: {market_context}")
    summary = _clean_text(row.get("summary_text"))
    if summary:
        segments.append(f"Summary: {summary}")
    relevant = _clean_text(row.get("relevant_text"))
    if relevant:
        segments.append(f"Relevant text: {relevant}")
    else:
        body = _clean_text(row.get("body_text"))
        if body:
            segments.append(f"Body: {body}")
    organizations = _clean_text(row.get("v2_organizations"))
    if organizations:
        segments.append(f"Organizations: {organizations}")
    persons = _clean_text(row.get("v2_persons"))
    if persons:
        segments.append(f"Persons: {persons}")
    text = "\n".join(segments).strip()
    if not text:
        text = str(row.get("document_identifier") or "")
    return text[:max_chars]


def _row_payload(row: dict[str, Any], max_embed_chars: int) -> dict[str, Any]:
    partition_date = str(row["partition_date"])
    return {
        "doc_id": int(row["doc_id"]),
        "partition_date": partition_date,
        "partition_ordinal": _date_to_ordinal(partition_date),
        "document_identifier": row["document_identifier"],
        "source_domain": row.get("source_domain"),
        "source_type": row.get("source_type"),
        "title": row.get("title"),
        "summary_text": row.get("summary_text"),
        "market_context_text": row.get("market_context_text"),
        "market_context_score": float(row.get("market_context_score") or 0.0),
        "theme_tags": _parse_theme_tags(row.get("v2_themes")),
        "embedding_text": _embedding_text(row, max_embed_chars),
    }


def load_bronze_rows(
    db_path: Path,
    start_date: str | None,
    end_date: str | None,
    row_limit: int | None,
) -> list[dict[str, Any]]:
    sql = """
        SELECT
            doc_id,
            CAST(partition_date AS VARCHAR) AS partition_date,
            source_domain,
            source_type,
            document_identifier,
            title,
            summary_text,
            body_text,
            relevant_text,
            market_context_text,
            market_context_score,
            v2_themes,
            v2_persons,
            v2_organizations
        FROM bronze_candidates
        WHERE (? IS NULL OR partition_date >= CAST(? AS DATE))
          AND (? IS NULL OR partition_date <= CAST(? AS DATE))
        ORDER BY partition_date ASC, doc_id ASC
    """
    if row_limit is not None:
        sql += f" LIMIT {int(row_limit)}"
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        cursor = con.execute(sql, [start_date, start_date, end_date, end_date])
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    finally:
        con.close()


def _chunked(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for index in range(0, len(values), batch_size):
        yield values[index : index + batch_size]


def _build_filter(
    start_date: str | None,
    end_date: str | None,
    source_domains: list[str],
    source_types: list[str],
    theme_tags: list[str],
) -> Any:
    from qdrant_client.http import models

    must: list[Any] = []
    if start_date or end_date:
        must.append(
            models.FieldCondition(
                key="partition_ordinal",
                range=models.Range(
                    gte=_date_to_ordinal(start_date) if start_date else None,
                    lte=_date_to_ordinal(end_date) if end_date else None,
                ),
            )
        )
    if source_domains:
        must.append(
            models.FieldCondition(
                key="source_domain",
                match=models.MatchAny(any=source_domains),
            )
        )
    if source_types:
        must.append(
            models.FieldCondition(
                key="source_type",
                match=models.MatchAny(any=source_types),
            )
        )
    if theme_tags:
        must.append(
            models.FieldCondition(
                key="theme_tags",
                match=models.MatchAny(any=[tag.upper() for tag in theme_tags]),
            )
        )
    if not must:
        return None
    return models.Filter(must=must)


def _qdrant_client(url: str, api_key: str | None) -> Any:
    from qdrant_client import QdrantClient

    if url == ":memory:":
        return QdrantClient(location=":memory:")
    if url.startswith("local:"):
        return QdrantClient(path=url.removeprefix("local:"))
    return QdrantClient(url=url, api_key=api_key)


def _openai_client(api_key: str | None) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def _embed_texts(
    provider: str,
    model: str,
    dimensions: int | None,
    texts: list[str],
    openai_api_key: str | None,
    ollama_url: str,
) -> list[list[float]]:
    if provider == "openai":
        client = _openai_client(openai_api_key)
        request: dict[str, Any] = {
            "model": model,
            "input": texts,
        }
        if dimensions is not None:
            request["dimensions"] = dimensions
        response = client.embeddings.create(**request)
        return [list(item.embedding) for item in response.data]
    if provider == "ollama":
        return _ollama_embed_texts(model, dimensions, texts, ollama_url)
    if provider == "sentence-transformers":
        from sentence_transformers import SentenceTransformer

        model_instance = SENTENCE_TRANSFORMER_CACHE.get(model)
        if model_instance is None:
            model_instance = SentenceTransformer(model)
            SENTENCE_TRANSFORMER_CACHE[model] = model_instance
        encoded = model_instance.encode(texts, normalize_embeddings=True)
        return [_truncate_vector(list(vector), dimensions) for vector in encoded.tolist()]
    raise ValueError(f"unsupported embedding provider: {provider}")


def _ollama_request_json(ollama_url: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{ollama_url.rstrip('/')}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def _ollama_embed_texts(
    model: str,
    dimensions: int | None,
    texts: list[str],
    ollama_url: str,
) -> list[list[float]]:
    try:
        payload = _ollama_request_json(
            ollama_url,
            "/api/embed",
            {"model": model, "input": texts},
        )
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError("Ollama /api/embed response did not contain `embeddings`")
        return [_truncate_vector(list(vector), dimensions) for vector in embeddings]
    except urllib.error.HTTPError as exc:
        if exc.code not in {404, 501}:
            raise RuntimeError(f"Ollama embedding request failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"failed to reach Ollama at {ollama_url}; ensure the daemon is running"
        ) from exc

    embeddings: list[list[float]] = []
    for text in texts:
        try:
            payload = _ollama_request_json(
                ollama_url,
                "/api/embeddings",
                {"model": model, "prompt": text},
            )
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Ollama model `{model}` did not serve embeddings via `/api/embed` or `/api/embeddings`"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"failed to reach Ollama at {ollama_url}; ensure the daemon is running"
            ) from exc
        embedding = payload.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError(
                "Ollama /api/embeddings response did not contain `embedding`"
            )
        embeddings.append(_truncate_vector(list(embedding), dimensions))
    return embeddings


def create_or_recreate_collection(
    client: Any,
    collection_name: str,
    dimensions: int,
    turbo_bits: str,
    recreate: bool,
    vector_on_disk: bool,
) -> None:
    from qdrant_client.http import models

    if recreate and client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)
    if client.collection_exists(collection_name=collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        on_disk_payload=True,
        vectors_config=models.VectorParams(
            size=dimensions,
            distance=models.Distance.COSINE,
            on_disk=vector_on_disk,
            datatype=models.Datatype.FLOAT32,
            hnsw_config=models.HnswConfigDiff(on_disk=vector_on_disk, inline_storage=True),
            quantization_config=models.TurboQuantization(
                turbo=models.TurboQuantQuantizationConfig(
                    bits=models.TurboQuantBitSize(turbo_bits),
                    always_ram=True,
                )
            ),
        ),
    )


def _codex_rerank_prompt(query: str, candidates: list[dict[str, Any]]) -> str:
    payload = {
        "query": query,
        "candidates": [
            {
                "id": candidate["id"],
                "title": candidate.get("title"),
                "summary_text": candidate.get("summary_text"),
                "market_context_text": candidate.get("market_context_text"),
                "theme_tags": candidate.get("theme_tags"),
                "source_type": candidate.get("source_type"),
                "source_domain": candidate.get("source_domain"),
                "partition_date": candidate.get("partition_date"),
            }
            for candidate in candidates
        ],
    }
    return (
        "You are reranking retrieved news evidence for a market-narrative system.\n"
        "Return the candidate ids sorted from best to worst for answering the query.\n"
        "Favor direct narrative relevance, market-context relevance, and specificity.\n"
        "Do not invent ids. Use every candidate id exactly once.\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=True)}"
    )


def _codex_rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    model: str,
) -> list[dict[str, Any]]:
    from openai_codex import Codex, Sandbox

    schema = {
        "type": "object",
        "properties": {
            "ranked_ids": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["ranked_ids"],
        "additionalProperties": False,
    }
    with Codex() as codex:
        thread = codex.thread_start(model=model, sandbox=Sandbox.read_only)
        result = thread.run(_codex_rerank_prompt(query, candidates), output_schema=schema)
    response = json.loads(result.final_response)
    ranked_ids = response.get("ranked_ids")
    if not isinstance(ranked_ids, list):
        raise RuntimeError("Codex reranker returned invalid `ranked_ids`")
    candidate_map = {candidate["id"]: candidate for candidate in candidates}
    reranked: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for candidate_id in ranked_ids:
        if candidate_id in candidate_map and candidate_id not in seen_ids:
            updated = dict(candidate_map[candidate_id])
            updated["rerank_provider"] = "codex-sdk"
            reranked.append(updated)
            seen_ids.add(candidate_id)
    for candidate in candidates:
        candidate_id = candidate["id"]
        if candidate_id not in seen_ids:
            updated = dict(candidate)
            updated["rerank_provider"] = "codex-sdk_fallback_tail"
            reranked.append(updated)
    return reranked


def _maybe_rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    provider: str,
    rerank_limit: int,
    codex_model: str,
) -> list[dict[str, Any]]:
    if provider == "none" or not hits:
        return hits
    head = hits[:rerank_limit]
    tail = hits[rerank_limit:]
    if provider == "codex-sdk":
        return _codex_rerank_candidates(query, head, codex_model) + tail
    raise ValueError(f"unsupported rerank provider: {provider}")


def run_index(args: argparse.Namespace) -> int:
    ensure_runtime_packages(args.embedding_provider)
    with resolve_query_db(Path(args.db), args.start_date, args.end_date) as query_db:
        rows = load_bronze_rows(Path(query_db), args.start_date, args.end_date, args.row_limit)
    if not rows:
        raise RuntimeError("no bronze rows matched the requested date window")
    payload_rows = [_row_payload(row, args.max_embed_chars) for row in rows]
    qdrant_client = _qdrant_client(args.qdrant_url, args.qdrant_api_key)

    from qdrant_client.http import models

    total_indexed = 0
    actual_dimensions: int | None = None
    for payload_batch in _chunked(payload_rows, args.batch_size):
        text_batch = [row["embedding_text"] for row in payload_batch]
        embeddings = _embed_texts(
            provider=args.embedding_provider,
            model=args.embedding_model,
            dimensions=args.dimensions,
            texts=text_batch,
            openai_api_key=args.openai_api_key,
            ollama_url=args.ollama_url,
        )
        if actual_dimensions is None:
            actual_dimensions = len(embeddings[0])
            create_or_recreate_collection(
                qdrant_client,
                args.collection,
                actual_dimensions,
                args.turbo_bits,
                args.recreate,
                args.vector_on_disk,
            )
        points = [
            models.PointStruct(
                id=_stable_point_id(payload),
                vector=embedding,
                payload=payload,
            )
            for payload, embedding in zip(payload_batch, embeddings, strict=True)
        ]
        qdrant_client.upsert(collection_name=args.collection, points=points, wait=True)
        total_indexed += len(points)

    print(
        json.dumps(
            {
                "collection": args.collection,
                "indexed_points": total_indexed,
                "dimensions": actual_dimensions,
                "requested_dimensions": args.dimensions,
                "embedding_provider": args.embedding_provider,
                "embedding_model": args.embedding_model,
                "turbo_bits": args.turbo_bits,
                "vector_on_disk": args.vector_on_disk,
                "source_db": str(args.db),
                "start_date": args.start_date,
                "end_date": args.end_date,
            },
            indent=2,
        )
    )
    return 0


def run_search(args: argparse.Namespace) -> int:
    ensure_runtime_packages(args.embedding_provider)
    ensure_rerank_runtime(args.rerank_provider)
    qdrant_client = _qdrant_client(args.qdrant_url, args.qdrant_api_key)
    query_vector = _embed_texts(
        provider=args.embedding_provider,
        model=args.embedding_model,
        dimensions=args.dimensions,
        texts=[args.query],
        openai_api_key=args.openai_api_key,
        ollama_url=args.ollama_url,
    )[0]

    from qdrant_client.http import models

    query_filter = _build_filter(
        start_date=args.start_date,
        end_date=args.end_date,
        source_domains=args.source_domain,
        source_types=args.source_type,
        theme_tags=args.theme_tag,
    )
    response = qdrant_client.query_points(
        collection_name=args.collection,
        query=query_vector,
        query_filter=query_filter,
        limit=args.limit,
        with_payload=True,
        with_vectors=False,
        search_params=models.SearchParams(
            hnsw_ef=args.hnsw_ef,
            quantization=models.QuantizationSearchParams(
                rescore=True,
                oversampling=args.oversampling,
            ),
        ),
    )
    hits = []
    for point in response.points:
        payload = dict(point.payload or {})
        hits.append(
            {
                "id": str(point.id),
                "score": point.score,
                "partition_date": payload.get("partition_date"),
                "source_domain": payload.get("source_domain"),
                "source_type": payload.get("source_type"),
                "document_identifier": payload.get("document_identifier"),
                "title": payload.get("title"),
                "summary_text": payload.get("summary_text"),
                "market_context_text": payload.get("market_context_text"),
                "theme_tags": payload.get("theme_tags"),
            }
        )
    hits = _maybe_rerank_hits(
        query=args.query,
        hits=hits,
        provider=args.rerank_provider,
        rerank_limit=min(args.rerank_limit, len(hits)),
        codex_model=args.codex_rerank_model,
    )
    print(
        json.dumps(
            {
                "collection": args.collection,
                "query": args.query,
                "embedding_provider": args.embedding_provider,
                "dimensions": args.dimensions,
                "embedding_model": args.embedding_model,
                "rerank_provider": args.rerank_provider,
                "rerank_limit": args.rerank_limit,
                "codex_rerank_model": args.codex_rerank_model if args.rerank_provider == "codex-sdk" else None,
                "oversampling": args.oversampling,
                "limit": args.limit,
                "hits": hits,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Embed bronze docs and upsert them into Qdrant.")
    index_parser.add_argument("--db", default=str(DEFAULT_DB))
    index_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    index_parser.add_argument("--qdrant-url", default="http://localhost:6333")
    index_parser.add_argument("--qdrant-api-key")
    index_parser.add_argument("--openai-api-key")
    index_parser.add_argument(
        "--embedding-provider",
        choices=["ollama", "openai", "sentence-transformers"],
        default=DEFAULT_EMBEDDING_PROVIDER,
    )
    index_parser.add_argument("--embedding-model")
    index_parser.add_argument(
        "--dimensions",
        type=int,
        help="Optional output dimension. OpenAI applies native shortening; local backends use prefix truncation.",
    )
    index_parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    index_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    index_parser.add_argument("--start-date")
    index_parser.add_argument("--end-date")
    index_parser.add_argument("--row-limit", type=int)
    index_parser.add_argument("--max-embed-chars", type=int, default=DEFAULT_MAX_EMBED_CHARS)
    index_parser.add_argument("--turbo-bits", choices=["bits1", "bits1_5", "bits2", "bits4"], default=DEFAULT_TURBO_BITS)
    index_parser.add_argument("--recreate", action="store_true")
    index_parser.add_argument("--vector-on-disk", action="store_true")
    index_parser.set_defaults(func=run_index)

    search_parser = subparsers.add_parser("search", help="Query the experimental Qdrant collection.")
    search_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    search_parser.add_argument("--qdrant-url", default="http://localhost:6333")
    search_parser.add_argument("--qdrant-api-key")
    search_parser.add_argument("--openai-api-key")
    search_parser.add_argument(
        "--embedding-provider",
        choices=["ollama", "openai", "sentence-transformers"],
        default=DEFAULT_EMBEDDING_PROVIDER,
    )
    search_parser.add_argument("--embedding-model")
    search_parser.add_argument(
        "--dimensions",
        type=int,
        help="Optional query embedding dimension. Must match the indexed collection width.",
    )
    search_parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    search_parser.add_argument("--oversampling", type=float, default=DEFAULT_OVERSAMPLING)
    search_parser.add_argument("--hnsw-ef", type=int, default=DEFAULT_HNSW_EF)
    search_parser.add_argument(
        "--rerank-provider",
        choices=["none", "codex-sdk"],
        default="none",
    )
    search_parser.add_argument("--rerank-limit", type=int, default=DEFAULT_RERANK_LIMIT)
    search_parser.add_argument("--codex-rerank-model", default=DEFAULT_CODEX_RERANK_MODEL)
    search_parser.add_argument("--start-date")
    search_parser.add_argument("--end-date")
    search_parser.add_argument("--source-domain", action="append", default=[])
    search_parser.add_argument("--source-type", action="append", default=[])
    search_parser.add_argument("--theme-tag", action="append", default=[])
    search_parser.set_defaults(func=run_search)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.embedding_model = _resolve_embedding_model(args.embedding_provider, args.embedding_model)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
