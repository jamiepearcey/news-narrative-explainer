use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use polars::prelude::*;
use qdrant_client::qdrant::{
    BinaryQuantizationBuilder, BinaryQuantizationEncoding, BinaryQuantizationQueryEncoding,
    CreateCollectionBuilder, Distance, PointStruct, QuantizationSearchParamsBuilder,
    QueryPointsBuilder, SearchParamsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};
use url::Url;
use uuid::Uuid;

const DEFAULT_BRONZE_ROOT: &str =
    "/Users/jamiepearcey/projects/research/news-narrative-explainer/data/narrative_graph_parquet_v3";
const DEFAULT_COLLECTION_PREFIX: &str = "news_narrative_v3";
const DEFAULT_QDRANT_URL: &str = "http://127.0.0.1:6334";
const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";
const DEFAULT_OLLAMA_MODEL: &str = "all-minilm";
const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_LIMIT: u64 = 12;
const DEFAULT_CANDIDATE_LIMIT: u64 = 64;
const DEFAULT_RERANK_LIMIT: usize = 8;
const DEFAULT_HNSW_EF: u64 = 128;
const DEFAULT_OVERSAMPLING: f64 = 2.0;
const DEFAULT_MAX_EMBED_CHARS: usize = 6000;
const DEFAULT_CODEX_MODEL: &str = "gpt-5.4-mini";
const DEFAULT_TRUNCATE_DIM: usize = 256;
const DEFAULT_BIND_ADDR: &str = "127.0.0.1:8788";
const SWAGGER_UI_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>qdrant-day API Docs</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({
      url: "/openapi.json",
      dom_id: "#swagger-ui",
      deepLinking: true,
      presets: [SwaggerUIBundle.presets.apis],
    });
  </script>
</body>
</html>
"##;

#[derive(Parser)]
#[command(name = "qdrant-day")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    IndexDay {
        #[arg(long)]
        date: String,
        #[arg(long, default_value = DEFAULT_BRONZE_ROOT)]
        bronze_root: String,
        #[arg(long, default_value = DEFAULT_QDRANT_URL)]
        qdrant_url: String,
        #[arg(long)]
        collection: Option<String>,
        #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
        batch_size: usize,
        #[arg(long, default_value_t = DEFAULT_MAX_EMBED_CHARS)]
        max_embed_chars: usize,
        #[arg(long, default_value = DEFAULT_OLLAMA_URL)]
        ollama_url: String,
        #[arg(long, default_value = DEFAULT_OLLAMA_MODEL)]
        embedding_model: String,
        #[arg(long, default_value_t = DEFAULT_TRUNCATE_DIM)]
        truncate_dim: usize,
        #[arg(long)]
        row_limit: Option<usize>,
        #[arg(long)]
        recreate: bool,
    },
    Search {
        #[arg(long)]
        collection: String,
        #[arg(long, default_value = DEFAULT_QDRANT_URL)]
        qdrant_url: String,
        #[arg(long)]
        query: String,
        #[arg(long, default_value = DEFAULT_OLLAMA_URL)]
        ollama_url: String,
        #[arg(long, default_value = DEFAULT_OLLAMA_MODEL)]
        embedding_model: String,
        #[arg(long, default_value_t = DEFAULT_TRUNCATE_DIM)]
        truncate_dim: usize,
        #[arg(long, default_value_t = DEFAULT_LIMIT)]
        limit: u64,
        #[arg(long, default_value_t = DEFAULT_CANDIDATE_LIMIT)]
        candidate_limit: u64,
        #[arg(long, default_value_t = DEFAULT_HNSW_EF)]
        hnsw_ef: u64,
        #[arg(long, default_value_t = DEFAULT_OVERSAMPLING)]
        oversampling: f64,
        #[arg(long)]
        min_source_score: Option<f64>,
        #[arg(long, value_enum, default_value_t = RerankProvider::None)]
        rerank_provider: RerankProvider,
        #[arg(long, default_value_t = DEFAULT_RERANK_LIMIT)]
        rerank_limit: usize,
        #[arg(long, default_value = DEFAULT_CODEX_MODEL)]
        codex_model: String,
    },
    Serve {
        #[arg(long, default_value = DEFAULT_BIND_ADDR)]
        bind: String,
        #[arg(long, default_value = DEFAULT_QDRANT_URL)]
        qdrant_url: String,
        #[arg(long, default_value = DEFAULT_OLLAMA_URL)]
        ollama_url: String,
        #[arg(long, default_value = DEFAULT_OLLAMA_MODEL)]
        embedding_model: String,
        #[arg(long, default_value_t = DEFAULT_TRUNCATE_DIM)]
        truncate_dim: usize,
        #[arg(long, default_value_t = DEFAULT_LIMIT)]
        limit: u64,
        #[arg(long, default_value_t = DEFAULT_CANDIDATE_LIMIT)]
        candidate_limit: u64,
        #[arg(long, default_value_t = DEFAULT_HNSW_EF)]
        hnsw_ef: u64,
        #[arg(long, default_value_t = DEFAULT_OVERSAMPLING)]
        oversampling: f64,
        #[arg(long, value_enum, default_value_t = RerankProvider::Codex)]
        rerank_provider: RerankProvider,
        #[arg(long, default_value_t = DEFAULT_RERANK_LIMIT)]
        rerank_limit: usize,
        #[arg(long, default_value = DEFAULT_CODEX_MODEL)]
        codex_model: String,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum RerankProvider {
    None,
    Codex,
}

#[derive(Debug, Clone)]
struct BronzeDoc {
    doc_id: u64,
    partition_date: String,
    source_domain: Option<String>,
    source_type: Option<String>,
    source_priority: i32,
    source_score: f64,
    document_identifier: String,
    title: Option<String>,
    summary_text: Option<String>,
    market_context_text: Option<String>,
    market_context_score: f64,
    theme_tags: Vec<String>,
    embedding_text: String,
    lexical_text: String,
    canonical_url: String,
    simhash_u64: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchHit {
    id: String,
    score: f32,
    dense_score: f32,
    lexical_score: f32,
    metadata_score: f32,
    fused_score: f32,
    partition_date: Option<String>,
    source_domain: Option<String>,
    source_type: Option<String>,
    source_priority: Option<i64>,
    source_score: f64,
    document_identifier: Option<String>,
    title: Option<String>,
    summary_text: Option<String>,
    market_context_text: Option<String>,
    market_context_score: f64,
    theme_tags: Vec<String>,
    canonical_url: Option<String>,
    simhash_u64: Option<u64>,
    rerank_provider: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct SearchRequest {
    collection: String,
    query: String,
    qdrant_url: Option<String>,
    ollama_url: Option<String>,
    embedding_model: Option<String>,
    truncate_dim: Option<usize>,
    limit: Option<u64>,
    candidate_limit: Option<u64>,
    hnsw_ef: Option<u64>,
    oversampling: Option<f64>,
    min_source_score: Option<f64>,
    rerank_provider: Option<RerankProvider>,
    rerank_limit: Option<usize>,
    codex_model: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RebuildDaysRequest {
    dates: Vec<String>,
    bronze_root: Option<String>,
    qdrant_url: Option<String>,
    collection_prefix: Option<String>,
    batch_size: Option<usize>,
    max_embed_chars: Option<usize>,
    ollama_url: Option<String>,
    embedding_model: Option<String>,
    truncate_dim: Option<usize>,
    row_limit: Option<usize>,
    recreate: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
struct SearchResponseBody {
    collection: String,
    query: String,
    qdrant_url: String,
    embedding_model: String,
    embedding_provider: &'static str,
    ollama_url: String,
    truncate_dim: usize,
    candidate_limit: u64,
    rerank_provider: String,
    codex_model: Option<String>,
    limit: u64,
    hits: Vec<SearchHit>,
}

#[derive(Debug, Clone, Serialize)]
struct RebuildDayResult {
    collection: String,
    indexed_points: usize,
    date: String,
    embedding_model: String,
    embedding_provider: &'static str,
    ollama_url: String,
    truncate_dim: usize,
    embedding_dimensions: Option<usize>,
    qdrant_url: String,
}

#[derive(Debug, Clone, Serialize)]
struct RebuildDaysResponseBody {
    requested_dates: Vec<String>,
    results: Vec<RebuildDayResult>,
}

#[derive(Debug, Clone)]
struct SearchConfig<'a> {
    qdrant_url: &'a str,
    collection: &'a str,
    query: &'a str,
    ollama_url: &'a str,
    embedding_model: &'a str,
    truncate_dim: usize,
    limit: u64,
    candidate_limit: u64,
    hnsw_ef: u64,
    oversampling: f64,
    min_source_score: Option<f64>,
    rerank_provider: RerankProvider,
    rerank_limit: usize,
    codex_model: &'a str,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::IndexDay {
            date,
            bronze_root,
            qdrant_url,
            collection,
            batch_size,
            max_embed_chars,
            ollama_url,
            embedding_model,
            truncate_dim,
            row_limit,
            recreate,
        } => {
            let collection = collection
                .unwrap_or_else(|| default_collection_name(&date, embedding_model.as_str()));
            let result = run_index_day(
                &date,
                Path::new(&bronze_root),
                &qdrant_url,
                &collection,
                batch_size,
                max_embed_chars,
                &ollama_url,
                embedding_model,
                truncate_dim,
                row_limit,
                recreate,
            )
            .await?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Commands::Search {
            collection,
            qdrant_url,
            query,
            ollama_url,
            embedding_model,
            truncate_dim,
            limit,
            candidate_limit,
            hnsw_ef,
            oversampling,
            min_source_score,
            rerank_provider,
            rerank_limit,
            codex_model,
        } => {
            run_search(
                &qdrant_url,
                &collection,
                &query,
                &ollama_url,
                embedding_model,
                truncate_dim,
                limit,
                candidate_limit,
                hnsw_ef,
                oversampling,
                min_source_score,
                rerank_provider,
                rerank_limit,
                &codex_model,
            )
            .await
        }
        Commands::Serve {
            bind,
            qdrant_url,
            ollama_url,
            embedding_model,
            truncate_dim,
            limit,
            candidate_limit,
            hnsw_ef,
            oversampling,
            rerank_provider,
            rerank_limit,
            codex_model,
        } => {
            run_server(
                &bind,
                &qdrant_url,
                &ollama_url,
                &embedding_model,
                truncate_dim,
                limit,
                candidate_limit,
                hnsw_ef,
                oversampling,
                rerank_provider,
                rerank_limit,
                &codex_model,
            )
            .await
        }
    }
}

async fn run_index_day(
    date: &str,
    bronze_root: &Path,
    qdrant_url: &str,
    collection: &str,
    batch_size: usize,
    max_embed_chars: usize,
    ollama_url: &str,
    embedding_model: String,
    truncate_dim: usize,
    row_limit: Option<usize>,
    recreate: bool,
) -> Result<RebuildDayResult> {
    let docs = load_bronze_day(bronze_root, date, row_limit, max_embed_chars)?;
    if docs.is_empty() {
        bail!("no bronze docs found for {}", date);
    }

    let embedder = OllamaEmbedder::new(ollama_url, &embedding_model);
    let client = Qdrant::from_url(qdrant_url).build()?;

    if recreate && client.collection_exists(collection).await? {
        client.delete_collection(collection).await?;
    }

    let mut total_indexed = 0usize;
    let mut embedding_dimension = None;

    for batch in docs.chunks(batch_size) {
        let texts: Vec<&str> = batch
            .iter()
            .map(|doc| doc.embedding_text.as_str())
            .collect();
        let embeddings = embedder
            .embed(&texts)?
            .into_iter()
            .map(|embedding| truncate_embedding(embedding, truncate_dim))
            .collect::<Result<Vec<_>>>()?;
        if embedding_dimension.is_none() {
            let dim = embeddings
                .first()
                .map(std::vec::Vec::len)
                .ok_or_else(|| anyhow!("embedding batch unexpectedly empty"))?;
            create_collection_if_needed(&client, collection, dim as u64).await?;
            embedding_dimension = Some(dim);
        }
        let points = batch
            .iter()
            .zip(embeddings.into_iter())
            .map(|(doc, embedding)| {
                let payload: Payload = json!({
                    "doc_id": doc.doc_id,
                    "partition_date": doc.partition_date,
                    "source_domain": doc.source_domain,
                    "source_type": doc.source_type,
                    "source_priority": doc.source_priority,
                    "source_score": doc.source_score,
                    "document_identifier": doc.document_identifier,
                    "title": doc.title,
                    "summary_text": doc.summary_text,
                    "market_context_text": doc.market_context_text,
                    "market_context_score": doc.market_context_score,
                    "theme_tags": doc.theme_tags,
                    "canonical_url": doc.canonical_url,
                    "lexical_text": doc.lexical_text,
                    "simhash_u64": doc.simhash_u64.to_string(),
                })
                .try_into()
                .context("payload conversion failed")?;
                Ok(PointStruct::new(stable_point_id(doc), embedding, payload))
            })
            .collect::<Result<Vec<_>>>()?;
        client
            .upsert_points(UpsertPointsBuilder::new(collection, points).wait(true))
            .await?;
        total_indexed += batch.len();
    }

    Ok(RebuildDayResult {
        collection: collection.to_string(),
        indexed_points: total_indexed,
        date: date.to_string(),
        embedding_model,
        embedding_provider: "ollama",
        ollama_url: ollama_url.to_string(),
        truncate_dim,
        embedding_dimensions: embedding_dimension,
        qdrant_url: qdrant_url.to_string(),
    })
}

async fn run_search(
    qdrant_url: &str,
    collection: &str,
    query: &str,
    ollama_url: &str,
    embedding_model: String,
    truncate_dim: usize,
    limit: u64,
    candidate_limit: u64,
    hnsw_ef: u64,
    oversampling: f64,
    min_source_score: Option<f64>,
    rerank_provider: RerankProvider,
    rerank_limit: usize,
    codex_model: &str,
) -> Result<()> {
    let payload = perform_search(SearchConfig {
        qdrant_url,
        collection,
        query,
        ollama_url,
        embedding_model: &embedding_model,
        truncate_dim,
        limit,
        candidate_limit,
        hnsw_ef,
        oversampling,
        min_source_score,
        rerank_provider,
        rerank_limit,
        codex_model,
    })
    .await?;
    println!("{}", serde_json::to_string_pretty(&payload)?);
    Ok(())
}

async fn run_server(
    bind: &str,
    qdrant_url: &str,
    ollama_url: &str,
    embedding_model: &str,
    truncate_dim: usize,
    limit: u64,
    candidate_limit: u64,
    hnsw_ef: u64,
    oversampling: f64,
    rerank_provider: RerankProvider,
    rerank_limit: usize,
    codex_model: &str,
) -> Result<()> {
    let server = Server::http(bind).map_err(|error| anyhow!("failed to bind {bind}: {error}"))?;
    eprintln!("qdrant-day server listening on http://{bind}");
    for request in server.incoming_requests() {
        let result = handle_request(
            request,
            qdrant_url,
            ollama_url,
            embedding_model,
            truncate_dim,
            limit,
            candidate_limit,
            hnsw_ef,
            oversampling,
            rerank_provider,
            rerank_limit,
            codex_model,
        )
        .await;
        if let Err(error) = result {
            eprintln!("request handling failed: {error:#}");
        }
    }
    Ok(())
}

async fn handle_request(
    mut request: Request,
    qdrant_url: &str,
    ollama_url: &str,
    embedding_model: &str,
    truncate_dim: usize,
    limit: u64,
    candidate_limit: u64,
    hnsw_ef: u64,
    oversampling: f64,
    rerank_provider: RerankProvider,
    rerank_limit: usize,
    codex_model: &str,
) -> Result<()> {
    match (request.method(), request.url()) {
        (&Method::Get, "/health") => respond_json(request, StatusCode(200), &json!({"ok": true})),
        (&Method::Get, "/openapi.json") => {
            respond_json(request, StatusCode(200), &build_openapi_spec())
        }
        (&Method::Get, "/docs") => respond_html(request, StatusCode(200), SWAGGER_UI_HTML),
        (&Method::Post, "/search") => {
            let mut body = String::new();
            request
                .as_reader()
                .read_to_string(&mut body)
                .context("failed to read request body")?;
            let payload: SearchRequest =
                serde_json::from_str(&body).context("failed to decode search request")?;
            let response = perform_search(SearchConfig {
                qdrant_url: payload.qdrant_url.as_deref().unwrap_or(qdrant_url),
                collection: &payload.collection,
                query: &payload.query,
                ollama_url: payload.ollama_url.as_deref().unwrap_or(ollama_url),
                embedding_model: payload
                    .embedding_model
                    .as_deref()
                    .unwrap_or(embedding_model),
                truncate_dim: payload.truncate_dim.unwrap_or(truncate_dim),
                limit: payload.limit.unwrap_or(limit),
                candidate_limit: payload.candidate_limit.unwrap_or(candidate_limit),
                hnsw_ef: payload.hnsw_ef.unwrap_or(hnsw_ef),
                oversampling: payload.oversampling.unwrap_or(oversampling),
                min_source_score: payload.min_source_score,
                rerank_provider: payload.rerank_provider.unwrap_or(rerank_provider),
                rerank_limit: payload.rerank_limit.unwrap_or(rerank_limit),
                codex_model: payload.codex_model.as_deref().unwrap_or(codex_model),
            })
            .await?;
            respond_json(request, StatusCode(200), &response)
        }
        (&Method::Post, "/index/rebuild-days") => {
            let mut body = String::new();
            request
                .as_reader()
                .read_to_string(&mut body)
                .context("failed to read request body")?;
            let payload: RebuildDaysRequest =
                serde_json::from_str(&body).context("failed to decode rebuild request")?;
            if payload.dates.is_empty() {
                return respond_json(
                    request,
                    StatusCode(400),
                    &json!({"error": "dates must not be empty"}),
                );
            }
            let bronze_root = payload
                .bronze_root
                .unwrap_or_else(|| DEFAULT_BRONZE_ROOT.to_string());
            let qdrant_url = payload
                .qdrant_url
                .unwrap_or_else(|| DEFAULT_QDRANT_URL.to_string());
            let collection_prefix = payload
                .collection_prefix
                .unwrap_or_else(|| DEFAULT_COLLECTION_PREFIX.to_string());
            let batch_size = payload.batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
            let max_embed_chars = payload.max_embed_chars.unwrap_or(DEFAULT_MAX_EMBED_CHARS);
            let ollama_url = payload
                .ollama_url
                .unwrap_or_else(|| DEFAULT_OLLAMA_URL.to_string());
            let embedding_model = payload
                .embedding_model
                .unwrap_or_else(|| DEFAULT_OLLAMA_MODEL.to_string());
            let truncate_dim = payload.truncate_dim.unwrap_or(DEFAULT_TRUNCATE_DIM);
            let row_limit = payload.row_limit;
            let recreate = payload.recreate.unwrap_or(false);

            let mut results = Vec::with_capacity(payload.dates.len());
            for date in &payload.dates {
                let collection =
                    collection_name_with_prefix(collection_prefix.as_str(), date, &embedding_model);
                let result = run_index_day(
                    date,
                    Path::new(&bronze_root),
                    &qdrant_url,
                    &collection,
                    batch_size,
                    max_embed_chars,
                    &ollama_url,
                    embedding_model.clone(),
                    truncate_dim,
                    row_limit,
                    recreate,
                )
                .await?;
                results.push(result);
            }
            respond_json(
                request,
                StatusCode(200),
                &RebuildDaysResponseBody {
                    requested_dates: payload.dates,
                    results,
                },
            )
        }
        _ => respond_json(
            request,
            StatusCode(404),
            &json!({"error": "not_found", "supported_paths": ["/health", "/openapi.json", "/docs", "/search", "/index/rebuild-days"]}),
        ),
    }
}

fn respond_json(request: Request, status: StatusCode, body: &impl Serialize) -> Result<()> {
    let response_body = serde_json::to_vec(body)?;
    let response = Response::from_data(response_body)
        .with_status_code(status)
        .with_header(
            Header::from_bytes("Content-Type", "application/json")
                .map_err(|_| anyhow!("failed to build content-type header"))?,
        );
    request
        .respond(response)
        .context("failed to send response")?;
    Ok(())
}

fn respond_html(request: Request, status: StatusCode, body: &str) -> Result<()> {
    let response = Response::from_string(body.to_string())
        .with_status_code(status)
        .with_header(
            Header::from_bytes("Content-Type", "text/html; charset=utf-8")
                .map_err(|_| anyhow!("failed to build content-type header"))?,
        );
    request
        .respond(response)
        .context("failed to send response")?;
    Ok(())
}

fn build_openapi_spec() -> Value {
    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "qdrant-day API",
            "version": "0.1.0",
            "description": "Hybrid retrieval API for news-narrative validation over Qdrant + Ollama with optional Codex reranking."
        },
        "servers": [
            {"url": "/"}
        ],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check",
                    "operationId": "healthCheck",
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "ok": {"type": "boolean"}
                                        },
                                        "required": ["ok"]
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/search": {
                "post": {
                    "summary": "Run hybrid news retrieval",
                    "operationId": "search",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/SearchRequest"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Ranked search results",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/SearchResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/index/rebuild-days": {
                "post": {
                    "summary": "Rebuild one or more day collections",
                    "operationId": "rebuildDays",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/RebuildDaysRequest"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Per-day indexing results",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/RebuildDaysResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "RerankProvider": {
                    "type": "string",
                    "enum": ["none", "codex"]
                },
                "SearchRequest": {
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string"},
                        "query": {"type": "string"},
                        "qdrant_url": {"type": "string"},
                        "ollama_url": {"type": "string"},
                        "embedding_model": {"type": "string"},
                        "truncate_dim": {"type": "integer", "minimum": 1},
                        "limit": {"type": "integer", "minimum": 1},
                        "candidate_limit": {"type": "integer", "minimum": 1},
                        "hnsw_ef": {"type": "integer", "minimum": 1},
                        "oversampling": {"type": "number", "minimum": 0},
                        "min_source_score": {"type": "number", "minimum": 0},
                        "rerank_provider": {"$ref": "#/components/schemas/RerankProvider"},
                        "rerank_limit": {"type": "integer", "minimum": 1},
                        "codex_model": {"type": "string"}
                    },
                    "required": ["collection", "query"]
                },
                "SearchHit": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "score": {"type": "number"},
                        "dense_score": {"type": "number"},
                        "lexical_score": {"type": "number"},
                        "metadata_score": {"type": "number"},
                        "fused_score": {"type": "number"},
                        "partition_date": {"type": ["string", "null"]},
                        "source_domain": {"type": ["string", "null"]},
                        "source_type": {"type": ["string", "null"]},
                        "source_priority": {"type": ["integer", "null"]},
                        "source_score": {"type": "number"},
                        "document_identifier": {"type": ["string", "null"]},
                        "title": {"type": ["string", "null"]},
                        "summary_text": {"type": ["string", "null"]},
                        "market_context_text": {"type": ["string", "null"]},
                        "market_context_score": {"type": "number"},
                        "theme_tags": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "canonical_url": {"type": ["string", "null"]},
                        "simhash_u64": {"type": ["integer", "null"]},
                        "rerank_provider": {"type": ["string", "null"]}
                    },
                    "required": [
                        "id", "score", "dense_score", "lexical_score", "metadata_score",
                        "fused_score", "source_score", "market_context_score", "theme_tags"
                    ]
                },
                "RebuildDaysRequest": {
                    "type": "object",
                    "properties": {
                        "dates": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "bronze_root": {"type": "string"},
                        "qdrant_url": {"type": "string"},
                        "collection_prefix": {"type": "string"},
                        "batch_size": {"type": "integer", "minimum": 1},
                        "max_embed_chars": {"type": "integer", "minimum": 1},
                        "ollama_url": {"type": "string"},
                        "embedding_model": {"type": "string"},
                        "truncate_dim": {"type": "integer", "minimum": 1},
                        "row_limit": {"type": "integer", "minimum": 1},
                        "recreate": {"type": "boolean"}
                    },
                    "required": ["dates"]
                },
                "RebuildDayResult": {
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string"},
                        "indexed_points": {"type": "integer"},
                        "date": {"type": "string"},
                        "embedding_model": {"type": "string"},
                        "embedding_provider": {"type": "string"},
                        "ollama_url": {"type": "string"},
                        "truncate_dim": {"type": "integer"},
                        "embedding_dimensions": {"type": ["integer", "null"]},
                        "qdrant_url": {"type": "string"}
                    },
                    "required": [
                        "collection", "indexed_points", "date", "embedding_model",
                        "embedding_provider", "ollama_url", "truncate_dim", "qdrant_url"
                    ]
                },
                "RebuildDaysResponse": {
                    "type": "object",
                    "properties": {
                        "requested_dates": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "results": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/RebuildDayResult"}
                        }
                    },
                    "required": ["requested_dates", "results"]
                },
                "SearchResponse": {
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string"},
                        "query": {"type": "string"},
                        "qdrant_url": {"type": "string"},
                        "embedding_model": {"type": "string"},
                        "embedding_provider": {"type": "string"},
                        "ollama_url": {"type": "string"},
                        "truncate_dim": {"type": "integer"},
                        "candidate_limit": {"type": "integer"},
                        "rerank_provider": {"type": "string"},
                        "codex_model": {"type": ["string", "null"]},
                        "limit": {"type": "integer"},
                        "hits": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/SearchHit"}
                        }
                    },
                    "required": [
                        "collection", "query", "qdrant_url", "embedding_model",
                        "embedding_provider", "ollama_url", "truncate_dim",
                        "candidate_limit", "rerank_provider", "limit", "hits"
                    ]
                }
            }
        }
    })
}

async fn perform_search(config: SearchConfig<'_>) -> Result<SearchResponseBody> {
    let client = Qdrant::from_url(config.qdrant_url).build()?;
    let embedder = OllamaEmbedder::new(config.ollama_url, config.embedding_model);
    let query_embedding = truncate_embedding(
        embedder
            .embed(&[config.query])?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("query embedding missing"))?,
        config.truncate_dim,
    )?;

    let response = client
        .query(
            QueryPointsBuilder::new(config.collection)
                .query(query_embedding)
                .limit(config.candidate_limit.max(config.limit))
                .with_payload(true)
                .params(
                    SearchParamsBuilder::default()
                        .hnsw_ef(config.hnsw_ef)
                        .quantization(
                            QuantizationSearchParamsBuilder::default()
                                .rescore(true)
                                .oversampling(config.oversampling),
                        ),
                ),
        )
        .await?;

    let mut hits = response
        .result
        .into_iter()
        .map(search_result_to_hit)
        .collect::<Result<Vec<_>>>()?;

    if let Some(min_source_score) = config.min_source_score {
        hits.retain(|hit| hit.source_score >= min_source_score);
    }

    score_hits(config.query, &mut hits);
    hits.sort_by(|left, right| right.fused_score.total_cmp(&left.fused_score));
    hits = dedup_and_diversify_hits(hits);

    if config.rerank_provider == RerankProvider::Codex && !hits.is_empty() {
        let head_len = config.rerank_limit.min(hits.len());
        let reranked = rerank_with_codex(config.query, &hits[..head_len], config.codex_model)?;
        hits.splice(0..head_len, reranked);
    }
    hits.truncate(config.limit as usize);

    Ok(SearchResponseBody {
        collection: config.collection.to_string(),
        query: config.query.to_string(),
        qdrant_url: config.qdrant_url.to_string(),
        embedding_model: config.embedding_model.to_string(),
        embedding_provider: "ollama",
        ollama_url: config.ollama_url.to_string(),
        truncate_dim: config.truncate_dim,
        candidate_limit: config.candidate_limit,
        rerank_provider: match config.rerank_provider {
            RerankProvider::None => "none".to_string(),
            RerankProvider::Codex => "codex".to_string(),
        },
        codex_model: if config.rerank_provider == RerankProvider::Codex {
            Some(config.codex_model.to_string())
        } else {
            None
        },
        limit: config.limit,
        hits,
    })
}

fn load_bronze_day(
    bronze_root: &Path,
    date: &str,
    row_limit: Option<usize>,
    max_embed_chars: usize,
) -> Result<Vec<BronzeDoc>> {
    let graph_path = bronze_root.join(format!(
        "graph_doc_nodes_daily/partition_date={date}/part-000.parquet"
    ));
    let review_path = bronze_root.join(format!(
        "doc_review_daily/partition_date={date}/part-000.parquet"
    ));
    let payload_path = bronze_root.join(format!(
        "doc_payload_daily/partition_date={date}/part-000.parquet"
    ));
    for path in [&graph_path, &review_path, &payload_path] {
        if !path.exists() {
            bail!(
                "required v3 artifact missing for {}: {}",
                date,
                path.display()
            );
        }
    }

    let review_lazy = LazyFrame::scan_parquet(
        review_path.to_string_lossy().into_owned(),
        ScanArgsParquet::default(),
    )
    .with_context(|| format!("failed to scan doc_review_daily for {}", date))?
    .select([
        col("doc_id"),
        col("title").alias("review_title"),
        col("summary_text").alias("review_summary_text"),
        col("relevant_text").alias("review_relevant_text"),
        col("metadata_json").alias("review_metadata_json"),
        col("quotations").alias("review_quotations"),
    ]);
    let payload_lazy = LazyFrame::scan_parquet(
        payload_path.to_string_lossy().into_owned(),
        ScanArgsParquet::default(),
    )
    .with_context(|| format!("failed to scan doc_payload_daily for {}", date))?
    .select([
        col("doc_id"),
        col("title").alias("payload_title"),
        col("summary_text").alias("payload_summary_text"),
        col("body_text").alias("payload_body_text"),
    ]);

    let lazy = LazyFrame::scan_parquet(
        graph_path.to_string_lossy().into_owned(),
        ScanArgsParquet::default(),
    )
    .with_context(|| format!("failed to scan graph_doc_nodes_daily for {}", date))?
    .join(
        review_lazy,
        [col("doc_id")],
        [col("doc_id")],
        JoinArgs::new(JoinType::Left),
    )
    .join(
        payload_lazy,
        [col("doc_id")],
        [col("doc_id")],
        JoinArgs::new(JoinType::Left),
    )
    .select([
        col("doc_id"),
        col("partition_date").cast(DataType::String),
        col("source_domain"),
        col("source_type"),
        col("source_priority"),
        col("document_identifier"),
        coalesce(&[col("title"), col("review_title"), col("payload_title")]).alias("title"),
        coalesce(&[col("review_summary_text"), col("payload_summary_text")]).alias("summary_text"),
        col("payload_body_text").alias("body_text"),
        col("review_relevant_text").alias("relevant_text"),
        col("market_context_text"),
        col("market_context_score"),
        col("review_metadata_json").alias("metadata_json"),
        col("review_quotations").alias("quotations"),
    ]);
    let lazy = if let Some(limit) = row_limit {
        lazy.limit(limit as u32)
    } else {
        lazy
    };
    let df = lazy.collect()?;

    let doc_ids = df.column("doc_id")?.u64()?;
    let partition_dates = df.column("partition_date")?.str()?;
    let source_domains = df.column("source_domain")?.str()?;
    let source_types = df.column("source_type")?.str()?;
    let source_priorities = df.column("source_priority")?.i32()?;
    let document_identifiers = df.column("document_identifier")?.str()?;
    let titles = df.column("title")?.str()?;
    let summaries = df.column("summary_text")?.str()?;
    let bodies = df.column("body_text")?.str()?;
    let relevant_texts = df.column("relevant_text")?.str()?;
    let market_contexts = df.column("market_context_text")?.str()?;
    let market_context_scores = df.column("market_context_score")?.f64()?;
    let metadata_jsons = df.column("metadata_json")?.str()?;
    let quotations = df.column("quotations")?.str()?;

    let mut docs = Vec::with_capacity(df.height());
    for idx in 0..df.height() {
        let partition_date = partition_dates
            .get(idx)
            .ok_or_else(|| anyhow!("partition_date missing at row {}", idx))?
            .to_string();
        let document_identifier = document_identifiers
            .get(idx)
            .ok_or_else(|| anyhow!("document_identifier missing at row {}", idx))?;
        let derived_title = derive_title(
            titles.get(idx),
            document_identifier,
            market_contexts.get(idx),
            relevant_texts.get(idx),
        );
        let derived_summary = derive_summary(
            summaries.get(idx),
            market_contexts.get(idx),
            relevant_texts.get(idx),
        );
        let derived_relevant = normalize_sparse_text(relevant_texts.get(idx));
        let derived_context = normalize_sparse_text(market_contexts.get(idx));
        let derived_quotes = normalize_sparse_text(quotations.get(idx));
        let metadata_terms = metadata_keywords(metadata_jsons.get(idx));
        let doc = BronzeDoc {
            doc_id: doc_ids
                .get(idx)
                .ok_or_else(|| anyhow!("doc_id missing at row {}", idx))?,
            partition_date,
            source_domain: source_domains.get(idx).map(|value| value.to_string()),
            source_type: source_types.get(idx).map(|value| value.to_string()),
            source_priority: source_priorities.get(idx).unwrap_or(1),
            source_score: source_weight_for_graph(
                source_domains.get(idx).unwrap_or(""),
                source_types.get(idx).unwrap_or(""),
                source_priorities.get(idx).unwrap_or(1),
            ),
            document_identifier: document_identifier.to_string(),
            title: derived_title.clone(),
            summary_text: derived_summary.clone(),
            market_context_text: derived_context.clone(),
            market_context_score: market_context_scores.get(idx).unwrap_or(0.0),
            theme_tags: Vec::new(),
            embedding_text: build_embedding_text(
                derived_title.as_deref(),
                derived_context.as_deref(),
                derived_summary.as_deref(),
                derived_relevant.as_deref(),
                bodies.get(idx),
                derived_quotes.as_deref(),
                metadata_terms.as_deref(),
                Some(document_identifier),
                max_embed_chars,
            ),
            lexical_text: build_lexical_text(
                derived_title.as_deref(),
                derived_summary.as_deref(),
                derived_context.as_deref(),
                derived_relevant.as_deref(),
                bodies.get(idx),
                derived_quotes.as_deref(),
                metadata_terms.as_deref(),
                max_embed_chars,
            ),
            canonical_url: canonicalize_identifier(document_identifier),
            simhash_u64: compute_simhash(&build_lexical_text(
                derived_title.as_deref(),
                derived_summary.as_deref(),
                derived_context.as_deref(),
                derived_relevant.as_deref(),
                bodies.get(idx),
                derived_quotes.as_deref(),
                metadata_terms.as_deref(),
                max_embed_chars,
            )),
        };
        docs.push(doc);
    }
    Ok(docs)
}

#[derive(Debug, Clone)]
struct OllamaEmbedder {
    url: String,
    model: String,
}

#[derive(Debug, Serialize)]
struct OllamaEmbedRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl OllamaEmbedder {
    fn new(url: &str, model: &str) -> Self {
        Self {
            url: url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let request = OllamaEmbedRequest {
            model: &self.model,
            input: texts.to_vec(),
        };
        let mut response = ureq::post(format!("{}/api/embed", self.url))
            .content_type("application/json")
            .send_json(request)
            .context("failed to call Ollama embed API")?;
        let payload: OllamaEmbedResponse = response
            .body_mut()
            .read_json()
            .context("failed to decode Ollama embed response")?;
        if payload.embeddings.len() != texts.len() {
            bail!(
                "ollama returned {} embeddings for {} texts",
                payload.embeddings.len(),
                texts.len()
            );
        }
        Ok(payload.embeddings)
    }
}

async fn create_collection_if_needed(
    client: &Qdrant,
    collection: &str,
    dimensions: u64,
) -> Result<()> {
    if client.collection_exists(collection).await? {
        return Ok(());
    }
    client
        .create_collection(
            CreateCollectionBuilder::new(collection)
                .vectors_config(VectorParamsBuilder::new(dimensions, Distance::Cosine))
                .quantization_config(
                    BinaryQuantizationBuilder::new(true)
                        .encoding(BinaryQuantizationEncoding::OneBit)
                        .query_encoding(BinaryQuantizationQueryEncoding::scalar8bits())
                        .always_ram(true),
                ),
        )
        .await?;
    Ok(())
}

fn truncate_embedding(mut embedding: Vec<f32>, truncate_dim: usize) -> Result<Vec<f32>> {
    if truncate_dim == 0 {
        bail!("truncate_dim must be greater than zero");
    }
    if truncate_dim > embedding.len() {
        bail!(
            "truncate_dim {} exceeds embedding dimension {}",
            truncate_dim,
            embedding.len()
        );
    }
    embedding.truncate(truncate_dim);
    Ok(embedding)
}

#[cfg(test)]
fn parse_theme_tags(value: Option<&str>) -> Vec<String> {
    value
        .unwrap_or_default()
        .split(';')
        .filter_map(|part| part.split(',').next())
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_uppercase)
        .collect()
}

fn clean_text(value: Option<&str>) -> Option<String> {
    let raw = value?;
    let cleaned = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

fn normalize_sparse_text(value: Option<&str>) -> Option<String> {
    let raw = value?;
    let mut parts = Vec::new();
    let mut seen = HashSet::new();
    for part in raw.split(';') {
        let token = part
            .rsplit_once(',')
            .and_then(|(head, tail)| tail.parse::<usize>().ok().map(|_| head))
            .unwrap_or(part)
            .replace('_', " ");
        let token = token.split_whitespace().collect::<Vec<_>>().join(" ");
        let token = token.trim().trim_matches('"').trim_matches('\'');
        if token.is_empty() {
            continue;
        }
        let lowered = token.to_lowercase();
        if seen.insert(lowered) {
            parts.push(token.to_string());
        }
        if parts.len() >= 48 {
            break;
        }
    }
    if parts.is_empty() {
        clean_text(Some(raw))
    } else {
        Some(parts.join("; "))
    }
}

fn metadata_keywords(value: Option<&str>) -> Option<String> {
    let raw = value?;
    let mut tokens = Vec::new();
    for needle in [
        "\"source_collection_identifier\":",
        "\"gcam\":\"",
        "\"v2_counts\":",
        "\"counts\":",
    ] {
        if raw.contains(needle) {
            tokens.push(
                needle
                    .trim_matches('"')
                    .trim_end_matches(':')
                    .replace('_', " "),
            );
        }
    }
    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join("; "))
    }
}

fn source_weight_for_graph(source_domain: &str, source_type: &str, source_priority: i32) -> f64 {
    let domain_weight = match source_domain {
        "bloomberg.com"
        | "cnbc.com"
        | "nikkei.com"
        | "ft.com"
        | "wsj.com"
        | "federalreserve.gov"
        | "treasury.gov"
        | "ecb.europa.eu"
        | "bankofengland.co.uk"
        | "opec.org"
        | "iea.org"
        | "imf.org"
        | "worldbank.org"
        | "reuters.com" => 1.0,
        "moneycontrol.com"
        | "livemint.com"
        | "business-standard.com"
        | "benzinga.com"
        | "seekingalpha.com"
        | "kitco.com"
        | "oilprice.com"
        | "hellenicshippingnews.com"
        | "shipandbunker.com"
        | "gcaptain.com"
        | "rigzone.com"
        | "worldoil.com"
        | "bullionvault.com"
        | "argusmedia.com"
        | "financialpost.com"
        | "afr.com"
        | "borsaitaliana.it"
        | "theglobeandmail.com"
        | "channelnewsasia.com"
        | "nasdaq.com"
        | "morningstar.com"
        | "investors.com"
        | "apnews.com"
        | "marketwatch.com" => 0.9,
        "prnewswire.com" | "openpr.com" | "financialcontent.com" | "tickerreport.com" => 0.15,
        _ => match source_type {
            "market_wrap" => 0.85,
            "commodity_specialist" => 0.8,
            "company_specific" => 0.45,
            _ => 0.55,
        },
    };
    let priority_adjustment = 0.03 * (source_priority - 1).max(0) as f64;
    (domain_weight + priority_adjustment).clamp(0.1, 1.1)
}

fn slug_to_title(slug: &str) -> Option<String> {
    let filtered = slug
        .trim_matches('/')
        .trim_end_matches(".html")
        .trim_end_matches(".htm")
        .trim_end_matches(".php");
    if filtered.is_empty() {
        return None;
    }
    let filtered = filtered
        .split('/')
        .next_back()
        .unwrap_or(filtered)
        .split('?')
        .next()
        .unwrap_or(filtered);
    let words = filtered
        .split(['-', '_', '+'])
        .filter_map(|part| {
            let cleaned = part.trim_matches(|c: char| !c.is_alphanumeric());
            (!cleaned.is_empty() && cleaned.chars().any(|c| c.is_alphabetic())).then_some(cleaned)
        })
        .take(16)
        .collect::<Vec<_>>();
    if words.is_empty() {
        return None;
    }
    Some(words.join(" "))
}

fn derive_title(
    title: Option<&str>,
    document_identifier: &str,
    market_context_text: Option<&str>,
    relevant_text: Option<&str>,
) -> Option<String> {
    clean_text(title)
        .or_else(|| {
            Url::parse(document_identifier)
                .ok()
                .and_then(|url| slug_to_title(url.path()))
        })
        .or_else(|| normalize_sparse_text(market_context_text))
        .or_else(|| normalize_sparse_text(relevant_text))
}

fn derive_summary(
    summary_text: Option<&str>,
    market_context_text: Option<&str>,
    relevant_text: Option<&str>,
) -> Option<String> {
    clean_text(summary_text)
        .or_else(|| normalize_sparse_text(market_context_text))
        .or_else(|| normalize_sparse_text(relevant_text))
        .map(|value| value.chars().take(280).collect())
}

#[allow(clippy::too_many_arguments)]
fn build_embedding_text(
    title: Option<&str>,
    market_context_text: Option<&str>,
    summary_text: Option<&str>,
    relevant_text: Option<&str>,
    body_text: Option<&str>,
    quotations: Option<&str>,
    metadata_terms: Option<&str>,
    document_identifier: Option<&str>,
    max_chars: usize,
) -> String {
    let mut segments = Vec::new();
    if let Some(title) = clean_text(title) {
        segments.push(format!("Title: {title}"));
    }
    if let Some(market_context) = clean_text(market_context_text) {
        segments.push(format!("Market context: {market_context}"));
    }
    if let Some(summary) = clean_text(summary_text) {
        segments.push(format!("Summary: {summary}"));
    }
    if let Some(relevant) = clean_text(relevant_text) {
        segments.push(format!("Relevant text: {relevant}"));
    } else if let Some(body) = clean_text(body_text) {
        segments.push(format!("Body: {body}"));
    }
    if let Some(quotations) = normalize_sparse_text(quotations) {
        segments.push(format!("Quotations: {quotations}"));
    }
    if let Some(metadata_terms) = clean_text(metadata_terms) {
        segments.push(format!("Metadata: {metadata_terms}"));
    }
    let mut joined = segments.join("\n");
    if joined.is_empty() {
        joined = document_identifier.unwrap_or_default().to_string();
    }
    joined.chars().take(max_chars).collect()
}

fn stable_point_id(doc: &BronzeDoc) -> String {
    let raw = format!(
        "{}\0{}\0{}",
        doc.doc_id, doc.partition_date, doc.document_identifier
    );
    let digest = Sha256::digest(raw.as_bytes());
    Uuid::new_v5(&Uuid::NAMESPACE_URL, digest.as_ref()).to_string()
}

fn default_collection_name(date: &str, embedding_model: &str) -> String {
    collection_name_with_prefix(DEFAULT_COLLECTION_PREFIX, date, embedding_model)
}

fn collection_name_with_prefix(
    collection_prefix: &str,
    date: &str,
    embedding_model: &str,
) -> String {
    format!(
        "{}_{}_{}",
        collection_prefix,
        date.replace('-', ""),
        normalize_model_name(embedding_model)
    )
}

fn normalize_model_name(embedding_model: &str) -> String {
    let base = embedding_model.split(':').next().unwrap_or(embedding_model);
    let normalized: String = base
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect();
    if normalized.is_empty() {
        "embedding".to_string()
    } else {
        normalized.to_lowercase()
    }
}

fn search_result_to_hit(scored: qdrant_client::qdrant::ScoredPoint) -> Result<SearchHit> {
    let payload = payload_to_json(scored.payload)?;
    let source_domain = payload
        .get("source_domain")
        .and_then(Value::as_str)
        .map(str::to_string);
    let source_type = payload
        .get("source_type")
        .and_then(Value::as_str)
        .map(str::to_string);
    let source_priority = payload.get("source_priority").and_then(Value::as_i64);
    let source_score = payload
        .get("source_score")
        .and_then(Value::as_f64)
        .unwrap_or_else(|| {
            source_weight_for_graph(
                source_domain.as_deref().unwrap_or(""),
                source_type.as_deref().unwrap_or(""),
                source_priority.unwrap_or(1) as i32,
            )
        }) as f32;
    Ok(SearchHit {
        id: scored
            .id
            .and_then(|id| id.point_id_options)
            .map(|value| match value {
                qdrant_client::qdrant::point_id::PointIdOptions::Num(num) => num.to_string(),
                qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid) => uuid,
            })
            .unwrap_or_default(),
        score: scored.score,
        dense_score: scored.score,
        lexical_score: 0.0,
        metadata_score: 0.0,
        fused_score: scored.score,
        partition_date: payload
            .get("partition_date")
            .and_then(Value::as_str)
            .map(str::to_string),
        source_domain,
        source_type,
        source_priority,
        source_score: source_score as f64,
        document_identifier: payload
            .get("document_identifier")
            .and_then(Value::as_str)
            .map(str::to_string),
        title: payload
            .get("title")
            .and_then(Value::as_str)
            .map(str::to_string),
        summary_text: payload
            .get("summary_text")
            .and_then(Value::as_str)
            .map(str::to_string),
        market_context_text: payload
            .get("market_context_text")
            .and_then(Value::as_str)
            .map(str::to_string),
        market_context_score: payload
            .get("market_context_score")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        theme_tags: payload
            .get("theme_tags")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
        canonical_url: payload
            .get("canonical_url")
            .and_then(Value::as_str)
            .map(str::to_string),
        simhash_u64: payload
            .get("simhash_u64")
            .and_then(Value::as_str)
            .and_then(|value| value.parse::<u64>().ok()),
        rerank_provider: None,
    })
}

fn payload_to_json(
    payload: std::collections::HashMap<String, qdrant_client::qdrant::Value>,
) -> Result<Value> {
    let map = payload
        .into_iter()
        .map(|(key, value)| (key, value.into_json()))
        .collect::<serde_json::Map<_, _>>();
    Ok(Value::Object(map))
}

fn rerank_with_codex(query: &str, hits: &[SearchHit], codex_model: &str) -> Result<Vec<SearchHit>> {
    let schema_path = std::env::temp_dir().join("codex-rerank-schema.json");
    fs::write(
        &schema_path,
        serde_json::to_vec_pretty(&json!({
            "type": "object",
            "properties": {
                "ranked_ids": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["ranked_ids"],
            "additionalProperties": false
        }))?,
    )?;
    let output_path = std::env::temp_dir().join("codex-rerank-output.json");
    let prompt = build_codex_rerank_prompt(query, hits);
    let status = Command::new("codex")
        .arg("exec")
        .arg("--skip-git-repo-check")
        .arg("--sandbox")
        .arg("read-only")
        .arg("--color")
        .arg("never")
        .arg("--model")
        .arg(codex_model)
        .arg("--output-schema")
        .arg(&schema_path)
        .arg("--output-last-message")
        .arg(&output_path)
        .arg(prompt)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("failed to invoke codex for reranking")?;
    if !status.success() {
        bail!("codex rerank exited with status {}", status);
    }
    let ranked: Value = serde_json::from_slice(&fs::read(&output_path)?)?;
    let ranked_ids = ranked
        .get("ranked_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("codex rerank response missing ranked_ids"))?;
    let mut id_set = HashSet::new();
    let mut reranked = Vec::new();
    for id in ranked_ids.iter().filter_map(Value::as_str) {
        if let Some(hit) = hits.iter().find(|hit| hit.id == id) {
            let mut cloned = hit.clone();
            cloned.rerank_provider = Some("codex".to_string());
            reranked.push(cloned);
            id_set.insert(id.to_string());
        }
    }
    for hit in hits {
        if !id_set.contains(&hit.id) {
            let mut cloned = hit.clone();
            cloned.rerank_provider = Some("codex_tail".to_string());
            reranked.push(cloned);
        }
    }
    Ok(reranked)
}

fn build_codex_rerank_prompt(query: &str, hits: &[SearchHit]) -> String {
    let payload = json!({
        "query": query,
        "candidates": hits,
    });
    format!(
        "You are reranking retrieved news evidence for a market-narrative system.\n\
Return candidate ids sorted from best to worst for answering the query.\n\
Favor direct narrative relevance, market-context relevance, and specificity.\n\
Use every candidate id exactly once and do not invent ids.\n\
Input JSON:\n{}",
        serde_json::to_string(&payload).unwrap_or_default()
    )
}

fn build_lexical_text(
    title: Option<&str>,
    summary_text: Option<&str>,
    market_context_text: Option<&str>,
    relevant_text: Option<&str>,
    body_text: Option<&str>,
    quotations: Option<&str>,
    metadata_terms: Option<&str>,
    max_chars: usize,
) -> String {
    let mut segments = Vec::new();
    if let Some(title) = clean_text(title) {
        segments.push(title);
    }
    if let Some(summary) = clean_text(summary_text) {
        segments.push(summary);
    }
    if let Some(context) = clean_text(market_context_text) {
        segments.push(context);
    }
    if let Some(relevant) = clean_text(relevant_text) {
        segments.push(relevant);
    } else if let Some(body) = clean_text(body_text) {
        segments.push(body);
    }
    if let Some(quotations) = normalize_sparse_text(quotations) {
        segments.push(quotations);
    }
    if let Some(metadata_terms) = clean_text(metadata_terms) {
        segments.push(metadata_terms);
    }
    segments.join(" ").chars().take(max_chars).collect()
}

fn canonicalize_identifier(identifier: &str) -> String {
    if let Ok(mut url) = url::Url::parse(identifier) {
        url.set_fragment(None);
        url.set_query(None);
        let _ = url.set_username("");
        let _ = url.set_password(None);
        let host = url
            .host_str()
            .map(|value| value.to_lowercase().trim_start_matches("www.").to_string())
            .unwrap_or_default();
        let mut path = url.path().trim_end_matches('/').to_lowercase();
        if path.is_empty() {
            path = "/".to_string();
        }
        format!("{host}{path}")
    } else {
        identifier.trim().to_lowercase()
    }
}

fn content_path_key(identifier: &str) -> String {
    if let Ok(url) = url::Url::parse(identifier) {
        return url.path().trim_end_matches('/').to_lowercase();
    }
    identifier
        .trim()
        .rsplit('/')
        .next()
        .unwrap_or_default()
        .to_lowercase()
}

fn compute_simhash(text: &str) -> u64 {
    let mut weights = [0i32; 64];
    for token in tokenize(text) {
        let digest = Sha256::digest(token.as_bytes());
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&digest[..8]);
        let hash = u64::from_be_bytes(bytes);
        for (idx, weight) in weights.iter_mut().enumerate() {
            if (hash >> idx) & 1 == 1 {
                *weight += 1;
            } else {
                *weight -= 1;
            }
        }
    }
    let mut fingerprint = 0u64;
    for (idx, weight) in weights.iter().enumerate() {
        if *weight >= 0 {
            fingerprint |= 1u64 << idx;
        }
    }
    fingerprint
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .map(|part| part.trim().to_lowercase())
        .filter(|part| part.len() > 1)
        .collect()
}

fn score_hits(query: &str, hits: &mut [SearchHit]) {
    if hits.is_empty() {
        return;
    }
    let query_tokens = tokenize(query);
    let dense_min = hits
        .iter()
        .map(|hit| hit.dense_score)
        .fold(f32::INFINITY, f32::min);
    let dense_max = hits
        .iter()
        .map(|hit| hit.dense_score)
        .fold(f32::NEG_INFINITY, f32::max);
    for hit in hits.iter_mut() {
        hit.lexical_score = lexical_score(query, &query_tokens, hit);
        hit.metadata_score = metadata_score(hit);
        let dense_norm = normalize_score(hit.dense_score, dense_min, dense_max);
        hit.fused_score = 0.55 * dense_norm + 0.25 * hit.lexical_score + 0.20 * hit.metadata_score;
        hit.score = hit.fused_score;
    }
}

fn lexical_score(query: &str, query_tokens: &[String], hit: &SearchHit) -> f32 {
    let doc_text = [
        hit.title.as_deref().unwrap_or_default(),
        hit.summary_text.as_deref().unwrap_or_default(),
        hit.market_context_text.as_deref().unwrap_or_default(),
        hit.document_identifier.as_deref().unwrap_or_default(),
        &hit.theme_tags.join(" "),
    ]
    .join(" ")
    .to_lowercase();
    let doc_tokens = tokenize(&doc_text);
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }
    let doc_set: HashSet<&str> = doc_tokens.iter().map(String::as_str).collect();
    let overlap = query_tokens
        .iter()
        .filter(|token| doc_set.contains(token.as_str()))
        .count() as f32
        / query_tokens.len() as f32;
    let phrase = if doc_text.contains(&query.to_lowercase()) {
        1.0
    } else {
        0.0
    };
    (0.8 * overlap + 0.2 * phrase).clamp(0.0, 1.0)
}

fn metadata_score(hit: &SearchHit) -> f32 {
    let source_bonus = (hit.source_score as f32).clamp(0.1, 1.1);
    let context_bonus = (hit.market_context_score as f32).clamp(0.0, 1.0);
    let finance_tag_bonus =
        if hit.theme_tags.iter().any(|tag| {
            tag.starts_with("ECON_") || tag.starts_with("EPU_") || tag.contains("TREASURY")
        }) {
            1.0
        } else {
            0.0
        };
    (0.45 * source_bonus + 0.35 * context_bonus + 0.20 * finance_tag_bonus).clamp(0.0, 1.0)
}

fn normalize_score(value: f32, min: f32, max: f32) -> f32 {
    if !min.is_finite() || !max.is_finite() || (max - min).abs() < f32::EPSILON {
        1.0
    } else {
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }
}

fn dedup_and_diversify_hits(hits: Vec<SearchHit>) -> Vec<SearchHit> {
    let mut kept = Vec::new();
    let mut seen_urls = HashSet::new();
    let mut seen_paths = HashSet::new();
    for mut hit in hits {
        let canonical = hit
            .canonical_url
            .clone()
            .or_else(|| {
                hit.document_identifier
                    .as_ref()
                    .map(|value| canonicalize_identifier(value))
            })
            .unwrap_or_default();
        let path_key = hit
            .document_identifier
            .as_deref()
            .map(content_path_key)
            .unwrap_or_default();
        let is_duplicate_url = !canonical.is_empty() && !seen_urls.insert(canonical.clone());
        let is_duplicate_path = !path_key.is_empty() && !seen_paths.insert(path_key);
        let is_duplicate_simhash = hit.simhash_u64.map_or(false, |candidate| {
            kept.iter().any(|existing: &SearchHit| {
                existing
                    .simhash_u64
                    .map(|value| (value ^ candidate).count_ones() <= 6)
                    .unwrap_or(false)
            })
        });
        if is_duplicate_url || is_duplicate_path || is_duplicate_simhash {
            continue;
        }
        let same_domain_count = kept
            .iter()
            .filter(|existing| existing.source_domain == hit.source_domain)
            .count();
        if same_domain_count >= 2 {
            hit.fused_score -= 0.05;
            hit.score = hit.fused_score;
        }
        kept.push(hit);
    }
    kept.sort_by(|left, right| right.fused_score.total_cmp(&left.fused_score));
    kept
}

#[cfg(test)]
mod tests {
    use super::{
        collection_name_with_prefix, default_collection_name, derive_summary, derive_title,
        normalize_model_name, normalize_sparse_text, parse_theme_tags, truncate_embedding,
    };

    #[test]
    fn truncate_embedding_reduces_dimension() {
        let embedding = vec![1.0, 2.0, 3.0, 4.0];
        let truncated = truncate_embedding(embedding, 2).expect("truncate should succeed");
        assert_eq!(truncated, vec![1.0, 2.0]);
    }

    #[test]
    fn truncate_embedding_rejects_invalid_dimension() {
        assert!(truncate_embedding(vec![1.0, 2.0], 0).is_err());
        assert!(truncate_embedding(vec![1.0, 2.0], 3).is_err());
    }

    #[test]
    fn parse_theme_tags_splits_and_normalizes() {
        let parsed = parse_theme_tags(Some("rates,macro;usd,fx; energy "));
        assert_eq!(parsed, vec!["RATES", "USD", "ENERGY"]);
    }

    #[test]
    fn normalize_sparse_text_removes_offsets_and_deduplicates() {
        let normalized = normalize_sparse_text(Some(
            "Hormuz disruption,12;WTI,18;Hormuz disruption,24;Oil supply,35",
        ))
        .expect("normalized text");
        assert_eq!(normalized, "Hormuz disruption; WTI; Oil supply");
    }

    #[test]
    fn derive_title_falls_back_to_slug() {
        let title = derive_title(
            None,
            "https://example.com/hormuz-disruption-lifts-oil-risk.html",
            None,
            None,
        )
        .expect("derived title");
        assert_eq!(title, "hormuz disruption lifts oil risk");
    }

    #[test]
    fn derive_summary_uses_sparse_text_when_missing() {
        let summary = derive_summary(None, Some("Hormuz disruption,12;WTI,18"), None)
            .expect("derived summary");
        assert_eq!(summary, "Hormuz disruption; WTI");
    }

    #[test]
    fn default_collection_name_matches_mcp_shape() {
        assert_eq!(
            default_collection_name("2026-06-05", "all-minilm"),
            "news_narrative_v3_20260605_allminilm"
        );
        assert_eq!(
            collection_name_with_prefix("custom_prefix", "2026-06-05", "embeddinggemma:latest"),
            "custom_prefix_20260605_embeddinggemma"
        );
        assert_eq!(
            normalize_model_name("embeddinggemma:latest"),
            "embeddinggemma"
        );
    }
}
