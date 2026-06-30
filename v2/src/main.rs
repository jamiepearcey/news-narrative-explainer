use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::Read;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration as StdDuration;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, TimeZone, Utc};
use clap::{Parser, Subcommand, ValueEnum};
use reqwest::blocking::Client;
use reqwest::Url;
use serde::de::{self, Deserializer};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tiny_http::{Header, Method, Response, Server, StatusCode};
use url::Url as ParsedUrl;

const DEFAULT_THEME_PATTERN: &str = "(ECON_|FINANCE|MARKET|MONETARY_POLICY|CENTRAL_BANK|INFLATION|INTEREST_RATE|YIELD|BOND|STOCK|EQUITY|COMMODITY|OIL|GAS|COPPER|GOLD|CRYPTO|BITCOIN|WAR|CONFLICT|SANCTION|TRADE|ELECTION|REGULATION)";
const SCHEMA_SQL: &str = include_str!("../sql/schema.sql");
const LOAD_GCS_SQL: &str = include_str!("../sql/load_gcs_into_clickhouse.sql");
const LOAD_LOCAL_SQL: &str = include_str!("../sql/load_local_parquet_into_clickhouse.sql");

const MARKET_WRAP_DOMAINS: &[&str] = &[
    "reuters.com",
    "wsj.com",
    "barrons.com",
    "marketwatch.com",
    "apnews.com",
    "ft.com",
    "finance.yahoo.com",
    "investopedia.com",
    "business-standard.com",
    "cnbcafrica.com",
    "moneycontrol.com",
];
const COMMODITY_SPECIALIST_DOMAINS: &[&str] = &[
    "oilandgas360.com",
    "kitco.com",
    "mining.com",
    "oilprice.com",
];
const MARKET_SENTENCE_TERMS: &[&str] = &[
    "YIELD",
    "TREASURY",
    "DOLLAR",
    "USD",
    "NASDAQ",
    "STOCK",
    "EQUITY",
    "OIL",
    "GOLD",
    "COPPER",
    "INFLATION",
    "FED",
    "RATE",
    "RISK OFF",
    "SAFE HAVEN",
    "REAL YIELD",
    "TERM PREMIUM",
    "AI",
    "SEMI",
    "S&P",
    "WALL STREET",
    "QQQ",
    "NASDAQ 100",
];
const ASSET_CONTEXT_REQUIRED: &[&str] = &[
    "WTI", "Brent", "HG", "BDI", "Gold", "BTC", "FXI", "NG", "TTF", "XLE", "XME", "GDX", "CAD",
    "FCX", "BHP", "RIO", "COIN", "NDX", "SPX",
];
const COMPANY_SPECIFIC_TERMS: &[&str] = &[
    "FINANCIAL COMPARISON",
    " VS. ",
    " VS ",
    "COMPARE",
    "EARNINGS PREVIEW",
    "QUARTER ENDED",
    "EBIT MARGIN",
    "(NASDAQ:",
    "(NYSE:",
    "INC.",
    " INC ",
    "CORP",
    "EARNINGS",
    "PLACEMENT",
    "GUIDANCE",
];
const SIMILAR_DAY_FACTORS: &[&str] = &[
    "central_bank_policy",
    "war_conflict",
    "oil",
    "inflation",
    "interest_rates",
    "shipping_disruption",
    "growth_activity",
    "gold_precious",
    "labour_market",
    "sanctions_trade",
];

#[derive(Parser)]
#[command(name = "news-narrative-v2")]
#[command(about = "ClickHouse-first narrative graph pipeline with Rust orchestration.")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Bootstrap,
    ExportBigqueryToGcs {
        #[arg(long)]
        project: String,
        #[arg(long, default_value = "US")]
        location: String,
        #[arg(long)]
        start: String,
        #[arg(long)]
        end: String,
        #[arg(long)]
        bucket: String,
        #[arg(long, default_value = "news-narrative-v2/gdelt")]
        prefix: String,
        #[arg(long, default_value_t = 2_000)]
        rows_per_day: usize,
        #[arg(long, default_value = DEFAULT_THEME_PATTERN)]
        theme_pattern: String,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        include_queries: bool,
    },
    LoadGcsIntoClickhouse {
        #[arg(long)]
        gcs_url: String,
        #[arg(long)]
        start_date: String,
        #[arg(long)]
        end_date: String,
        #[arg(long, env = "GCS_HMAC_ACCESS_KEY")]
        gcs_access_key: String,
        #[arg(long, env = "GCS_HMAC_SECRET_KEY")]
        gcs_secret_key: String,
    },
    LoadLocalParquetIntoClickhouse {
        #[arg(long)]
        input_glob: String,
        #[arg(long)]
        start_date: String,
        #[arg(long)]
        end_date: String,
    },
    EnrichBronze {
        #[arg(long)]
        start_date: String,
        #[arg(long)]
        end_date: String,
        #[arg(long, default_value_t = 2_000)]
        batch_size: usize,
    },
    BuildClickhouseGraph {
        #[arg(long)]
        start_date: String,
        #[arg(long)]
        end_date: String,
        #[arg(long, default_value = "gcs")]
        source_uri: String,
    },
    Query {
        #[arg(long)]
        view: QueryView,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long)]
        asset_label: Option<String>,
        #[arg(long)]
        factor_label: Option<String>,
        #[arg(long)]
        start_date: Option<String>,
        #[arg(long)]
        end_date: Option<String>,
    },
    BenchmarkRustWork {
        #[arg(long, default_value_t = 100_000)]
        iterations: usize,
    },
    McpStdio,
    McpProxy {
        #[arg(long)]
        api_base_url: String,
    },
    ServeApi {
        #[arg(long, default_value = "127.0.0.1:8788")]
        bind: String,
    },
}

#[derive(Clone, ValueEnum)]
enum QueryView {
    Summary,
    TopFactors,
    TopAssets,
    AssetNarratives,
    AssetCrossovers,
    SupportingDocs,
    ExplainMove,
}

#[derive(Debug, Clone, Deserialize)]
struct Taxonomy {
    factors: Vec<FactorDefinition>,
}

#[derive(Debug, Clone, Deserialize)]
struct FactorDefinition {
    id: u32,
    label: String,
    group: String,
    patterns: Vec<String>,
    #[serde(default)]
    asset_hints: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ExportDayResult {
    date: String,
    uri: String,
    query: Option<String>,
    count_query: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BronzeRawRow {
    record_datetime: String,
    partition_date: String,
    ingested_at: String,
    source_common_name: Option<String>,
    document_identifier: String,
    title: Option<String>,
    summary: Option<String>,
    text: Option<String>,
    v2_themes: Option<String>,
    v2_tone: Option<String>,
    v2_locations: Option<String>,
    v2_persons: Option<String>,
    v2_organizations: Option<String>,
    all_names: Option<String>,
    metadata_json: Option<String>,
}

#[derive(Debug, Serialize)]
struct BronzeCandidateRow {
    doc_id: u64,
    record_datetime: String,
    event_time: String,
    partition_date: String,
    source_domain: String,
    document_identifier: String,
    v2_themes: Option<String>,
    v2_tone: Option<String>,
    v2_locations: Option<String>,
    v2_persons: Option<String>,
    v2_organizations: Option<String>,
    all_names: Option<String>,
    title: Option<String>,
    summary_text: Option<String>,
    body_text: Option<String>,
    relevant_text: Option<String>,
    metadata_json: Option<String>,
    gkg_extras: Option<String>,
    sharing_image: Option<String>,
    related_images: Option<String>,
    social_image_embeds: Option<String>,
    social_video_embeds: Option<String>,
    quotations: Option<String>,
    amounts: Option<String>,
    dates: Option<String>,
    gcam: Option<String>,
    translation_info: Option<String>,
    source_type: String,
    source_priority: u8,
    market_context_text: Option<String>,
    market_context_score: f64,
    tone: Option<f64>,
    geo_labels: Vec<String>,
    match_text: String,
    asset_match_text: String,
}

#[derive(Debug, Deserialize)]
struct SingleValueRow {
    value: String,
}

#[derive(Debug, Deserialize)]
struct CountRow {
    #[serde(deserialize_with = "de_u64_from_string_or_number")]
    count: u64,
}

#[derive(Debug, Deserialize)]
struct DocIdRow {
    #[serde(deserialize_with = "de_u64_from_string_or_number")]
    doc_id: u64,
}

#[derive(Debug, Clone)]
struct BronzeCursor {
    document_identifier: String,
    ingested_at: String,
}

#[derive(Debug, Deserialize)]
struct CatalogHitRow {
    #[serde(deserialize_with = "de_u64_from_string_or_number")]
    count: u64,
}

#[derive(Debug, Serialize)]
struct IngestCatalogRecord {
    source_kind: String,
    source_path: String,
    content_sha256: Option<String>,
    file_size_bytes: Option<u64>,
    partition_date: String,
    row_count: Option<u64>,
    status: String,
}

#[derive(Debug, Serialize)]
struct SupportingDoc {
    event_time: String,
    asset_label: String,
    factor_label: String,
    geo_label: String,
    source_domain: String,
    source_type: Option<String>,
    source_priority: Option<u8>,
    document_identifier: String,
    title: Option<String>,
    summary_text: Option<String>,
    body_excerpt: Option<String>,
    market_context_text: Option<String>,
    market_context_score: Option<f64>,
    classification_confidence: Option<f64>,
    relevance_score: f64,
}

#[derive(Debug, Serialize)]
struct BenchmarkResult {
    benchmark: String,
    iterations: usize,
    total_ms: f64,
    per_iteration_us: f64,
    ops_per_sec: f64,
}

static TAXONOMY_CACHE: OnceLock<Taxonomy> = OnceLock::new();
static PRIORITIES_CACHE: OnceLock<HashMap<String, HashMap<String, f64>>> = OnceLock::new();
static ASSET_TEXT_PATTERNS_CACHE: OnceLock<HashMap<String, Vec<String>>> = OnceLock::new();
static ASSET_CUES_CACHE: OnceLock<HashMap<String, Vec<String>>> = OnceLock::new();
static FACTOR_CUES_BY_LABEL_CACHE: OnceLock<HashMap<String, Vec<String>>> = OnceLock::new();

struct ClickHouseClient {
    client: Client,
    base_url: String,
    database: String,
    user: Option<String>,
    password: Option<String>,
}

impl ClickHouseClient {
    fn from_env() -> Result<Self> {
        let base_url =
            env::var("CLICKHOUSE_URL").unwrap_or_else(|_| "http://localhost:8123".to_string());
        let database = env::var("CLICKHOUSE_DATABASE").unwrap_or_else(|_| "default".to_string());
        let user = env::var("CLICKHOUSE_USER").ok();
        let password = env::var("CLICKHOUSE_PASSWORD").ok();
        let timeout_seconds = env::var("CLICKHOUSE_TIMEOUT_SECONDS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(600);
        Ok(Self {
            client: Client::builder()
                .timeout(StdDuration::from_secs(timeout_seconds))
                .build()?,
            base_url,
            database,
            user,
            password,
        })
    }

    fn execute(&self, sql: &str) -> Result<()> {
        let mut url = Url::parse(&self.base_url)?;
        url.query_pairs_mut()
            .append_pair("database", &self.database);
        let mut request = self.client.post(url).body(sql.to_string());
        if let Some(user) = &self.user {
            request = request.basic_auth(user, self.password.clone());
        }
        let response = request
            .send()
            .context("failed to send ClickHouse request")?;
        let status = response.status();
        let body = response.text().unwrap_or_default();
        if !status.is_success() {
            bail!("ClickHouse query failed with {status}: {body}");
        }
        Ok(())
    }

    fn execute_many(&self, sql: &str) -> Result<()> {
        for statement in sql.split(';') {
            let trimmed = statement.trim();
            if trimmed.is_empty() {
                continue;
            }
            self.execute(trimmed)?;
        }
        Ok(())
    }

    fn select_rows<T: DeserializeOwned>(&self, sql: &str) -> Result<Vec<T>> {
        let mut url = Url::parse(&self.base_url)?;
        let query = format!("{sql} FORMAT JSONEachRow");
        url.query_pairs_mut()
            .append_pair("database", &self.database)
            .append_pair("query", &query);
        let mut request = self.client.post(url).body(String::new());
        if let Some(user) = &self.user {
            request = request.basic_auth(user, self.password.clone());
        }
        let response = request.send().context("failed to send ClickHouse select")?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            bail!("ClickHouse select failed with {status}: {body}");
        }
        let mut rows = Vec::new();
        let reader = BufReader::new(response);
        for line in reader.lines() {
            let line = line.context("failed to read ClickHouse response line")?;
            if line.trim().is_empty() {
                continue;
            }
            rows.push(
                serde_json::from_str::<T>(&line)
                    .with_context(|| format!("invalid JSONEachRow line: {line}"))?,
            );
        }
        Ok(rows)
    }

    fn insert_json_each_row<T: Serialize>(&self, table: &str, rows: &[T]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut url = Url::parse(&self.base_url)?;
        let query = format!("INSERT INTO {table} FORMAT JSONEachRow");
        url.query_pairs_mut()
            .append_pair("database", &self.database)
            .append_pair("query", &query);
        let mut payload = String::new();
        for row in rows {
            payload.push_str(&serde_json::to_string(row)?);
            payload.push('\n');
        }
        let mut request = self.client.post(url).body(payload);
        if let Some(user) = &self.user {
            request = request.basic_auth(user, self.password.clone());
        }
        let response = request.send().context("failed to insert JSONEachRow")?;
        let status = response.status();
        let body = response.text().unwrap_or_default();
        if !status.is_success() {
            bail!("ClickHouse insert failed with {status}: {body}");
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Bootstrap => bootstrap(),
        Commands::ExportBigqueryToGcs {
            project,
            location,
            start,
            end,
            bucket,
            prefix,
            rows_per_day,
            theme_pattern,
            dry_run,
            include_queries,
        } => export_bigquery_to_gcs(
            &project,
            &location,
            &start,
            &end,
            &bucket,
            &prefix,
            rows_per_day,
            &theme_pattern,
            dry_run,
            include_queries,
        ),
        Commands::LoadGcsIntoClickhouse {
            gcs_url,
            start_date,
            end_date,
            gcs_access_key,
            gcs_secret_key,
        } => load_gcs_into_clickhouse(
            &gcs_url,
            &start_date,
            &end_date,
            &gcs_access_key,
            &gcs_secret_key,
        ),
        Commands::LoadLocalParquetIntoClickhouse {
            input_glob,
            start_date,
            end_date,
        } => load_local_parquet_into_clickhouse(&input_glob, &start_date, &end_date),
        Commands::EnrichBronze {
            start_date,
            end_date,
            batch_size,
        } => enrich_bronze(&start_date, &end_date, batch_size),
        Commands::BuildClickhouseGraph {
            start_date,
            end_date,
            source_uri,
        } => build_clickhouse_graph(&start_date, &end_date, &source_uri),
        Commands::Query {
            view,
            limit,
            asset_label,
            factor_label,
            start_date,
            end_date,
        } => run_query_view(view, limit, asset_label, factor_label, start_date, end_date),
        Commands::BenchmarkRustWork { iterations } => benchmark_rust_work(iterations),
        Commands::McpStdio => run_mcp_stdio_local(),
        Commands::McpProxy { api_base_url } => run_mcp_stdio_proxy(&api_base_url),
        Commands::ServeApi { bind } => serve_mcp_api(&bind),
    }
}

fn bootstrap() -> Result<()> {
    let client = ClickHouseClient::from_env()?;
    client.execute_many(SCHEMA_SQL)?;
    load_taxonomy_tables(&client)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "ok",
            "database": client.database,
            "schema": "applied",
            "taxonomy": "loaded"
        }))?
    );
    Ok(())
}

fn export_bigquery_to_gcs(
    project: &str,
    location: &str,
    start: &str,
    end: &str,
    bucket: &str,
    prefix: &str,
    rows_per_day: usize,
    theme_pattern: &str,
    dry_run: bool,
    include_queries: bool,
) -> Result<()> {
    let start = parse_datetime(start)?;
    let end = parse_datetime(end)?;
    if start >= end {
        bail!("--start must be before --end");
    }
    let days = iter_window_days(start, end)?;
    let mut results = Vec::new();
    for day in days {
        let uri = format!(
            "gs://{bucket}/{}/dt={}/part-*.parquet",
            prefix.trim_matches('/'),
            day.format("%Y-%m-%d")
        );
        let export_sql = build_bigquery_export_sql(day, &uri, theme_pattern, rows_per_day);
        let count_sql = build_bigquery_count_sql(day, theme_pattern);
        if !dry_run {
            run_bq_query(project, location, &export_sql)?;
        }
        results.push(ExportDayResult {
            date: day.format("%Y-%m-%d").to_string(),
            uri,
            query: include_queries.then_some(export_sql),
            count_query: include_queries.then_some(count_sql),
        });
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "project": project,
            "location": location,
            "start": start.to_rfc3339(),
            "end": end.to_rfc3339(),
            "rows_per_day": rows_per_day,
            "dry_run": dry_run,
            "day_results": results
        }))?
    );
    Ok(())
}

fn load_gcs_into_clickhouse(
    gcs_url: &str,
    start_date: &str,
    end_date: &str,
    access_key: &str,
    secret_key: &str,
) -> Result<()> {
    let client = ClickHouseClient::from_env()?;
    let sql = LOAD_GCS_SQL
        .replace("__GCS_URL__", gcs_url)
        .replace("__GCS_ACCESS_KEY__", access_key)
        .replace("__GCS_SECRET_KEY__", secret_key)
        .replace("__START_DATE__", start_date)
        .replace("__END_DATE__", end_date);
    client.execute(&sql)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "ok",
            "loaded_from": gcs_url,
            "start_date": start_date,
            "end_date": end_date
        }))?
    );
    Ok(())
}

fn load_local_parquet_into_clickhouse(
    input_glob: &str,
    start_date: &str,
    end_date: &str,
) -> Result<()> {
    let client = ClickHouseClient::from_env()?;
    let files = resolve_local_input_files(input_glob, start_date, end_date)?;
    let mut loaded_files = Vec::new();
    let mut skipped_files = Vec::new();
    for file in files {
        let checksum = sha256_file_hex(&file.host_path)?;
        if local_file_already_loaded(
            &client,
            &file.clickhouse_path,
            &checksum,
            &file.partition_date,
        )? {
            skipped_files.push(file.clickhouse_path);
            continue;
        }
        let row_count =
            count_local_parquet_rows(&client, &file.clickhouse_path, &file.partition_date)?;
        let sql = LOAD_LOCAL_SQL
            .replace("__LOCAL_GLOB__", &file.clickhouse_path)
            .replace("__START_DATE__", start_date)
            .replace("__END_DATE__", end_date);
        client.execute(&sql)?;
        let record = IngestCatalogRecord {
            source_kind: "local_parquet".to_string(),
            source_path: file.clickhouse_path.clone(),
            content_sha256: Some(checksum),
            file_size_bytes: Some(file.file_size_bytes),
            partition_date: file.partition_date.clone(),
            row_count: Some(row_count),
            status: "loaded".to_string(),
        };
        client.insert_json_each_row("ingest_file_catalog", &[record])?;
        loaded_files.push(json!({
            "path": file.clickhouse_path,
            "partition_date": file.partition_date,
            "file_size_bytes": file.file_size_bytes,
            "row_count": row_count
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "ok",
            "loaded_from": input_glob,
            "start_date": start_date,
            "end_date": end_date,
            "loaded_files": loaded_files,
            "skipped_already_loaded_files": skipped_files
        }))?
    );
    Ok(())
}

fn enrich_bronze(start_date: &str, end_date: &str, batch_size: usize) -> Result<()> {
    if batch_size == 0 {
        bail!("--batch-size must be positive");
    }
    let client = ClickHouseClient::from_env()?;
    let mut inserted_rows = 0usize;
    for partition_date in partition_dates_in_window(start_date, end_date)? {
        let existing_ids = existing_doc_ids_for_partition(&client, &partition_date)?;
        let mut cursor: Option<BronzeCursor> = None;
        loop {
            let raw_rows =
                fetch_bronze_raw_batch(&client, &partition_date, cursor.as_ref(), batch_size)?;
            if raw_rows.is_empty() {
                break;
            }
            let mut transformed = Vec::with_capacity(raw_rows.len());
            for row in raw_rows.iter() {
                let candidate = transform_bronze_row(row)?;
                if existing_ids.contains(&candidate.doc_id) {
                    continue;
                }
                transformed.push(candidate);
            }
            client.insert_json_each_row("bronze_candidates", &transformed)?;
            inserted_rows += transformed.len();
            cursor = raw_rows.last().map(|row| BronzeCursor {
                document_identifier: row.document_identifier.clone(),
                ingested_at: row.ingested_at.clone(),
            });
            if raw_rows.len() < batch_size {
                break;
            }
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "ok",
            "inserted_rows": inserted_rows,
            "start_date": start_date,
            "end_date": end_date,
            "batch_size": batch_size
        }))?
    );
    Ok(())
}

fn build_clickhouse_graph(start_date: &str, end_date: &str, source_uri: &str) -> Result<()> {
    let client = ClickHouseClient::from_env()?;
    let bronze_partitions = client.select_rows::<SingleValueRow>(&format!(
        "SELECT DISTINCT toString(partition_date) AS value
         FROM bronze_candidates
         WHERE partition_date >= toDate('{start_date}')
           AND partition_date < toDate('{end_date}')
         ORDER BY partition_date"
    ))?;
    let already_processed: HashSet<String> = client
        .select_rows::<SingleValueRow>(&format!(
            "SELECT DISTINCT toString(partition_date) AS value
             FROM graph_build_partitions
             WHERE partition_date >= toDate('{start_date}')
               AND partition_date < toDate('{end_date}')"
        ))?
        .into_iter()
        .map(|row| row.value)
        .collect();
    let partitions: Vec<String> = bronze_partitions
        .into_iter()
        .map(|row| row.value)
        .filter(|value| !already_processed.contains(value))
        .collect();
    if partitions.is_empty() {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "status": "noop",
                "reason": "no unprocessed partitions in window"
            }))?
        );
        return Ok(());
    }
    let silver_event_sql = build_silver_event_graph_sql(&partitions);
    let silver_factor_sql = build_silver_factor_mentions_sql(&partitions);
    let silver_asset_sql = build_silver_asset_factor_mentions_sql(&partitions);
    let market_context_sql = build_silver_market_context_mentions_sql(&partitions);
    client.execute(&silver_event_sql)?;
    client.execute(&silver_factor_sql)?;
    client.execute(&silver_asset_sql)?;
    client.execute(&market_context_sql)?;
    let build_rows: Vec<Value> = partitions
        .iter()
        .map(|partition| {
            json!({
                "partition_date": partition,
                "source_uri": source_uri
            })
        })
        .collect();
    client.insert_json_each_row(
        "graph_build_partitions (partition_date, source_uri)",
        &build_rows,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "ok",
            "processed_partitions": partitions,
            "source_uri": source_uri
        }))?
    );
    Ok(())
}

fn partition_dates_in_window(start_date: &str, end_date: &str) -> Result<Vec<String>> {
    let start = parse_datetime(start_date)?;
    let end = parse_datetime(end_date)?;
    Ok(iter_window_days(start, end)?
        .into_iter()
        .map(|day| day.format("%Y-%m-%d").to_string())
        .collect())
}

fn existing_doc_ids_for_partition(
    client: &ClickHouseClient,
    partition_date: &str,
) -> Result<HashSet<u64>> {
    Ok(client
        .select_rows::<DocIdRow>(&format!(
            "SELECT doc_id
             FROM bronze_candidates
             WHERE partition_date = toDate('{partition_date}')"
        ))?
        .into_iter()
        .map(|row| row.doc_id)
        .collect())
}

fn fetch_bronze_raw_batch(
    client: &ClickHouseClient,
    partition_date: &str,
    cursor: Option<&BronzeCursor>,
    batch_size: usize,
) -> Result<Vec<BronzeRawRow>> {
    let cursor_filter = cursor
        .map(|cursor| {
            format!(
                "AND (
                    bronze_raw_gdelt.document_identifier > '{doc}'
                    OR (
                        bronze_raw_gdelt.document_identifier = '{doc}'
                        AND bronze_raw_gdelt.ingested_at > parseDateTime64BestEffort('{ingested_at}')
                    )
                )",
                doc = escape_sql(&cursor.document_identifier),
                ingested_at = escape_sql(&cursor.ingested_at),
            )
        })
        .unwrap_or_default();
    client.select_rows::<BronzeRawRow>(&format!(
        "SELECT
            record_datetime,
            toString(bronze_raw_gdelt.partition_date) AS partition_date,
            toString(ingested_at) AS ingested_at,
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
            all_names,
            metadata_json
        FROM bronze_raw_gdelt
        WHERE bronze_raw_gdelt.partition_date = toDate('{partition_date}')
          {cursor_filter}
        ORDER BY bronze_raw_gdelt.document_identifier ASC, bronze_raw_gdelt.ingested_at ASC
        LIMIT {batch_size}"
    ))
}

fn run_query_view(
    view: QueryView,
    limit: usize,
    asset_label: Option<String>,
    factor_label: Option<String>,
    start_date: Option<String>,
    end_date: Option<String>,
) -> Result<()> {
    let client = ClickHouseClient::from_env()?;
    let payload = match view {
        QueryView::Summary => query_summary(&client)?,
        QueryView::TopFactors => json!(query_top_factors(&client, limit)?),
        QueryView::TopAssets => json!(query_top_assets(&client, limit)?),
        QueryView::AssetNarratives => {
            let asset_label = asset_label.context("--asset-label is required")?;
            json!(query_asset_narratives(
                &client,
                &asset_label,
                start_date.as_deref(),
                end_date.as_deref(),
                limit
            )?)
        }
        QueryView::AssetCrossovers => {
            let asset_label = asset_label.context("--asset-label is required")?;
            json!(query_asset_crossovers(
                &client,
                &asset_label,
                factor_label.as_deref(),
                start_date.as_deref(),
                end_date.as_deref(),
                limit,
            )?)
        }
        QueryView::SupportingDocs => {
            let asset_label = asset_label.context("--asset-label is required")?;
            json!(query_supporting_docs(
                &client,
                &asset_label,
                factor_label.as_deref(),
                start_date.as_deref(),
                end_date.as_deref(),
                limit,
            )?)
        }
        QueryView::ExplainMove => {
            let asset_label = asset_label.context("--asset-label is required")?;
            json!({
                "asset_label": asset_label,
                "window": {
                    "start_date": start_date,
                    "end_date": end_date,
                },
                "top_narratives": query_asset_narratives(&client, &asset_label, start_date.as_deref(), end_date.as_deref(), limit)?,
                "supporting_docs": query_supporting_docs(&client, &asset_label, None, start_date.as_deref(), end_date.as_deref(), limit)?,
                "crossovers": query_asset_crossovers(&client, &asset_label, None, start_date.as_deref(), end_date.as_deref(), limit)?,
            })
        }
    };
    println!("{}", serde_json::to_string_pretty(&payload)?);
    Ok(())
}

fn benchmark_rust_work(iterations: usize) -> Result<()> {
    if iterations == 0 {
        bail!("--iterations must be positive");
    }
    let sample = sample_bronze_raw_row();
    let sample_doc = sample_supporting_doc_row();
    let factor_patterns = factor_cues_by_label()
        .get("oil")
        .map(Vec::as_slice)
        .unwrap_or(&[]);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = transform_bronze_row(&sample)?;
    }
    let transform_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = supporting_doc_relevance(&sample_doc, "WTI", factor_patterns);
    }
    let relevance_elapsed = start.elapsed();

    let results = vec![
        to_benchmark_result("transform_bronze_row", iterations, transform_elapsed),
        to_benchmark_result("supporting_doc_relevance", iterations, relevance_elapsed),
    ];
    println!("{}", serde_json::to_string_pretty(&results)?);
    Ok(())
}

fn to_benchmark_result(
    name: &str,
    iterations: usize,
    elapsed: std::time::Duration,
) -> BenchmarkResult {
    let total_ms = elapsed.as_secs_f64() * 1_000.0;
    let per_iteration_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
    BenchmarkResult {
        benchmark: name.to_string(),
        iterations,
        total_ms,
        per_iteration_us,
        ops_per_sec,
    }
}

fn load_taxonomy_tables(client: &ClickHouseClient) -> Result<()> {
    let taxonomy = load_taxonomy()?;
    client.execute("TRUNCATE TABLE factor_dictionary")?;
    client.execute("TRUNCATE TABLE factor_rule_patterns")?;
    client.execute("TRUNCATE TABLE factor_rule_assets")?;
    client.execute("TRUNCATE TABLE asset_rule_patterns")?;
    client.execute("TRUNCATE TABLE asset_context_required")?;

    let factor_rows: Vec<Value> = taxonomy
        .factors
        .iter()
        .map(|factor| {
            json!({
                "factor_id": factor.id,
                "factor_label": factor.label,
                "factor_group": factor.group
            })
        })
        .collect();
    client.insert_json_each_row("factor_dictionary", &factor_rows)?;

    let mut pattern_rows = Vec::new();
    let mut asset_rows = Vec::new();
    for factor in &taxonomy.factors {
        for pattern in &factor.patterns {
            pattern_rows.push(json!({
                "factor_id": factor.id,
                "factor_label": factor.label,
                "factor_group": factor.group,
                "pattern": pattern.to_uppercase()
            }));
        }
        for asset in &factor.asset_hints {
            asset_rows.push(json!({
                "factor_id": factor.id,
                "asset_label": asset
            }));
        }
    }
    client.insert_json_each_row("factor_rule_patterns", &pattern_rows)?;
    client.insert_json_each_row("factor_rule_assets", &asset_rows)?;

    let mut asset_pattern_rows = Vec::new();
    for (asset, patterns) in asset_text_patterns() {
        for pattern in patterns {
            asset_pattern_rows.push(json!({
                "asset_label": asset,
                "pattern": pattern.to_uppercase()
            }));
        }
        asset_pattern_rows.push(json!({
            "asset_label": asset,
            "pattern": asset.to_uppercase()
        }));
    }
    client.insert_json_each_row("asset_rule_patterns", &asset_pattern_rows)?;

    let context_rows: Vec<Value> = ASSET_CONTEXT_REQUIRED
        .iter()
        .map(|asset| json!({ "asset_label": asset }))
        .collect();
    client.insert_json_each_row("asset_context_required", &context_rows)?;
    Ok(())
}

fn load_taxonomy() -> Result<Taxonomy> {
    if let Some(cached) = TAXONOMY_CACHE.get() {
        return Ok(cached.clone());
    }
    let path = project_root().join("config/news_narrative_taxonomy.json");
    let parsed: Taxonomy = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let _ = TAXONOMY_CACHE.set(parsed.clone());
    Ok(parsed)
}

fn load_priorities() -> Result<HashMap<String, HashMap<String, f64>>> {
    if let Some(cached) = PRIORITIES_CACHE.get() {
        return Ok(cached.clone());
    }
    let path = project_root().join("config/asset_narrative_priorities.json");
    let text = std::fs::read_to_string(path)?;
    let parsed: HashMap<String, HashMap<String, f64>> = serde_json::from_str(&text)?;
    let _ = PRIORITIES_CACHE.set(parsed.clone());
    Ok(parsed)
}

#[derive(Debug)]
struct LocalInputFile {
    host_path: PathBuf,
    clickhouse_path: String,
    partition_date: String,
    file_size_bytes: u64,
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("v2 should have parent project directory")
        .to_path_buf()
}

fn escape_sql(value: &str) -> String {
    value.replace('\'', "''")
}

fn de_u64_from_string_or_number<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::Number(number) => number
            .as_u64()
            .ok_or_else(|| de::Error::custom("expected unsigned integer")),
        Value::String(text) => text
            .parse::<u64>()
            .map_err(|error| de::Error::custom(format!("invalid u64 string: {error}"))),
        _ => Err(de::Error::custom("expected string or number for u64")),
    }
}

fn resolve_local_input_files(
    input_glob: &str,
    start_date: &str,
    end_date: &str,
) -> Result<Vec<LocalInputFile>> {
    let host_glob = resolve_host_glob(input_glob)?;
    let start = NaiveDate::parse_from_str(start_date, "%Y-%m-%d")?;
    let end = NaiveDate::parse_from_str(end_date, "%Y-%m-%d")?;
    let mut files = Vec::new();
    for entry in glob::glob(&host_glob)? {
        let path = entry?;
        let partition_date = partition_date_from_path(&path)?;
        let day = NaiveDate::parse_from_str(&partition_date, "%Y-%m-%d")?;
        if day < start || day >= end {
            continue;
        }
        let metadata = std::fs::metadata(&path)?;
        files.push(LocalInputFile {
            clickhouse_path: host_relative_clickhouse_path(&path)?,
            host_path: path,
            partition_date,
            file_size_bytes: metadata.len(),
        });
    }
    files.sort_by(|left, right| left.clickhouse_path.cmp(&right.clickhouse_path));
    Ok(files)
}

fn resolve_host_glob(input_glob: &str) -> Result<String> {
    let data_root = project_root().join("data");
    if let Some(rest) = input_glob.strip_prefix("gdelt_candidates_20d_full/") {
        return Ok(data_root
            .join("gdelt_candidates_20d_full")
            .join(rest)
            .to_string_lossy()
            .to_string());
    }
    if let Some(rest) = input_glob.strip_prefix("gdelt_candidates/") {
        return Ok(data_root
            .join("gdelt_candidates")
            .join(rest)
            .to_string_lossy()
            .to_string());
    }
    bail!("unsupported local input glob root: {input_glob}");
}

fn host_relative_clickhouse_path(host_path: &Path) -> Result<String> {
    let data_root = project_root().join("data");
    let corpus_20d_root = data_root.join("gdelt_candidates_20d_full");
    let candidates_root = data_root.join("gdelt_candidates");
    if let Ok(relative) = host_path.strip_prefix(&corpus_20d_root) {
        return Ok(Path::new("gdelt_candidates_20d_full")
            .join(relative)
            .to_string_lossy()
            .to_string());
    }
    if let Ok(relative) = host_path.strip_prefix(&candidates_root) {
        return Ok(Path::new("gdelt_candidates")
            .join(relative)
            .to_string_lossy()
            .to_string());
    }
    bail!(
        "could not map host path into clickhouse user_files path: {}",
        host_path.display()
    )
}

fn partition_date_from_path(path: &Path) -> Result<String> {
    for component in path.components() {
        let text = component.as_os_str().to_string_lossy();
        if let Some(value) = text.strip_prefix("dt=") {
            return Ok(value.to_string());
        }
    }
    bail!(
        "could not detect partition date from path {}",
        path.display()
    )
}

fn sha256_file_hex(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn local_file_already_loaded(
    client: &ClickHouseClient,
    clickhouse_path: &str,
    checksum: &str,
    partition_date: &str,
) -> Result<bool> {
    let rows = client.select_rows::<CatalogHitRow>(&format!(
        "SELECT count() AS count
         FROM ingest_file_catalog
         WHERE source_kind = 'local_parquet'
           AND source_path = '{path}'
           AND content_sha256 = '{checksum}'
           AND partition_date = toDate('{partition_date}')
           AND status = 'loaded'",
        path = escape_sql(clickhouse_path),
        checksum = escape_sql(checksum),
    ))?;
    Ok(rows.first().map(|row| row.count > 0).unwrap_or(false))
}

fn count_local_parquet_rows(
    client: &ClickHouseClient,
    clickhouse_path: &str,
    partition_date: &str,
) -> Result<u64> {
    let rows = client.select_rows::<CountRow>(&format!(
        "SELECT count() AS count
         FROM file('{path}', 'Parquet')
         WHERE toDate(partition_date) = toDate('{partition_date}')",
        path = escape_sql(clickhouse_path),
        partition_date = escape_sql(partition_date),
    ))?;
    Ok(rows.first().map(|row| row.count).unwrap_or(0))
}

fn parse_datetime(value: &str) -> Result<DateTime<Utc>> {
    if let Ok(parsed) = DateTime::parse_from_rfc3339(value) {
        return Ok(parsed.with_timezone(&Utc));
    }
    if let Ok(date) = NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        return Ok(Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0).unwrap()));
    }
    bail!("unsupported datetime format: {value}")
}

fn iter_window_days(start: DateTime<Utc>, end: DateTime<Utc>) -> Result<Vec<NaiveDate>> {
    if start >= end {
        bail!("start must be before end");
    }
    let mut days = Vec::new();
    let mut cursor = start.date_naive();
    while cursor < end.date_naive() {
        days.push(cursor);
        cursor += Duration::days(1);
    }
    Ok(days)
}

fn build_bigquery_export_sql(
    day: NaiveDate,
    uri: &str,
    theme_pattern: &str,
    rows_per_day: usize,
) -> String {
    let start = format!("{}T00:00:00Z", day.format("%Y-%m-%d"));
    let end = format!("{}T00:00:00Z", (day + Duration::days(1)).format("%Y-%m-%d"));
    format!(
        "EXPORT DATA OPTIONS(
  uri='{uri}',
  format='PARQUET',
  overwrite=true
) AS
SELECT
  CAST(DATE AS STRING) AS record_datetime,
  CAST(DATE(_PARTITIONTIME) AS STRING) AS partition_date,
  SourceCommonName AS source_common_name,
  DocumentIdentifier AS document_identifier,
  CAST(NULL AS STRING) AS title,
  CAST(NULL AS STRING) AS summary,
  CAST(NULL AS STRING) AS text,
  V2Themes AS v2_themes,
  V2Tone AS v2_tone,
  V2Locations AS v2_locations,
  V2Persons AS v2_persons,
  V2Organizations AS v2_organizations,
  AllNames AS all_names,
  TO_JSON_STRING(STRUCT(
    SourceCollectionIdentifier AS source_collection_identifier,
    Counts AS counts,
    V2Counts AS v2_counts,
    Dates AS dates,
    GCAM AS gcam,
    SharingImage AS sharing_image,
    RelatedImages AS related_images,
    SocialImageEmbeds AS social_image_embeds,
    SocialVideoEmbeds AS social_video_embeds,
    Quotations AS quotations,
    Amounts AS amounts,
    TranslationInfo AS translation_info,
    Extras AS extras,
    CURRENT_TIMESTAMP() AS fetched_at,
    'gdelt-bq.gdeltv2.gkg_partitioned' AS source_table
  )) AS metadata_json
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= TIMESTAMP('{start}')
  AND _PARTITIONTIME < TIMESTAMP('{end}')
  AND REGEXP_CONTAINS(IFNULL(V2Themes, ''), r'{theme_pattern}')
LIMIT {rows_per_day}"
    )
}

fn build_bigquery_count_sql(day: NaiveDate, theme_pattern: &str) -> String {
    let start = format!("{}T00:00:00Z", day.format("%Y-%m-%d"));
    let end = format!("{}T00:00:00Z", (day + Duration::days(1)).format("%Y-%m-%d"));
    format!(
        "SELECT COUNT(*) AS total_rows
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= TIMESTAMP('{start}')
  AND _PARTITIONTIME < TIMESTAMP('{end}')
  AND REGEXP_CONTAINS(IFNULL(V2Themes, ''), r'{theme_pattern}')"
    )
}

fn run_bq_query(project: &str, location: &str, sql: &str) -> Result<()> {
    let output = Command::new("bq")
        .args([
            "query",
            "--use_legacy_sql=false",
            "--project_id",
            project,
            "--location",
            location,
            sql,
        ])
        .output()
        .context("failed to execute bq query")?;
    if !output.status.success() {
        bail!(
            "bq query failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

fn transform_bronze_row(row: &BronzeRawRow) -> Result<BronzeCandidateRow> {
    let metadata = row
        .metadata_json
        .as_ref()
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .unwrap_or(Value::Null);
    let gkg_extras = metadata_string(&metadata, &["extras", "gkg_extras"]);
    let source_domain =
        extract_source_domain(row.source_common_name.as_deref(), &row.document_identifier)?;
    let page_title = extract_page_title(gkg_extras.as_deref());
    let title = normalize_html_text(row.title.as_deref().or(page_title.as_deref()));
    let summary_text = normalize_html_text(row.summary.as_deref());
    let body_text = normalize_html_text(row.text.as_deref());
    let relevant_text = build_relevant_text(
        title.as_deref(),
        summary_text.as_deref(),
        body_text.as_deref(),
        row.all_names.as_deref(),
        row.v2_organizations.as_deref(),
        row.v2_persons.as_deref(),
        row.v2_themes.as_deref(),
        row.v2_locations.as_deref(),
    );
    let (source_type, source_priority) = classify_source_type(
        &source_domain,
        title.as_deref(),
        Some(&row.document_identifier),
    );
    let (market_context_text, market_context_score) = extract_market_context_text(
        title.as_deref(),
        summary_text.as_deref(),
        body_text.as_deref(),
        relevant_text.as_deref(),
    );
    let geo_labels = extract_geo_labels(row.v2_locations.as_deref());
    let match_text = uppercase_join(&[
        row.v2_themes.as_deref(),
        row.v2_persons.as_deref(),
        row.v2_organizations.as_deref(),
        row.all_names.as_deref(),
        row.v2_locations.as_deref(),
    ]);
    let asset_match_text = uppercase_join(&[
        title.as_deref(),
        summary_text.as_deref(),
        body_text.as_deref(),
        relevant_text.as_deref(),
        row.v2_themes.as_deref(),
        row.v2_persons.as_deref(),
        row.v2_organizations.as_deref(),
        row.all_names.as_deref(),
        row.v2_locations.as_deref(),
        gkg_extras.as_deref(),
        Some(row.document_identifier.as_str()),
    ]);
    let event_time = parse_record_datetime(&row.record_datetime, &row.partition_date)?;
    let event_time_text = event_time.format("%Y-%m-%d %H:%M:%S%.3f").to_string();
    let doc_id = stable_u64(&row.document_identifier);
    Ok(BronzeCandidateRow {
        doc_id,
        record_datetime: row.record_datetime.clone(),
        event_time: event_time_text,
        partition_date: row.partition_date.clone(),
        source_domain,
        document_identifier: row.document_identifier.clone(),
        v2_themes: row.v2_themes.clone(),
        v2_tone: row.v2_tone.clone(),
        v2_locations: row.v2_locations.clone(),
        v2_persons: row.v2_persons.clone(),
        v2_organizations: row.v2_organizations.clone(),
        all_names: row.all_names.clone(),
        title,
        summary_text,
        body_text,
        relevant_text,
        metadata_json: row.metadata_json.clone(),
        gkg_extras,
        sharing_image: metadata_string(&metadata, &["sharing_image", "SharingImage"]),
        related_images: metadata_string(&metadata, &["related_images", "RelatedImages"]),
        social_image_embeds: metadata_string(
            &metadata,
            &["social_image_embeds", "SocialImageEmbeds"],
        ),
        social_video_embeds: metadata_string(
            &metadata,
            &["social_video_embeds", "SocialVideoEmbeds"],
        ),
        quotations: metadata_string(&metadata, &["quotations", "Quotations"]),
        amounts: metadata_string(&metadata, &["amounts", "Amounts"]),
        dates: metadata_string(&metadata, &["dates", "Dates"]),
        gcam: metadata_string(&metadata, &["gcam", "GCAM"]),
        translation_info: metadata_string(&metadata, &["translation_info", "TranslationInfo"]),
        source_type,
        source_priority,
        market_context_text,
        market_context_score,
        tone: parse_tone(row.v2_tone.as_deref()),
        geo_labels,
        match_text,
        asset_match_text,
    })
}

fn metadata_string(metadata: &Value, keys: &[&str]) -> Option<String> {
    let object = metadata.as_object()?;
    for key in keys {
        if let Some(value) = object.get(*key) {
            if let Some(text) = value.as_str() {
                return normalize_html_text(Some(text));
            }
            if !value.is_null() {
                return normalize_html_text(Some(&value.to_string()));
            }
        }
    }
    None
}

fn stable_u64(text: &str) -> u64 {
    let mut digest = Sha256::new();
    digest.update(text.as_bytes());
    let bytes = digest.finalize();
    u64::from_be_bytes(
        bytes[..8]
            .try_into()
            .expect("sha256 digest should be 32 bytes"),
    )
}

fn parse_tone(raw: Option<&str>) -> Option<f64> {
    let raw = raw?.split(',').next()?.trim();
    raw.parse::<f64>().ok()
}

fn parse_record_datetime(raw: &str, partition_date: &str) -> Result<DateTime<Utc>> {
    for format in ["%Y%m%d%H%M%S", "%Y%m%d"] {
        if let Ok(parsed) = NaiveDateTime::parse_from_str(raw, format) {
            return Ok(Utc.from_utc_datetime(&parsed));
        }
    }
    let date = NaiveDate::parse_from_str(partition_date, "%Y-%m-%d")?;
    Ok(Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0).unwrap()))
}

fn extract_source_domain(
    source_common_name: Option<&str>,
    document_identifier: &str,
) -> Result<String> {
    if let Some(name) = source_common_name {
        let trimmed = name.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_lowercase());
        }
    }
    let parsed = ParsedUrl::parse(document_identifier).ok();
    Ok(parsed
        .and_then(|url| url.host_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string())
        .to_lowercase())
}

fn normalize_html_text(raw: Option<&str>) -> Option<String> {
    let raw = raw?;
    let decoded = html_escape::decode_html_entities(raw).to_string();
    let collapsed = decoded.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        None
    } else {
        Some(collapsed)
    }
}

fn extract_page_title(gkg_extras: Option<&str>) -> Option<String> {
    let extras = gkg_extras?;
    let start_tag = "<PAGE_TITLE>";
    let end_tag = "</PAGE_TITLE>";
    let start = extras.find(start_tag)? + start_tag.len();
    let end = extras[start..].find(end_tag)? + start;
    normalize_html_text(Some(&extras[start..end]))
}

fn build_relevant_text(
    title: Option<&str>,
    summary_text: Option<&str>,
    body_text: Option<&str>,
    all_names: Option<&str>,
    organizations: Option<&str>,
    persons: Option<&str>,
    themes: Option<&str>,
    locations: Option<&str>,
) -> Option<String> {
    let mut parts = Vec::new();
    for part in [
        title,
        summary_text,
        body_text.map(|text| {
            let end = text.len().min(4_000);
            &text[..end]
        }),
        all_names,
        organizations,
        persons,
        themes,
        locations,
    ] {
        if let Some(value) = part {
            if !value.trim().is_empty() {
                parts.push(value.trim());
            }
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" || "))
    }
}

fn classify_source_type(
    source_domain: &str,
    title: Option<&str>,
    document_identifier: Option<&str>,
) -> (String, u8) {
    let title_upper = title.unwrap_or_default().to_uppercase();
    let url = document_identifier.unwrap_or_default().to_lowercase();
    if MARKET_WRAP_DOMAINS.contains(&source_domain) {
        return ("market_wrap".to_string(), 5);
    }
    if COMMODITY_SPECIALIST_DOMAINS.contains(&source_domain) {
        return ("commodity_specialist".to_string(), 4);
    }
    if COMPANY_SPECIFIC_TERMS
        .iter()
        .any(|term| title_upper.contains(term))
        || url.contains("/markets/stocks/")
    {
        return ("company_specific".to_string(), 2);
    }
    if ["ETF", "YIELDS", "TREASURY", "DOLLAR", "NASDAQ", "MARKETS"]
        .iter()
        .any(|term| title_upper.contains(term))
    {
        return ("market_wrap".to_string(), 4);
    }
    if ["CRUDE", "OIL", "GOLD", "COPPER", "BULLION", "MINING"]
        .iter()
        .any(|term| title_upper.contains(term))
    {
        return ("commodity_specialist".to_string(), 3);
    }
    ("general_news".to_string(), 1)
}

fn split_sentences(text: Option<&str>) -> Vec<String> {
    let Some(cleaned) = normalize_html_text(text) else {
        return Vec::new();
    };
    cleaned
        .split("||")
        .flat_map(|part| part.split_terminator(['.', '!', '?', '\n']))
        .filter_map(|part| normalize_html_text(Some(part)))
        .filter(|candidate| candidate.len() >= 40)
        .collect()
}

fn extract_market_context_text(
    title: Option<&str>,
    summary_text: Option<&str>,
    body_text: Option<&str>,
    relevant_text: Option<&str>,
) -> (Option<String>, f64) {
    let mut sentences = Vec::new();
    sentences.extend(split_sentences(title));
    sentences.extend(split_sentences(summary_text));
    sentences.extend(split_sentences(body_text));
    if sentences.is_empty() {
        sentences.extend(split_sentences(relevant_text));
    }
    let mut scored = Vec::new();
    let mut seen = HashSet::new();
    for sentence in sentences {
        let normalized = sentence.to_uppercase();
        let score = MARKET_SENTENCE_TERMS
            .iter()
            .filter(|term| normalized.contains(**term))
            .count() as i32;
        if score <= 0 || !seen.insert(sentence.clone()) {
            continue;
        }
        scored.push((score, sentence));
    }
    scored.sort_by(|a, b| b.cmp(a));
    let kept: Vec<String> = scored
        .iter()
        .take(5)
        .map(|(_, sentence)| sentence.clone())
        .collect();
    if kept.is_empty() {
        (None, 0.0)
    } else {
        let total_score: i32 = scored.iter().take(5).map(|(score, _)| *score).sum();
        (Some(kept.join(" || ")), total_score as f64)
    }
}

fn extract_geo_labels(raw: Option<&str>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut labels = Vec::new();
    for entry in raw.unwrap_or_default().split(';') {
        for part in entry.split('#') {
            let upper = part.trim().to_uppercase();
            if upper.len() == 2
                && upper.chars().all(|ch| ch.is_ascii_alphabetic())
                && seen.insert(upper.clone())
            {
                labels.push(upper);
            }
        }
    }
    if labels.is_empty() {
        vec!["GLOBAL".to_string()]
    } else {
        labels.sort();
        labels
    }
}

fn uppercase_join(parts: &[Option<&str>]) -> String {
    parts
        .iter()
        .filter_map(|part| *part)
        .filter(|part| !part.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" | ")
        .to_uppercase()
}

fn quote_date_list(partitions: &[String]) -> String {
    partitions
        .iter()
        .map(|partition| format!("toDate('{partition}')"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn build_common_ctes(partitions: &[String]) -> String {
    let partition_list = quote_date_list(partitions);
    format!(
        "WITH
bronze_window AS (
    SELECT *
    FROM bronze_candidates
    WHERE partition_date IN ({partition_list})
),
matched_factors AS (
    SELECT DISTINCT
        b.doc_id AS doc_id,
        p.factor_id AS factor_id,
        p.factor_label AS factor_label
    FROM bronze_window b
    CROSS JOIN factor_rule_patterns p
    WHERE position(b.match_text, p.pattern) > 0
),
matched_assets AS (
    SELECT DISTINCT
        m.doc_id AS doc_id,
        m.factor_id AS factor_id,
        m.factor_label AS factor_label,
        a.asset_label AS asset_label
    FROM matched_factors m
    INNER JOIN factor_rule_assets a ON a.factor_id = m.factor_id
    LEFT JOIN asset_context_required r ON r.asset_label = a.asset_label
    WHERE isNull(r.asset_label)

    UNION DISTINCT

    SELECT DISTINCT
        m.doc_id AS doc_id,
        m.factor_id AS factor_id,
        m.factor_label AS factor_label,
        a.asset_label AS asset_label
    FROM matched_factors m
    INNER JOIN factor_rule_assets a ON a.factor_id = m.factor_id
    INNER JOIN asset_context_required r ON r.asset_label = a.asset_label
    INNER JOIN bronze_window b ON b.doc_id = m.doc_id
    CROSS JOIN asset_rule_patterns p
    WHERE p.asset_label = a.asset_label
      AND position(b.asset_match_text, p.pattern) > 0
),
factor_rollup AS (
    SELECT
        doc_id,
        arraySort(groupUniqArray(factor_id)) AS factor_ids,
        arraySort(groupUniqArray(factor_label)) AS factor_labels,
        count() AS factor_count
    FROM matched_factors
    GROUP BY doc_id
),
asset_rollup AS (
    SELECT
        doc_id,
        arraySort(groupUniqArray(cityHash64(asset_label))) AS asset_ids,
        arraySort(groupUniqArray(asset_label)) AS asset_labels
    FROM matched_assets
    GROUP BY doc_id
),
geo_rollup AS (
    SELECT
        doc_id,
        arraySort(groupUniqArray(cityHash64(geo_label))) AS geo_ids,
        arraySort(groupUniqArray(geo_label)) AS geo_labels
    FROM (
        SELECT
            doc_id,
            arrayJoin(geo_labels) AS geo_label
        FROM bronze_window
    )
    GROUP BY doc_id
),
asset_pattern_hits AS (
    SELECT DISTINCT
        m.doc_id AS doc_id,
        m.factor_id AS factor_id,
        m.asset_label AS asset_label,
        arp.pattern AS pattern
    FROM matched_assets m
    INNER JOIN bronze_window b ON b.doc_id = m.doc_id
    CROSS JOIN asset_rule_patterns arp
    WHERE arp.asset_label = m.asset_label
      AND position(b.asset_match_text, arp.pattern) > 0
),
factor_pattern_hits AS (
    SELECT DISTINCT
        m.doc_id AS doc_id,
        m.factor_id AS factor_id,
        frp.pattern AS pattern
    FROM matched_assets m
    INNER JOIN bronze_window b ON b.doc_id = m.doc_id
    CROSS JOIN factor_rule_patterns frp
    WHERE frp.factor_id = m.factor_id
      AND position(b.asset_match_text, frp.pattern) > 0
),
asset_factor_scores AS (
    SELECT
        m.doc_id AS doc_id,
        m.factor_id AS factor_id,
        m.factor_label AS factor_label,
        m.asset_label AS asset_label,
        toFloat64((2 * countDistinctIf(aph.pattern, isNotNull(aph.pattern))) + countDistinctIf(fph.pattern, isNotNull(fph.pattern))) AS asset_factor_relevance
    FROM matched_assets m
    LEFT JOIN asset_pattern_hits aph
        ON aph.doc_id = m.doc_id
       AND aph.factor_id = m.factor_id
       AND aph.asset_label = m.asset_label
    LEFT JOIN factor_pattern_hits fph
        ON fph.doc_id = m.doc_id
       AND fph.factor_id = m.factor_id
    GROUP BY m.doc_id, m.factor_id, m.factor_label, m.asset_label
)"
    )
}

fn build_silver_event_graph_sql(partitions: &[String]) -> String {
    format!(
        "{}
INSERT INTO silver_event_graph
SELECT
    b.event_time,
    toDate(b.event_time) AS bucket_time,
    b.partition_date,
    cityHash64(concat(b.source_domain, '|', b.document_identifier)) AS cluster_id,
    b.doc_id,
    f.factor_ids,
    f.factor_labels,
    ifNull(a.asset_ids, CAST([], 'Array(UInt64)')) AS asset_ids,
    ifNull(a.asset_labels, CAST([], 'Array(String)')) AS asset_labels,
    g.geo_ids,
    g.geo_labels,
    cityHash64(b.source_domain) AS source_id,
    b.source_domain,
    b.tone,
    1.0 AS novelty,
    1.0 AS source_weight,
    least(0.95, 0.55 + (0.08 * f.factor_count)) AS classification_confidence,
    'narrative_graph.phase2.clickhouse.v1' AS model_version,
    'deterministic-narrative-taxonomy' AS prompt_version,
    now64(3) AS created_at
FROM bronze_window b
INNER JOIN factor_rollup f USING (doc_id)
INNER JOIN geo_rollup g USING (doc_id)
LEFT JOIN asset_rollup a USING (doc_id)
SETTINGS join_algorithm = 'grace_hash'",
        build_common_ctes(partitions)
    )
}

fn build_silver_factor_mentions_sql(partitions: &[String]) -> String {
    format!(
        "{}
INSERT INTO silver_factor_mentions
SELECT
    toDate(b.event_time) AS bucket_time,
    b.event_time,
    b.partition_date,
    b.doc_id,
    cityHash64(concat(b.source_domain, '|', b.document_identifier)) AS cluster_id,
    m.factor_id,
    m.factor_label,
    cityHash64(geo_label) AS geo_id,
    geo_label,
    cityHash64(b.source_domain) AS source_id,
    b.source_domain,
    b.tone,
    1.0 AS novelty,
    1.0 AS source_weight,
    least(0.95, 0.55 + (0.08 * f.factor_count)) AS classification_confidence
FROM bronze_window b
INNER JOIN factor_rollup f USING (doc_id)
INNER JOIN matched_factors m USING (doc_id)
ARRAY JOIN b.geo_labels AS geo_label
SETTINGS join_algorithm = 'grace_hash'",
        build_common_ctes(partitions)
    )
}

fn build_silver_asset_factor_mentions_sql(partitions: &[String]) -> String {
    format!(
        "{}
INSERT INTO silver_asset_factor_mentions
SELECT
    toDate(b.event_time) AS bucket_time,
    b.event_time,
    b.partition_date,
    b.doc_id,
    cityHash64(concat(b.source_domain, '|', b.document_identifier)) AS cluster_id,
    m.factor_id,
    m.factor_label,
    cityHash64(m.asset_label) AS asset_id,
    m.asset_label,
    cityHash64(geo_label) AS geo_id,
    geo_label,
    cityHash64(b.source_domain) AS source_id,
    b.source_domain,
    b.tone,
    1.0 AS novelty,
    1.0 AS source_weight,
    least(0.95, 0.55 + (0.08 * f.factor_count)) AS classification_confidence,
    ifNull(s.asset_factor_relevance, 0.0) AS asset_factor_relevance
FROM bronze_window b
INNER JOIN factor_rollup f USING (doc_id)
INNER JOIN matched_assets m USING (doc_id)
LEFT JOIN asset_factor_scores s
    ON s.doc_id = m.doc_id
   AND s.factor_id = m.factor_id
   AND s.asset_label = m.asset_label
ARRAY JOIN b.geo_labels AS geo_label
SETTINGS join_algorithm = 'grace_hash'",
        build_common_ctes(partitions)
    )
}

fn build_silver_market_context_mentions_sql(partitions: &[String]) -> String {
    format!(
        "{}
INSERT INTO silver_market_context_mentions
SELECT DISTINCT
    toDate(b.event_time) AS bucket_time,
    b.event_time,
    b.partition_date,
    b.doc_id,
    cityHash64(concat(b.source_domain, '|', b.document_identifier)) AS cluster_id,
    m.factor_label,
    m.asset_label,
    b.source_domain,
    b.source_type,
    b.source_priority,
    b.market_context_text,
    b.market_context_score,
    least(0.95, 0.55 + (0.08 * f.factor_count)) AS classification_confidence
FROM bronze_window b
INNER JOIN factor_rollup f USING (doc_id)
INNER JOIN matched_assets m USING (doc_id)
WHERE isNotNull(b.market_context_text)
  AND b.market_context_text != ''
SETTINGS join_algorithm = 'grace_hash'",
        build_common_ctes(partitions)
    )
}

fn query_summary(client: &ClickHouseClient) -> Result<Value> {
    let tables = [
        "bronze_candidates",
        "silver_event_graph",
        "silver_factor_mentions",
        "silver_asset_factor_mentions",
        "gold_factor_buckets_daily",
        "gold_asset_factor_panel_daily",
        "gold_factor_crossover_links_daily",
        "gold_asset_factor_crossover_links_daily",
    ];
    let mut counts = BTreeMap::new();
    for table in tables {
        let rows =
            client.select_rows::<CountRow>(&format!("SELECT count() AS count FROM {table}"))?;
        counts.insert(table, rows.first().map(|row| row.count).unwrap_or(0));
    }
    let event_span = client.select_rows::<Map<String, Value>>(
        "SELECT min(event_time) AS min_event_time, max(event_time) AS max_event_time FROM silver_event_graph",
    )?;
    let bucket_span = client.select_rows::<Map<String, Value>>(
        "SELECT min(bucket_time) AS min_bucket_time, max(bucket_time) AS max_bucket_time, uniqExact(bucket_time) AS bucket_dates FROM gold_factor_buckets_daily",
    )?;
    let build_state = client.select_rows::<Map<String, Value>>(
        "SELECT uniqExact(partition_date) AS partition_count, min(partition_date) AS min_partition_date, max(partition_date) AS max_partition_date FROM graph_build_partitions",
    )?;
    Ok(json!({
        "database": client.database,
        "table_counts": counts,
        "event_span": event_span.into_iter().next().unwrap_or_default(),
        "bucket_span": bucket_span.into_iter().next().unwrap_or_default(),
        "build_partitions": build_state.into_iter().next().unwrap_or_default(),
    }))
}

fn query_top_factors(client: &ClickHouseClient, limit: usize) -> Result<Vec<Map<String, Value>>> {
    client.select_rows(&format!(
        "SELECT
            factor_label,
            sum(doc_count) AS doc_count,
            sum(mention_count) AS mention_count,
            avg(source_dispersion) AS avg_source_dispersion,
            avg(tone_mean) AS avg_tone_mean,
            avg(narrative_score) AS avg_narrative_score
        FROM gold_factor_buckets_daily
        GROUP BY factor_label
        ORDER BY doc_count DESC, avg_narrative_score DESC, factor_label ASC
        LIMIT {limit}"
    ))
}

fn query_top_assets(client: &ClickHouseClient, limit: usize) -> Result<Vec<Map<String, Value>>> {
    client.select_rows(&format!(
        "SELECT
            asset_label,
            sum(doc_count) AS doc_count,
            sum(mention_count) AS mention_count,
            avg(source_dispersion) AS avg_source_dispersion,
            avg(event_intensity) AS avg_event_intensity,
            avg(narrative_score) AS avg_narrative_score
        FROM gold_asset_factor_panel_daily
        GROUP BY asset_label
        ORDER BY doc_count DESC, avg_narrative_score DESC, asset_label ASC
        LIMIT {limit}"
    ))
}

fn query_factor_daily(
    client: &ClickHouseClient,
    factor_label: &str,
    limit: usize,
) -> Result<Vec<Map<String, Value>>> {
    client.select_rows(&format!(
        "SELECT
            bucket_time,
            geo_label,
            doc_count,
            mention_count,
            unique_sources,
            tone_mean,
            tone_zscore_30d,
            avg_abs_tone,
            novelty_mean,
            source_dispersion,
            confidence_mean,
            narrative_score
        FROM gold_factor_buckets_daily
        WHERE factor_label = '{factor_label}'
        ORDER BY bucket_time DESC, narrative_score DESC, geo_label ASC
        LIMIT {limit}"
    ))
}

fn query_tone_tails(client: &ClickHouseClient, limit: usize) -> Result<Vec<Map<String, Value>>> {
    client.select_rows(&format!(
        "SELECT
            factor_label,
            geo_label,
            bucket_time,
            doc_count,
            mention_count,
            negative_tail_count,
            positive_tail_count,
            tone_mean,
            source_dispersion,
            narrative_score
        FROM gold_factor_buckets_daily
        ORDER BY
            negative_tail_count DESC,
            positive_tail_count DESC,
            narrative_score DESC,
            factor_label ASC
        LIMIT {limit}"
    ))
}

fn query_asset_narratives(
    client: &ClickHouseClient,
    asset_label: &str,
    start_date: Option<&str>,
    end_date: Option<&str>,
    limit: usize,
) -> Result<Vec<Map<String, Value>>> {
    let start_filter = start_date
        .map(|value| format!(" AND bucket_time >= toDate('{value}')"))
        .unwrap_or_default();
    let end_filter = end_date
        .map(|value| format!(" AND bucket_time <= toDate('{value}')"))
        .unwrap_or_default();
    let mut rows: Vec<Map<String, Value>> = client.select_rows(&format!(
        "SELECT
            asset_label,
            factor_label,
            sum(doc_count) AS doc_count,
            sum(mention_count) AS mention_count,
            avg(unique_sources) AS avg_unique_sources,
            avg(geo_count) AS avg_geo_count,
            avg(tone_mean) AS avg_tone_mean,
            avg(tone_zscore_30d) AS avg_tone_zscore_30d,
            avg(avg_abs_tone) AS avg_abs_tone,
            avg(source_dispersion) AS avg_source_dispersion,
            avg(event_intensity) AS avg_event_intensity,
            avg(narrative_score) AS avg_narrative_score,
            avg(confidence) AS avg_confidence,
            min(bucket_time) AS first_bucket,
            max(bucket_time) AS last_bucket
        FROM gold_asset_factor_panel_daily
        WHERE asset_label = '{asset_label}'{start_filter}{end_filter}
        GROUP BY asset_label, factor_label
        ORDER BY avg_narrative_score DESC, doc_count DESC, factor_label ASC
        LIMIT {limit}"
    ))?;
    let priorities = load_priorities().unwrap_or_default();
    for row in &mut rows {
        let factor = row
            .get("factor_label")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let base_score = row
            .get("avg_narrative_score")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let multiplier = priorities
            .get(asset_label)
            .and_then(|map| map.get(factor))
            .copied()
            .unwrap_or(1.0);
        row.insert(
            "asset_factor_priority_multiplier".to_string(),
            Value::from(multiplier),
        );
        row.insert(
            "adjusted_narrative_score".to_string(),
            Value::from(base_score * multiplier),
        );
    }
    rows.sort_by(|left, right| {
        let left_score = left
            .get("adjusted_narrative_score")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let right_score = right
            .get("adjusted_narrative_score")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        right_score
            .partial_cmp(&left_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(rows)
}

fn query_asset_timeline(
    client: &ClickHouseClient,
    asset_label: &str,
    factor_label: Option<&str>,
    start_date: Option<&str>,
    end_date: Option<&str>,
    limit: usize,
) -> Result<Vec<Map<String, Value>>> {
    let factor_filter = factor_label
        .map(|value| format!(" AND factor_label = '{value}'"))
        .unwrap_or_default();
    let start_filter = start_date
        .map(|value| format!(" AND bucket_time >= toDate('{value}')"))
        .unwrap_or_default();
    let end_filter = end_date
        .map(|value| format!(" AND bucket_time <= toDate('{value}')"))
        .unwrap_or_default();
    client.select_rows(&format!(
        "SELECT
            bucket_time,
            asset_label,
            factor_label,
            doc_count,
            mention_count,
            unique_sources,
            geo_count,
            tone_mean,
            tone_zscore_30d,
            avg_abs_tone,
            source_dispersion,
            event_intensity,
            confidence,
            narrative_score
        FROM gold_asset_factor_panel_daily
        WHERE asset_label = '{asset_label}'{factor_filter}{start_filter}{end_filter}
        ORDER BY bucket_time DESC, narrative_score DESC, factor_label ASC
        LIMIT {limit}"
    ))
}

fn query_factor_crossovers(
    client: &ClickHouseClient,
    factor_label: Option<&str>,
    start_date: Option<&str>,
    end_date: Option<&str>,
    limit: usize,
) -> Result<Vec<Map<String, Value>>> {
    let factor_filter = factor_label
        .map(|value| format!(" AND factor_label = '{value}'"))
        .unwrap_or_default();
    let start_filter = start_date
        .map(|value| format!(" AND bucket_time >= toDate('{value}')"))
        .unwrap_or_default();
    let end_filter = end_date
        .map(|value| format!(" AND bucket_time <= toDate('{value}')"))
        .unwrap_or_default();
    client.select_rows(&format!(
        "SELECT
            prior_bucket_time,
            bucket_time,
            factor_label,
            geo_label,
            prior_doc_count,
            doc_count,
            prior_narrative_score,
            narrative_score,
            doc_count_delta,
            narrative_score_delta
        FROM gold_factor_crossover_links_daily
        WHERE 1 = 1{factor_filter}{start_filter}{end_filter}
        ORDER BY bucket_time DESC, abs(narrative_score_delta) DESC, abs(doc_count_delta) DESC, factor_label ASC, geo_label ASC
        LIMIT {limit}"
    ))
}

fn query_asset_crossovers(
    client: &ClickHouseClient,
    asset_label: &str,
    factor_label: Option<&str>,
    start_date: Option<&str>,
    end_date: Option<&str>,
    limit: usize,
) -> Result<Vec<Map<String, Value>>> {
    let factor_filter = factor_label
        .map(|value| format!(" AND factor_label = '{value}'"))
        .unwrap_or_default();
    let start_filter = start_date
        .map(|value| format!(" AND bucket_time >= toDate('{value}')"))
        .unwrap_or_default();
    let end_filter = end_date
        .map(|value| format!(" AND bucket_time <= toDate('{value}')"))
        .unwrap_or_default();
    client.select_rows(&format!(
        "SELECT
            prior_bucket_time,
            bucket_time,
            asset_label,
            factor_label,
            geo_label,
            prior_doc_count,
            doc_count,
            prior_narrative_score,
            narrative_score,
            doc_count_delta,
            narrative_score_delta
        FROM gold_asset_factor_crossover_links_daily
        WHERE asset_label = '{asset_label}'{factor_filter}{start_filter}{end_filter}
        ORDER BY bucket_time DESC, abs(narrative_score_delta) DESC, abs(doc_count_delta) DESC, factor_label ASC, geo_label ASC
        LIMIT {limit}"
    ))
}

fn query_supporting_docs(
    client: &ClickHouseClient,
    asset_label: &str,
    factor_label: Option<&str>,
    start_date: Option<&str>,
    end_date: Option<&str>,
    limit: usize,
) -> Result<Vec<SupportingDoc>> {
    let candidate_limit = if factor_label.is_some() {
        std::cmp::max(limit * 200, 1_000)
    } else {
        std::cmp::max(limit * 100, 500)
    };
    let factor_filter = factor_label
        .map(|value| format!(" AND m.factor_label = '{value}'"))
        .unwrap_or_default();
    let start_filter = start_date
        .map(|value| format!(" AND toDate(m.event_time) >= toDate('{value}')"))
        .unwrap_or_default();
    let end_filter = end_date
        .map(|value| format!(" AND toDate(m.event_time) <= toDate('{value}')"))
        .unwrap_or_default();
    let rows: Vec<Map<String, Value>> = client.select_rows(&format!(
        "SELECT DISTINCT
            toString(m.event_time) AS event_time,
            m.asset_label,
            m.factor_label,
            m.geo_label,
            m.source_domain,
            b.source_type,
            b.source_priority,
            b.document_identifier,
            b.title,
            b.summary_text,
            substring(b.body_text, 1, 4000) AS body_excerpt,
            b.market_context_text,
            b.market_context_score,
            m.classification_confidence
        FROM silver_asset_factor_mentions m
        INNER JOIN bronze_candidates b ON b.doc_id = m.doc_id
        WHERE m.asset_label = '{asset_label}'{factor_filter}{start_filter}{end_filter}
        ORDER BY m.event_time DESC
        LIMIT {candidate_limit}"
    ))?;
    let factor_patterns_by_label = factor_cues_by_label();
    let mut docs = Vec::new();
    let mut seen = HashSet::new();
    for row in rows {
        let factor = row
            .get("factor_label")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let doc = SupportingDoc {
            event_time: row
                .get("event_time")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            asset_label: row
                .get("asset_label")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            factor_label: factor.clone(),
            geo_label: row
                .get("geo_label")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            source_domain: row
                .get("source_domain")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            source_type: row
                .get("source_type")
                .and_then(Value::as_str)
                .map(str::to_string),
            source_priority: row
                .get("source_priority")
                .and_then(Value::as_u64)
                .map(|value| value as u8),
            document_identifier: row
                .get("document_identifier")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            title: row.get("title").and_then(Value::as_str).map(str::to_string),
            summary_text: row
                .get("summary_text")
                .and_then(Value::as_str)
                .map(str::to_string),
            body_excerpt: row
                .get("body_excerpt")
                .and_then(Value::as_str)
                .map(str::to_string),
            market_context_text: row
                .get("market_context_text")
                .and_then(Value::as_str)
                .map(str::to_string),
            market_context_score: row.get("market_context_score").and_then(Value::as_f64),
            classification_confidence: row.get("classification_confidence").and_then(Value::as_f64),
            relevance_score: supporting_doc_relevance(
                &row,
                asset_label,
                factor_patterns_by_label
                    .get(&factor)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
            ),
        };
        let key = (
            doc.document_identifier.clone(),
            doc.asset_label.clone(),
            doc.factor_label.clone(),
        );
        if !seen.insert(key) {
            continue;
        }
        docs.push(doc);
    }
    docs.sort_by(|left, right| {
        right
            .relevance_score
            .partial_cmp(&left.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    docs.truncate(limit);
    Ok(docs)
}

fn supporting_doc_relevance(
    row: &Map<String, Value>,
    requested_asset_label: &str,
    factor_patterns: &[String],
) -> f64 {
    let title = row.get("title").and_then(Value::as_str);
    let summary = row.get("summary_text").and_then(Value::as_str);
    let body = row.get("body_excerpt").and_then(Value::as_str);
    let market_context = row.get("market_context_text").and_then(Value::as_str);
    let evidence_text = [market_context, summary, body]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ");
    let factor_hits =
        match_count(title, factor_patterns) + match_count(Some(&evidence_text), factor_patterns);
    let asset_cues = asset_cues(requested_asset_label);
    let asset_hits = match_count(title, asset_cues) + match_count(Some(&evidence_text), asset_cues);
    let source_type = row
        .get("source_type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let source_bonus = match source_type {
        "market_wrap" => 3.0,
        "commodity_specialist" => 2.0,
        _ => 0.0,
    };
    let confidence = row
        .get("classification_confidence")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    (factor_hits as f64 * 4.0)
        + (asset_hits as f64 * 3.0)
        + source_bonus
        + confidence * 2.0
        + if title.is_some() { 1.0 } else { 0.0 }
        + if summary.is_some() { 0.5 } else { 0.0 }
        + if body.is_some() { 0.5 } else { 0.0 }
}

fn asset_text_patterns() -> &'static HashMap<String, Vec<String>> {
    ASSET_TEXT_PATTERNS_CACHE.get_or_init(|| {
        HashMap::from([
            (
                "WTI".to_string(),
                vec!["WTI", "CRUDE", "OIL", "OPEC", "BARREL", "TANKER", "RED SEA"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            ),
            (
                "Brent".to_string(),
                vec![
                    "BRENT", "CRUDE", "OIL", "OPEC", "BARREL", "TANKER", "RED SEA",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ),
            (
                "HG".to_string(),
                vec![
                    "COPPER",
                    "COMEX COPPER",
                    "LME COPPER",
                    "SMELTER",
                    "MINE SUPPLY",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ),
            (
                "FXI".to_string(),
                vec![
                    "FXI",
                    "CHINA",
                    "CHINESE",
                    "HONG KONG",
                    "CSI 300",
                    "MAINLAND",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ),
            (
                "BTC".to_string(),
                vec![
                    "BITCOIN",
                    "BTC",
                    "CRYPTO",
                    "TOKEN",
                    "STABLECOIN",
                    "EXCHANGE",
                    "ETF",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ),
            (
                "BDI".to_string(),
                vec![
                    "BALTIC DRY",
                    "DRY BULK",
                    "BULK CARRIER",
                    "FREIGHT",
                    "SHIPPING",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ),
            (
                "Gold".to_string(),
                vec!["GOLD", "BULLION", "XAU", "PRECIOUS METAL", "SAFE HAVEN"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            ),
            (
                "DXY".to_string(),
                vec!["DOLLAR", "USD", "GREENBACK", "US DOLLAR"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            ),
            (
                "US2Y".to_string(),
                vec![
                    "2Y",
                    "2-YEAR",
                    "TWO-YEAR",
                    "SHORT-DATED TREASURY",
                    "FRONT-END YIELD",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ),
            (
                "US10Y".to_string(),
                vec![
                    "10Y",
                    "10-YEAR",
                    "TEN-YEAR",
                    "LONG-DATED TREASURY",
                    "BENCHMARK YIELD",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ),
            (
                "NDX".to_string(),
                vec![
                    "NASDAQ 100",
                    "NASDAQ-100",
                    "NDX",
                    "QQQ",
                    "NASDAQ FALLS",
                    "NASDAQ DROPS",
                    "NASDAQ SLIDES",
                    "TECH SELLOFF",
                    "CHIP STOCKS",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ),
            (
                "SPX".to_string(),
                vec![
                    "S&P 500",
                    "SP 500",
                    "SPX",
                    "WALL STREET",
                    "US STOCKS",
                    "U.S. STOCKS",
                    "STOCKS SLUMP",
                    "STOCKS FALL",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ),
        ])
    })
}

fn asset_cues(asset_label: &str) -> &'static [String] {
    let map = ASSET_CUES_CACHE.get_or_init(|| {
        let mut map = HashMap::new();
        for (asset, patterns) in asset_text_patterns() {
            let mut cues = patterns.clone();
            cues.push(asset.clone());
            map.insert(asset.clone(), cues);
        }
        map
    });
    map.get(asset_label).map(Vec::as_slice).unwrap_or(&[])
}

fn factor_cues_by_label() -> &'static HashMap<String, Vec<String>> {
    FACTOR_CUES_BY_LABEL_CACHE.get_or_init(|| {
        let taxonomy = load_taxonomy().unwrap_or(Taxonomy {
            factors: Vec::new(),
        });
        taxonomy
            .factors
            .into_iter()
            .map(|factor| {
                let mut cues = factor.patterns.clone();
                cues.push(factor.label.replace('_', " "));
                (factor.label, cues)
            })
            .collect()
    })
}

fn match_count(text: Option<&str>, cues: &[String]) -> usize {
    let normalized = normalize_for_match(text.unwrap_or_default());
    if normalized.is_empty() {
        return 0;
    }
    let padded = format!(" {normalized} ");
    let tokens: HashSet<&str> = normalized.split_whitespace().collect();
    cues.iter()
        .filter_map(|cue| {
            let normalized_cue = normalize_for_match(cue);
            if normalized_cue.is_empty() {
                None
            } else if normalized_cue.contains(' ') {
                Some(usize::from(padded.contains(&format!(" {normalized_cue} "))))
            } else {
                Some(usize::from(tokens.contains(normalized_cue.as_str())))
            }
        })
        .sum()
}

fn normalize_for_match(text: &str) -> String {
    let replaced = text.to_uppercase().replace('_', " ");
    let filtered: String = replaced
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch.is_ascii_whitespace() {
                ch
            } else {
                ' '
            }
        })
        .collect();
    filtered.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_clickhouse_identifier(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn client_for_tool_arguments(arguments: &Map<String, Value>) -> Result<ClickHouseClient> {
    let mut client = ClickHouseClient::from_env()?;
    let override_name = arguments
        .get("database")
        .and_then(Value::as_str)
        .or_else(|| arguments.get("db").and_then(Value::as_str));
    if let Some(database) = override_name {
        let looks_like_legacy_path = database.contains('/')
            || database.contains('\\')
            || database.ends_with(".duckdb")
            || database.contains("parquet");
        if !looks_like_legacy_path {
            if !is_clickhouse_identifier(database) {
                bail!("invalid ClickHouse database override: {database}");
            }
            client.database = database.to_string();
        }
    }
    Ok(client)
}

fn required_string_argument(arguments: &Map<String, Value>, key: &str) -> Result<String> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .with_context(|| format!("missing required argument: {key}"))
}

fn optional_string_argument(arguments: &Map<String, Value>, key: &str) -> Option<String> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn usize_argument(arguments: &Map<String, Value>, key: &str, default: usize) -> usize {
    arguments
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .unwrap_or(default)
}

fn string_list_argument(arguments: &Map<String, Value>, key: &str) -> Result<Vec<String>> {
    let Some(values) = arguments.get(key) else {
        bail!("missing required argument: {key}");
    };
    let array = values
        .as_array()
        .with_context(|| format!("argument {key} must be an array"))?;
    Ok(array
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect())
}

fn mcp_tool_specs() -> Vec<Value> {
    vec![
        json!({
            "name": "explain_move",
            "description": "Return the raw ClickHouse narrative bundle for an asset and date window, without server-side reasoning.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "asset_label": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                    "database": {"type": "string"},
                    "db": {"type": "string"}
                },
                "required": ["asset_label"]
            }
        }),
        json!({
            "name": "summarize_narrative",
            "description": "Compatibility alias that now returns the raw explain-move bundle. Narrative summarization must happen on the client.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "asset_label": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                    "database": {"type": "string"},
                    "db": {"type": "string"}
                },
                "required": ["asset_label"]
            }
        }),
        json!({
            "name": "supporting_docs",
            "description": "Return supporting document candidates for an asset and optional factor in a date window.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "asset_label": {"type": "string"},
                    "factor_label": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                    "database": {"type": "string"},
                    "db": {"type": "string"}
                },
                "required": ["asset_label"]
            }
        }),
        json!({
            "name": "explain_day",
            "description": "Return the raw per-asset day bundle for a requested universe, without narrative framing.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "universe": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "default": 5},
                    "database": {"type": "string"},
                    "db": {"type": "string"}
                },
                "required": ["date", "universe"]
            }
        }),
        json!({
            "name": "explain_cross_asset_move",
            "description": "Return raw cross-asset context bundles and overlap metrics for multiple assets on a date.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "assets": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "default": 5},
                    "database": {"type": "string"},
                    "db": {"type": "string"}
                },
                "required": ["date", "assets"]
            }
        }),
        json!({
            "name": "build_narrative_frame",
            "description": "Compatibility alias that returns the raw day bundle clients should use to construct their own frame.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "universe": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "default": 5},
                    "database": {"type": "string"},
                    "db": {"type": "string"}
                },
                "required": ["date", "universe"]
            }
        }),
        json!({
            "name": "find_contradictory_assets",
            "description": "Return low-overlap asset pairs and their raw factor sets so the client can reason about contradictions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "universe": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "default": 5},
                    "database": {"type": "string"},
                    "db": {"type": "string"}
                },
                "required": ["date", "universe"]
            }
        }),
        json!({
            "name": "explain_asset_via_day_context",
            "description": "Return raw asset and day context bundles so the client can explain the asset itself.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "asset_label": {"type": "string"},
                    "universe": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "default": 5},
                    "database": {"type": "string"},
                    "db": {"type": "string"}
                },
                "required": ["date", "asset_label"]
            }
        }),
        json!({
            "name": "query_clickhouse",
            "description": "Run a guarded read-only ClickHouse query for edge-case inspection.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
                    "database": {"type": "string"},
                    "db": {"type": "string"}
                },
                "required": ["sql"]
            }
        }),
        json!({
            "name": "query_duckdb",
            "description": "Deprecated compatibility alias for query_clickhouse.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
                    "database": {"type": "string"},
                    "db": {"type": "string"}
                },
                "required": ["sql"]
            }
        }),
        json!({
            "name": "similar_days",
            "description": "Return prior days with the closest factor-mix distance to the requested date.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                    "database": {"type": "string"},
                    "db": {"type": "string"}
                },
                "required": ["date"]
            }
        }),
        json!({
            "name": "intraday_evolution",
            "description": "Return hourly event counts and top hourly factors for a date.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                    "database": {"type": "string"},
                    "db": {"type": "string"}
                },
                "required": ["date"]
            }
        }),
        json!({
            "name": "summary",
            "description": "Return the ClickHouse graph summary.",
            "inputSchema": {"type": "object", "properties": {"database": {"type": "string"}, "db": {"type": "string"}}}
        }),
        json!({
            "name": "top_factors",
            "description": "Return top factors from the ClickHouse graph.",
            "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10}, "database": {"type": "string"}, "db": {"type": "string"}}}
        }),
        json!({
            "name": "top_assets",
            "description": "Return top assets from the ClickHouse graph.",
            "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10}, "database": {"type": "string"}, "db": {"type": "string"}}}
        }),
        json!({
            "name": "factor_daily",
            "description": "Return daily factor rows for a factor label.",
            "inputSchema": {"type": "object", "properties": {"factor_label": {"type": "string"}, "limit": {"type": "integer", "default": 10}, "database": {"type": "string"}, "db": {"type": "string"}}, "required": ["factor_label"]}
        }),
        json!({
            "name": "tone_tails",
            "description": "Return factor buckets with the largest tone tails.",
            "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10}, "database": {"type": "string"}, "db": {"type": "string"}}}
        }),
        json!({
            "name": "asset_narratives",
            "description": "Return raw narrative rows for an asset window.",
            "inputSchema": {"type": "object", "properties": {"asset_label": {"type": "string"}, "start_date": {"type": "string"}, "end_date": {"type": "string"}, "limit": {"type": "integer", "default": 10}, "database": {"type": "string"}, "db": {"type": "string"}}, "required": ["asset_label"]}
        }),
        json!({
            "name": "asset_timeline",
            "description": "Return daily timeline rows for an asset and optional factor.",
            "inputSchema": {"type": "object", "properties": {"asset_label": {"type": "string"}, "factor_label": {"type": "string"}, "start_date": {"type": "string"}, "end_date": {"type": "string"}, "limit": {"type": "integer", "default": 10}, "database": {"type": "string"}, "db": {"type": "string"}}, "required": ["asset_label"]}
        }),
        json!({
            "name": "factor_crossovers",
            "description": "Return factor crossover rows.",
            "inputSchema": {"type": "object", "properties": {"factor_label": {"type": "string"}, "start_date": {"type": "string"}, "end_date": {"type": "string"}, "limit": {"type": "integer", "default": 10}, "database": {"type": "string"}, "db": {"type": "string"}}}
        }),
        json!({
            "name": "asset_crossovers",
            "description": "Return asset crossover rows.",
            "inputSchema": {"type": "object", "properties": {"asset_label": {"type": "string"}, "factor_label": {"type": "string"}, "start_date": {"type": "string"}, "end_date": {"type": "string"}, "limit": {"type": "integer", "default": 10}, "database": {"type": "string"}, "db": {"type": "string"}}, "required": ["asset_label"]}
        }),
    ]
}

fn factor_labels_from_bundle(bundle: &Value) -> Vec<String> {
    bundle
        .get("top_narratives")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|row| row.get("factor_label").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn pairwise_factor_overlap(asset_contexts: &[Value]) -> Vec<Value> {
    let mut rows = Vec::new();
    for left_index in 0..asset_contexts.len() {
        for right_index in (left_index + 1)..asset_contexts.len() {
            let left = &asset_contexts[left_index];
            let right = &asset_contexts[right_index];
            let left_asset = left
                .get("asset_label")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let right_asset = right
                .get("asset_label")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let left_factors: HashSet<String> =
                factor_labels_from_bundle(left).into_iter().collect();
            let right_factors: HashSet<String> =
                factor_labels_from_bundle(right).into_iter().collect();
            let shared: Vec<String> = left_factors
                .intersection(&right_factors)
                .cloned()
                .collect::<Vec<_>>();
            let left_only: Vec<String> = left_factors
                .difference(&right_factors)
                .cloned()
                .collect::<Vec<_>>();
            let right_only: Vec<String> = right_factors
                .difference(&left_factors)
                .cloned()
                .collect::<Vec<_>>();
            rows.push(json!({
                "left_asset": left_asset,
                "right_asset": right_asset,
                "shared_factor_count": shared.len(),
                "shared_factors": shared,
                "left_only_factors": left_only,
                "right_only_factors": right_only,
            }));
        }
    }
    rows.sort_by(|left, right| {
        left.get("shared_factor_count")
            .and_then(Value::as_u64)
            .cmp(&right.get("shared_factor_count").and_then(Value::as_u64))
    });
    rows
}

fn cross_asset_factor_frequency(asset_contexts: &[Value]) -> Vec<Value> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for context in asset_contexts {
        let unique: HashSet<String> = factor_labels_from_bundle(context).into_iter().collect();
        for factor in unique {
            *counts.entry(factor).or_default() += 1;
        }
    }
    let mut rows: Vec<Value> = counts
        .into_iter()
        .map(|(factor_label, asset_count)| json!({"factor_label": factor_label, "asset_count": asset_count}))
        .collect();
    rows.sort_by(|left, right| {
        right
            .get("asset_count")
            .and_then(Value::as_u64)
            .cmp(&left.get("asset_count").and_then(Value::as_u64))
            .then_with(|| {
                left.get("factor_label")
                    .and_then(Value::as_str)
                    .cmp(&right.get("factor_label").and_then(Value::as_str))
            })
    });
    rows
}

fn explain_move_bundle(
    client: &ClickHouseClient,
    asset_label: &str,
    start_date: Option<&str>,
    end_date: Option<&str>,
    limit: usize,
) -> Result<Value> {
    Ok(json!({
        "asset_label": asset_label,
        "window": {"start_date": start_date, "end_date": end_date},
        "top_narratives": query_asset_narratives(client, asset_label, start_date, end_date, limit)?,
        "timeline": query_asset_timeline(client, asset_label, None, start_date, end_date, limit)?,
        "crossovers": query_asset_crossovers(client, asset_label, None, start_date, end_date, limit)?,
        "supporting_docs": query_supporting_docs(client, asset_label, None, start_date, end_date, limit)?,
    }))
}

fn day_context_bundle(
    client: &ClickHouseClient,
    date: &str,
    universe: &[String],
    limit: usize,
) -> Result<Value> {
    let asset_contexts: Vec<Value> = universe
        .iter()
        .map(|asset_label| explain_move_bundle(client, asset_label, Some(date), Some(date), limit))
        .collect::<Result<_>>()?;
    Ok(json!({
        "date": date,
        "universe": universe,
        "asset_contexts": asset_contexts,
        "pairwise_factor_overlap": pairwise_factor_overlap(&asset_contexts),
        "cross_asset_factor_frequency": cross_asset_factor_frequency(&asset_contexts),
    }))
}

fn query_similar_days(
    client: &ClickHouseClient,
    date: &str,
    limit: usize,
) -> Result<Vec<Map<String, Value>>> {
    let select_terms = SIMILAR_DAY_FACTORS
        .iter()
        .map(|factor| format!("sumIf(score, factor_label = '{factor}') AS {factor}"))
        .collect::<Vec<_>>()
        .join(",\n            ");
    let distance_terms = SIMILAR_DAY_FACTORS
        .iter()
        .map(|factor| format!("pow(d.{factor} - b.{factor}, 2)"))
        .collect::<Vec<_>>()
        .join(" + ");
    client.select_rows(&format!(
        "WITH daily_factor AS (
            SELECT
                bucket_time,
                factor_label,
                sum(narrative_score) AS score
            FROM gold_factor_buckets_daily
            WHERE factor_label IN ({factor_list})
            GROUP BY bucket_time, factor_label
        ),
        daily_vec AS (
            SELECT
                bucket_time,
                {select_terms}
            FROM daily_factor
            GROUP BY bucket_time
        ),
        base_day AS (
            SELECT *
            FROM daily_vec
            WHERE bucket_time = toDate('{date}')
        )
        SELECT
            d.bucket_time,
            sqrt({distance_terms}) AS distance
        FROM daily_vec d
        CROSS JOIN base_day b
        WHERE d.bucket_time < toDate('{date}')
        ORDER BY distance ASC, d.bucket_time DESC
        LIMIT {limit}",
        factor_list = SIMILAR_DAY_FACTORS
            .iter()
            .map(|factor| format!("'{factor}'"))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

fn query_intraday_evolution(client: &ClickHouseClient, date: &str, limit: usize) -> Result<Value> {
    let hour_event_counts: Vec<Map<String, Value>> = client.select_rows(&format!(
        "SELECT
            toStartOfHour(event_time) AS hour_bucket,
            count() AS event_count
        FROM silver_event_graph
        WHERE toDate(event_time) = toDate('{date}')
        GROUP BY hour_bucket
        ORDER BY hour_bucket ASC"
    ))?;
    let hourly_top_factors: Vec<Map<String, Value>> = client.select_rows(&format!(
        "WITH hourly AS (
            SELECT
                toStartOfHour(event_time) AS hour_bucket,
                factor_label,
                count() AS mention_count,
                avg(tone) AS tone_mean
            FROM silver_factor_mentions
            WHERE toDate(event_time) = toDate('{date}')
            GROUP BY hour_bucket, factor_label
        ),
        ranked AS (
            SELECT
                hour_bucket,
                factor_label,
                mention_count,
                tone_mean,
                row_number() OVER (PARTITION BY hour_bucket ORDER BY mention_count DESC, factor_label ASC) AS rn
            FROM hourly
        )
        SELECT
            hour_bucket,
            factor_label,
            mention_count,
            tone_mean
        FROM ranked
        WHERE rn <= {limit}
        ORDER BY hour_bucket ASC, rn ASC"
    ))?;
    Ok(json!({
        "date": date,
        "hour_event_counts": hour_event_counts,
        "hourly_top_factors": hourly_top_factors,
    }))
}

fn normalize_read_only_clickhouse_sql(sql: &str, limit: usize) -> Result<String> {
    let trimmed = sql.trim();
    if trimmed.is_empty() {
        bail!("sql must not be empty");
    }
    let upper = trimmed.to_uppercase();
    for banned in [
        ";",
        " ATTACH ",
        " COPY ",
        " CREATE ",
        " DELETE ",
        " DETACH ",
        " DROP ",
        " EXPORT ",
        " INSERT ",
        " LOAD ",
        " OPTIMIZE ",
        " REPLACE ",
        " SET ",
        " SYSTEM ",
        " TRUNCATE ",
        " UPDATE ",
        " USE ",
    ] {
        if upper.contains(banned) || trimmed == banned.trim() {
            bail!("disallowed SQL pattern in query");
        }
    }
    let is_read_only = upper.starts_with("SELECT ")
        || upper.starts_with("WITH ")
        || upper.starts_with("SHOW ")
        || upper.starts_with("DESCRIBE ")
        || upper.starts_with("EXPLAIN ");
    if !is_read_only {
        bail!("only read-only SELECT/WITH/SHOW/DESCRIBE/EXPLAIN queries are allowed");
    }
    if upper.contains(" LIMIT ") || upper.ends_with("LIMIT") {
        Ok(trimmed.to_string())
    } else {
        Ok(format!("{trimmed} LIMIT {limit}"))
    }
}

fn call_mcp_tool_local(name: &str, arguments: &Map<String, Value>) -> Result<Value> {
    let client = client_for_tool_arguments(arguments)?;
    let limit = usize_argument(arguments, "limit", 10);
    match name {
        "summary" => Ok(query_summary(&client)?),
        "top_factors" => Ok(json!(query_top_factors(&client, limit)?)),
        "top_assets" => Ok(json!(query_top_assets(&client, limit)?)),
        "factor_daily" => {
            let factor_label = required_string_argument(arguments, "factor_label")?;
            Ok(json!(query_factor_daily(&client, &factor_label, limit)?))
        }
        "tone_tails" => Ok(json!(query_tone_tails(&client, limit)?)),
        "asset_narratives" => {
            let asset_label = required_string_argument(arguments, "asset_label")?;
            Ok(json!(query_asset_narratives(
                &client,
                &asset_label,
                optional_string_argument(arguments, "start_date").as_deref(),
                optional_string_argument(arguments, "end_date").as_deref(),
                limit,
            )?))
        }
        "asset_timeline" => {
            let asset_label = required_string_argument(arguments, "asset_label")?;
            Ok(json!(query_asset_timeline(
                &client,
                &asset_label,
                optional_string_argument(arguments, "factor_label").as_deref(),
                optional_string_argument(arguments, "start_date").as_deref(),
                optional_string_argument(arguments, "end_date").as_deref(),
                limit,
            )?))
        }
        "factor_crossovers" => Ok(json!(query_factor_crossovers(
            &client,
            optional_string_argument(arguments, "factor_label").as_deref(),
            optional_string_argument(arguments, "start_date").as_deref(),
            optional_string_argument(arguments, "end_date").as_deref(),
            limit,
        )?)),
        "asset_crossovers" => {
            let asset_label = required_string_argument(arguments, "asset_label")?;
            Ok(json!(query_asset_crossovers(
                &client,
                &asset_label,
                optional_string_argument(arguments, "factor_label").as_deref(),
                optional_string_argument(arguments, "start_date").as_deref(),
                optional_string_argument(arguments, "end_date").as_deref(),
                limit,
            )?))
        }
        "supporting_docs" => {
            let asset_label = required_string_argument(arguments, "asset_label")?;
            Ok(json!(query_supporting_docs(
                &client,
                &asset_label,
                optional_string_argument(arguments, "factor_label").as_deref(),
                optional_string_argument(arguments, "start_date").as_deref(),
                optional_string_argument(arguments, "end_date").as_deref(),
                limit,
            )?))
        }
        "explain_move" => {
            let asset_label = required_string_argument(arguments, "asset_label")?;
            explain_move_bundle(
                &client,
                &asset_label,
                optional_string_argument(arguments, "start_date").as_deref(),
                optional_string_argument(arguments, "end_date").as_deref(),
                limit,
            )
        }
        "summarize_narrative" => {
            let asset_label = required_string_argument(arguments, "asset_label")?;
            Ok(json!({
                "tool": "summarize_narrative",
                "server_reasoning": "disabled",
                "note": "Use the returned raw bundle to reason on the client.",
                "raw_context": explain_move_bundle(
                    &client,
                    &asset_label,
                    optional_string_argument(arguments, "start_date").as_deref(),
                    optional_string_argument(arguments, "end_date").as_deref(),
                    limit,
                )?,
            }))
        }
        "explain_day" | "build_narrative_frame" => {
            let date = required_string_argument(arguments, "date")?;
            let universe = string_list_argument(arguments, "universe")?;
            let bundle = day_context_bundle(&client, &date, &universe, limit)?;
            if name == "build_narrative_frame" {
                Ok(json!({
                    "tool": "build_narrative_frame",
                    "server_reasoning": "disabled",
                    "note": "Use the raw day bundle to build your own frame on the client.",
                    "raw_context": bundle,
                }))
            } else {
                Ok(bundle)
            }
        }
        "explain_cross_asset_move" => {
            let date = required_string_argument(arguments, "date")?;
            let assets = string_list_argument(arguments, "assets")?;
            Ok(json!({
                "date": date,
                "assets": assets,
                "raw_context": day_context_bundle(&client, &date, &assets, limit)?,
            }))
        }
        "find_contradictory_assets" => {
            let date = required_string_argument(arguments, "date")?;
            let universe = string_list_argument(arguments, "universe")?;
            let bundle = day_context_bundle(&client, &date, &universe, limit)?;
            Ok(json!({
                "date": date,
                "universe": universe,
                "contradiction_candidates": bundle
                    .get("pairwise_factor_overlap")
                    .cloned()
                    .unwrap_or(Value::Array(Vec::new())),
                "raw_context": bundle,
            }))
        }
        "explain_asset_via_day_context" => {
            let date = required_string_argument(arguments, "date")?;
            let asset_label = required_string_argument(arguments, "asset_label")?;
            let mut universe = optional_string_argument(arguments, "asset_label")
                .map(|_| Vec::new())
                .unwrap_or_default();
            if let Some(requested) = arguments.get("universe") {
                universe = requested
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect();
            }
            if !universe.iter().any(|asset| asset == &asset_label) {
                universe.insert(0, asset_label.clone());
            }
            Ok(json!({
                "date": date,
                "asset_label": asset_label,
                "asset_context": explain_move_bundle(&client, &asset_label, Some(&date), Some(&date), limit)?,
                "day_context": day_context_bundle(&client, &date, &universe, limit)?,
            }))
        }
        "query_clickhouse" | "query_duckdb" => {
            let sql = required_string_argument(arguments, "sql")?;
            let normalized =
                normalize_read_only_clickhouse_sql(&sql, usize_argument(arguments, "limit", 50))?;
            let rows: Vec<Map<String, Value>> = client.select_rows(&normalized)?;
            Ok(json!({
                "database": client.database,
                "sql": normalized,
                "row_count": rows.len(),
                "rows": rows,
            }))
        }
        "similar_days" => {
            let date = required_string_argument(arguments, "date")?;
            Ok(json!({
                "date": date,
                "rows": query_similar_days(&client, &date, limit)?,
            }))
        }
        "intraday_evolution" => {
            let date = required_string_argument(arguments, "date")?;
            query_intraday_evolution(&client, &date, limit)
        }
        _ => bail!("unknown tool: {name}"),
    }
}

fn mcp_result_payload(payload: Value) -> Result<Value> {
    let text = serde_json::to_string_pretty(&payload)?;
    Ok(json!({
        "content": [{"type": "text", "text": text}],
        "structuredContent": payload
    }))
}

fn mcp_error(id: Value, code: i64, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message.into()}
    })
}

fn handle_mcp_request_local(request: Value) -> Value {
    let method = request
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let id = request.get("id").cloned().unwrap_or(Value::Null);
    match method {
        "initialize" => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "protocolVersion": "2025-06-18",
                "serverInfo": {"name": "news-narrative-v2-clickhouse-mcp", "version": "0.1.0"},
                "capabilities": {"tools": {}}
            }
        }),
        "notifications/initialized" => Value::Null,
        "tools/list" => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {"tools": mcp_tool_specs()}
        }),
        "tools/call" => {
            let params = request
                .get("params")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let tool_name = params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let arguments = params
                .get("arguments")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            match call_mcp_tool_local(tool_name, &arguments).and_then(mcp_result_payload) {
                Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
                Err(error) => mcp_error(id, -32000, error.to_string()),
            }
        }
        _ => mcp_error(id, -32601, format!("method not found: {method}")),
    }
}

fn write_mcp_message(payload: &Value) -> Result<()> {
    if payload.is_null() {
        return Ok(());
    }
    let body = serde_json::to_vec(payload)?;
    let mut stdout = std::io::stdout().lock();
    write!(stdout, "Content-Length: {}\r\n\r\n", body.len())?;
    stdout.write_all(&body)?;
    stdout.flush()?;
    Ok(())
}

fn read_mcp_message() -> Result<Option<Value>> {
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Ok(None);
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((key, value)) = line.split_once(':') {
            headers.insert(key.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }
    let length = headers
        .get("content-length")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    if length == 0 {
        return Ok(None);
    }
    let mut body = vec![0_u8; length];
    reader.read_exact(&mut body)?;
    Ok(Some(serde_json::from_slice(&body)?))
}

fn run_mcp_stdio_local() -> Result<()> {
    loop {
        let Some(request) = read_mcp_message()? else {
            return Ok(());
        };
        let response = handle_mcp_request_local(request);
        if !response.is_null() {
            write_mcp_message(&response)?;
        }
    }
}

fn run_mcp_stdio_proxy(api_base_url: &str) -> Result<()> {
    let client = Client::builder()
        .timeout(StdDuration::from_secs(600))
        .build()?;
    let endpoint = format!("{}/mcp", api_base_url.trim_end_matches('/'));
    loop {
        let Some(request) = read_mcp_message()? else {
            return Ok(());
        };
        let response = client.post(&endpoint).json(&request).send()?;
        if response.status().as_u16() == 204 {
            continue;
        }
        let payload: Value = response.json()?;
        if !payload.is_null() {
            write_mcp_message(&payload)?;
        }
    }
}

fn json_header() -> Header {
    Header::from_bytes(b"Content-Type".as_slice(), b"application/json".as_slice())
        .expect("static header should be valid")
}

fn respond_json(request: tiny_http::Request, status_code: u16, payload: &Value) {
    let body = serde_json::to_string_pretty(payload).unwrap_or_else(|_| "{}".to_string());
    let response = Response::from_string(body)
        .with_status_code(StatusCode(status_code))
        .with_header(json_header());
    let _ = request.respond(response);
}

fn respond_text(request: tiny_http::Request, status_code: u16, message: &str) {
    let response =
        Response::from_string(message.to_string()).with_status_code(StatusCode(status_code));
    let _ = request.respond(response);
}

fn serve_mcp_api(bind: &str) -> Result<()> {
    let server = Server::http(bind).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    for mut request in server.incoming_requests() {
        let method = request.method().clone();
        let path = request.url().to_string();
        match (method, path.as_str()) {
            (Method::Get, "/health") => respond_json(
                request,
                200,
                &json!({"status": "ok", "server": "news-narrative-v2-clickhouse-mcp"}),
            ),
            (Method::Get, "/tools") => {
                respond_json(request, 200, &json!({"tools": mcp_tool_specs()}));
            }
            (Method::Post, "/mcp") => {
                let mut body = String::new();
                if let Err(error) = request.as_reader().read_to_string(&mut body) {
                    respond_text(request, 400, &format!("failed to read body: {error}"));
                    continue;
                }
                let parsed: Value = match serde_json::from_str(&body) {
                    Ok(value) => value,
                    Err(error) => {
                        respond_text(request, 400, &format!("invalid json: {error}"));
                        continue;
                    }
                };
                let response = handle_mcp_request_local(parsed);
                if response.is_null() {
                    let _ = request.respond(Response::empty(StatusCode(204)));
                } else {
                    respond_json(request, 200, &response);
                }
            }
            _ => respond_text(request, 404, "not found"),
        }
    }
    Ok(())
}

fn sample_bronze_raw_row() -> BronzeRawRow {
    BronzeRawRow {
        record_datetime: "20260618153000".to_string(),
        partition_date: "2026-06-18".to_string(),
        ingested_at: "2026-06-28 12:00:00.000".to_string(),
        source_common_name: Some("reuters.com".to_string()),
        document_identifier: "https://www.reuters.com/markets/commodities/oil-prices-rise-as-red-sea-risk-and-dollar-moves-shape-trade-2026-06-18/".to_string(),
        title: Some("Oil prices rise as Red Sea risk, Treasury yields and dollar moves shape trade".to_string()),
        summary: Some("Market participants weighed shipping disruption, OPEC rhetoric and U.S. yields as crude advanced.".to_string()),
        text: Some("Wall Street watched Treasury yields and the dollar while crude rose on Red Sea shipping risk and tighter OPEC supply language.".to_string()),
        v2_themes: Some("OIL;CRUDE;SHIPPING;TREASURY;DOLLAR".to_string()),
        v2_tone: Some("-1.5,0.2,0.1".to_string()),
        v2_locations: Some("1#Saudi Arabia#SA#SA#23#45#;1#United States#US#US#38#-97#".to_string()),
        v2_persons: Some("POWELL".to_string()),
        v2_organizations: Some("OPEC".to_string()),
        all_names: Some("WTI;BRENT;RED SEA".to_string()),
        metadata_json: Some(r#"{"extras":"<PAGE_TITLE>Oil prices rise on shipping risk</PAGE_TITLE>","gcam":"foo","dates":"20260618","amounts":"$80"}"#.to_string()),
    }
}

fn sample_supporting_doc_row() -> Map<String, Value> {
    Map::from_iter([
        ("title".to_string(), Value::from("Oil prices rise as Red Sea shipping risk lifts crude while Treasury yields move")),
        ("summary_text".to_string(), Value::from("WTI and Brent advanced as shipping disruption and dollar moves stayed in focus.")),
        ("body_excerpt".to_string(), Value::from("The market tracked crude, OPEC, Treasury yields and the dollar during Wall Street trading.")),
        ("market_context_text".to_string(), Value::from("Treasury yields, the dollar and oil stayed central to the session.")),
        ("source_type".to_string(), Value::from("market_wrap")),
        ("classification_confidence".to_string(), Value::from(0.87)),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn market_context_prefers_scored_sentences() {
        let (text, score) = extract_market_context_text(
            Some("Nasdaq falls sharply as Treasury yields rise and Wall Street cuts risk into the close."),
            None,
            None,
            None,
        );
        assert!(text.unwrap().contains("Nasdaq falls"));
        assert!(score >= 2.0);
    }

    #[test]
    fn geo_labels_default_to_global() {
        assert_eq!(extract_geo_labels(None), vec!["GLOBAL".to_string()]);
    }

    #[test]
    fn stable_u64_is_deterministic() {
        assert_eq!(stable_u64("abc"), stable_u64("abc"));
    }

    #[test]
    fn read_only_sql_guard_rejects_mutation() {
        assert!(normalize_read_only_clickhouse_sql("DROP TABLE bronze_candidates", 10).is_err());
    }

    #[test]
    fn read_only_sql_guard_adds_limit() {
        let normalized = normalize_read_only_clickhouse_sql("SELECT * FROM bronze_candidates", 25)
            .expect("query should normalize");
        assert!(normalized.ends_with("LIMIT 25"));
    }
}
