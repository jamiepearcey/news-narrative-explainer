use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use chrono::{Duration, NaiveDate, NaiveDateTime};
use clap::{Parser, Subcommand};
use glob::glob;
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};
use url::Url;

const DEFAULT_OUTPUT_ROOT: &str =
    "/Users/jamiepearcey/projects/research/news-narrative-explainer/data/narrative_graph_parquet_v3";
const DEFAULT_INPUT_GLOB: &str = "gdelt_candidates_20d_full/dt=*/part-*.parquet";
const DEFAULT_TAXONOMY_PATH: &str =
    "/Users/jamiepearcey/projects/research/news-narrative-explainer/config/news_narrative_taxonomy.json";
const DEFAULT_RESULTS_DIR: &str =
    "/Users/jamiepearcey/projects/research/news-narrative-explainer/v3/results";
const DEFAULT_FETCH_ROOT: &str =
    "/Users/jamiepearcey/projects/research/news-narrative-explainer/data";
const DEFAULT_GCS_URI_TEMPLATE: &str = "gs://market-news-datwa/gdelt/day-{date}/*.parquet";
const COLD_PAYLOAD_CHUNK_SIZE: usize = 4096;
const ARTIFACT_SWAGGER_UI_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>news-narrative-v3 artifact API Docs</title>
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

const MARKET_WRAP_DOMAINS: &[&str] = &[
    "reuters.com",
    "bloomberg.com",
    "cnbc.com",
    "nasdaq.com",
    "morningstar.com",
    "investors.com",
    "nikkei.com",
    "ftchinese.com",
    "wsj.com",
    "barrons.com",
    "marketwatch.com",
    "apnews.com",
    "ft.com",
    "finance.yahoo.com",
    "investopedia.com",
    "business-standard.com",
    "livemint.com",
    "cnbcafrica.com",
    "moneycontrol.com",
    "channelnewsasia.com",
    "theglobeandmail.com",
    "borsaitaliana.it",
    "handelsblatt.com",
];
const COMMODITY_SPECIALIST_DOMAINS: &[&str] = &[
    "oilandgas360.com",
    "kitco.com",
    "mining.com",
    "oilprice.com",
    "argusmedia.com",
    "hellenicshippingnews.com",
    "shipandbunker.com",
    "gcaptain.com",
    "rigzone.com",
    "worldoil.com",
    "bullionvault.com",
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

#[derive(Parser)]
#[command(name = "news-narrative-v3")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    BuildLocalDay {
        #[arg(long)]
        date: String,
        #[arg(long, default_value = DEFAULT_INPUT_GLOB)]
        input_glob: String,
        #[arg(long, default_value = DEFAULT_OUTPUT_ROOT)]
        output_root: String,
        #[arg(long, default_value = DEFAULT_TAXONOMY_PATH)]
        taxonomy_path: String,
        #[arg(long)]
        overwrite_day: bool,
        #[arg(long)]
        benchmark_out: Option<String>,
    },
    BenchmarkLocalDay {
        #[arg(long)]
        date: String,
        #[arg(long, default_value = DEFAULT_INPUT_GLOB)]
        input_glob: String,
        #[arg(long, default_value = DEFAULT_OUTPUT_ROOT)]
        output_root: String,
        #[arg(long, default_value = DEFAULT_TAXONOMY_PATH)]
        taxonomy_path: String,
        #[arg(long)]
        overwrite_day: bool,
    },
    ServeArtifacts {
        #[arg(long, default_value = DEFAULT_OUTPUT_ROOT)]
        output_root: String,
        #[arg(long, default_value = "127.0.0.1:8789")]
        bind: String,
    },
    PrintDuckdbClientSql {
        #[arg(long, default_value = "http://127.0.0.1:8789")]
        base_url: String,
        #[arg(long)]
        date: Option<String>,
    },
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

#[derive(Debug, Clone)]
struct BronzeCandidate {
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
    source_priority: i32,
    market_context_text: Option<String>,
    market_context_score: f64,
    tone: Option<f64>,
    geo_labels: Vec<String>,
    match_text: String,
    asset_match_text: String,
}

#[derive(Debug, Clone)]
struct SilverEventRow {
    event_time: String,
    event_date: String,
    cluster_id: u64,
    doc_id: u64,
    factor_ids: Vec<u32>,
    factor_labels: Vec<String>,
    asset_ids: Vec<u64>,
    asset_labels: Vec<String>,
    geo_ids: Vec<u64>,
    geo_labels: Vec<String>,
    source_id: u64,
    source_domain: String,
    tone: Option<f64>,
    novelty: f64,
    source_weight: f64,
    classification_confidence: f64,
    model_version: String,
    prompt_version: String,
    created_at: String,
}

#[derive(Debug, Clone)]
struct SilverFactorMentionRow {
    bucket_time: String,
    event_time: String,
    doc_id: u64,
    cluster_id: u64,
    factor_id: u32,
    factor_label: String,
    geo_id: u64,
    geo_label: String,
    source_id: u64,
    source_domain: String,
    tone: Option<f64>,
    novelty: f64,
    source_weight: f64,
    classification_confidence: f64,
}

#[derive(Debug, Clone)]
struct SilverAssetFactorMentionRow {
    bucket_time: String,
    event_time: String,
    doc_id: u64,
    cluster_id: u64,
    factor_id: u32,
    factor_label: String,
    asset_id: u64,
    asset_label: String,
    geo_id: u64,
    geo_label: String,
    source_id: u64,
    source_domain: String,
    tone: Option<f64>,
    novelty: f64,
    source_weight: f64,
    classification_confidence: f64,
    asset_factor_relevance: f64,
}

#[derive(Debug, Clone)]
struct SilverMarketContextRow {
    bucket_time: String,
    event_time: String,
    doc_id: u64,
    cluster_id: u64,
    factor_label: String,
    asset_label: String,
    source_domain: String,
    source_type: String,
    source_priority: i32,
    market_context_text: String,
    market_context_score: f64,
    classification_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoldFactorBucketRow {
    bucket_time: String,
    factor_id: u32,
    factor_label: String,
    geo_id: u64,
    geo_label: String,
    doc_count: u32,
    mention_count: u32,
    unique_sources: u32,
    geo_count: u32,
    tone_mean: Option<f64>,
    tone_zscore_30d: Option<f64>,
    avg_abs_tone: f64,
    novelty_mean: f64,
    negative_tail_count: u32,
    positive_tail_count: u32,
    source_dispersion: Option<f64>,
    weighted_source_mass: f64,
    weighted_source_dispersion: Option<f64>,
    confidence_mean: f64,
    first_seen: String,
    last_seen: String,
    narrative_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoldAssetFactorPanelRow {
    bucket_time: String,
    asset_id: u64,
    asset_label: String,
    factor_id: u32,
    factor_label: String,
    geo_id: u64,
    geo_label: String,
    doc_count: u32,
    mention_count: u32,
    unique_sources: u32,
    geo_count: u32,
    tone_mean: Option<f64>,
    tone_zscore_30d: Option<f64>,
    avg_abs_tone: f64,
    novelty_mean: f64,
    event_intensity: f64,
    source_dispersion: Option<f64>,
    weighted_source_mass: f64,
    weighted_source_dispersion: Option<f64>,
    confidence: f64,
    narrative_score: f64,
}

#[derive(Debug, Clone, Serialize)]
struct GoldFactorCrossoverRow {
    prior_bucket_time: String,
    bucket_time: String,
    factor_id: u32,
    factor_label: String,
    geo_id: u64,
    geo_label: String,
    prior_doc_count: u32,
    doc_count: u32,
    prior_narrative_score: f64,
    narrative_score: f64,
    doc_count_delta: i64,
    narrative_score_delta: f64,
}

#[derive(Debug, Clone, Serialize)]
struct GoldAssetFactorCrossoverRow {
    prior_bucket_time: String,
    bucket_time: String,
    asset_id: u64,
    asset_label: String,
    factor_id: u32,
    factor_label: String,
    geo_id: u64,
    geo_label: String,
    prior_doc_count: u32,
    doc_count: u32,
    prior_narrative_score: f64,
    narrative_score: f64,
    doc_count_delta: i64,
    narrative_score_delta: f64,
}

#[derive(Debug, Clone, Serialize)]
struct StageMetric {
    stage: String,
    rows_in: usize,
    rows_out: usize,
    elapsed_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PayloadChunkIndexEntry {
    file_name: String,
    min_doc_id: u64,
    max_doc_id: u64,
    row_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PayloadChunkIndexManifest {
    table: String,
    partition_date: String,
    chunk_size: usize,
    chunks: Vec<PayloadChunkIndexEntry>,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkArtifact {
    captured_at_utc: String,
    date: String,
    input_glob: String,
    input_files: Vec<String>,
    output_root: String,
    stage_metrics: Vec<StageMetric>,
    output_counts: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize)]
struct GraphDocNodeRow {
    doc_id: u64,
    partition_date: String,
    event_time: String,
    source_domain: String,
    document_identifier: String,
    title: Option<String>,
    source_type: String,
    source_priority: i32,
    market_context_text: Option<String>,
    market_context_score: f64,
    tone: Option<f64>,
    geo_labels_json: String,
}

#[derive(Debug, Clone, Serialize)]
struct DocPayloadRow {
    doc_id: u64,
    partition_date: String,
    document_identifier: String,
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
}

#[derive(Debug, Clone)]
struct BuildOutput {
    artifact: BenchmarkArtifact,
}

#[derive(Debug, Clone, Deserialize)]
struct FetchDaysRequest {
    dates: Vec<String>,
    gcs_uri_template: Option<String>,
    fetch_root: Option<String>,
    overwrite: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct LoadDaysRequest {
    dates: Vec<String>,
    fetch_root: Option<String>,
    output_root: Option<String>,
    taxonomy_path: Option<String>,
    overwrite_day: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct FetchAndLoadDaysRequest {
    dates: Vec<String>,
    gcs_uri_template: Option<String>,
    fetch_root: Option<String>,
    overwrite_fetch: Option<bool>,
    output_root: Option<String>,
    taxonomy_path: Option<String>,
    overwrite_day: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
struct FetchDayResult {
    date: String,
    gcs_uri: String,
    local_output_path: String,
    fetched_file_count: usize,
    fetched_total_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct LoadDayResult {
    date: String,
    input_glob: String,
    output_root: String,
    artifact: BenchmarkArtifact,
}

#[derive(Debug, Clone, Serialize)]
struct FetchDaysResponse {
    requested_dates: Vec<String>,
    results: Vec<FetchDayResult>,
}

#[derive(Debug, Clone, Serialize)]
struct LoadDaysResponse {
    requested_dates: Vec<String>,
    results: Vec<LoadDayResult>,
}

#[derive(Debug, Clone, Serialize)]
struct FetchAndLoadDayResult {
    fetch: FetchDayResult,
    load: LoadDayResult,
}

#[derive(Debug, Clone, Serialize)]
struct FetchAndLoadDaysResponse {
    requested_dates: Vec<String>,
    results: Vec<FetchAndLoadDayResult>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::BuildLocalDay {
            date,
            input_glob,
            output_root,
            taxonomy_path,
            overwrite_day,
            benchmark_out,
        } => {
            let output = build_local_day(
                &date,
                &input_glob,
                Path::new(&output_root),
                Path::new(&taxonomy_path),
                overwrite_day,
            )?;
            if let Some(path) = benchmark_out {
                write_benchmark_artifact(Path::new(&path), &output.artifact)?;
            }
            println!("{}", serde_json::to_string_pretty(&output.artifact)?);
        }
        Commands::BenchmarkLocalDay {
            date,
            input_glob,
            output_root,
            taxonomy_path,
            overwrite_day,
        } => {
            let output = build_local_day(
                &date,
                &input_glob,
                Path::new(&output_root),
                Path::new(&taxonomy_path),
                overwrite_day,
            )?;
            let out_path = default_benchmark_artifact_path(&date);
            write_benchmark_artifact(&out_path, &output.artifact)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "saved_benchmark": out_path,
                    "artifact": output.artifact
                }))?
            );
        }
        Commands::ServeArtifacts { output_root, bind } => {
            serve_artifacts(Path::new(&output_root), &bind)?;
        }
        Commands::PrintDuckdbClientSql { base_url, date } => {
            print_duckdb_client_sql(&base_url, date.as_deref());
        }
    }
    Ok(())
}

fn build_local_day(
    date: &str,
    input_glob: &str,
    output_root: &Path,
    taxonomy_path: &Path,
    overwrite_day: bool,
) -> Result<BuildOutput> {
    let day = NaiveDate::parse_from_str(date, "%Y-%m-%d")?;
    let input_files = resolve_input_files_for_day(input_glob, day)?;
    if input_files.is_empty() {
        bail!("no input parquet files matched {input_glob} for {date}");
    }
    if overwrite_day {
        clear_day_outputs(output_root, date)?;
    }
    let taxonomy = load_taxonomy(taxonomy_path)?;
    let mut stage_metrics = Vec::new();

    let start = Instant::now();
    let df = load_input_dataframe(&input_files).context("load_input_dataframe failed")?;
    stage_metrics.push(StageMetric {
        stage: "load_input".to_string(),
        rows_in: input_files.len(),
        rows_out: df.height(),
        elapsed_ms: elapsed_ms(start),
    });

    let start = Instant::now();
    let bronze =
        build_bronze_candidates(&df, &taxonomy, date).context("build_bronze_candidates failed")?;
    stage_metrics.push(StageMetric {
        stage: "build_bronze".to_string(),
        rows_in: df.height(),
        rows_out: bronze.len(),
        elapsed_ms: elapsed_ms(start),
    });

    let start = Instant::now();
    let silver =
        build_silver_graph(&bronze, &taxonomy, date).context("build_silver_graph failed")?;
    stage_metrics.push(StageMetric {
        stage: "build_silver".to_string(),
        rows_in: bronze.len(),
        rows_out: silver.factor_mentions.len() + silver.asset_factor_mentions.len(),
        elapsed_ms: elapsed_ms(start),
    });

    let start = Instant::now();
    let gold = build_gold_rollups(
        &silver.factor_mentions,
        &silver.asset_factor_mentions,
        output_root,
        day,
    )
    .context("build_gold_rollups failed")?;
    stage_metrics.push(StageMetric {
        stage: "build_gold".to_string(),
        rows_in: silver.factor_mentions.len() + silver.asset_factor_mentions.len(),
        rows_out: gold.factor_buckets.len() + gold.asset_factor_panel.len(),
        elapsed_ms: elapsed_ms(start),
    });

    let start = Instant::now();
    write_outputs(output_root, date, &bronze, &silver, &gold).context("write_outputs failed")?;
    update_manifest(output_root, date).context("update_manifest failed")?;
    stage_metrics.push(StageMetric {
        stage: "write_outputs".to_string(),
        rows_in: bronze.len()
            + silver.event_graph.len()
            + silver.factor_mentions.len()
            + silver.asset_factor_mentions.len()
            + silver.market_context_mentions.len()
            + gold.factor_buckets.len()
            + gold.asset_factor_panel.len(),
        rows_out: 9,
        elapsed_ms: elapsed_ms(start),
    });

    let artifact = BenchmarkArtifact {
        captured_at_utc: chrono::Utc::now().to_rfc3339(),
        date: date.to_string(),
        input_glob: input_glob.to_string(),
        input_files: input_files
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect(),
        output_root: output_root.to_string_lossy().to_string(),
        stage_metrics,
        output_counts: BTreeMap::from([
            ("graph_doc_nodes_daily".to_string(), bronze.len()),
            ("doc_payload_daily".to_string(), bronze.len()),
            ("doc_review_daily".to_string(), bronze.len()),
            ("doc_detail_daily".to_string(), bronze.len()),
            ("silver_event_graph".to_string(), silver.event_graph.len()),
            (
                "silver_factor_mentions".to_string(),
                silver.factor_mentions.len(),
            ),
            (
                "silver_asset_factor_mentions".to_string(),
                silver.asset_factor_mentions.len(),
            ),
            (
                "silver_market_context_mentions".to_string(),
                silver.market_context_mentions.len(),
            ),
            (
                "gold_factor_buckets_daily".to_string(),
                gold.factor_buckets.len(),
            ),
            (
                "gold_asset_factor_panel_daily".to_string(),
                gold.asset_factor_panel.len(),
            ),
            (
                "gold_factor_crossover_links_daily".to_string(),
                gold.factor_crossovers.len(),
            ),
            (
                "gold_asset_factor_crossover_links_daily".to_string(),
                gold.asset_crossovers.len(),
            ),
        ]),
    };
    Ok(BuildOutput { artifact })
}

#[derive(Debug)]
struct SilverBuild {
    event_graph: Vec<SilverEventRow>,
    factor_mentions: Vec<SilverFactorMentionRow>,
    asset_factor_mentions: Vec<SilverAssetFactorMentionRow>,
    market_context_mentions: Vec<SilverMarketContextRow>,
}

#[derive(Debug)]
struct GoldBuild {
    factor_buckets: Vec<GoldFactorBucketRow>,
    asset_factor_panel: Vec<GoldAssetFactorPanelRow>,
    factor_crossovers: Vec<GoldFactorCrossoverRow>,
    asset_crossovers: Vec<GoldAssetFactorCrossoverRow>,
}

fn resolve_input_files_for_day(input_glob: &str, day: NaiveDate) -> Result<Vec<PathBuf>> {
    let host_glob = resolve_host_glob(input_glob)?;
    let mut out = Vec::new();
    for entry in glob(&host_glob)? {
        let path = entry?;
        if partition_date_from_path(&path).as_deref() == Some(&day.to_string()) {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

fn resolve_host_glob(input_glob: &str) -> Result<String> {
    let root = PathBuf::from("/Users/jamiepearcey/projects/research/news-narrative-explainer/data");
    if Path::new(input_glob).is_absolute() {
        return Ok(input_glob.to_string());
    }
    if let Some(rest) = input_glob.strip_prefix("gdelt_candidates_20d_full/") {
        return Ok(root
            .join("gdelt_candidates_20d_full")
            .join(rest)
            .to_string_lossy()
            .to_string());
    }
    if let Some(rest) = input_glob.strip_prefix("gdelt_candidates/") {
        return Ok(root
            .join("gdelt_candidates")
            .join(rest)
            .to_string_lossy()
            .to_string());
    }
    Ok(root.join(input_glob).to_string_lossy().to_string())
}

fn partition_date_from_path(path: &Path) -> Option<String> {
    for component in path.components() {
        let text = component.as_os_str().to_string_lossy();
        if let Some(value) = text.strip_prefix("dt=") {
            return Some(value.to_string());
        }
        if let Some(value) = text.strip_prefix("gdelt_candidates_etl_day_") {
            let prefix = value.get(..10)?;
            let normalized = prefix.replace('_', "-");
            if NaiveDate::parse_from_str(&normalized, "%Y-%m-%d").is_ok() {
                return Some(normalized);
            }
        }
    }
    None
}

fn load_taxonomy(path: &Path) -> Result<Taxonomy> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn load_input_dataframe(paths: &[PathBuf]) -> Result<DataFrame> {
    let mut frames = Vec::new();
    for path in paths {
        let file = File::open(path)?;
        let frame = ParquetReader::new(file).finish()?;
        frames.push(frame);
    }
    let mut iter = frames.into_iter();
    let mut base = iter.next().context("no input frames loaded")?;
    for frame in iter {
        base.vstack_mut(&frame)?;
    }
    Ok(base)
}

fn build_bronze_candidates(
    df: &DataFrame,
    taxonomy: &Taxonomy,
    date: &str,
) -> Result<Vec<BronzeCandidate>> {
    let required_context: HashSet<&str> = ASSET_CONTEXT_REQUIRED.iter().copied().collect();
    let asset_patterns = asset_text_patterns();
    let rules = taxonomy.factors.clone();
    let rows: Result<Vec<_>> = (0..df.height())
        .into_par_iter()
        .map(|idx| {
            build_bronze_row(df, idx, date, &rules, &asset_patterns, &required_context)
                .with_context(|| format!("bronze row {idx} failed"))
        })
        .collect();
    rows
}

fn build_bronze_row(
    df: &DataFrame,
    idx: usize,
    date: &str,
    _rules: &[FactorDefinition],
    _asset_patterns: &HashMap<String, Vec<String>>,
    _required_context: &HashSet<&str>,
) -> Result<BronzeCandidate> {
    let record_datetime = get_string(df, "record_datetime", idx)
        .or_else(|| get_string(df, "date", idx))
        .unwrap_or_else(|| date.replace('-', ""));
    let partition_date = get_string(df, "partition_date", idx).unwrap_or_else(|| date.to_string());
    let document_identifier =
        get_string(df, "document_identifier", idx).context("document_identifier is required")?;
    let source_common_name = get_string(df, "source_common_name", idx);
    let title = normalized_optional(
        get_string(df, "title", idx)
            .or_else(|| get_string(df, "article_title", idx))
            .or_else(|| page_title_from_extras(get_string(df, "gkg_extras", idx).as_deref())),
    );
    let summary_text = normalized_optional(
        get_string(df, "summary", idx)
            .or_else(|| get_string(df, "snippet", idx))
            .or_else(|| get_string(df, "description", idx)),
    );
    let body_text = normalized_optional(
        get_string(df, "text", idx)
            .or_else(|| get_string(df, "article_text", idx))
            .or_else(|| get_string(df, "body_text", idx))
            .or_else(|| get_string(df, "content", idx)),
    );
    let v2_themes = get_string(df, "v2_themes", idx);
    let v2_tone = get_string(df, "v2_tone", idx);
    let v2_locations = get_string(df, "v2_locations", idx);
    let v2_persons = get_string(df, "v2_persons", idx);
    let v2_organizations = get_string(df, "v2_organizations", idx);
    let all_names = get_string(df, "all_names", idx);
    let metadata_json = get_string(df, "metadata_json", idx);
    let gkg_extras = normalized_optional(get_string(df, "gkg_extras", idx));
    let relevant_text = build_relevant_text(
        title.as_deref(),
        summary_text.as_deref(),
        body_text.as_deref(),
        all_names.as_deref(),
        v2_organizations.as_deref(),
        v2_persons.as_deref(),
        v2_themes.as_deref(),
        v2_locations.as_deref(),
    );
    let source_domain = extract_source_domain(source_common_name.as_deref(), &document_identifier)?;
    let (source_type, source_priority) =
        classify_source_type(&source_domain, title.as_deref(), Some(&document_identifier));
    let (market_context_text, market_context_score) = extract_market_context_text(
        title.as_deref(),
        summary_text.as_deref(),
        body_text.as_deref(),
        relevant_text.as_deref(),
    );
    let geo_labels = {
        let labels = extract_geo_labels(v2_locations.as_deref());
        if labels.is_empty() {
            vec!["GLOBAL".to_string()]
        } else {
            labels
        }
    };
    let match_text = uppercase_join(&[
        v2_themes.as_deref(),
        v2_persons.as_deref(),
        v2_organizations.as_deref(),
        all_names.as_deref(),
        v2_locations.as_deref(),
    ]);
    let asset_match_text = uppercase_join(&[
        title.as_deref(),
        summary_text.as_deref(),
        body_text.as_deref(),
        relevant_text.as_deref(),
        v2_themes.as_deref(),
        v2_persons.as_deref(),
        v2_organizations.as_deref(),
        all_names.as_deref(),
        v2_locations.as_deref(),
        gkg_extras.as_deref(),
        Some(document_identifier.as_str()),
    ]);
    let event_time = parse_record_datetime(&record_datetime, &partition_date).with_context(|| {
        format!(
            "invalid event time fields for row {idx}: record_datetime={record_datetime:?} partition_date={partition_date:?}"
        )
    })?;
    Ok(BronzeCandidate {
        doc_id: stable_u64(&document_identifier),
        record_datetime,
        event_time,
        partition_date,
        source_domain,
        document_identifier,
        v2_themes,
        v2_tone: v2_tone.clone(),
        v2_locations,
        v2_persons,
        v2_organizations,
        all_names,
        title,
        summary_text,
        body_text,
        relevant_text,
        metadata_json,
        gkg_extras,
        sharing_image: get_string(df, "sharing_image", idx),
        related_images: get_string(df, "related_images", idx),
        social_image_embeds: get_string(df, "social_image_embeds", idx),
        social_video_embeds: get_string(df, "social_video_embeds", idx),
        quotations: get_string(df, "quotations", idx),
        amounts: get_string(df, "amounts", idx),
        dates: get_string(df, "dates", idx),
        gcam: get_string(df, "gcam", idx),
        translation_info: get_string(df, "translation_info", idx),
        source_type,
        source_priority,
        market_context_text,
        market_context_score,
        tone: parse_tone(v2_tone.as_deref()),
        geo_labels,
        match_text,
        asset_match_text,
    })
}

fn build_silver_graph(
    bronze: &[BronzeCandidate],
    taxonomy: &Taxonomy,
    _date: &str,
) -> Result<SilverBuild> {
    let asset_patterns = asset_text_patterns();
    let required_context: HashSet<&str> = ASSET_CONTEXT_REQUIRED.iter().copied().collect();
    let now = chrono::Utc::now()
        .format("%Y-%m-%d %H:%M:%S%.3f")
        .to_string();
    let built: Vec<_> = bronze
        .par_iter()
        .map(|row| {
            let matched_factors = taxonomy
                .factors
                .iter()
                .filter(|factor| {
                    factor
                        .patterns
                        .iter()
                        .any(|pattern| row.match_text.contains(&pattern.to_uppercase()))
                })
                .cloned()
                .collect::<Vec<_>>();
            let factor_count = matched_factors.len();
            let confidence = classification_confidence(factor_count);
            let factor_ids = matched_factors.iter().map(|f| f.id).collect::<Vec<_>>();
            let factor_labels = matched_factors
                .iter()
                .map(|f| f.label.clone())
                .collect::<Vec<_>>();

            let mut matched_assets = Vec::<(u32, String, String)>::new();
            for factor in &matched_factors {
                for asset in &factor.asset_hints {
                    let needs_context = required_context.contains(asset.as_str());
                    let has_context = if !needs_context {
                        true
                    } else {
                        asset_patterns
                            .get(asset)
                            .map(|patterns| {
                                patterns
                                    .iter()
                                    .any(|pattern| row.asset_match_text.contains(pattern))
                            })
                            .unwrap_or(false)
                    };
                    if has_context {
                        matched_assets.push((factor.id, factor.label.clone(), asset.clone()));
                    }
                }
            }
            matched_assets.sort();
            matched_assets.dedup();

            let asset_labels_unique = unique_sorted_strings(
                matched_assets
                    .iter()
                    .map(|(_, _, asset)| asset.clone())
                    .collect::<Vec<_>>(),
            );
            let asset_ids = asset_labels_unique
                .iter()
                .map(|label| stable_u64(label))
                .collect::<Vec<_>>();
            let geo_ids = row
                .geo_labels
                .iter()
                .map(|label| stable_u64(label))
                .collect::<Vec<_>>();
            let source_id = stable_u64(&row.source_domain);
            let source_weight =
                source_weight_for_graph(&row.source_domain, &row.source_type, row.source_priority);
            let cluster_id = stable_u64(&format!(
                "{}|{}",
                row.source_domain, row.document_identifier
            ));

            let event = (!matched_factors.is_empty()).then(|| SilverEventRow {
                event_time: row.event_time.clone(),
                event_date: row.partition_date.clone(),
                cluster_id,
                doc_id: row.doc_id,
                factor_ids: factor_ids.clone(),
                factor_labels: factor_labels.clone(),
                asset_ids: asset_ids.clone(),
                asset_labels: asset_labels_unique.clone(),
                geo_ids: geo_ids.clone(),
                geo_labels: row.geo_labels.clone(),
                source_id,
                source_domain: row.source_domain.clone(),
                tone: row.tone,
                novelty: 1.0,
                source_weight,
                classification_confidence: confidence,
                model_version: "narrative_graph.v3.rust.v1".to_string(),
                prompt_version: "deterministic-narrative-taxonomy".to_string(),
                created_at: now.clone(),
            });

            let mut factor_mentions = Vec::new();
            for factor in &matched_factors {
                for geo_label in &row.geo_labels {
                    factor_mentions.push(SilverFactorMentionRow {
                        bucket_time: row.partition_date.clone(),
                        event_time: row.event_time.clone(),
                        doc_id: row.doc_id,
                        cluster_id,
                        factor_id: factor.id,
                        factor_label: factor.label.clone(),
                        geo_id: stable_u64(geo_label),
                        geo_label: geo_label.clone(),
                        source_id,
                        source_domain: row.source_domain.clone(),
                        tone: row.tone,
                        novelty: 1.0,
                        source_weight,
                        classification_confidence: confidence,
                    });
                }
            }

            let mut asset_factor_mentions = Vec::new();
            let mut market_context_mentions = Vec::new();
            for (factor_id, factor_label, asset_label) in matched_assets {
                let asset_id = stable_u64(&asset_label);
                let relevance = asset_factor_relevance(
                    row.asset_match_text.as_str(),
                    Some(asset_label.as_str()),
                    Some(factor_label.as_str()),
                );
                for geo_label in &row.geo_labels {
                    asset_factor_mentions.push(SilverAssetFactorMentionRow {
                        bucket_time: row.partition_date.clone(),
                        event_time: row.event_time.clone(),
                        doc_id: row.doc_id,
                        cluster_id,
                        factor_id,
                        factor_label: factor_label.clone(),
                        asset_id,
                        asset_label: asset_label.clone(),
                        geo_id: stable_u64(geo_label),
                        geo_label: geo_label.clone(),
                        source_id,
                        source_domain: row.source_domain.clone(),
                        tone: row.tone,
                        novelty: 1.0,
                        source_weight,
                        classification_confidence: confidence,
                        asset_factor_relevance: relevance,
                    });
                }
                if let Some(context_text) = &row.market_context_text {
                    market_context_mentions.push(SilverMarketContextRow {
                        bucket_time: row.partition_date.clone(),
                        event_time: row.event_time.clone(),
                        doc_id: row.doc_id,
                        cluster_id,
                        factor_label: factor_label.clone(),
                        asset_label: asset_label.clone(),
                        source_domain: row.source_domain.clone(),
                        source_type: row.source_type.clone(),
                        source_priority: row.source_priority,
                        market_context_text: context_text.clone(),
                        market_context_score: row.market_context_score,
                        classification_confidence: confidence,
                    });
                }
            }
            (
                event,
                factor_mentions,
                asset_factor_mentions,
                market_context_mentions,
            )
        })
        .collect();

    let mut event_graph = Vec::new();
    let mut factor_mentions = Vec::new();
    let mut asset_factor_mentions = Vec::new();
    let mut market_context_mentions = Vec::new();
    for (event, factors, assets, context) in built {
        if let Some(event) = event {
            event_graph.push(event);
        }
        factor_mentions.extend(factors);
        asset_factor_mentions.extend(assets);
        market_context_mentions.extend(context);
    }
    Ok(SilverBuild {
        event_graph,
        factor_mentions,
        asset_factor_mentions,
        market_context_mentions,
    })
}

fn build_gold_rollups(
    factor_mentions: &[SilverFactorMentionRow],
    asset_factor_mentions: &[SilverAssetFactorMentionRow],
    output_root: &Path,
    day: NaiveDate,
) -> Result<GoldBuild> {
    let factor_buckets = build_gold_factor_buckets(factor_mentions);
    let asset_factor_panel = build_gold_asset_factor_panel(asset_factor_mentions);
    let factor_crossovers = build_factor_crossovers(output_root, day, &factor_buckets)?;
    let asset_crossovers = build_asset_crossovers(output_root, day, &asset_factor_panel)?;
    Ok(GoldBuild {
        factor_buckets,
        asset_factor_panel,
        factor_crossovers,
        asset_crossovers,
    })
}

fn build_gold_factor_buckets(rows: &[SilverFactorMentionRow]) -> Vec<GoldFactorBucketRow> {
    #[derive(Default)]
    struct Agg {
        doc_ids: HashSet<u64>,
        source_ids: HashSet<u64>,
        source_weights: HashMap<u64, f64>,
        geo_ids: HashSet<u64>,
        mention_count: u32,
        tone_sum: f64,
        tone_count: u32,
        abs_tone_sum: f64,
        novelty_sum: f64,
        negative_tail_count: HashSet<u64>,
        positive_tail_count: HashSet<u64>,
        confidence_sum: f64,
        first_seen: Option<String>,
        last_seen: Option<String>,
    }
    let mut groups: BTreeMap<(String, u32, String, u64, String), Agg> = BTreeMap::new();
    for row in rows {
        let key = (
            row.bucket_time.clone(),
            row.factor_id,
            row.factor_label.clone(),
            row.geo_id,
            row.geo_label.clone(),
        );
        let agg = groups.entry(key).or_default();
        agg.doc_ids.insert(row.doc_id);
        agg.source_ids.insert(row.source_id);
        agg.source_weights
            .entry(row.source_id)
            .and_modify(|weight| *weight = weight.max(row.source_weight))
            .or_insert(row.source_weight);
        agg.geo_ids.insert(row.geo_id);
        agg.mention_count += 1;
        if let Some(tone) = row.tone {
            agg.tone_sum += tone;
            agg.abs_tone_sum += tone.abs();
            agg.tone_count += 1;
            if tone <= -5.0 {
                agg.negative_tail_count.insert(row.doc_id);
            }
            if tone >= 5.0 {
                agg.positive_tail_count.insert(row.doc_id);
            }
        }
        agg.novelty_sum += row.novelty;
        agg.confidence_sum += row.classification_confidence;
        min_string_assign(&mut agg.first_seen, &row.event_time);
        max_string_assign(&mut agg.last_seen, &row.event_time);
    }
    let mut by_factor: HashMap<u32, Vec<usize>> = HashMap::new();
    let mut out = Vec::new();
    for ((bucket_time, factor_id, factor_label, geo_id, geo_label), agg) in groups {
        let doc_count = agg.doc_ids.len() as u32;
        let unique_sources = agg.source_ids.len() as u32;
        let tone_mean = (agg.tone_count > 0).then_some(agg.tone_sum / agg.tone_count as f64);
        let avg_abs_tone = if agg.tone_count > 0 {
            agg.abs_tone_sum / agg.tone_count as f64
        } else {
            0.0
        };
        let novelty_mean = if agg.mention_count > 0 {
            agg.novelty_sum / agg.mention_count as f64
        } else {
            0.0
        };
        let source_dispersion = (doc_count > 0).then_some(unique_sources as f64 / doc_count as f64);
        let weighted_source_mass = agg.source_weights.values().sum::<f64>();
        let weighted_source_dispersion =
            (doc_count > 0).then_some(weighted_source_mass / doc_count as f64);
        let confidence_mean = if agg.mention_count > 0 {
            agg.confidence_sum / agg.mention_count as f64
        } else {
            0.0
        };
        let narrative_score = doc_count as f64
            * (0.5 + weighted_source_dispersion.unwrap_or(0.0))
            * (1.0 + (avg_abs_tone / 5.0));
        out.push(GoldFactorBucketRow {
            bucket_time,
            factor_id,
            factor_label,
            geo_id,
            geo_label,
            doc_count,
            mention_count: agg.mention_count,
            unique_sources,
            geo_count: agg.geo_ids.len() as u32,
            tone_mean,
            tone_zscore_30d: None,
            avg_abs_tone,
            novelty_mean,
            negative_tail_count: agg.negative_tail_count.len() as u32,
            positive_tail_count: agg.positive_tail_count.len() as u32,
            source_dispersion,
            weighted_source_mass,
            weighted_source_dispersion,
            confidence_mean,
            first_seen: agg.first_seen.unwrap_or_default(),
            last_seen: agg.last_seen.unwrap_or_default(),
            narrative_score,
        });
        by_factor.entry(factor_id).or_default().push(out.len() - 1);
    }
    apply_factor_zscores(&mut out, &by_factor);
    out
}

fn build_gold_asset_factor_panel(
    rows: &[SilverAssetFactorMentionRow],
) -> Vec<GoldAssetFactorPanelRow> {
    #[derive(Default)]
    struct Agg {
        doc_ids: HashSet<u64>,
        source_ids: HashSet<u64>,
        source_weights: HashMap<u64, f64>,
        geo_ids: HashSet<u64>,
        mention_count: u32,
        tone_sum: f64,
        tone_count: u32,
        abs_tone_sum: f64,
        novelty_sum: f64,
        confidence_sum: f64,
        relevance_sum: f64,
    }
    let mut groups: BTreeMap<(String, u64, String, u32, String, u64, String), Agg> =
        BTreeMap::new();
    for row in rows {
        let key = (
            row.bucket_time.clone(),
            row.asset_id,
            row.asset_label.clone(),
            row.factor_id,
            row.factor_label.clone(),
            row.geo_id,
            row.geo_label.clone(),
        );
        let agg = groups.entry(key).or_default();
        agg.doc_ids.insert(row.doc_id);
        agg.source_ids.insert(row.source_id);
        agg.source_weights
            .entry(row.source_id)
            .and_modify(|weight| *weight = weight.max(row.source_weight))
            .or_insert(row.source_weight);
        agg.geo_ids.insert(row.geo_id);
        agg.mention_count += 1;
        if let Some(tone) = row.tone {
            agg.tone_sum += tone;
            agg.abs_tone_sum += tone.abs();
            agg.tone_count += 1;
        }
        agg.novelty_sum += row.novelty;
        agg.confidence_sum += row.classification_confidence;
        agg.relevance_sum += row.asset_factor_relevance;
    }
    let mut by_asset_factor: HashMap<(u64, u32), Vec<usize>> = HashMap::new();
    let mut out = Vec::new();
    for ((bucket_time, asset_id, asset_label, factor_id, factor_label, geo_id, geo_label), agg) in
        groups
    {
        let doc_count = agg.doc_ids.len() as u32;
        let unique_sources = agg.source_ids.len() as u32;
        let tone_mean = (agg.tone_count > 0).then_some(agg.tone_sum / agg.tone_count as f64);
        let avg_abs_tone = if agg.tone_count > 0 {
            agg.abs_tone_sum / agg.tone_count as f64
        } else {
            0.0
        };
        let novelty_mean = if agg.mention_count > 0 {
            agg.novelty_sum / agg.mention_count as f64
        } else {
            0.0
        };
        let source_dispersion = (doc_count > 0).then_some(unique_sources as f64 / doc_count as f64);
        let confidence = if agg.mention_count > 0 {
            agg.confidence_sum / agg.mention_count as f64
        } else {
            0.0
        };
        let relevance_mean = if agg.mention_count > 0 {
            agg.relevance_sum / agg.mention_count as f64
        } else {
            0.0
        };
        let weighted_source_mass = agg.source_weights.values().sum::<f64>();
        let weighted_source_dispersion =
            (doc_count > 0).then_some(weighted_source_mass / doc_count as f64);
        let event_intensity = doc_count as f64
            * weighted_source_dispersion.unwrap_or(0.0)
            * (0.5 + relevance_mean / 8.0);
        let narrative_score = doc_count as f64
            * (0.5 + weighted_source_dispersion.unwrap_or(0.0))
            * (0.5 + relevance_mean / 8.0)
            * (1.0 + (avg_abs_tone / 5.0));
        out.push(GoldAssetFactorPanelRow {
            bucket_time,
            asset_id,
            asset_label,
            factor_id,
            factor_label,
            geo_id,
            geo_label,
            doc_count,
            mention_count: agg.mention_count,
            unique_sources,
            geo_count: agg.geo_ids.len() as u32,
            tone_mean,
            tone_zscore_30d: None,
            avg_abs_tone,
            novelty_mean,
            event_intensity,
            source_dispersion,
            weighted_source_mass,
            weighted_source_dispersion,
            confidence,
            narrative_score,
        });
        by_asset_factor
            .entry((asset_id, factor_id))
            .or_default()
            .push(out.len() - 1);
    }
    apply_asset_factor_zscores(&mut out, &by_asset_factor);
    out
}

fn build_factor_crossovers(
    output_root: &Path,
    day: NaiveDate,
    current: &[GoldFactorBucketRow],
) -> Result<Vec<GoldFactorCrossoverRow>> {
    let previous_day = day - Duration::days(1);
    let prev_path = partition_output_path(
        output_root,
        "gold_factor_buckets_daily",
        "bucket_time",
        &previous_day.to_string(),
    )
    .join("part-000.parquet");
    if !prev_path.exists() {
        return Ok(Vec::new());
    }
    let prev = load_gold_factor_bucket_rows(&prev_path)?;
    let prev_map: HashMap<(u32, u64), GoldFactorBucketRow> = prev
        .into_iter()
        .map(|row| ((row.factor_id, row.geo_id), row))
        .collect();
    let mut out = Vec::new();
    for row in current {
        if let Some(prev_row) = prev_map.get(&(row.factor_id, row.geo_id)) {
            out.push(GoldFactorCrossoverRow {
                prior_bucket_time: prev_row.bucket_time.clone(),
                bucket_time: row.bucket_time.clone(),
                factor_id: row.factor_id,
                factor_label: row.factor_label.clone(),
                geo_id: row.geo_id,
                geo_label: row.geo_label.clone(),
                prior_doc_count: prev_row.doc_count,
                doc_count: row.doc_count,
                prior_narrative_score: prev_row.narrative_score,
                narrative_score: row.narrative_score,
                doc_count_delta: row.doc_count as i64 - prev_row.doc_count as i64,
                narrative_score_delta: row.narrative_score - prev_row.narrative_score,
            });
        }
    }
    Ok(out)
}

fn build_asset_crossovers(
    output_root: &Path,
    day: NaiveDate,
    current: &[GoldAssetFactorPanelRow],
) -> Result<Vec<GoldAssetFactorCrossoverRow>> {
    let previous_day = day - Duration::days(1);
    let prev_path = partition_output_path(
        output_root,
        "gold_asset_factor_panel_daily",
        "bucket_time",
        &previous_day.to_string(),
    )
    .join("part-000.parquet");
    if !prev_path.exists() {
        return Ok(Vec::new());
    }
    let prev = load_gold_asset_factor_panel_rows(&prev_path)?;
    let prev_map: HashMap<(u64, u32, u64), GoldAssetFactorPanelRow> = prev
        .into_iter()
        .map(|row| ((row.asset_id, row.factor_id, row.geo_id), row))
        .collect();
    let mut out = Vec::new();
    for row in current {
        if let Some(prev_row) = prev_map.get(&(row.asset_id, row.factor_id, row.geo_id)) {
            out.push(GoldAssetFactorCrossoverRow {
                prior_bucket_time: prev_row.bucket_time.clone(),
                bucket_time: row.bucket_time.clone(),
                asset_id: row.asset_id,
                asset_label: row.asset_label.clone(),
                factor_id: row.factor_id,
                factor_label: row.factor_label.clone(),
                geo_id: row.geo_id,
                geo_label: row.geo_label.clone(),
                prior_doc_count: prev_row.doc_count,
                doc_count: row.doc_count,
                prior_narrative_score: prev_row.narrative_score,
                narrative_score: row.narrative_score,
                doc_count_delta: row.doc_count as i64 - prev_row.doc_count as i64,
                narrative_score_delta: row.narrative_score - prev_row.narrative_score,
            });
        }
    }
    Ok(out)
}

fn write_outputs(
    output_root: &Path,
    date: &str,
    bronze: &[BronzeCandidate],
    silver: &SilverBuild,
    gold: &GoldBuild,
) -> Result<()> {
    let mut sorted_bronze = bronze.to_vec();
    sorted_bronze.sort_by_key(|row| row.doc_id);
    write_df(
        &partition_output_path(output_root, "graph_doc_nodes_daily", "partition_date", date)
            .join("part-000.parquet"),
        graph_doc_nodes_df(bronze)?,
    )?;
    write_df(
        &partition_output_path(output_root, "doc_payload_daily", "partition_date", date)
            .join("part-000.parquet"),
        doc_payload_df(bronze)?,
    )?;
    write_df(
        &partition_output_path(output_root, "doc_review_daily", "partition_date", date)
            .join("part-000.parquet"),
        doc_review_df(bronze)?,
    )?;
    write_chunked_payload_artifact(
        &partition_output_path(output_root, "doc_review_daily", "partition_date", date),
        "doc_review_daily",
        date,
        &sorted_bronze,
        doc_review_df,
    )?;
    write_df(
        &partition_output_path(output_root, "doc_detail_daily", "partition_date", date)
            .join("part-000.parquet"),
        doc_detail_df(bronze)?,
    )?;
    write_chunked_payload_artifact(
        &partition_output_path(output_root, "doc_detail_daily", "partition_date", date),
        "doc_detail_daily",
        date,
        &sorted_bronze,
        doc_detail_df,
    )?;
    write_df(
        &partition_output_path(output_root, "silver_event_graph", "event_date", date)
            .join("part-000.parquet"),
        silver_event_graph_df(&silver.event_graph)?,
    )?;
    write_df(
        &partition_output_path(output_root, "silver_factor_mentions", "bucket_time", date)
            .join("part-000.parquet"),
        silver_factor_mentions_df(&silver.factor_mentions)?,
    )?;
    write_df(
        &partition_output_path(
            output_root,
            "silver_asset_factor_mentions",
            "bucket_time",
            date,
        )
        .join("part-000.parquet"),
        silver_asset_factor_mentions_df(&silver.asset_factor_mentions)?,
    )?;
    write_df(
        &partition_output_path(
            output_root,
            "silver_market_context_mentions",
            "bucket_time",
            date,
        )
        .join("part-000.parquet"),
        silver_market_context_mentions_df(&silver.market_context_mentions)?,
    )?;
    write_df(
        &partition_output_path(
            output_root,
            "gold_factor_buckets_daily",
            "bucket_time",
            date,
        )
        .join("part-000.parquet"),
        gold_factor_buckets_df(&gold.factor_buckets)?,
    )?;
    write_df(
        &partition_output_path(
            output_root,
            "gold_asset_factor_panel_daily",
            "bucket_time",
            date,
        )
        .join("part-000.parquet"),
        gold_asset_factor_panel_df(&gold.asset_factor_panel)?,
    )?;
    write_df(
        &partition_output_path(
            output_root,
            "gold_factor_crossover_links_daily",
            "bucket_time",
            date,
        )
        .join("part-000.parquet"),
        gold_factor_crossovers_df(&gold.factor_crossovers)?,
    )?;
    write_df(
        &partition_output_path(
            output_root,
            "gold_asset_factor_crossover_links_daily",
            "bucket_time",
            date,
        )
        .join("part-000.parquet"),
        gold_asset_crossovers_df(&gold.asset_crossovers)?,
    )?;
    Ok(())
}

fn write_df(path: &Path, mut df: DataFrame) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    ParquetWriter::new(file).finish(&mut df)?;
    Ok(())
}

fn write_chunked_payload_artifact<F>(
    partition_root: &Path,
    table: &str,
    partition_date: &str,
    rows: &[BronzeCandidate],
    builder: F,
) -> Result<()>
where
    F: Fn(&[BronzeCandidate]) -> Result<DataFrame>,
{
    fs::create_dir_all(partition_root)?;
    let mut chunks = Vec::new();
    for (idx, batch) in rows.chunks(COLD_PAYLOAD_CHUNK_SIZE).enumerate() {
        if batch.is_empty() {
            continue;
        }
        let file_name = format!("part-{idx:05}.parquet");
        write_df(&partition_root.join(&file_name), builder(batch)?)?;
        chunks.push(PayloadChunkIndexEntry {
            file_name,
            min_doc_id: batch.first().map(|row| row.doc_id).unwrap_or_default(),
            max_doc_id: batch.last().map(|row| row.doc_id).unwrap_or_default(),
            row_count: batch.len(),
        });
    }
    fs::write(
        partition_root.join("index.json"),
        serde_json::to_string_pretty(&PayloadChunkIndexManifest {
            table: table.to_string(),
            partition_date: partition_date.to_string(),
            chunk_size: COLD_PAYLOAD_CHUNK_SIZE,
            chunks,
        })?,
    )?;
    Ok(())
}

fn partition_output_path(
    output_root: &Path,
    table: &str,
    partition_column: &str,
    value: &str,
) -> PathBuf {
    output_root
        .join(table)
        .join(format!("{partition_column}={value}"))
}

fn clear_day_outputs(output_root: &Path, date: &str) -> Result<()> {
    let targets = [
        ("graph_doc_nodes_daily", "partition_date"),
        ("doc_payload_daily", "partition_date"),
        ("doc_review_daily", "partition_date"),
        ("doc_detail_daily", "partition_date"),
        ("silver_event_graph", "event_date"),
        ("silver_factor_mentions", "bucket_time"),
        ("silver_asset_factor_mentions", "bucket_time"),
        ("silver_market_context_mentions", "bucket_time"),
        ("gold_factor_buckets_daily", "bucket_time"),
        ("gold_asset_factor_panel_daily", "bucket_time"),
        ("gold_factor_crossover_links_daily", "bucket_time"),
        ("gold_asset_factor_crossover_links_daily", "bucket_time"),
    ];
    for (table, partition) in targets {
        let path = partition_output_path(output_root, table, partition, date);
        if path.exists() {
            fs::remove_dir_all(path)?;
        }
    }
    Ok(())
}

fn update_manifest(output_root: &Path, date: &str) -> Result<()> {
    let path = output_root.join("manifest.json");
    let mut dates = if path.exists() {
        serde_json::from_str::<serde_json::Value>(&fs::read_to_string(&path)?)?
            .get("materialized_dates")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|value| value.as_str().map(str::to_string))
            .collect::<BTreeSet<_>>()
    } else {
        BTreeSet::new()
    };
    dates.insert(date.to_string());
    fs::create_dir_all(output_root)?;
    fs::write(
        path,
        serde_json::to_string_pretty(&serde_json::json!({
            "materialized_dates": dates.into_iter().collect::<Vec<_>>()
        }))?,
    )?;
    Ok(())
}

fn default_benchmark_artifact_path(date: &str) -> PathBuf {
    PathBuf::from(DEFAULT_RESULTS_DIR).join(format!("{date}-v3-benchmark.json"))
}

fn write_benchmark_artifact(path: &Path, artifact: &BenchmarkArtifact) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(artifact)?)?;
    Ok(())
}

fn serve_artifacts(output_root: &Path, bind: &str) -> Result<()> {
    let server = Server::http(bind).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "status": "serving",
            "bind": bind,
            "output_root": output_root,
            "range_requests": true,
            "docs_url": format!("http://{bind}/docs")
        }))?
    );
    for request in server.incoming_requests() {
        if let Err(error) = handle_artifact_request(output_root, request) {
            eprintln!("artifact server error: {error:#}");
        }
    }
    Ok(())
}

fn handle_artifact_request(output_root: &Path, request: Request) -> Result<()> {
    match (request.method(), request.url()) {
        (&Method::Get, "/health") => {
            return respond_json(request, StatusCode(200), &serde_json::json!({"ok": true}));
        }
        (&Method::Get, "/openapi.json") => {
            return respond_json(request, StatusCode(200), &build_artifact_openapi_spec());
        }
        (&Method::Get, "/docs") => {
            return respond_html(request, StatusCode(200), ARTIFACT_SWAGGER_UI_HTML);
        }
        (&Method::Post, "/admin/fetch-days") => return handle_fetch_days_request(request),
        (&Method::Post, "/admin/load-days") => {
            return handle_load_days_request(output_root, request);
        }
        (&Method::Post, "/admin/fetch-and-load-days") => {
            return handle_fetch_and_load_days_request(output_root, request);
        }
        _ => {}
    }
    match request.method() {
        Method::Get | Method::Head => {}
        _ => {
            return respond_json(
                request,
                StatusCode(405),
                &serde_json::json!({
                    "error": "method_not_allowed",
                    "supported_paths": [
                        "/health",
                        "/openapi.json",
                        "/docs",
                        "/admin/fetch-days",
                        "/admin/load-days",
                        "/admin/fetch-and-load-days"
                    ]
                }),
            );
        }
    }
    let rel = request.url().trim_start_matches('/');
    let candidate = output_root.join(rel);
    let canonical_root = output_root.canonicalize()?;
    let canonical_candidate = candidate
        .canonicalize()
        .ok()
        .filter(|path| path.starts_with(&canonical_root));
    let Some(path) = canonical_candidate else {
        request.respond(Response::from_string("not found").with_status_code(StatusCode(404)))?;
        return Ok(());
    };
    if !path.is_file() {
        request.respond(Response::from_string("not found").with_status_code(StatusCode(404)))?;
        return Ok(());
    }
    let file_size = fs::metadata(&path)?.len();
    let range_header = request
        .headers()
        .iter()
        .find(|header| header.field.equiv("Range"))
        .map(|header| header.value.as_str().to_string());
    let is_head = matches!(request.method(), Method::Head);
    match range_header {
        Some(value) => respond_with_range(request, &path, file_size, &value, is_head),
        None => respond_full(request, &path, file_size, is_head),
    }
}

fn handle_fetch_days_request(mut request: Request) -> Result<()> {
    let payload: FetchDaysRequest = decode_json_request(&mut request, "fetch-days request")?;
    if payload.dates.is_empty() {
        return respond_json(
            request,
            StatusCode(400),
            &serde_json::json!({"error": "dates must not be empty"}),
        );
    }
    let fetch_root = PathBuf::from(
        payload
            .fetch_root
            .unwrap_or_else(|| DEFAULT_FETCH_ROOT.to_string()),
    );
    let gcs_uri_template = payload
        .gcs_uri_template
        .unwrap_or_else(|| DEFAULT_GCS_URI_TEMPLATE.to_string());
    let overwrite = payload.overwrite.unwrap_or(false);
    let mut results = Vec::with_capacity(payload.dates.len());
    for date in &payload.dates {
        results.push(fetch_day_from_gcs(
            date,
            &gcs_uri_template,
            &fetch_root,
            overwrite,
        )?);
    }
    respond_json(
        request,
        StatusCode(200),
        &FetchDaysResponse {
            requested_dates: payload.dates,
            results,
        },
    )
}

fn handle_load_days_request(output_root: &Path, mut request: Request) -> Result<()> {
    let payload: LoadDaysRequest = decode_json_request(&mut request, "load-days request")?;
    if payload.dates.is_empty() {
        return respond_json(
            request,
            StatusCode(400),
            &serde_json::json!({"error": "dates must not be empty"}),
        );
    }
    let fetch_root = PathBuf::from(
        payload
            .fetch_root
            .unwrap_or_else(|| DEFAULT_FETCH_ROOT.to_string()),
    );
    let output_root = PathBuf::from(
        payload
            .output_root
            .unwrap_or_else(|| output_root.to_string_lossy().to_string()),
    );
    let taxonomy_path = PathBuf::from(
        payload
            .taxonomy_path
            .unwrap_or_else(|| DEFAULT_TAXONOMY_PATH.to_string()),
    );
    let overwrite_day = payload.overwrite_day.unwrap_or(false);
    let mut results = Vec::with_capacity(payload.dates.len());
    for date in &payload.dates {
        results.push(load_day_into_graph(
            date,
            &fetch_root,
            &output_root,
            &taxonomy_path,
            overwrite_day,
        )?);
    }
    respond_json(
        request,
        StatusCode(200),
        &LoadDaysResponse {
            requested_dates: payload.dates,
            results,
        },
    )
}

fn handle_fetch_and_load_days_request(output_root: &Path, mut request: Request) -> Result<()> {
    let payload: FetchAndLoadDaysRequest =
        decode_json_request(&mut request, "fetch-and-load-days request")?;
    if payload.dates.is_empty() {
        return respond_json(
            request,
            StatusCode(400),
            &serde_json::json!({"error": "dates must not be empty"}),
        );
    }
    let fetch_root = PathBuf::from(
        payload
            .fetch_root
            .unwrap_or_else(|| DEFAULT_FETCH_ROOT.to_string()),
    );
    let gcs_uri_template = payload
        .gcs_uri_template
        .unwrap_or_else(|| DEFAULT_GCS_URI_TEMPLATE.to_string());
    let output_root = PathBuf::from(
        payload
            .output_root
            .unwrap_or_else(|| output_root.to_string_lossy().to_string()),
    );
    let taxonomy_path = PathBuf::from(
        payload
            .taxonomy_path
            .unwrap_or_else(|| DEFAULT_TAXONOMY_PATH.to_string()),
    );
    let overwrite_fetch = payload.overwrite_fetch.unwrap_or(false);
    let overwrite_day = payload.overwrite_day.unwrap_or(false);
    let mut results = Vec::with_capacity(payload.dates.len());
    for date in &payload.dates {
        let fetch = fetch_day_from_gcs(date, &gcs_uri_template, &fetch_root, overwrite_fetch)?;
        let load = load_day_into_graph(
            date,
            &fetch_root,
            &output_root,
            &taxonomy_path,
            overwrite_day,
        )?;
        results.push(FetchAndLoadDayResult { fetch, load });
    }
    respond_json(
        request,
        StatusCode(200),
        &FetchAndLoadDaysResponse {
            requested_dates: payload.dates,
            results,
        },
    )
}

fn decode_json_request<T: for<'de> Deserialize<'de>>(
    request: &mut Request,
    label: &str,
) -> Result<T> {
    let mut body = String::new();
    request
        .as_reader()
        .read_to_string(&mut body)
        .with_context(|| format!("failed to read {label} body"))?;
    serde_json::from_str(&body).with_context(|| format!("failed to decode {label}"))
}

fn respond_json(request: Request, status: StatusCode, body: &impl Serialize) -> Result<()> {
    let response_body = serde_json::to_vec(body)?;
    let response = Response::from_data(response_body)
        .with_status_code(status)
        .with_header(
            Header::from_bytes("Content-Type", "application/json")
                .map_err(|_| anyhow!("invalid content-type header"))?,
        );
    request.respond(response)?;
    Ok(())
}

fn respond_html(request: Request, status: StatusCode, body: &str) -> Result<()> {
    let response = Response::from_string(body.to_string())
        .with_status_code(status)
        .with_header(
            Header::from_bytes("Content-Type", "text/html; charset=utf-8")
                .map_err(|_| anyhow!("invalid content-type header"))?,
        );
    request.respond(response)?;
    Ok(())
}

fn build_artifact_openapi_spec() -> serde_json::Value {
    serde_json::json!({
        "openapi": "3.1.0",
        "info": {
            "title": "news-narrative-v3 artifact API",
            "version": "0.1.0",
            "description": "Static parquet artifact serving plus day fetch/load administration for v3."
        },
        "servers": [{"url": "/"}],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check",
                    "responses": {"200": {"description": "Service is healthy"}}
                }
            },
            "/admin/fetch-days": {
                "post": {
                    "summary": "Fetch one or more day exports from GCS into local day directories",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/FetchDaysRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Per-day fetch results",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/FetchDaysResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/admin/load-days": {
                "post": {
                    "summary": "Build one or more days into the v3 graph parquet output",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/LoadDaysRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Per-day load results",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/LoadDaysResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/admin/fetch-and-load-days": {
                "post": {
                    "summary": "Fetch then load one or more days sequentially",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/FetchAndLoadDaysRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Per-day fetch and load results",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/FetchAndLoadDaysResponse"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "FetchDaysRequest": {
                    "type": "object",
                    "properties": {
                        "dates": {"type": "array", "items": {"type": "string"}},
                        "gcs_uri_template": {"type": "string"},
                        "fetch_root": {"type": "string"},
                        "overwrite": {"type": "boolean"}
                    },
                    "required": ["dates"]
                },
                "LoadDaysRequest": {
                    "type": "object",
                    "properties": {
                        "dates": {"type": "array", "items": {"type": "string"}},
                        "fetch_root": {"type": "string"},
                        "output_root": {"type": "string"},
                        "taxonomy_path": {"type": "string"},
                        "overwrite_day": {"type": "boolean"}
                    },
                    "required": ["dates"]
                },
                "FetchAndLoadDaysRequest": {
                    "type": "object",
                    "properties": {
                        "dates": {"type": "array", "items": {"type": "string"}},
                        "gcs_uri_template": {"type": "string"},
                        "fetch_root": {"type": "string"},
                        "overwrite_fetch": {"type": "boolean"},
                        "output_root": {"type": "string"},
                        "taxonomy_path": {"type": "string"},
                        "overwrite_day": {"type": "boolean"}
                    },
                    "required": ["dates"]
                },
                "FetchDayResult": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "gcs_uri": {"type": "string"},
                        "local_output_path": {"type": "string"},
                        "fetched_file_count": {"type": "integer"},
                        "fetched_total_bytes": {"type": "integer"}
                    },
                    "required": ["date", "gcs_uri", "local_output_path", "fetched_file_count", "fetched_total_bytes"]
                },
                "LoadDayResult": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "input_glob": {"type": "string"},
                        "output_root": {"type": "string"},
                        "artifact": {"type": "object"}
                    },
                    "required": ["date", "input_glob", "output_root", "artifact"]
                },
                "FetchDaysResponse": {
                    "type": "object",
                    "properties": {
                        "requested_dates": {"type": "array", "items": {"type": "string"}},
                        "results": {"type": "array", "items": {"$ref": "#/components/schemas/FetchDayResult"}}
                    },
                    "required": ["requested_dates", "results"]
                },
                "LoadDaysResponse": {
                    "type": "object",
                    "properties": {
                        "requested_dates": {"type": "array", "items": {"type": "string"}},
                        "results": {"type": "array", "items": {"$ref": "#/components/schemas/LoadDayResult"}}
                    },
                    "required": ["requested_dates", "results"]
                },
                "FetchAndLoadDayResult": {
                    "type": "object",
                    "properties": {
                        "fetch": {"$ref": "#/components/schemas/FetchDayResult"},
                        "load": {"$ref": "#/components/schemas/LoadDayResult"}
                    },
                    "required": ["fetch", "load"]
                },
                "FetchAndLoadDaysResponse": {
                    "type": "object",
                    "properties": {
                        "requested_dates": {"type": "array", "items": {"type": "string"}},
                        "results": {"type": "array", "items": {"$ref": "#/components/schemas/FetchAndLoadDayResult"}}
                    },
                    "required": ["requested_dates", "results"]
                }
            }
        }
    })
}

fn fetch_day_from_gcs(
    date: &str,
    gcs_uri_template: &str,
    fetch_root: &Path,
    overwrite: bool,
) -> Result<FetchDayResult> {
    let gcs_uri = apply_date_template(gcs_uri_template, date);
    let local_output_path = local_fetch_day_dir(fetch_root, date);
    if overwrite && local_output_path.exists() {
        fs::remove_dir_all(&local_output_path)?;
    }
    fs::create_dir_all(&local_output_path)?;
    run_gcs_copy(&gcs_uri, &local_output_path)?;
    let (fetched_file_count, fetched_total_bytes) = summarize_parquet_dir(&local_output_path)?;
    Ok(FetchDayResult {
        date: date.to_string(),
        gcs_uri,
        local_output_path: local_output_path.to_string_lossy().to_string(),
        fetched_file_count,
        fetched_total_bytes,
    })
}

fn load_day_into_graph(
    date: &str,
    fetch_root: &Path,
    output_root: &Path,
    taxonomy_path: &Path,
    overwrite_day: bool,
) -> Result<LoadDayResult> {
    let input_glob = local_fetch_day_input_glob(fetch_root, date);
    let output = build_local_day(date, &input_glob, output_root, taxonomy_path, overwrite_day)?;
    Ok(LoadDayResult {
        date: date.to_string(),
        input_glob,
        output_root: output_root.to_string_lossy().to_string(),
        artifact: output.artifact,
    })
}

fn apply_date_template(template: &str, date: &str) -> String {
    template
        .replace("{date}", date)
        .replace("{date_underscored}", &date.replace('-', "_"))
}

fn local_fetch_day_dir(fetch_root: &Path, date: &str) -> PathBuf {
    fetch_root.join(format!(
        "gdelt_candidates_etl_day_{}",
        date.replace('-', "_")
    ))
}

fn local_fetch_day_input_glob(fetch_root: &Path, date: &str) -> String {
    local_fetch_day_dir(fetch_root, date)
        .join("*.parquet")
        .to_string_lossy()
        .to_string()
}

fn run_gcs_copy(gcs_uri: &str, local_output_path: &Path) -> Result<()> {
    if let Ok(status) = Command::new("gcloud")
        .args(["storage", "cp", "--recursive", gcs_uri])
        .arg(local_output_path)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
    {
        if status.success() {
            return Ok(());
        }
        bail!("gcloud storage cp failed for {gcs_uri} with status {status}");
    }
    let status = Command::new("gsutil")
        .args(["-m", "cp", gcs_uri])
        .arg(local_output_path)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .context("failed to launch gcloud or gsutil for GCS copy")?;
    if !status.success() {
        bail!("gsutil cp failed for {gcs_uri} with status {status}");
    }
    Ok(())
}

fn summarize_parquet_dir(path: &Path) -> Result<(usize, u64)> {
    let mut file_count = 0usize;
    let mut total_bytes = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let candidate = entry.path();
        if candidate.extension().and_then(|ext| ext.to_str()) != Some("parquet") {
            continue;
        }
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            file_count += 1;
            total_bytes += metadata.len();
        }
    }
    Ok((file_count, total_bytes))
}

fn respond_full(request: Request, path: &Path, file_size: u64, is_head: bool) -> Result<()> {
    if is_head {
        let response = Response::empty(200)
            .with_header(content_type_header(path)?)
            .with_header(accept_ranges_header()?)
            .with_header(content_length_header(file_size)?);
        request.respond(response)?;
    } else {
        let response = Response::from_file(File::open(path)?)
            .with_header(content_type_header(path)?)
            .with_header(accept_ranges_header()?)
            .with_header(content_length_header(file_size)?);
        request.respond(response)?;
    }
    Ok(())
}

fn respond_with_range(
    request: Request,
    path: &Path,
    file_size: u64,
    header_value: &str,
    is_head: bool,
) -> Result<()> {
    let Some((start, end)) = parse_range_header(header_value, file_size) else {
        request
            .respond(Response::from_string("invalid range").with_status_code(StatusCode(416)))?;
        return Ok(());
    };
    let length = end - start + 1;
    if is_head {
        let response = Response::empty(206)
            .with_status_code(StatusCode(206))
            .with_header(content_type_header(path)?)
            .with_header(accept_ranges_header()?)
            .with_header(content_length_header(length)?)
            .with_header(content_range_header(start, end, file_size)?);
        request.respond(response)?;
    } else {
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(start))?;
        let mut buffer = vec![0_u8; length as usize];
        file.read_exact(&mut buffer)?;
        let response = Response::from_data(buffer)
            .with_status_code(StatusCode(206))
            .with_header(content_type_header(path)?)
            .with_header(accept_ranges_header()?)
            .with_header(content_length_header(length)?)
            .with_header(content_range_header(start, end, file_size)?);
        request.respond(response)?;
    }
    Ok(())
}

fn parse_range_header(header_value: &str, file_size: u64) -> Option<(u64, u64)> {
    let bytes = header_value.strip_prefix("bytes=")?;
    let (start_raw, end_raw) = bytes.split_once('-')?;
    if start_raw.is_empty() {
        let suffix = end_raw.parse::<u64>().ok()?;
        if suffix == 0 || suffix > file_size {
            return None;
        }
        return Some((file_size - suffix, file_size - 1));
    }
    let start = start_raw.parse::<u64>().ok()?;
    let end = if end_raw.is_empty() {
        file_size.checked_sub(1)?
    } else {
        end_raw.parse::<u64>().ok()?
    };
    if start > end || end >= file_size {
        return None;
    }
    Some((start, end))
}

fn content_type_header(path: &Path) -> Result<Header> {
    let value = if path.extension().and_then(|ext| ext.to_str()) == Some("parquet") {
        "application/vnd.apache.parquet"
    } else if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
        "application/json"
    } else {
        "application/octet-stream"
    };
    Header::from_bytes("Content-Type", value)
        .map_err(|_| anyhow::anyhow!("invalid content-type header"))
}

fn accept_ranges_header() -> Result<Header> {
    Header::from_bytes("Accept-Ranges", "bytes")
        .map_err(|_| anyhow::anyhow!("invalid accept-ranges header"))
}

fn content_length_header(length: u64) -> Result<Header> {
    Header::from_bytes("Content-Length", length.to_string())
        .map_err(|_| anyhow::anyhow!("invalid content-length header"))
}

fn content_range_header(start: u64, end: u64, total: u64) -> Result<Header> {
    Header::from_bytes("Content-Range", format!("bytes {start}-{end}/{total}"))
        .map_err(|_| anyhow::anyhow!("invalid content-range header"))
}

fn print_duckdb_client_sql(base_url: &str, date: Option<&str>) {
    let date = date.unwrap_or("2026-06-05");
    let sql = format!(
        "INSTALL httpfs;\nLOAD httpfs;\n\n-- Hot graph scan\nSELECT factor_label, sum(doc_count) AS doc_count\nFROM read_parquet('{base_url}/gold_factor_buckets_daily/bucket_time={date}/part-000.parquet')\nGROUP BY factor_label\nORDER BY doc_count DESC;\n\n-- Hydrate light review payloads for broader reranking\nSELECT *\nFROM read_parquet('{base_url}/doc_review_daily/partition_date={date}/part-000.parquet')\nWHERE doc_id IN (/* shortlisted doc ids */);\n\n-- Hydrate heavier detail payloads only for the final supporting docs\nSELECT *\nFROM read_parquet('{base_url}/doc_detail_daily/partition_date={date}/part-000.parquet')\nWHERE doc_id IN (/* final doc ids */);\n"
    );
    println!("{sql}");
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn stable_u64(text: &str) -> u64 {
    let mut digest = Sha256::new();
    digest.update(text.as_bytes());
    let bytes = digest.finalize();
    u64::from_be_bytes(bytes[..8].try_into().expect("sha256 digest length"))
}

fn parse_record_datetime(raw: &str, partition_date: &str) -> Result<String> {
    let raw = raw.trim();
    let raw_digits = raw
        .chars()
        .filter(|ch| ch.is_ascii_digit())
        .collect::<String>();
    for candidate in [raw, raw_digits.as_str()] {
        if candidate.is_empty() {
            continue;
        }
        if let Ok(parsed) = NaiveDateTime::parse_from_str(candidate, "%Y%m%d%H%M%S") {
            return Ok(parsed.format("%Y-%m-%d %H:%M:%S%.3f").to_string());
        }
        if let Ok(parsed) = NaiveDate::parse_from_str(candidate, "%Y%m%d") {
            return Ok(parsed
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .format("%Y-%m-%d %H:%M:%S%.3f")
                .to_string());
        }
    }
    let partition_date = partition_date.trim();
    if let Ok(date) = NaiveDate::parse_from_str(partition_date, "%Y-%m-%d") {
        return Ok(date
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .format("%Y-%m-%d %H:%M:%S%.3f")
            .to_string());
    }
    let partition_digits = partition_date
        .chars()
        .filter(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if let Ok(date) = NaiveDate::parse_from_str(&partition_digits, "%Y%m%d") {
        return Ok(date
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .format("%Y-%m-%d %H:%M:%S%.3f")
            .to_string());
    }
    bail!(
        "unable to parse record_datetime {:?} with partition_date {:?}",
        raw,
        partition_date
    )
}

fn parse_tone(raw: Option<&str>) -> Option<f64> {
    let raw = raw?.split(',').next()?.trim();
    raw.parse::<f64>().ok()
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
    Ok(Url::parse(document_identifier)
        .ok()
        .and_then(|url| url.host_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string())
        .to_lowercase())
}

fn normalized_optional(raw: Option<String>) -> Option<String> {
    raw.and_then(|value| {
        let decoded = if value.contains('&') {
            html_escape::decode_html_entities(&value).to_string()
        } else {
            value
        };
        let collapsed = decoded.split_whitespace().collect::<Vec<_>>().join(" ");
        (!collapsed.is_empty()).then_some(collapsed)
    })
}

fn page_title_from_extras(extras: Option<&str>) -> Option<String> {
    let extras = extras?;
    let start = extras.find("<PAGE_TITLE>")? + "<PAGE_TITLE>".len();
    let end = extras[start..].find("</PAGE_TITLE>")? + start;
    normalized_optional(Some(extras[start..end].to_string()))
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
        body_text.map(|text| &text[..text.len().min(4000)]),
        all_names,
        organizations,
        persons,
        themes,
        locations,
    ] {
        if let Some(value) = part {
            if !value.trim().is_empty() {
                parts.push(value.trim().to_string());
            }
        }
    }
    (!parts.is_empty()).then_some(parts.join(" || "))
}

fn classify_source_type(
    source_domain: &str,
    title: Option<&str>,
    document_identifier: Option<&str>,
) -> (String, i32) {
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

fn extract_geo_labels(raw: Option<&str>) -> Vec<String> {
    let mut labels = BTreeSet::new();
    for entry in raw.unwrap_or_default().split(';') {
        for part in entry.split('#') {
            let upper = part.trim().to_uppercase();
            if upper.len() == 2 && upper.chars().all(|ch| ch.is_ascii_alphabetic()) {
                labels.insert(upper);
            }
        }
    }
    labels.into_iter().collect()
}

fn split_sentences(text: Option<&str>) -> Vec<String> {
    let Some(cleaned) = text.and_then(|text| normalized_optional(Some(text.to_string()))) else {
        return Vec::new();
    };
    cleaned
        .split("||")
        .flat_map(|part| part.split_terminator(['.', '!', '?', '\n']))
        .filter_map(|part| normalized_optional(Some(part.to_string())))
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
    let mut seen = HashSet::new();
    let mut scored = Vec::new();
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
    let kept = scored
        .iter()
        .take(5)
        .map(|(_, s)| s.clone())
        .collect::<Vec<_>>();
    if kept.is_empty() {
        (None, 0.0)
    } else {
        let total = scored.iter().take(5).map(|(score, _)| *score).sum::<i32>();
        (Some(kept.join(" || ")), total as f64)
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

fn unique_sorted_strings(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn asset_factor_relevance(
    text: &str,
    asset_label: Option<&str>,
    factor_label: Option<&str>,
) -> f64 {
    let asset_hits = asset_cues(asset_label)
        .iter()
        .filter(|cue| text.contains(cue.as_str()))
        .count();
    let factor_hits = factor_cues(factor_label)
        .iter()
        .filter(|cue| text.contains(cue.as_str()))
        .count();
    if asset_hits == 0 && factor_hits == 0 {
        0.0
    } else {
        (asset_hits * 2 + factor_hits) as f64
    }
}

fn classification_confidence(match_count: usize) -> f64 {
    (0.55 + 0.08 * match_count as f64).min(0.95)
}

fn asset_text_patterns() -> HashMap<String, Vec<String>> {
    HashMap::from([
        (
            "WTI".to_string(),
            vec![
                "WTI".to_string(),
                "WEST TEXAS INTERMEDIATE".to_string(),
                "CRUDE".to_string(),
                "OIL".to_string(),
            ],
        ),
        (
            "Brent".to_string(),
            vec![
                "BRENT".to_string(),
                "BRENT CRUDE".to_string(),
                "OIL".to_string(),
            ],
        ),
        (
            "Gold".to_string(),
            vec!["GOLD".to_string(), "BULLION".to_string(), "XAU".to_string()],
        ),
        (
            "BTC".to_string(),
            vec![
                "BTC".to_string(),
                "BITCOIN".to_string(),
                "CRYPTO".to_string(),
            ],
        ),
        (
            "NDX".to_string(),
            vec![
                "NDX".to_string(),
                "NASDAQ".to_string(),
                "NASDAQ 100".to_string(),
                "QQQ".to_string(),
            ],
        ),
        (
            "SPX".to_string(),
            vec![
                "SPX".to_string(),
                "S&P".to_string(),
                "S&P 500".to_string(),
                "SP 500".to_string(),
            ],
        ),
        (
            "HG".to_string(),
            vec!["COPPER".to_string(), "HG".to_string()],
        ),
        (
            "NG".to_string(),
            vec![
                "NATURAL GAS".to_string(),
                "NG".to_string(),
                "GAS".to_string(),
            ],
        ),
        (
            "TTF".to_string(),
            vec!["TTF".to_string(), "EUROPEAN GAS".to_string()],
        ),
        (
            "XLE".to_string(),
            vec!["XLE".to_string(), "ENERGY ETF".to_string()],
        ),
        (
            "XME".to_string(),
            vec!["XME".to_string(), "METALS ETF".to_string()],
        ),
        (
            "GDX".to_string(),
            vec!["GDX".to_string(), "GOLD MINERS".to_string()],
        ),
        (
            "FXI".to_string(),
            vec!["FXI".to_string(), "CHINA ETF".to_string()],
        ),
        (
            "CAD".to_string(),
            vec!["CAD".to_string(), "CANADIAN DOLLAR".to_string()],
        ),
        (
            "BDI".to_string(),
            vec!["BDI".to_string(), "BALTIC DRY".to_string()],
        ),
        (
            "FCX".to_string(),
            vec!["FREEPORT".to_string(), "FCX".to_string()],
        ),
        ("BHP".to_string(), vec!["BHP".to_string()]),
        (
            "RIO".to_string(),
            vec!["RIO".to_string(), "RIO TINTO".to_string()],
        ),
        (
            "COIN".to_string(),
            vec!["COIN".to_string(), "COINBASE".to_string()],
        ),
    ])
}

fn factor_cues(factor_label: Option<&str>) -> Vec<String> {
    match factor_label.unwrap_or_default() {
        "oil" => vec![
            "OIL".to_string(),
            "CRUDE".to_string(),
            "BRENT".to_string(),
            "WTI".to_string(),
        ],
        "war_conflict" => vec![
            "WAR".to_string(),
            "CONFLICT".to_string(),
            "STRIKE".to_string(),
        ],
        "sanctions_trade" => vec![
            "SANCTION".to_string(),
            "TRADE".to_string(),
            "TARIFF".to_string(),
        ],
        "shipping_disruption" => vec![
            "SHIPPING".to_string(),
            "FREIGHT".to_string(),
            "PORT".to_string(),
        ],
        "growth_activity" => vec![
            "GROWTH".to_string(),
            "PMI".to_string(),
            "DEMAND".to_string(),
        ],
        _ => factor_label
            .map(|label| vec![label.replace('_', " ").to_uppercase()])
            .unwrap_or_default(),
    }
}

fn asset_cues(asset_label: Option<&str>) -> Vec<String> {
    asset_text_patterns()
        .get(asset_label.unwrap_or_default())
        .cloned()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{apply_date_template, local_fetch_day_dir, local_fetch_day_input_glob};
    use std::path::Path;

    #[test]
    fn apply_date_template_supports_both_date_tokens() {
        let rendered = apply_date_template(
            "gs://bucket/day-{date}/{date_underscored}/*.parquet",
            "2026-06-05",
        );
        assert_eq!(rendered, "gs://bucket/day-2026-06-05/2026_06_05/*.parquet");
    }

    #[test]
    fn local_fetch_paths_match_existing_day_layout() {
        let root = Path::new("/tmp/news-narrative");
        assert_eq!(
            local_fetch_day_dir(root, "2026-06-05"),
            root.join("gdelt_candidates_etl_day_2026_06_05")
        );
        assert_eq!(
            local_fetch_day_input_glob(root, "2026-06-05"),
            "/tmp/news-narrative/gdelt_candidates_etl_day_2026_06_05/*.parquet"
        );
    }
}

fn get_string(df: &DataFrame, column: &str, idx: usize) -> Option<String> {
    let series = df.column(column).ok()?;
    if let Ok(strings) = series.str() {
        let rendered = strings.get(idx)?;
        return (!rendered.is_empty() && rendered != "null").then(|| rendered.to_string());
    }
    let casted = series.cast(&DataType::String).ok()?;
    let strings = casted.str().ok()?;
    let rendered = strings.get(idx)?;
    if rendered.is_empty() || rendered == "null" {
        None
    } else {
        Some(rendered.to_string())
    }
}

fn min_string_assign(slot: &mut Option<String>, value: &str) {
    match slot {
        Some(existing) if existing.as_str() <= value => {}
        _ => *slot = Some(value.to_string()),
    }
}

fn max_string_assign(slot: &mut Option<String>, value: &str) {
    match slot {
        Some(existing) if existing.as_str() >= value => {}
        _ => *slot = Some(value.to_string()),
    }
}

fn apply_factor_zscores(rows: &mut [GoldFactorBucketRow], groups: &HashMap<u32, Vec<usize>>) {
    for indexes in groups.values() {
        let values = indexes
            .iter()
            .filter_map(|idx| rows[*idx].tone_mean)
            .collect::<Vec<_>>();
        let Some((mean, stddev)) = mean_stddev(&values) else {
            continue;
        };
        if stddev == 0.0 {
            continue;
        }
        for idx in indexes {
            if let Some(value) = rows[*idx].tone_mean {
                rows[*idx].tone_zscore_30d = Some((value - mean) / stddev);
            }
        }
    }
}

fn apply_asset_factor_zscores(
    rows: &mut [GoldAssetFactorPanelRow],
    groups: &HashMap<(u64, u32), Vec<usize>>,
) {
    for indexes in groups.values() {
        let values = indexes
            .iter()
            .filter_map(|idx| rows[*idx].tone_mean)
            .collect::<Vec<_>>();
        let Some((mean, stddev)) = mean_stddev(&values) else {
            continue;
        };
        if stddev == 0.0 {
            continue;
        }
        for idx in indexes {
            if let Some(value) = rows[*idx].tone_mean {
                rows[*idx].tone_zscore_30d = Some((value - mean) / stddev);
            }
        }
    }
}

fn mean_stddev(values: &[f64]) -> Option<(f64, f64)> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / (values.len() as f64 - 1.0);
    Some((mean, variance.sqrt()))
}

fn bronze_candidates_df(rows: &[BronzeCandidate]) -> Result<DataFrame> {
    DataFrame::new(vec![
        Series::new(
            "doc_id".into(),
            rows.iter().map(|r| r.doc_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "record_datetime".into(),
            rows.iter()
                .map(|r| r.record_datetime.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "event_time".into(),
            rows.iter()
                .map(|r| r.event_time.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "partition_date".into(),
            rows.iter()
                .map(|r| r.partition_date.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_domain".into(),
            rows.iter()
                .map(|r| r.source_domain.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "document_identifier".into(),
            rows.iter()
                .map(|r| r.document_identifier.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "title".into(),
            rows.iter().map(|r| r.title.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "summary_text".into(),
            rows.iter()
                .map(|r| r.summary_text.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "body_text".into(),
            rows.iter().map(|r| r.body_text.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "relevant_text".into(),
            rows.iter()
                .map(|r| r.relevant_text.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "v2_themes".into(),
            rows.iter().map(|r| r.v2_themes.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "v2_tone".into(),
            rows.iter().map(|r| r.v2_tone.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "v2_locations".into(),
            rows.iter()
                .map(|r| r.v2_locations.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "v2_persons".into(),
            rows.iter()
                .map(|r| r.v2_persons.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "v2_organizations".into(),
            rows.iter()
                .map(|r| r.v2_organizations.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "all_names".into(),
            rows.iter().map(|r| r.all_names.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_type".into(),
            rows.iter()
                .map(|r| r.source_type.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_priority".into(),
            rows.iter().map(|r| r.source_priority).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "market_context_text".into(),
            rows.iter()
                .map(|r| r.market_context_text.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "market_context_score".into(),
            rows.iter()
                .map(|r| r.market_context_score)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "tone".into(),
            rows.iter().map(|r| r.tone).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "geo_labels".into(),
            rows.iter()
                .map(|r| serde_json::to_string(&r.geo_labels))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        )
        .into(),
        Series::new(
            "match_text".into(),
            rows.iter()
                .map(|r| r.match_text.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "asset_match_text".into(),
            rows.iter()
                .map(|r| r.asset_match_text.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
    ])
    .map_err(Into::into)
}

fn graph_doc_nodes_df(rows: &[BronzeCandidate]) -> Result<DataFrame> {
    DataFrame::new(vec![
        Series::new(
            "doc_id".into(),
            rows.iter().map(|r| r.doc_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "partition_date".into(),
            rows.iter()
                .map(|r| r.partition_date.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "event_time".into(),
            rows.iter()
                .map(|r| r.event_time.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_domain".into(),
            rows.iter()
                .map(|r| r.source_domain.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "document_identifier".into(),
            rows.iter()
                .map(|r| r.document_identifier.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "title".into(),
            rows.iter().map(|r| r.title.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_type".into(),
            rows.iter()
                .map(|r| r.source_type.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_priority".into(),
            rows.iter().map(|r| r.source_priority).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "market_context_text".into(),
            rows.iter()
                .map(|r| r.market_context_text.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "market_context_score".into(),
            rows.iter()
                .map(|r| r.market_context_score)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "tone".into(),
            rows.iter().map(|r| r.tone).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "geo_labels_json".into(),
            rows.iter()
                .map(|r| serde_json::to_string(&r.geo_labels))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        )
        .into(),
    ])
    .map_err(Into::into)
}

fn doc_payload_df(rows: &[BronzeCandidate]) -> Result<DataFrame> {
    DataFrame::new(vec![
        Series::new(
            "doc_id".into(),
            rows.iter().map(|r| r.doc_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "partition_date".into(),
            rows.iter()
                .map(|r| r.partition_date.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "document_identifier".into(),
            rows.iter()
                .map(|r| r.document_identifier.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "title".into(),
            rows.iter().map(|r| r.title.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "summary_text".into(),
            rows.iter()
                .map(|r| r.summary_text.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "body_text".into(),
            rows.iter().map(|r| r.body_text.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "relevant_text".into(),
            rows.iter()
                .map(|r| r.relevant_text.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "metadata_json".into(),
            rows.iter()
                .map(|r| r.metadata_json.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "gkg_extras".into(),
            rows.iter()
                .map(|r| r.gkg_extras.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "sharing_image".into(),
            rows.iter()
                .map(|r| r.sharing_image.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "related_images".into(),
            rows.iter()
                .map(|r| r.related_images.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "social_image_embeds".into(),
            rows.iter()
                .map(|r| r.social_image_embeds.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "social_video_embeds".into(),
            rows.iter()
                .map(|r| r.social_video_embeds.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "quotations".into(),
            rows.iter()
                .map(|r| r.quotations.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "amounts".into(),
            rows.iter().map(|r| r.amounts.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "dates".into(),
            rows.iter().map(|r| r.dates.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "gcam".into(),
            rows.iter().map(|r| r.gcam.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "translation_info".into(),
            rows.iter()
                .map(|r| r.translation_info.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
    ])
    .map_err(Into::into)
}

fn doc_review_df(rows: &[BronzeCandidate]) -> Result<DataFrame> {
    DataFrame::new(vec![
        Series::new(
            "doc_id".into(),
            rows.iter().map(|r| r.doc_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "partition_date".into(),
            rows.iter()
                .map(|r| r.partition_date.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "document_identifier".into(),
            rows.iter()
                .map(|r| r.document_identifier.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "title".into(),
            rows.iter().map(|r| r.title.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "summary_text".into(),
            rows.iter()
                .map(|r| r.summary_text.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "relevant_text".into(),
            rows.iter()
                .map(|r| r.relevant_text.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "metadata_json".into(),
            rows.iter()
                .map(|r| r.metadata_json.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "gkg_extras".into(),
            rows.iter()
                .map(|r| r.gkg_extras.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "quotations".into(),
            rows.iter()
                .map(|r| r.quotations.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
    ])
    .map_err(Into::into)
}

fn doc_detail_df(rows: &[BronzeCandidate]) -> Result<DataFrame> {
    DataFrame::new(vec![
        Series::new(
            "doc_id".into(),
            rows.iter().map(|r| r.doc_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "partition_date".into(),
            rows.iter()
                .map(|r| r.partition_date.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "body_text".into(),
            rows.iter().map(|r| r.body_text.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "sharing_image".into(),
            rows.iter()
                .map(|r| r.sharing_image.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "related_images".into(),
            rows.iter()
                .map(|r| r.related_images.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "social_image_embeds".into(),
            rows.iter()
                .map(|r| r.social_image_embeds.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "social_video_embeds".into(),
            rows.iter()
                .map(|r| r.social_video_embeds.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "amounts".into(),
            rows.iter().map(|r| r.amounts.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "dates".into(),
            rows.iter().map(|r| r.dates.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "gcam".into(),
            rows.iter().map(|r| r.gcam.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "translation_info".into(),
            rows.iter()
                .map(|r| r.translation_info.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
    ])
    .map_err(Into::into)
}

fn silver_event_graph_df(rows: &[SilverEventRow]) -> Result<DataFrame> {
    DataFrame::new(vec![
        Series::new(
            "event_time".into(),
            rows.iter()
                .map(|r| r.event_time.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "event_date".into(),
            rows.iter()
                .map(|r| r.event_date.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "cluster_id".into(),
            rows.iter().map(|r| r.cluster_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "doc_id".into(),
            rows.iter().map(|r| r.doc_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "factor_ids".into(),
            rows.iter()
                .map(|r| serde_json::to_string(&r.factor_ids))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        )
        .into(),
        Series::new(
            "factor_labels".into(),
            rows.iter()
                .map(|r| serde_json::to_string(&r.factor_labels))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        )
        .into(),
        Series::new(
            "asset_ids".into(),
            rows.iter()
                .map(|r| serde_json::to_string(&r.asset_ids))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        )
        .into(),
        Series::new(
            "asset_labels".into(),
            rows.iter()
                .map(|r| serde_json::to_string(&r.asset_labels))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        )
        .into(),
        Series::new(
            "geo_ids".into(),
            rows.iter()
                .map(|r| serde_json::to_string(&r.geo_ids))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        )
        .into(),
        Series::new(
            "geo_labels".into(),
            rows.iter()
                .map(|r| serde_json::to_string(&r.geo_labels))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        )
        .into(),
        Series::new(
            "source_id".into(),
            rows.iter().map(|r| r.source_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_domain".into(),
            rows.iter()
                .map(|r| r.source_domain.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "tone".into(),
            rows.iter().map(|r| r.tone).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "novelty".into(),
            rows.iter().map(|r| r.novelty).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_weight".into(),
            rows.iter().map(|r| r.source_weight).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "classification_confidence".into(),
            rows.iter()
                .map(|r| r.classification_confidence)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "model_version".into(),
            rows.iter()
                .map(|r| r.model_version.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "prompt_version".into(),
            rows.iter()
                .map(|r| r.prompt_version.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "created_at".into(),
            rows.iter()
                .map(|r| r.created_at.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
    ])
    .map_err(Into::into)
}

fn silver_factor_mentions_df(rows: &[SilverFactorMentionRow]) -> Result<DataFrame> {
    DataFrame::new(vec![
        Series::new(
            "bucket_time".into(),
            rows.iter()
                .map(|r| r.bucket_time.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "event_time".into(),
            rows.iter()
                .map(|r| r.event_time.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "doc_id".into(),
            rows.iter().map(|r| r.doc_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "cluster_id".into(),
            rows.iter().map(|r| r.cluster_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "factor_id".into(),
            rows.iter().map(|r| r.factor_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "factor_label".into(),
            rows.iter()
                .map(|r| r.factor_label.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "geo_id".into(),
            rows.iter().map(|r| r.geo_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "geo_label".into(),
            rows.iter().map(|r| r.geo_label.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_id".into(),
            rows.iter().map(|r| r.source_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_domain".into(),
            rows.iter()
                .map(|r| r.source_domain.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "tone".into(),
            rows.iter().map(|r| r.tone).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "novelty".into(),
            rows.iter().map(|r| r.novelty).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_weight".into(),
            rows.iter().map(|r| r.source_weight).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "classification_confidence".into(),
            rows.iter()
                .map(|r| r.classification_confidence)
                .collect::<Vec<_>>(),
        )
        .into(),
    ])
    .map_err(Into::into)
}

fn silver_asset_factor_mentions_df(rows: &[SilverAssetFactorMentionRow]) -> Result<DataFrame> {
    DataFrame::new(vec![
        Series::new(
            "bucket_time".into(),
            rows.iter()
                .map(|r| r.bucket_time.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "event_time".into(),
            rows.iter()
                .map(|r| r.event_time.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "doc_id".into(),
            rows.iter().map(|r| r.doc_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "cluster_id".into(),
            rows.iter().map(|r| r.cluster_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "factor_id".into(),
            rows.iter().map(|r| r.factor_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "factor_label".into(),
            rows.iter()
                .map(|r| r.factor_label.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "asset_id".into(),
            rows.iter().map(|r| r.asset_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "asset_label".into(),
            rows.iter()
                .map(|r| r.asset_label.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "geo_id".into(),
            rows.iter().map(|r| r.geo_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "geo_label".into(),
            rows.iter().map(|r| r.geo_label.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_id".into(),
            rows.iter().map(|r| r.source_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_domain".into(),
            rows.iter()
                .map(|r| r.source_domain.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "tone".into(),
            rows.iter().map(|r| r.tone).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "novelty".into(),
            rows.iter().map(|r| r.novelty).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_weight".into(),
            rows.iter().map(|r| r.source_weight).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "classification_confidence".into(),
            rows.iter()
                .map(|r| r.classification_confidence)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "asset_factor_relevance".into(),
            rows.iter()
                .map(|r| r.asset_factor_relevance)
                .collect::<Vec<_>>(),
        )
        .into(),
    ])
    .map_err(Into::into)
}

fn silver_market_context_mentions_df(rows: &[SilverMarketContextRow]) -> Result<DataFrame> {
    DataFrame::new(vec![
        Series::new(
            "bucket_time".into(),
            rows.iter()
                .map(|r| r.bucket_time.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "event_time".into(),
            rows.iter()
                .map(|r| r.event_time.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "doc_id".into(),
            rows.iter().map(|r| r.doc_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "cluster_id".into(),
            rows.iter().map(|r| r.cluster_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "factor_label".into(),
            rows.iter()
                .map(|r| r.factor_label.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "asset_label".into(),
            rows.iter()
                .map(|r| r.asset_label.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_domain".into(),
            rows.iter()
                .map(|r| r.source_domain.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_type".into(),
            rows.iter()
                .map(|r| r.source_type.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_priority".into(),
            rows.iter().map(|r| r.source_priority).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "market_context_text".into(),
            rows.iter()
                .map(|r| r.market_context_text.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "market_context_score".into(),
            rows.iter()
                .map(|r| r.market_context_score)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "classification_confidence".into(),
            rows.iter()
                .map(|r| r.classification_confidence)
                .collect::<Vec<_>>(),
        )
        .into(),
    ])
    .map_err(Into::into)
}

fn gold_factor_buckets_df(rows: &[GoldFactorBucketRow]) -> Result<DataFrame> {
    DataFrame::new(vec![
        Series::new(
            "bucket_time".into(),
            rows.iter()
                .map(|r| r.bucket_time.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "factor_id".into(),
            rows.iter().map(|r| r.factor_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "factor_label".into(),
            rows.iter()
                .map(|r| r.factor_label.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "geo_id".into(),
            rows.iter().map(|r| r.geo_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "geo_label".into(),
            rows.iter().map(|r| r.geo_label.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "doc_count".into(),
            rows.iter().map(|r| r.doc_count).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "mention_count".into(),
            rows.iter().map(|r| r.mention_count).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "unique_sources".into(),
            rows.iter().map(|r| r.unique_sources).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "geo_count".into(),
            rows.iter().map(|r| r.geo_count).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "tone_mean".into(),
            rows.iter().map(|r| r.tone_mean).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "tone_zscore_30d".into(),
            rows.iter().map(|r| r.tone_zscore_30d).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "avg_abs_tone".into(),
            rows.iter().map(|r| r.avg_abs_tone).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "novelty_mean".into(),
            rows.iter().map(|r| r.novelty_mean).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "negative_tail_count".into(),
            rows.iter()
                .map(|r| r.negative_tail_count)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "positive_tail_count".into(),
            rows.iter()
                .map(|r| r.positive_tail_count)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_dispersion".into(),
            rows.iter().map(|r| r.source_dispersion).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "weighted_source_mass".into(),
            rows.iter()
                .map(|r| r.weighted_source_mass)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "weighted_source_dispersion".into(),
            rows.iter()
                .map(|r| r.weighted_source_dispersion)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "confidence_mean".into(),
            rows.iter().map(|r| r.confidence_mean).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "first_seen".into(),
            rows.iter()
                .map(|r| r.first_seen.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "last_seen".into(),
            rows.iter().map(|r| r.last_seen.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "narrative_score".into(),
            rows.iter().map(|r| r.narrative_score).collect::<Vec<_>>(),
        )
        .into(),
    ])
    .map_err(Into::into)
}

fn gold_asset_factor_panel_df(rows: &[GoldAssetFactorPanelRow]) -> Result<DataFrame> {
    DataFrame::new(vec![
        Series::new(
            "bucket_time".into(),
            rows.iter()
                .map(|r| r.bucket_time.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "asset_id".into(),
            rows.iter().map(|r| r.asset_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "asset_label".into(),
            rows.iter()
                .map(|r| r.asset_label.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "factor_id".into(),
            rows.iter().map(|r| r.factor_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "factor_label".into(),
            rows.iter()
                .map(|r| r.factor_label.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "geo_id".into(),
            rows.iter().map(|r| r.geo_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "geo_label".into(),
            rows.iter().map(|r| r.geo_label.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "doc_count".into(),
            rows.iter().map(|r| r.doc_count).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "mention_count".into(),
            rows.iter().map(|r| r.mention_count).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "unique_sources".into(),
            rows.iter().map(|r| r.unique_sources).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "geo_count".into(),
            rows.iter().map(|r| r.geo_count).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "tone_mean".into(),
            rows.iter().map(|r| r.tone_mean).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "tone_zscore_30d".into(),
            rows.iter().map(|r| r.tone_zscore_30d).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "avg_abs_tone".into(),
            rows.iter().map(|r| r.avg_abs_tone).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "novelty_mean".into(),
            rows.iter().map(|r| r.novelty_mean).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "event_intensity".into(),
            rows.iter().map(|r| r.event_intensity).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_dispersion".into(),
            rows.iter().map(|r| r.source_dispersion).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "weighted_source_mass".into(),
            rows.iter()
                .map(|r| r.weighted_source_mass)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "weighted_source_dispersion".into(),
            rows.iter()
                .map(|r| r.weighted_source_dispersion)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "confidence".into(),
            rows.iter().map(|r| r.confidence).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "narrative_score".into(),
            rows.iter().map(|r| r.narrative_score).collect::<Vec<_>>(),
        )
        .into(),
    ])
    .map_err(Into::into)
}

fn gold_factor_crossovers_df(rows: &[GoldFactorCrossoverRow]) -> Result<DataFrame> {
    DataFrame::new(vec![
        Series::new(
            "prior_bucket_time".into(),
            rows.iter()
                .map(|r| r.prior_bucket_time.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "bucket_time".into(),
            rows.iter()
                .map(|r| r.bucket_time.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "factor_id".into(),
            rows.iter().map(|r| r.factor_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "factor_label".into(),
            rows.iter()
                .map(|r| r.factor_label.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "geo_id".into(),
            rows.iter().map(|r| r.geo_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "geo_label".into(),
            rows.iter().map(|r| r.geo_label.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "prior_doc_count".into(),
            rows.iter().map(|r| r.prior_doc_count).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "doc_count".into(),
            rows.iter().map(|r| r.doc_count).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "prior_narrative_score".into(),
            rows.iter()
                .map(|r| r.prior_narrative_score)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "narrative_score".into(),
            rows.iter().map(|r| r.narrative_score).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "doc_count_delta".into(),
            rows.iter().map(|r| r.doc_count_delta).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "narrative_score_delta".into(),
            rows.iter()
                .map(|r| r.narrative_score_delta)
                .collect::<Vec<_>>(),
        )
        .into(),
    ])
    .map_err(Into::into)
}

fn gold_asset_crossovers_df(rows: &[GoldAssetFactorCrossoverRow]) -> Result<DataFrame> {
    DataFrame::new(vec![
        Series::new(
            "prior_bucket_time".into(),
            rows.iter()
                .map(|r| r.prior_bucket_time.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "bucket_time".into(),
            rows.iter()
                .map(|r| r.bucket_time.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "asset_id".into(),
            rows.iter().map(|r| r.asset_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "asset_label".into(),
            rows.iter()
                .map(|r| r.asset_label.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "factor_id".into(),
            rows.iter().map(|r| r.factor_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "factor_label".into(),
            rows.iter()
                .map(|r| r.factor_label.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "geo_id".into(),
            rows.iter().map(|r| r.geo_id).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "geo_label".into(),
            rows.iter().map(|r| r.geo_label.clone()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "prior_doc_count".into(),
            rows.iter().map(|r| r.prior_doc_count).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "doc_count".into(),
            rows.iter().map(|r| r.doc_count).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "prior_narrative_score".into(),
            rows.iter()
                .map(|r| r.prior_narrative_score)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "narrative_score".into(),
            rows.iter().map(|r| r.narrative_score).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "doc_count_delta".into(),
            rows.iter().map(|r| r.doc_count_delta).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "narrative_score_delta".into(),
            rows.iter()
                .map(|r| r.narrative_score_delta)
                .collect::<Vec<_>>(),
        )
        .into(),
    ])
    .map_err(Into::into)
}

fn load_gold_factor_bucket_rows(path: &Path) -> Result<Vec<GoldFactorBucketRow>> {
    let file = File::open(path)?;
    let df = ParquetReader::new(file).finish()?;
    let mut out = Vec::with_capacity(df.height());
    for idx in 0..df.height() {
        out.push(GoldFactorBucketRow {
            bucket_time: get_string(&df, "bucket_time", idx).unwrap_or_default(),
            factor_id: get_string(&df, "factor_id", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            factor_label: get_string(&df, "factor_label", idx).unwrap_or_default(),
            geo_id: get_string(&df, "geo_id", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            geo_label: get_string(&df, "geo_label", idx).unwrap_or_default(),
            doc_count: get_string(&df, "doc_count", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            mention_count: get_string(&df, "mention_count", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            unique_sources: get_string(&df, "unique_sources", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            geo_count: get_string(&df, "geo_count", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            tone_mean: get_string(&df, "tone_mean", idx).and_then(|v| v.parse().ok()),
            tone_zscore_30d: get_string(&df, "tone_zscore_30d", idx).and_then(|v| v.parse().ok()),
            avg_abs_tone: get_string(&df, "avg_abs_tone", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            novelty_mean: get_string(&df, "novelty_mean", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            negative_tail_count: get_string(&df, "negative_tail_count", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            positive_tail_count: get_string(&df, "positive_tail_count", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            source_dispersion: get_string(&df, "source_dispersion", idx)
                .and_then(|v| v.parse().ok()),
            weighted_source_mass: get_string(&df, "weighted_source_mass", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            weighted_source_dispersion: get_string(&df, "weighted_source_dispersion", idx)
                .and_then(|v| v.parse().ok()),
            confidence_mean: get_string(&df, "confidence_mean", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            first_seen: get_string(&df, "first_seen", idx).unwrap_or_default(),
            last_seen: get_string(&df, "last_seen", idx).unwrap_or_default(),
            narrative_score: get_string(&df, "narrative_score", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
        });
    }
    Ok(out)
}

fn load_gold_asset_factor_panel_rows(path: &Path) -> Result<Vec<GoldAssetFactorPanelRow>> {
    let file = File::open(path)?;
    let df = ParquetReader::new(file).finish()?;
    let mut out = Vec::with_capacity(df.height());
    for idx in 0..df.height() {
        out.push(GoldAssetFactorPanelRow {
            bucket_time: get_string(&df, "bucket_time", idx).unwrap_or_default(),
            asset_id: get_string(&df, "asset_id", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            asset_label: get_string(&df, "asset_label", idx).unwrap_or_default(),
            factor_id: get_string(&df, "factor_id", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            factor_label: get_string(&df, "factor_label", idx).unwrap_or_default(),
            geo_id: get_string(&df, "geo_id", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            geo_label: get_string(&df, "geo_label", idx).unwrap_or_default(),
            doc_count: get_string(&df, "doc_count", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            mention_count: get_string(&df, "mention_count", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            unique_sources: get_string(&df, "unique_sources", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            geo_count: get_string(&df, "geo_count", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            tone_mean: get_string(&df, "tone_mean", idx).and_then(|v| v.parse().ok()),
            tone_zscore_30d: get_string(&df, "tone_zscore_30d", idx).and_then(|v| v.parse().ok()),
            avg_abs_tone: get_string(&df, "avg_abs_tone", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            novelty_mean: get_string(&df, "novelty_mean", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            event_intensity: get_string(&df, "event_intensity", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            source_dispersion: get_string(&df, "source_dispersion", idx)
                .and_then(|v| v.parse().ok()),
            weighted_source_mass: get_string(&df, "weighted_source_mass", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            weighted_source_dispersion: get_string(&df, "weighted_source_dispersion", idx)
                .and_then(|v| v.parse().ok()),
            confidence: get_string(&df, "confidence", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
            narrative_score: get_string(&df, "narrative_score", idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or_default(),
        });
    }
    Ok(out)
}
