import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const projectRoot = "/Users/jamiepearcey/projects";
const resultsDir = path.join(
  projectRoot,
  "research/news-narrative-explainer/v3/results",
);
const csvPath = path.join(resultsDir, "source_quality_external.csv");
const summaryPath = path.join(resultsDir, "source_quality_external_summary.json");
const outputDir = path.join(projectRoot, "outputs", "source_quality");
const outputPath = path.join(outputDir, "source_quality.xlsx");

const csvText = await fs.readFile(csvPath, "utf8");
const summary = JSON.parse(await fs.readFile(summaryPath, "utf8"));

const workbook = await Workbook.fromCSV(csvText, { sheetName: "SourceQuality" });
const sourceSheet = workbook.worksheets.getItem("SourceQuality");
const summarySheet = workbook.worksheets.add("Summary");
const sourceLastRow = summary.rows + 1;
const sourceHeaders = csvText.split(/\r?\n/, 1)[0].split(",");

sourceSheet.showGridLines = false;
summarySheet.showGridLines = false;

const usedRange = sourceSheet.getUsedRange();
usedRange.format.font.name = "Aptos";
usedRange.format.font.size = 10;
usedRange.format.wrapText = false;
usedRange.getRow(0).format.font.bold = true;
usedRange.getRow(0).format.fill.color = "#DCE6F1";
usedRange.getRow(0).format.borders = { preset: "all", style: "thin", color: "#9FBAD0" };
sourceSheet.freezePanes.freezeRows(1);
sourceSheet.getUsedRange().format.autofitColumns();

const integerColumns = new Set([
  "min_source_priority",
  "max_source_priority",
  "row_count",
  "day_count",
  "external_match_count",
  "signal_useful_count",
  "signal_cited_by_hq_count",
  "signal_contradicted_count",
  "signal_user_selected_count",
  "signal_user_dismissed_count",
  "signal_original_reporting_count",
  "signal_duplicate_count",
  "signal_hallucination_count",
  "signal_total_count",
  "signal_positive_count",
  "signal_negative_count",
]);
const decimalColumns = new Set([
  "heuristic_score",
  "static_prior_score",
  "mbfc_quality_score",
  "adfontes_reliability",
  "adfontes_bias",
  "adfontes_quality_score",
  "external_reference_score",
  "reference_dynamic_adjustment",
  "final_source_quality_score",
]);

for (let index = 0; index < sourceHeaders.length; index += 1) {
  const header = sourceHeaders[index];
  const range = sourceSheet.getRangeByIndexes(1, index, sourceLastRow - 1, 1);
  range.format.horizontalAlignment = integerColumns.has(header) || decimalColumns.has(header) ? "Right" : "Left";
  if (integerColumns.has(header)) {
    range.format.numberFormat = [["#,##0"]];
  }
  if (decimalColumns.has(header)) {
    range.format.numberFormat = [["0.0000"]];
  }
}

const summaryRows = [
  ["Source Quality", null, null, null],
  ["Rows", summary.rows, "Externally matched", summary.externally_matched_rows],
  ["Static prior only", summary.static_prior_only_rows, "MBFC matches", summary.mbfc_matches],
  ["Ad Fontes matches", summary.adfontes_matches, "AllSides URL candidates", summary.allsides_url_candidates],
  ["", null, null, null],
  ["Score Basis", "Count", "Notes", ""],
];

for (const [basis, count] of Object.entries(summary.score_basis)) {
  summaryRows.push([basis, count, "", ""]);
}

summaryRows.push(["", null, null, null]);
summaryRows.push(["Tier Counts", "Count", "Notes", ""]);
for (const [tier, count] of Object.entries(summary.tier_label_counts ?? {})) {
  summaryRows.push([tier, count, "", ""]);
}

summaryRows.push(["", null, null, null]);
summaryRows.push(["Signal Totals", "Count", "Notes", ""]);
for (const [signalName, count] of Object.entries(summary.signal_totals ?? {})) {
  summaryRows.push([signalName, count, "", ""]);
}

summaryRows.push(["Domains with events", summary.domains_with_events ?? 0, "", ""]);
summaryRows.push(["", null, null, null]);
summaryRows.push(["Top 20 Rows", null, null, null]);
summaryRows.push(["source_domain", "tier_label", "score_basis", "final_source_quality_score"]);
for (const row of summary.top_50.slice(0, 20)) {
  summaryRows.push([
    row.source_domain,
    row.tier_label,
    row.score_basis,
    row.final_source_quality_score,
  ]);
}

const summaryRange = summarySheet.getRangeByIndexes(0, 0, summaryRows.length, 4);
summaryRange.values = summaryRows;
summaryRange.format.font.name = "Aptos";
summaryRange.format.font.size = 10;
summaryRange.format.wrapText = false;
summarySheet.getRange("A1:D1").merge();
summarySheet.getRange("A1").format.font.bold = true;
summarySheet.getRange("A1").format.font.size = 15;
summarySheet.getRange("A6:D6").format.font.bold = true;
summarySheet.getRange("A6:D6").format.fill.color = "#DCE6F1";
for (let rowIndex = 0; rowIndex < summaryRows.length; rowIndex += 1) {
  const firstCell = summaryRows[rowIndex][0];
  if (["Tier Counts", "Signal Totals"].includes(firstCell)) {
    const excelRow = rowIndex + 1;
    summarySheet.getRange(`A${excelRow}:D${excelRow}`).format.font.bold = true;
    summarySheet.getRange(`A${excelRow}:D${excelRow}`).format.fill.color = "#EAF2F8";
  }
}
const topHeaderIndex = summaryRows.findIndex((row) => row[0] === "Top 20 Rows");
const topHeaderRow = topHeaderIndex + 1;
summarySheet.getRange(`A${topHeaderRow}:D${topHeaderRow}`).merge();
summarySheet.getRange(`A${topHeaderRow}`).format.font.bold = true;
summarySheet.getRange(`A${topHeaderRow + 1}:D${topHeaderRow + 1}`).format.font.bold = true;
summarySheet.getRange(`A${topHeaderRow + 1}:D${topHeaderRow + 1}`).format.fill.color = "#EAF2F8";
summarySheet.getUsedRange().format.autofitColumns();
summarySheet.freezePanes.freezeRows(6);

await fs.mkdir(outputDir, { recursive: true });

const summaryRender = await workbook.render({
  sheetName: "Summary",
  range: "A1:D22",
  scale: 2,
  format: "png",
});
const sourceRender = await workbook.render({
  sheetName: "SourceQuality",
  range: "A1:AE25",
  scale: 2,
  format: "png",
});
await fs.writeFile(
  path.join(outputDir, "summary_preview.png"),
  new Uint8Array(await summaryRender.arrayBuffer()),
);
await fs.writeFile(
  path.join(outputDir, "sources_preview.png"),
  new Uint8Array(await sourceRender.arrayBuffer()),
);

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(JSON.stringify({ outputPath }, null, 2));
