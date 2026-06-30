import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const projectRoot = "/Users/jamiepearcey/projects";
const resultsDir = path.join(
  projectRoot,
  "research/news-narrative-explainer/v3/results",
);
const csvPath = path.join(resultsDir, "source_score_inventory.csv");
const summaryPath = path.join(resultsDir, "source_score_inventory_summary.json");
const outputDir = path.join(projectRoot, "outputs", "source_score_inventory");
const outputPath = path.join(outputDir, "source_score_inventory.xlsx");

const csvText = await fs.readFile(csvPath, "utf8");
const summary = JSON.parse(await fs.readFile(summaryPath, "utf8"));

const workbook = await Workbook.fromCSV(csvText, { sheetName: "Sources" });
const sourceSheet = workbook.worksheets.getItem("Sources");
const summarySheet = workbook.worksheets.add("Summary");
const sourceLastRow = summary.rows + 1;

sourceSheet.showGridLines = false;
summarySheet.showGridLines = false;

const usedRange = sourceSheet.getUsedRange();
usedRange.format.font.name = "Aptos";
usedRange.format.font.size = 10;
usedRange.getRow(0).format.font.bold = true;
usedRange.getRow(0).format.fill.color = "#DCE6F1";
usedRange.getRow(0).format.borders = { preset: "all", style: "thin", color: "#9FBAD0" };
usedRange.format.borders = { preset: "outside", style: "thin", color: "#D9E2F3" };
sourceSheet.freezePanes.freezeRows(1);
sourceSheet.getRange("A1:J1").format.horizontalAlignment = "Center";
sourceSheet.getRange(`C2:J${sourceLastRow}`).format.horizontalAlignment = "Right";
sourceSheet.getRange("A:B").format.columnWidth = 24;
sourceSheet.getRange("C:F").format.columnWidth = 14;
sourceSheet.getRange("G:G").format.columnWidth = 28;
sourceSheet.getRange("H:J").format.columnWidth = 14;
sourceSheet.getRange(`E2:F${sourceLastRow}`).format.numberFormat = [["#,##0"]];
sourceSheet.getRange(`H2:J${sourceLastRow}`).format.numberFormat = [["0.00"]];
sourceSheet.getUsedRange().format.autofitColumns();
usedRange.format.wrapText = false;

const summaryRows = [
  ["Source Score Inventory", null, null, null],
  ["Distinct domain/type rows", summary.rows, "Distinct domains", summary.distinct_domains],
  ["", null, null, null],
  ["Source Type", "Count", "Mapping Basis", "Count"],
];

const typeEntries = Object.entries(summary.source_types);
const basisEntries = Object.entries(summary.mapping_basis);
const maxLen = Math.max(typeEntries.length, basisEntries.length);
for (let idx = 0; idx < maxLen; idx += 1) {
  const typeEntry = typeEntries[idx] ?? ["", null];
  const basisEntry = basisEntries[idx] ?? ["", null];
  summaryRows.push([typeEntry[0], typeEntry[1], basisEntry[0], basisEntry[1]]);
}

summaryRows.push(["", null, null, null]);
summaryRows.push(["Top Domains By Row Count", null, null, null]);
summaryRows.push(["source_domain", "source_type", "row_count", "current_actual_score"]);
for (const row of summary.top_20_by_row_count) {
  summaryRows.push([
    row.source_domain,
    row.source_type,
    row.row_count,
    row.current_actual_score,
  ]);
}

const summaryRange = summarySheet.getRangeByIndexes(0, 0, summaryRows.length, 4);
summaryRange.values = summaryRows;
summaryRange.format.font.name = "Aptos";
summaryRange.format.font.size = 10;
summarySheet.getRange("A1:D1").merge();
summarySheet.getRange("A1").format.font.bold = true;
summarySheet.getRange("A1").format.font.size = 15;
summarySheet.getRange("A4:D4").format.font.bold = true;
summarySheet.getRange("A4:D4").format.fill.color = "#DCE6F1";
const topHeaderRow = 6 + maxLen;
summarySheet.getRange(`A${topHeaderRow}:D${topHeaderRow}`).merge();
summarySheet.getRange(`A${topHeaderRow}`).format.font.bold = true;
summarySheet.getRange(`A${topHeaderRow + 1}:D${topHeaderRow + 1}`).format.font.bold = true;
summarySheet.getRange(`A${topHeaderRow + 1}:D${topHeaderRow + 1}`).format.fill.color = "#EAF2F8";
summarySheet.getUsedRange().format.autofitColumns();
summarySheet.getUsedRange().format.wrapText = false;
summarySheet.freezePanes.freezeRows(4);

await fs.mkdir(outputDir, { recursive: true });

const inspect = await workbook.inspect({
  kind: "table",
  sheetId: "Sources",
  range: "A1:J12",
  include: "values",
  tableMaxRows: 12,
  tableMaxCols: 10,
});
console.log(inspect.ndjson);

const summaryRender = await workbook.render({
  sheetName: "Summary",
  range: "A1:D20",
  scale: 2,
  format: "png",
});
const sourceRender = await workbook.render({
  sheetName: "Sources",
  range: "A1:J25",
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
