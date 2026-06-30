# fasttext classifier

Conservative finance/news junk filter built from weak labels over the existing
GDELT candidate parquet corpus.

There are now two parallel objectives:

- the original multi-class taxonomy for routing analysis
- a binary `finance_influential` objective for the broader question
  "could this story plausibly matter for markets, macro, sectors, companies,
  supply chains, policy, or risk sentiment?"

## Labels

- `keep_finance`
- `keep_macro`
- `keep_geopolitics`
- `keep_company_event`
- `drop_sports`
- `drop_entertainment`
- `drop_lifestyle`
- `drop_local_crime`
- `drop_low_quality`
- `drop_press_release`

## Workflow

1. Build weak labels from the local GDELT parquet plus a conservative
   per-source probe profile.
2. Train a supervised `fastText` classifier from those weak labels.
3. Optionally build a vector-teacher training set from the local Qdrant
   embedding collection, then train `fastText` on the high-confidence
   teacher-labeled rows.
3. Score new documents with confidence bands:
   - `> 0.85`: auto keep/drop
   - `0.55-0.85`: trust-aware band, keep if the source is strong or the text
     still looks finance-cluster-like
   - `< 0.55`: send to review / fallback
4. Persist borderline rows and corrected labels for retraining.

## Files

- `build_weak_labels.py`
  - Exports a slim local corpus from parquet.
  - Applies conservative weak-label rules.
  - Writes `data/train.txt`, `data/valid.txt`, `data/weak_labels.csv`, and
    `feedback/review_queue.csv`.
- `build_source_probe_profiles.py`
  - Samples per-source finance/macro/geopolitics/company/industry/junk signal
    from the local corpus.
  - Writes `results/source_probe_profiles.csv` and
    `results/source_probe_profiles_summary.json`.
- `train_fasttext.py`
  - Trains and evaluates `fastText`.
  - Writes `models/news_filter.bin`, `models/news_filter.ftz`,
    `results/training_summary.json`, and `results/training_benchmark.json`.
  - Also accepts alternate `--train-txt`, `--valid-txt`, and output paths so
    the vector-teacher corpus can be trained without replacing the weak-label
    baseline files.
- `build_vector_teacher_labels.py`
  - Pulls vectors and payloads from a local Qdrant collection.
  - Clusters the embedding space, assigns conservative teacher labels at the
  cluster level, and propagates only high-confidence row labels.
  - Writes `results/vector_teacher_clusters.csv`,
    `results/vector_teacher_labels.csv`, `results/vector_teacher_summary.json`,
    `data/train_vector_teacher.txt`, and `data/valid_vector_teacher.txt`.
- `score_fasttext.py`
  - Scores rows and applies the confidence band policy.
- `build_gpt_eval_candidates.py`
  - Builds the first stratified GPT/manual eval candidate set from vector
    search plus domain disagreement slices.
- `build_gpt_eval_hard_cases.py`
  - Builds a second-stage hard-case eval batch directly from the scored corpus,
    excluding already labeled eval rows.
- `build_gpt_eval_next_wave.py`
  - Builds a targeted expansion batch from the remaining weak-label failure
    families seen in `results/gpt_eval_set_detailed.csv`.
  - Focuses the next labeling wave on macro-vs-geopolitics, finance-vs-company,
    press-release boundaries, and finance-vs-macro boundaries rather than
    repeating broad search slices.
- `build_gpt_labeling_packets.py`
  - Splits an unlabeled eval batch into prompt, template, and CSV packet files
    for GPT or manual labeling.
  - Mixes rows across strata with a round-robin packer so most batches contain
    a balanced set of hard-case types.
- `merge_gpt_eval_labels.py`
  - Merges completed labeled rows back into `feedback/gpt_labeled_eval_set.csv`
    with label validation and duplicate-safe updates.
  - Also reads completed `feedback/labeling_packets/batch_*.template.json`
    files directly and syncs accepted labels back into
    `feedback/gpt_eval_hard_cases.csv`.
  - Backfills canonical eval rows from the hard-case and scored corpora so the
    eval set keeps article context such as title, partition date, weak label,
    model prediction, and scoring metadata instead of only id/label pairs.
  - Also falls back to `feedback/review_queue.csv` and
    `data/corpus_projection.csv` so eval rows that were routed to review still
    retain their article context in the canonical eval set.
- `import_gpt_packet_response.py`
  - Applies a raw GPT/manual JSON response into a packet
    `batch_*.template.json` file.
  - Accepts either a plain JSON array or a fenced `json` code block.
- `import_gpt_packet_responses.py`
  - Bulk-imports response files from `feedback/labeling_responses/` into all
    matching packet templates.
  - Looks for `batch_XXX.response.json`, `.md`, or `.txt`.
  - Ignores `batch_XXX.response.stub.json` by default, so empty scaffolding
    files do not count as completed responses.
- `build_labeling_response_manifest.py`
  - Creates `feedback/labeling_responses/manifest.json` and `manifest.md`
    listing the expected response filenames and source packet artifacts for
    every batch.
  - Also creates `manifest.csv` plus per-batch `batch_XXX.response.stub.json`
    files that can be filled directly or used as output targets.
- `labeling_status.py`
  - Reports packet completion, hard-case coverage, merge progress, and
    remaining unlabeled rows.
- `smoke_test_labeling_workflow.py`
  - Runs a tempdir end-to-end test of bulk response import, merge, and status
    using a synthetic response file.
- `smoke_test_next_wave_labeling_workflow.py`
  - Runs a tempdir end-to-end test of the dedicated next-wave packet and
    response workspace using a synthetic response file.
  - Verifies that next-wave responses import correctly, merge into the
    canonical eval set, refresh next-wave status, and update the active brief
    without mutating the live workspace.
- `select_next_labeling_batch.py`
  - Chooses the next batch needing work from live status.
  - Can promote a stub into the active `batch_XXX.response.json` target.
- `build_active_labeling_brief.py`
  - Builds a concise operator brief for the current active batch, including
    paths, stratum mix, and a short preview of rows to label.
- `build_labeling_review_sheet.py`
  - Builds a compact full-batch adjudication sheet for the current active
    batch, surfacing weak-vs-model disagreement, signal counts, and the
    available text fields row by row.
  - Also emits a compact CSV ordered by adjudication priority so the operator
    can sort or filter the batch without working from the full prompt.
  - Also emits an unresolved shortlist in markdown, JSON, and CSV for the
    highest-priority rows still needing labels.
  - Also emits a shortlist response stub in JSON and markdown so the top
    unresolved rows can be filled directly in import-ready shape.
- `apply_active_shortlist_response.py`
  - Applies a filled shortlist response payload back into the current active
    packet template.
  - Can also sync those same filled rows into the active batch response file
    for auditability and later bulk-finalize paths.
- `advance_labeling_queue.py`
  - Refreshes status and the active brief, and activates the next batch once
    the current active batch has no remaining rows.
- `prepare_labeling_workspace.py`
  - Rebuilds the response manifest, refreshes status, and rebuilds the active
    brief plus review sheet in one command.
  - Can activate the next batch, but only when no batch is already active
    unless `--force-activate` is used.
- `prepare_next_wave_labeling_workspace.py`
  - Builds packets for `feedback/gpt_eval_next_wave.csv` into a dedicated
    next-wave packet/response workspace.
  - Uses separate packet, response, status, selection, and active-brief files
    so the targeted expansion set does not collide with the original hard-case
    labeling loop.
- `finalize_labeling_run.py`
  - Runs the safe post-labeling path in one command:
    bulk response import, label merge, queue advance, status/brief refresh,
    and eval recomputation.
  - Useful after a GPT/manual batch has been filled because it preserves the
    current active batch when no substantive labels were added.
- `finalize_next_wave_labeling_run.py`
  - Runs the same safe post-labeling path for the dedicated next-wave packet
  and response workspace.
  - Imports next-wave responses, merges labels into the canonical eval set,
    refreshes the next-wave queue/brief, and recomputes eval metrics without
    touching the original hard-case workspace state.
- `finalize_next_wave_shortlist_run.py`
  - Runs the shortlist-specific operator path for the dedicated next-wave
    workspace.
  - Applies the active shortlist response payload, then refreshes status,
    active brief, full review sheet, and the unresolved shortlist in one step.
- `evaluate_gpt_eval_set.py`
  - Evaluates weak labels and model predictions against the canonical labeled
    eval set.
  - Also writes row-level outputs at `results/gpt_eval_set_detailed.csv` and
    `results/gpt_eval_set_missing.csv` so review can focus on actual misses,
    weak-label errors, and corpus-coverage gaps.
  - Distinguishes rows that are absent from `scored_weak_labels.csv` but still
    present in `review_queue.csv` from rows that are truly absent from the
    local corpus.
  - `results/gpt_eval_set_detailed.csv` is now the unified eval surface for
    all labeled rows, with `coverage_status` showing whether a row was
    `matched_scored` or `present_review_only`.
- The JSON summary now also reports overall and per-stratum scored coverage,
  review-only coverage, and corpus coverage.
- The current rule set includes conservative targeted promotions for IPO
  market-structure stories and energy/trade theme combinations so obviously
  market-relevant review rows do not stay stranded outside the scored path.
- The current rule set also includes:
  - strong catalyst promotion for company-event headlines such as FDA feedback
    even when generic company token counts are thin
  - a narrow low-signal macro override in `score_fasttext.py` to stop
    theme-only energy/trade rows from being over-predicted as `keep_finance`
- `prioritize_review_queue.py`
  - Builds a focused review batch from ambiguous sector stories.
  - Mines hard negatives where drop labels are being predicted as keep labels.
- `evaluate_quality.py`
  - Builds a consolidated classifier/domain-score quality report.
  - Writes `results/quality_evaluation.json` and
    `results/quality_evaluation.md`.
- `feedback/labeled_feedback.csv`
  - Optional human corrections. If present, these override weak labels on the
    next build.

## Run

```bash
cd /Users/jamiepearcey/projects/research/news-narrative-explainer
uv run --with fasttext python3 'fasttext classifier/build_weak_labels.py'
uv run --with fasttext python3 'fasttext classifier/train_fasttext.py'
python3 'fasttext classifier/build_vector_teacher_labels.py' --collection news_narrative_v3_20260605_allminilm
uv run --with fasttext python3 'fasttext classifier/train_fasttext.py' \
  --train-txt '/Users/jamiepearcey/projects/research/news-narrative-explainer/fasttext classifier/data/train_vector_teacher.txt' \
  --valid-txt '/Users/jamiepearcey/projects/research/news-narrative-explainer/fasttext classifier/data/valid_vector_teacher.txt' \
  --model-bin '/Users/jamiepearcey/projects/research/news-narrative-explainer/fasttext classifier/models/news_filter_vector_teacher.bin' \
  --model-ftz '/Users/jamiepearcey/projects/research/news-narrative-explainer/fasttext classifier/models/news_filter_vector_teacher.ftz' \
  --summary-json '/Users/jamiepearcey/projects/research/news-narrative-explainer/fasttext classifier/results/training_summary_vector_teacher.json' \
  --benchmark-json '/Users/jamiepearcey/projects/research/news-narrative-explainer/fasttext classifier/results/training_benchmark_vector_teacher.json' \
  --thresholds-json '/Users/jamiepearcey/projects/research/news-narrative-explainer/fasttext classifier/results/thresholds_vector_teacher.json'
python3 'fasttext classifier/prioritize_review_queue.py'
python3 'fasttext classifier/evaluate_quality.py'
```

## Binary Finance-Influential Path

```bash
cd /Users/jamiepearcey/projects/research/news-narrative-explainer
python3 'fasttext classifier/build_finance_influential_dataset.py'
uv run --with fasttext python3 'fasttext classifier/train_finance_influential_fasttext.py'
uv run --with fasttext python3 'fasttext classifier/score_finance_influential_fasttext.py'
python3 'fasttext classifier/evaluate_finance_influential_holdout.py'
```

- `build_finance_influential_dataset.py`
  - Collapses the existing weak labels into:
    - `finance_influential`
    - `not_finance_influential`
  - Writes `data/finance_influential_labels.csv`,
    `data/finance_influential_train.txt`,
    `data/finance_influential_valid.txt`, and
    `results/finance_influential_dataset_summary.json`.
- `train_finance_influential_fasttext.py`
  - Trains a binary fastText model and saves:
    `models/finance_influential.bin`,
    `models/finance_influential.ftz`,
    `results/finance_influential_training_summary.json`, and
    `results/finance_influential_training_benchmark.json`.
- `score_finance_influential_fasttext.py`
  - Scores the full collapsed corpus and writes
    `results/finance_influential_scored.csv`.
- `evaluate_finance_influential_holdout.py`
  - Evaluates the binary objective against the manually judged rows already
    present in `feedback/gpt_eval_holdout_1000.csv`.
  - Writes
    `results/finance_influential_holdout_metrics.json` and
    `results/finance_influential_holdout_report.md`.

## Eval labeling workflow

```bash
python3 'fasttext classifier/build_gpt_eval_hard_cases.py'
python3 'fasttext classifier/build_gpt_labeling_packets.py'
python3 'fasttext classifier/prepare_labeling_workspace.py' --activate-next
python3 'fasttext classifier/import_gpt_packet_response.py' \
  --template-json '/Users/jamiepearcey/projects/research/news-narrative-explainer/fasttext classifier/feedback/labeling_packets/batch_001.template.json' \
  --response-file '/path/to/gpt_response.json'
python3 'fasttext classifier/import_gpt_packet_responses.py'
python3 'fasttext classifier/merge_gpt_eval_labels.py'
python3 'fasttext classifier/advance_labeling_queue.py'
python3 'fasttext classifier/evaluate_gpt_eval_set.py'
python3 'fasttext classifier/finalize_labeling_run.py'
```

- Packet artifacts are written to
  `feedback/labeling_packets/batch_*.prompt.md`,
  `batch_*.template.json`, and `batch_*.rows.csv`.
- Preferred editing surface is the packet `batch_*.template.json` files rather
  than the full hard-case CSV; merge will propagate completed packet labels
  back into both the hard-case CSV and the canonical eval set.
- `import_gpt_packet_response.py` can populate packet templates directly from a
  model response, avoiding manual row-by-row copy/paste.
- `import_gpt_packet_responses.py` provides the bulk path when responses are
  saved under `feedback/labeling_responses/` using the `batch_XXX.response.*`
  naming convention.
- `build_labeling_response_manifest.py` creates the responses directory and a
  manifest of expected response files, so batch handoff does not rely on
  remembering filenames.
- The generated `batch_XXX.response.stub.json` files are safe starting points
  for model output and already contain the required `document_identifier`
  entries.
- Status distinguishes between real response files and stub scaffolding:
  `response_present` means a real importable response exists, while
  `stub_response_present` only means the scaffold file exists.
- `smoke_test_labeling_workflow.py` can be used to verify the whole labeling
  pipeline on temporary copies before applying any real labels to the live
  eval set.
- `select_next_labeling_batch.py --activate-response` creates the real
  `batch_XXX.response.json` from the stub for the next recommended batch, so
  there is a single active response target to fill.
- `build_active_labeling_brief.py` writes
  `results/active_labeling_brief.md` and `.json` for the currently active
  batch, so the operator does not need to cross-reference multiple files.
- `build_labeling_review_sheet.py` writes
  `results/active_labeling_review_sheet.md` and `.json` with every row in the
  active batch, which is the faster adjudication surface when prompt guidance
  is already understood and the main need is seeing boundary signals quickly.
  It also writes `results/active_labeling_review_sheet.csv`, ordered so
  disagreements and `review`-band rows are handled first.
  The same script also writes
  `results/active_labeling_shortlist.{md,json,csv}` for the top unresolved
  rows so the operator can focus on the next few decisions without scanning
  the whole batch.
  It also writes
  `results/active_labeling_shortlist.response.{json,md}` as a fill-ready
  response payload for those shortlisted rows.
- `apply_active_shortlist_response.py` is the one-command bridge back from
  that shortlist payload into the active packet template. This avoids copying
  shortlisted labels manually into `batch_XXX.template.json`.
- `finalize_next_wave_shortlist_run.py` is the next-wave wrapper around that
  bridge. It is the shortest safe path after filling the shortlist response
  because it applies the shortlist and immediately refreshes all live queue
  artifacts.
- `advance_labeling_queue.py` is the safe post-merge step: if the active batch
  is complete it promotes the next batch, otherwise it leaves the current
  batch in place and simply refreshes status artifacts.
- `finalize_labeling_run.py` is the one-command post-labeling step. It chains
  import, merge, queue advancement, status refresh, active-brief refresh, and
  eval recomputation into a single safe operation.
- `prepare_labeling_workspace.py --activate-next` is the safe pre-labeling
  step: it refreshes all workspace artifacts and only activates a new batch if
  there is no current active response target.
- `labeling_status.py` writes `results/labeling_status.json` for a live status
  snapshot of packets, response-file presence, and merged eval coverage.
- The canonical labeled eval set remains
  `feedback/gpt_labeled_eval_set.csv`.
- That canonical eval CSV is now intentionally enriched rather than minimal:
  it preserves label truth plus joined article/model context so error analysis
  does not require jumping back through multiple intermediate files.
- Rows that fell into the review path are not treated as corpus misses. The
  eval metrics now report them separately as `review_only_rows`.
- The hard-case batch excludes rows already present in the canonical eval set,
  so the workflow can be rerun iteratively without relabeling the same rows.

## Notes

- The weak-label layer is intentionally conservative. It is designed mainly to
  drop obvious junk while preserving rare market-relevant stories, specialist
  trade coverage, and industry-adjacent evidence for review.
- Ambiguous industry stories should tend to `review`, not `drop`.
- The raw weak-label evidence and source probe profile are stored alongside the
  final label so score thresholds can be rebalanced later without regenerating
  the corpus logic.
- The vector-teacher path is intentionally conservative. It is meant to supply
  better training rows, not to act as unquestioned ground truth.
