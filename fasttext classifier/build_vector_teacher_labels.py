#!/usr/bin/env python3
"""Build vector-cluster teacher labels from the local Qdrant collection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path("/Users/jamiepearcey/projects/research/news-narrative-explainer")
WORK_DIR = PROJECT_ROOT / "fasttext classifier"
DATA_DIR = WORK_DIR / "data"
RESULTS_DIR = WORK_DIR / "results"
SCORED_CSV = RESULTS_DIR / "scored_weak_labels.csv"
DOMAIN_SCORES_CSV = RESULTS_DIR / "effective_domain_scores.csv"
CLUSTERS_CSV = RESULTS_DIR / "vector_teacher_clusters.csv"
LABELS_CSV = RESULTS_DIR / "vector_teacher_labels.csv"
SUMMARY_JSON = RESULTS_DIR / "vector_teacher_summary.json"
BENCHMARK_JSON = RESULTS_DIR / "vector_teacher_benchmark.json"
TRAIN_TXT = DATA_DIR / "train_vector_teacher.txt"
VALID_TXT = DATA_DIR / "valid_vector_teacher.txt"

KEEP_LABELS = {
    "keep_finance",
    "keep_macro",
    "keep_geopolitics",
    "keep_company_event",
}

DROP_LABELS = {
    "drop_sports",
    "drop_entertainment",
    "drop_lifestyle",
    "drop_local_crime",
    "drop_low_quality",
    "drop_press_release",
}

REVIEW_LABEL = "review"
PRESS_RELEASE_ARCHETYPES = {"press_release_or_mirror"}
LOW_QUALITY_ARCHETYPES = {"low_quality_or_scraper"}
HIGH_TRUST_ARCHETYPES = {"premium_primary", "specialist_trade", "mainstream_business_or_general"}
ANNOUNCEMENT_MARKERS = (
    " announces ",
    " announced ",
    " voting results ",
    " annual general meeting ",
    " initial public offering ",
    " closing of ",
    " priced its ",
    " press release ",
)
ROW_DROP_MIN_SCORE = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, help="Qdrant collection name")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333", help="Base Qdrant URL")
    parser.add_argument("--cluster-count", type=int, default=64, help="Number of vector clusters")
    parser.add_argument("--fit-sample-size", type=int, default=25000, help="Rows used to fit centroids")
    parser.add_argument("--iterations", type=int, default=8, help="K-means iterations")
    parser.add_argument("--batch-size", type=int, default=2048, help="Vector assignment batch size")
    parser.add_argument("--page-size", type=int, default=1024, help="Qdrant scroll page size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap for debugging")
    return parser.parse_args()


def normalize_text(*parts: str) -> str:
    return " ".join((part or "").strip() for part in parts if part).strip().lower()


def weak_split_bucket(document_identifier: str) -> str:
    digest = hashlib.sha256(document_identifier.encode("utf-8")).hexdigest()
    return "valid" if int(digest[:8], 16) % 5 == 0 else "train"


def to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except ValueError:
        return default


def to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value or default))
    except ValueError:
        return default


def load_domain_scores() -> dict[str, dict[str, str]]:
    if not DOMAIN_SCORES_CSV.exists():
        return {}
    with DOMAIN_SCORES_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["source_domain"].strip().lower(): row for row in reader}


def load_rows(domain_scores: dict[str, dict[str, str]], max_rows: int) -> list[dict[str, object]]:
    if not SCORED_CSV.exists():
        raise SystemExit("Missing scored rows. Run score_fasttext.py first.")

    rows: list[dict[str, object]] = []
    with SCORED_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            source_domain = (row.get("source_domain") or "").strip().lower()
            domain_meta = domain_scores.get(source_domain, {})
            row["row_index"] = index
            row["source_domain"] = source_domain
            row["weak_label"] = row["label"]
            row["predicted_score"] = to_float(row.get("predicted_score"))
            row["finance_cluster_score"] = to_float(row.get("finance_cluster_score"))
            row["market_relevance_rate"] = to_float(row.get("market_relevance_rate"))
            row["industry_signal_rate"] = to_float(row.get("industry_signal_rate"))
            row["junk_rate"] = to_float(row.get("junk_rate"))
            row["finance_hits"] = to_int(row.get("finance_hits"))
            row["macro_hits"] = to_int(row.get("macro_hits"))
            row["geo_hits"] = to_int(row.get("geo_hits"))
            row["company_hits"] = to_int(row.get("company_hits"))
            row["sports_hits"] = to_int(row.get("sports_hits"))
            row["entertainment_hits"] = to_int(row.get("entertainment_hits"))
            row["lifestyle_hits"] = to_int(row.get("lifestyle_hits"))
            row["crime_hits"] = to_int(row.get("crime_hits"))
            row["press_hits"] = to_int(row.get("press_hits"))
            row["keep_theme_hits"] = to_int(row.get("keep_theme_hits"))
            row["macro_theme_hits"] = to_int(row.get("macro_theme_hits"))
            row["geo_theme_hits"] = to_int(row.get("geo_theme_hits"))
            row["drop_theme_hits"] = to_int(row.get("drop_theme_hits"))
            row["effective_archetype"] = domain_meta.get("effective_archetype", "")
            row["effective_score_0_1"] = to_float(domain_meta.get("effective_score_0_1"))
            row["teacher_seed_text"] = normalize_text(row.get("title", ""), row.get("summary", ""), row.get("text", ""))
            rows.append(row)
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def qdrant_post(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_qdrant_vectors(
    qdrant_url: str,
    collection: str,
    needed_ids: set[str],
    page_size: int,
) -> dict[str, dict[str, object]]:
    base_url = qdrant_url.rstrip("/")
    scroll_url = f"{base_url}/collections/{collection}/points/scroll"
    offset: str | None = None
    matched: dict[str, dict[str, object]] = {}
    total_scanned = 0

    while True:
        payload: dict[str, object] = {
            "limit": page_size,
            "with_payload": [
                "document_identifier",
                "source_domain",
                "canonical_url",
                "title",
                "simhash_u64",
            ],
            "with_vector": True,
        }
        if offset is not None:
            payload["offset"] = offset
        response = qdrant_post(scroll_url, payload)
        result = response.get("result", {})
        points = result.get("points", [])
        total_scanned += len(points)
        for point in points:
            payload_data = point.get("payload") or {}
            document_identifier = payload_data.get("document_identifier")
            if document_identifier in needed_ids:
                matched[document_identifier] = {
                    "vector": point.get("vector"),
                    "payload": payload_data,
                }
        offset = result.get("next_page_offset")
        if offset is None or len(matched) >= len(needed_ids):
            break
    if not matched:
        raise SystemExit(
            f"No matching rows were found in collection {collection}. "
            f"Expected {len(needed_ids)} ids, scanned {total_scanned} points."
        )
    return matched


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def fit_centroids(
    vectors: np.ndarray,
    cluster_count: int,
    fit_sample_size: int,
    iterations: int,
    seed: int,
) -> np.ndarray:
    sample_size = min(len(vectors), fit_sample_size)
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(vectors), size=sample_size, replace=False)
    sample = vectors[sample_indices]
    seed_indices = rng.choice(sample_size, size=min(cluster_count, sample_size), replace=False)
    centroids = sample[seed_indices].copy()

    for _ in range(iterations):
        scores = sample @ centroids.T
        assignments = np.argmax(scores, axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.bincount(assignments, minlength=len(centroids))
        for cluster_id in range(len(centroids)):
            if counts[cluster_id] == 0:
                new_centroids[cluster_id] = sample[rng.integers(0, sample_size)]
                continue
            cluster_vectors = sample[assignments == cluster_id]
            centroid = cluster_vectors.mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm == 0:
                new_centroids[cluster_id] = sample[rng.integers(0, sample_size)]
            else:
                new_centroids[cluster_id] = centroid / centroid_norm
        centroids = new_centroids
    return centroids


def assign_clusters(vectors: np.ndarray, centroids: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    assignments = np.empty(len(vectors), dtype=np.int32)
    similarities = np.empty(len(vectors), dtype=np.float32)
    for start in range(0, len(vectors), batch_size):
        end = min(start + batch_size, len(vectors))
        batch_scores = vectors[start:end] @ centroids.T
        assignments[start:end] = np.argmax(batch_scores, axis=1)
        similarities[start:end] = np.max(batch_scores, axis=1)
    return assignments, similarities


def dominant_label(counter: Counter[str]) -> tuple[str, float]:
    if not counter:
        return REVIEW_LABEL, 0.0
    total = sum(counter.values())
    label, count = counter.most_common(1)[0]
    return label, (count / total) if total else 0.0


def build_cluster_metrics(rows: list[dict[str, object]], assignments: np.ndarray, similarities: np.ndarray) -> dict[int, dict[str, object]]:
    clusters: dict[int, dict[str, object]] = {}
    simhash_counts: dict[int, Counter[str]] = defaultdict(Counter)
    canonical_counts: dict[int, Counter[str]] = defaultdict(Counter)

    for row, cluster_id in zip(rows, assignments, strict=True):
        payload = row["vector_payload"]
        simhash = str(payload.get("simhash_u64") or "")
        canonical = str(payload.get("canonical_url") or "")
        if simhash:
            simhash_counts[int(cluster_id)][simhash] += 1
        if canonical:
            canonical_counts[int(cluster_id)][canonical] += 1

    grouped_rows: dict[int, list[dict[str, object]]] = defaultdict(list)
    grouped_sims: dict[int, list[float]] = defaultdict(list)
    for row, cluster_id, similarity in zip(rows, assignments, similarities, strict=True):
        grouped_rows[int(cluster_id)].append(row)
        grouped_sims[int(cluster_id)].append(float(similarity))

    for cluster_id, cluster_rows in grouped_rows.items():
        weak_counts = Counter(str(row["weak_label"]) for row in cluster_rows)
        predicted_counts = Counter(str(row["predicted_label"]) for row in cluster_rows)
        decision_counts = Counter(str(row["decision_band"]) for row in cluster_rows)
        label_agreement = sum(1 for row in cluster_rows if row["weak_label"] == row["predicted_label"])
        press_like = sum(
            1
            for row in cluster_rows
            if row["press_hits"] > 0
            or row["effective_archetype"] in PRESS_RELEASE_ARCHETYPES
            or row["predicted_label"] == "drop_press_release"
            or row["weak_label"] == "drop_press_release"
        )
        low_quality_like = sum(
            1
            for row in cluster_rows
            if row["effective_archetype"] in LOW_QUALITY_ARCHETYPES
            or row["predicted_label"] == "drop_low_quality"
            or row["junk_rate"] >= 0.5
        )
        finance_signal = sum(
            1
            for row in cluster_rows
            if row["finance_hits"] > 0 or row["predicted_label"] == "keep_finance"
        )
        macro_signal = sum(
            1
            for row in cluster_rows
            if row["macro_hits"] > 0 or row["predicted_label"] == "keep_macro"
        )
        geo_signal = sum(
            1
            for row in cluster_rows
            if row["geo_hits"] > 0 or row["predicted_label"] == "keep_geopolitics"
        )
        company_signal = sum(
            1
            for row in cluster_rows
            if row["company_hits"] > 0 or row["predicted_label"] == "keep_company_event"
        )
        sports_signal = sum(1 for row in cluster_rows if row["sports_hits"] > 0 or row["predicted_label"] == "drop_sports")
        entertainment_signal = sum(
            1 for row in cluster_rows if row["entertainment_hits"] > 0 or row["predicted_label"] == "drop_entertainment"
        )
        lifestyle_signal = sum(
            1 for row in cluster_rows if row["lifestyle_hits"] > 0 or row["predicted_label"] == "drop_lifestyle"
        )
        crime_signal = sum(1 for row in cluster_rows if row["crime_hits"] > 0 or row["predicted_label"] == "drop_local_crime")
        announcement_signal = sum(
            1
            for row in cluster_rows
            if any(marker in f" {str(row['teacher_seed_text'])} " for marker in ANNOUNCEMENT_MARKERS)
        )
        trusted_keep_seeds = sum(
            1
            for row in cluster_rows
            if row["weak_label"] == row["predicted_label"]
            and str(row["predicted_label"]).startswith("keep_")
            and float(row["predicted_score"]) >= 0.85
            and (
                float(row["effective_score_0_1"]) >= 0.6
                or float(row["market_relevance_rate"]) >= 0.6
                or float(row["industry_signal_rate"]) >= 0.3
            )
        )
        trusted_drop_seeds = sum(
            1
            for row in cluster_rows
            if row["weak_label"] == row["predicted_label"]
            and str(row["predicted_label"]).startswith("drop_")
            and float(row["predicted_score"]) >= 0.85
        )

        size = len(cluster_rows)
        duplicate_points = sum(count for count in simhash_counts[cluster_id].values() if count > 1)
        canonical_duplicate_points = sum(count for count in canonical_counts[cluster_id].values() if count > 1)
        top_domains = Counter(str(row["source_domain"]) for row in cluster_rows).most_common(5)
        top_titles = [
            str(row.get("title") or "")
            for row in sorted(
                cluster_rows,
                key=lambda item: (item["predicted_score"], item["finance_cluster_score"]),
                reverse=True,
            )[:3]
        ]

        clusters[cluster_id] = {
            "cluster_id": cluster_id,
            "rows": cluster_rows,
            "size": size,
            "weak_counts": weak_counts,
            "predicted_counts": predicted_counts,
            "decision_counts": decision_counts,
            "label_agreement_rate": label_agreement / size if size else 0.0,
            "press_like_rate": press_like / size if size else 0.0,
            "low_quality_rate": low_quality_like / size if size else 0.0,
            "finance_signal_rate": finance_signal / size if size else 0.0,
            "macro_signal_rate": macro_signal / size if size else 0.0,
            "geo_signal_rate": geo_signal / size if size else 0.0,
            "company_signal_rate": company_signal / size if size else 0.0,
            "sports_signal_rate": sports_signal / size if size else 0.0,
            "entertainment_signal_rate": entertainment_signal / size if size else 0.0,
            "lifestyle_signal_rate": lifestyle_signal / size if size else 0.0,
            "crime_signal_rate": crime_signal / size if size else 0.0,
            "announcement_rate": announcement_signal / size if size else 0.0,
            "mean_similarity": float(np.mean(grouped_sims[cluster_id])) if grouped_sims[cluster_id] else 0.0,
            "mean_predicted_score": sum(float(row["predicted_score"]) for row in cluster_rows) / size if size else 0.0,
            "mean_domain_quality": sum(float(row["effective_score_0_1"]) for row in cluster_rows) / size if size else 0.0,
            "high_trust_rate": (
                sum(1 for row in cluster_rows if row["effective_archetype"] in HIGH_TRUST_ARCHETYPES) / size if size else 0.0
            ),
            "duplicate_rate": duplicate_points / size if size else 0.0,
            "canonical_duplicate_rate": canonical_duplicate_points / size if size else 0.0,
            "trusted_keep_seed_rate": trusted_keep_seeds / size if size else 0.0,
            "trusted_drop_seed_rate": trusted_drop_seeds / size if size else 0.0,
            "top_domains": top_domains,
            "top_titles": top_titles,
        }
    return clusters


def choose_keep_label(metrics: dict[str, object]) -> str:
    scores = {
        "keep_finance": float(metrics["finance_signal_rate"]) + 0.2 * float(metrics["mean_domain_quality"]),
        "keep_macro": float(metrics["macro_signal_rate"]) + 0.15 * float(metrics["high_trust_rate"]),
        "keep_geopolitics": float(metrics["geo_signal_rate"]) + 0.05 * float(metrics["high_trust_rate"]),
        "keep_company_event": float(metrics["company_signal_rate"]) + 0.1 * float(metrics["mean_domain_quality"]),
    }
    return max(scores.items(), key=lambda item: item[1])[0]


def choose_teacher_label(metrics: dict[str, object]) -> tuple[str, float, str]:
    size = int(metrics["size"])
    weak_label, weak_share = dominant_label(metrics["weak_counts"])
    predicted_label, predicted_share = dominant_label(metrics["predicted_counts"])
    press_like_rate = float(metrics["press_like_rate"])
    duplicate_rate = max(float(metrics["duplicate_rate"]), float(metrics["canonical_duplicate_rate"]))
    low_quality_rate = float(metrics["low_quality_rate"])
    label_agreement_rate = float(metrics["label_agreement_rate"])
    mean_domain_quality = float(metrics["mean_domain_quality"])
    high_trust_rate = float(metrics["high_trust_rate"])
    trusted_keep_seed_rate = float(metrics["trusted_keep_seed_rate"])
    trusted_drop_seed_rate = float(metrics["trusted_drop_seed_rate"])
    announcement_rate = float(metrics["announcement_rate"])

    drop_signals = {
        "drop_press_release": press_like_rate + 0.8 * duplicate_rate,
        "drop_low_quality": low_quality_rate + max(0.0, 0.35 - mean_domain_quality),
        "drop_sports": float(metrics["sports_signal_rate"]),
        "drop_entertainment": float(metrics["entertainment_signal_rate"]),
        "drop_lifestyle": float(metrics["lifestyle_signal_rate"]),
        "drop_local_crime": float(metrics["crime_signal_rate"]),
    }
    best_drop_label, best_drop_score = max(drop_signals.items(), key=lambda item: item[1])
    keep_signal = max(
        float(metrics["finance_signal_rate"]),
        float(metrics["macro_signal_rate"]),
        float(metrics["geo_signal_rate"]),
        float(metrics["company_signal_rate"]),
    )

    if size < 10 and label_agreement_rate < 0.7:
        return REVIEW_LABEL, 0.0, "small_ambiguous_cluster"

    if (
        press_like_rate >= 0.35
        or announcement_rate >= 0.18
        or (best_drop_label == "drop_press_release" and best_drop_score >= 0.48)
    ):
        confidence = min(0.99, 0.45 + 0.3 * press_like_rate + 0.15 * duplicate_rate + 0.1 * label_agreement_rate)
        return "drop_press_release", confidence, "press_release_cluster"

    if best_drop_score >= 0.62 and (trusted_drop_seed_rate >= 0.08 or best_drop_label == "drop_low_quality"):
        confidence = min(0.99, 0.4 + 0.35 * best_drop_score + 0.15 * label_agreement_rate)
        return best_drop_label, confidence, "drop_signal_cluster"

    if high_trust_rate < 0.08 and trusted_keep_seed_rate < 0.1 and trusted_drop_seed_rate < 0.08:
        return REVIEW_LABEL, 0.0, "low_trust_cluster"

    if keep_signal >= 0.55 and trusted_keep_seed_rate >= 0.12 and (
        mean_domain_quality >= 0.3 or high_trust_rate >= 0.12
    ):
        keep_label = choose_keep_label(metrics)
        if keep_label in {"keep_macro", "keep_geopolitics"} and not (
            high_trust_rate >= 0.18
            or mean_domain_quality >= 0.5
            or (trusted_keep_seed_rate >= 0.22 and label_agreement_rate >= 0.82)
        ):
            return REVIEW_LABEL, 0.0, "macro_cluster_needs_trust"
        confidence = min(
            0.99,
            0.35
            + 0.2 * keep_signal
            + 0.15 * label_agreement_rate
            + 0.15 * mean_domain_quality
            + 0.1 * predicted_share
            + 0.15 * trusted_keep_seed_rate,
        )
        if weak_label == predicted_label == keep_label:
            confidence = min(0.99, confidence + 0.05)
        return keep_label, confidence, "keep_signal_cluster"

    if weak_label == predicted_label and weak_share >= 0.8 and predicted_share >= 0.8 and (
        weak_label.startswith("drop_") or mean_domain_quality >= 0.55 or high_trust_rate >= 0.18
    ):
        confidence = min(0.95, 0.35 + 0.25 * weak_share + 0.25 * predicted_share + 0.15 * label_agreement_rate)
        return weak_label, confidence, "weak_predicted_consensus"

    return REVIEW_LABEL, 0.0, "ambiguous_cluster"


def build_row_teacher_labels(
    rows: list[dict[str, object]],
    assignments: np.ndarray,
    similarities: np.ndarray,
    cluster_metrics: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    cluster_rows = defaultdict(list)
    for row, cluster_id in zip(rows, assignments, strict=True):
        cluster_rows[int(cluster_id)].append(row)

    similarity_cutoffs: dict[int, float] = {}
    for cluster_id, members in cluster_rows.items():
        sims = [float(row["_similarity"]) for row in members]
        similarity_cutoffs[cluster_id] = float(np.quantile(np.array(sims, dtype=np.float32), 0.2)) if len(sims) >= 5 else min(sims)

    simhash_families: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        simhash = str((row.get("vector_payload") or {}).get("simhash_u64") or "")
        if simhash:
            simhash_families[simhash].append(row)

    press_family_seeds: dict[str, bool] = {}
    low_quality_family_seeds: dict[str, bool] = {}
    for simhash, family_rows in simhash_families.items():
        text_blobs = [f" {str(item['teacher_seed_text'])} " for item in family_rows]
        press_family_seeds[simhash] = any(
            (
                item["predicted_label"] == "drop_press_release"
                and item["predicted_score"] >= ROW_DROP_MIN_SCORE
                and (
                    item["press_hits"] > 0
                    or item["effective_archetype"] in PRESS_RELEASE_ARCHETYPES
                    or any(marker in text_blob for marker in ANNOUNCEMENT_MARKERS)
                )
            )
            for item, text_blob in zip(family_rows, text_blobs, strict=True)
        )
        low_quality_family_seeds[simhash] = any(
            (
                item["predicted_label"] == "drop_low_quality"
                and item["predicted_score"] >= ROW_DROP_MIN_SCORE
                and (
                    item["effective_archetype"] in LOW_QUALITY_ARCHETYPES
                    or item["junk_rate"] >= 0.5
                    or item["effective_score_0_1"] <= 0.25
                )
            )
            for item in family_rows
        )

    output_rows: list[dict[str, object]] = []
    for row, cluster_id, similarity in zip(rows, assignments, similarities, strict=True):
        metrics = cluster_metrics[int(cluster_id)]
        teacher_label = str(metrics["teacher_label"])
        cluster_confidence = float(metrics["teacher_confidence"])
        similarity_floor = similarity_cutoffs[int(cluster_id)]
        row_similarity = float(similarity)

        row_teacher_label = teacher_label
        row_reason = str(metrics["teacher_reason"])
        adjusted_confidence = cluster_confidence
        simhash = str((row.get("vector_payload") or {}).get("simhash_u64") or "")

        text_blob = f" {str(row['teacher_seed_text'])} "
        strong_press_row = (
            row["predicted_label"] == "drop_press_release"
            and row["weak_label"] == "drop_press_release"
            and row["predicted_score"] >= ROW_DROP_MIN_SCORE
            and (
                row["press_hits"] > 0
                or row["effective_archetype"] in PRESS_RELEASE_ARCHETYPES
                or any(marker in text_blob for marker in ANNOUNCEMENT_MARKERS)
            )
        )
        strong_low_quality_row = (
            row["predicted_label"] == "drop_low_quality"
            and row["weak_label"] == "drop_low_quality"
            and row["predicted_score"] >= ROW_DROP_MIN_SCORE
            and (
                row["effective_archetype"] in LOW_QUALITY_ARCHETYPES
                or row["junk_rate"] >= 0.5
                or row["effective_score_0_1"] <= 0.25
            )
        )
        strong_junk_topic_row = (
            row["predicted_label"] in {"drop_sports", "drop_entertainment", "drop_lifestyle", "drop_local_crime"}
            and row["weak_label"] == row["predicted_label"]
            and row["predicted_score"] >= 0.94
        )
        mirror_press_family_row = (
            bool(simhash)
            and press_family_seeds.get(simhash, False)
            and (
                row["press_hits"] > 0
                or row["effective_archetype"] in PRESS_RELEASE_ARCHETYPES
                or any(marker in text_blob for marker in ANNOUNCEMENT_MARKERS)
                or row["source_domain"] in {"newswire.ca", "manilatimes.net", "itnewsonline.com", "searchlight.vc", "pr.com"}
            )
        )
        mirror_low_quality_family_row = (
            bool(simhash)
            and low_quality_family_seeds.get(simhash, False)
            and (
                row["effective_archetype"] in LOW_QUALITY_ARCHETYPES
                or row["junk_rate"] >= 0.5
                or row["effective_score_0_1"] <= 0.25
            )
        )

        if strong_press_row:
            row_teacher_label = "drop_press_release"
            adjusted_confidence = max(adjusted_confidence, 0.93)
            row_reason = "row_press_release_override"
        elif mirror_press_family_row:
            row_teacher_label = "drop_press_release"
            adjusted_confidence = max(adjusted_confidence, 0.88)
            row_reason = "mirror_press_family_override"
        elif strong_low_quality_row:
            row_teacher_label = "drop_low_quality"
            adjusted_confidence = max(adjusted_confidence, 0.94)
            row_reason = "row_low_quality_override"
        elif mirror_low_quality_family_row:
            row_teacher_label = "drop_low_quality"
            adjusted_confidence = max(adjusted_confidence, 0.87)
            row_reason = "mirror_low_quality_family_override"
        elif strong_junk_topic_row:
            row_teacher_label = str(row["predicted_label"])
            adjusted_confidence = max(adjusted_confidence, 0.95)
            row_reason = "row_junk_override"
        elif teacher_label == REVIEW_LABEL:
            row_teacher_label = REVIEW_LABEL
            adjusted_confidence = 0.0
        else:
            if row_similarity < similarity_floor:
                row_teacher_label = REVIEW_LABEL
                adjusted_confidence = 0.0
                row_reason = "low_similarity_outlier"
            elif teacher_label.startswith("keep_") and row["predicted_label"].startswith("drop_") and row["predicted_score"] >= 0.85:
                row_teacher_label = REVIEW_LABEL
                adjusted_confidence = 0.0
                row_reason = "student_drop_conflict"
            elif teacher_label.startswith("drop_") and row["predicted_label"].startswith("keep_") and row["predicted_score"] >= 0.9:
                if teacher_label == "drop_press_release" and (row["press_hits"] > 0 or row["effective_archetype"] in PRESS_RELEASE_ARCHETYPES):
                    adjusted_confidence = min(0.99, cluster_confidence + 0.05)
                else:
                    row_teacher_label = REVIEW_LABEL
                    adjusted_confidence = 0.0
                    row_reason = "student_keep_conflict"
            else:
                agreement_bonus = 0.05 if row["weak_label"] == teacher_label or row["predicted_label"] == teacher_label else 0.0
                similarity_bonus = max(0.0, min(0.1, (row_similarity - similarity_floor) * 0.25))
                adjusted_confidence = min(0.99, cluster_confidence + agreement_bonus + similarity_bonus)

        min_confidence = 0.65 if row_teacher_label.startswith("drop_") else 0.72
        teacher_trainable = row_teacher_label != REVIEW_LABEL and adjusted_confidence >= min_confidence
        output_rows.append(
            {
                **row,
                "cluster_id": int(cluster_id),
                "cluster_similarity": f"{row_similarity:.4f}",
                "teacher_cluster_label": teacher_label,
                "teacher_label": row_teacher_label,
                "teacher_confidence": f"{adjusted_confidence:.4f}",
                "teacher_reason": row_reason,
                "teacher_trainable": "1" if teacher_trainable else "0",
            }
        )
    return output_rows


def write_outputs(cluster_metrics: dict[int, dict[str, object]], output_rows: list[dict[str, object]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cluster_fieldnames = [
        "cluster_id",
        "size",
        "teacher_label",
        "teacher_confidence",
        "teacher_reason",
        "label_agreement_rate",
        "mean_similarity",
        "mean_predicted_score",
        "mean_domain_quality",
        "high_trust_rate",
        "press_like_rate",
        "low_quality_rate",
        "duplicate_rate",
        "canonical_duplicate_rate",
        "trusted_keep_seed_rate",
        "trusted_drop_seed_rate",
        "finance_signal_rate",
        "macro_signal_rate",
        "geo_signal_rate",
        "company_signal_rate",
        "sports_signal_rate",
        "entertainment_signal_rate",
        "lifestyle_signal_rate",
        "crime_signal_rate",
        "announcement_rate",
        "weak_counts_json",
        "predicted_counts_json",
        "decision_counts_json",
        "top_domains_json",
        "top_titles_json",
    ]
    with CLUSTERS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cluster_fieldnames)
        writer.writeheader()
        for cluster_id in sorted(cluster_metrics):
            metrics = cluster_metrics[cluster_id]
            writer.writerow(
                {
                    "cluster_id": cluster_id,
                    "size": metrics["size"],
                    "teacher_label": metrics["teacher_label"],
                    "teacher_confidence": f"{float(metrics['teacher_confidence']):.4f}",
                    "teacher_reason": metrics["teacher_reason"],
                    "label_agreement_rate": f"{float(metrics['label_agreement_rate']):.4f}",
                    "mean_similarity": f"{float(metrics['mean_similarity']):.4f}",
                    "mean_predicted_score": f"{float(metrics['mean_predicted_score']):.4f}",
                    "mean_domain_quality": f"{float(metrics['mean_domain_quality']):.4f}",
                    "high_trust_rate": f"{float(metrics['high_trust_rate']):.4f}",
                    "press_like_rate": f"{float(metrics['press_like_rate']):.4f}",
                    "low_quality_rate": f"{float(metrics['low_quality_rate']):.4f}",
                    "duplicate_rate": f"{float(metrics['duplicate_rate']):.4f}",
                    "canonical_duplicate_rate": f"{float(metrics['canonical_duplicate_rate']):.4f}",
                    "trusted_keep_seed_rate": f"{float(metrics['trusted_keep_seed_rate']):.4f}",
                    "trusted_drop_seed_rate": f"{float(metrics['trusted_drop_seed_rate']):.4f}",
                    "finance_signal_rate": f"{float(metrics['finance_signal_rate']):.4f}",
                    "macro_signal_rate": f"{float(metrics['macro_signal_rate']):.4f}",
                    "geo_signal_rate": f"{float(metrics['geo_signal_rate']):.4f}",
                    "company_signal_rate": f"{float(metrics['company_signal_rate']):.4f}",
                    "sports_signal_rate": f"{float(metrics['sports_signal_rate']):.4f}",
                    "entertainment_signal_rate": f"{float(metrics['entertainment_signal_rate']):.4f}",
                    "lifestyle_signal_rate": f"{float(metrics['lifestyle_signal_rate']):.4f}",
                    "crime_signal_rate": f"{float(metrics['crime_signal_rate']):.4f}",
                    "announcement_rate": f"{float(metrics['announcement_rate']):.4f}",
                    "weak_counts_json": json.dumps(metrics["weak_counts"], sort_keys=True),
                    "predicted_counts_json": json.dumps(metrics["predicted_counts"], sort_keys=True),
                    "decision_counts_json": json.dumps(metrics["decision_counts"], sort_keys=True),
                    "top_domains_json": json.dumps(metrics["top_domains"]),
                    "top_titles_json": json.dumps(metrics["top_titles"]),
                }
            )

    label_fieldnames = [
        "document_identifier",
        "partition_date",
        "source_domain",
        "cluster_id",
        "cluster_similarity",
        "weak_label",
        "predicted_label",
        "predicted_score",
        "decision_band",
        "teacher_cluster_label",
        "teacher_label",
        "teacher_confidence",
        "teacher_reason",
        "teacher_trainable",
        "effective_archetype",
        "effective_score_0_1",
        "title",
        "summary",
    ]
    with LABELS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=label_fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({name: row.get(name, "") for name in label_fieldnames})

    train_lines: list[str] = []
    valid_lines: list[str] = []
    for row in output_rows:
        if row["teacher_trainable"] != "1":
            continue
        line = f"__label__{row['teacher_label']} {row['teacher_seed_text']}".strip()
        if weak_split_bucket(str(row["document_identifier"])) == "valid":
            valid_lines.append(line)
        else:
            train_lines.append(line)

    TRAIN_TXT.write_text("\n".join(train_lines) + ("\n" if train_lines else ""), encoding="utf-8")
    VALID_TXT.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    domain_scores = load_domain_scores()
    rows = load_rows(domain_scores, args.max_rows)
    rows_by_id = {str(row["document_identifier"]): row for row in rows}
    qdrant_start = time.perf_counter()
    matched = fetch_qdrant_vectors(args.qdrant_url, args.collection, set(rows_by_id), args.page_size)
    qdrant_seconds = time.perf_counter() - qdrant_start

    joined_rows: list[dict[str, object]] = []
    vector_list: list[list[float]] = []
    missing_rows = 0
    for document_identifier, row in rows_by_id.items():
        point = matched.get(document_identifier)
        if not point or not point.get("vector"):
            missing_rows += 1
            continue
        joined_row = dict(row)
        joined_row["vector_payload"] = point["payload"]
        joined_rows.append(joined_row)
        vector_list.append(point["vector"])

    if not joined_rows:
        raise SystemExit("No rows could be joined against the Qdrant collection.")

    vectors = normalize_vectors(np.asarray(vector_list, dtype=np.float32))
    cluster_start = time.perf_counter()
    centroids = fit_centroids(vectors, args.cluster_count, args.fit_sample_size, args.iterations, args.seed)
    assignments, similarities = assign_clusters(vectors, centroids, args.batch_size)
    cluster_seconds = time.perf_counter() - cluster_start

    for row, similarity in zip(joined_rows, similarities, strict=True):
        row["_similarity"] = float(similarity)

    metrics = build_cluster_metrics(joined_rows, assignments, similarities)
    for cluster_id, cluster_metrics in metrics.items():
        teacher_label, teacher_confidence, teacher_reason = choose_teacher_label(cluster_metrics)
        cluster_metrics["teacher_label"] = teacher_label
        cluster_metrics["teacher_confidence"] = teacher_confidence
        cluster_metrics["teacher_reason"] = teacher_reason

    output_rows = build_row_teacher_labels(joined_rows, assignments, similarities, metrics)
    write_outputs(metrics, output_rows)

    trainable_rows = [row for row in output_rows if row["teacher_trainable"] == "1"]
    teacher_counts = Counter(str(row["teacher_label"]) for row in output_rows)
    trainable_counts = Counter(str(row["teacher_label"]) for row in trainable_rows)
    summary = {
        "collection": args.collection,
        "qdrant_url": args.qdrant_url,
        "input_rows": len(rows),
        "joined_rows": len(joined_rows),
        "missing_rows": missing_rows,
        "cluster_count": len(metrics),
        "teacher_label_counts": dict(sorted(teacher_counts.items())),
        "trainable_label_counts": dict(sorted(trainable_counts.items())),
        "trainable_rows": len(trainable_rows),
        "trainable_rate": round(len(trainable_rows) / len(output_rows), 4) if output_rows else 0.0,
        "review_rows": teacher_counts.get(REVIEW_LABEL, 0),
        "top_teacher_clusters": [
            {
                "cluster_id": cluster_id,
                "size": metric["size"],
                "teacher_label": metric["teacher_label"],
                "teacher_confidence": round(float(metric["teacher_confidence"]), 4),
                "teacher_reason": metric["teacher_reason"],
                "top_domains": metric["top_domains"][:3],
                "top_titles": metric["top_titles"][:2],
            }
            for cluster_id, metric in sorted(metrics.items(), key=lambda item: item[1]["size"], reverse=True)[:10]
        ],
        "output_files": {
            "clusters_csv": str(CLUSTERS_CSV),
            "labels_csv": str(LABELS_CSV),
            "train_txt": str(TRAIN_TXT),
            "valid_txt": str(VALID_TXT),
        },
    }
    benchmark = {
        "qdrant_fetch_seconds": round(qdrant_seconds, 4),
        "clustering_seconds": round(cluster_seconds, 4),
        "total_seconds": round(time.perf_counter() - start, 4),
        "rows_per_second": round(len(joined_rows) / max(time.perf_counter() - start, 1e-9), 2),
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    BENCHMARK_JSON.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "benchmark": benchmark}, indent=2))


if __name__ == "__main__":
    main()
