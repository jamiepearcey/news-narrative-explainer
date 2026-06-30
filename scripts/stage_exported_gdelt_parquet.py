#!/usr/bin/env python3
"""Stage exported GDELT parquet files into the local candidate layout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "gdelt_candidates"


def ensure_dependencies() -> None:
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except ModuleNotFoundError:
        uv = shutil.which("uv") or "/opt/homebrew/bin/uv"
        if not Path(uv).exists():
            raise RuntimeError("pyarrow is required, and `uv` was not found to bootstrap it") from None
        if os.environ.get("NEWS_NARRATIVE_STAGE_UV_BOOTSTRAPPED") == "1":
            raise
        env = os.environ.copy()
        env["NEWS_NARRATIVE_STAGE_UV_BOOTSTRAPPED"] = "1"
        os.execvpe(
            uv,
            [uv, "run", "--with", "pyarrow>=16", str(Path(__file__).resolve()), *sys.argv[1:]],
            env,
        )


def parquet_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.parquet") if path.is_file())


def detect_partition_date(path: Path) -> str:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["partition_date"])
    values = {
        str(value)
        for value in table.column("partition_date").to_pylist()
        if value is not None and str(value).strip()
    }
    if not values:
        raise ValueError(f"{path} has no non-null partition_date values")
    if len(values) != 1:
        raise ValueError(f"{path} spans multiple partition_date values: {sorted(values)}")
    return next(iter(values))


def unique_target_path(target_dir: Path, file_name: str) -> Path:
    candidate = target_dir / file_name
    if not candidate.exists():
        return candidate
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    counter = 1
    while True:
        candidate = target_dir / f"{stem}-{counter:06d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def stage_export(input_root: Path, output_root: Path, move: bool, overwrite: bool) -> dict[str, object]:
    files = parquet_files(input_root)
    if not files:
        raise FileNotFoundError(f"no parquet files found under {input_root}")

    staged: list[dict[str, object]] = []
    total_bytes = 0
    for source in files:
        partition_date = detect_partition_date(source)
        target_dir = output_root / f"dt={partition_date}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        if target.exists() and overwrite:
            target.unlink()
        elif target.exists():
            target = unique_target_path(target_dir, source.name)
        if move:
            shutil.move(str(source), str(target))
        else:
            shutil.copy2(source, target)
        file_bytes = target.stat().st_size
        total_bytes += file_bytes
        staged.append(
            {
                "source": str(source),
                "target": str(target),
                "partition_date": partition_date,
                "bytes": file_bytes,
            }
        )
    payload = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "staged_files": staged,
        "staged_file_count": len(staged),
        "staged_total_bytes": total_bytes,
        "move": move,
    }
    manifest_path = output_root / "stage-manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2))
    payload["manifest_path"] = str(manifest_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--move", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    ensure_dependencies()
    args = parse_args()
    payload = stage_export(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        move=args.move,
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
