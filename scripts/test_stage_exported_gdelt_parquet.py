#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage_exported_gdelt_parquet import stage_export


class StageExportedGdeltParquetTests(unittest.TestCase):
    def _write_parquet(self, path: Path, partition_date: str) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "partition_date": [partition_date],
                "document_identifier": ["https://example.com/story"],
                "v2_themes": ["OIL"],
            }
        )
        pq.write_table(table, path)

    def test_stage_export_copies_files_into_dt_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_root = tmp / "input"
            output_root = tmp / "output"
            input_root.mkdir()
            source = input_root / "part-000000.parquet"
            self._write_parquet(source, "2026-06-24")

            payload = stage_export(input_root, output_root, move=False, overwrite=False)

            target = output_root / "dt=2026-06-24" / "part-000000.parquet"
            self.assertTrue(target.exists())
            self.assertTrue(source.exists())
            self.assertEqual(payload["staged_file_count"], 1)
            self.assertTrue((output_root / "stage-manifest.json").exists())

    def test_stage_export_can_move_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_root = tmp / "input"
            output_root = tmp / "output"
            input_root.mkdir()
            source = input_root / "part-000000.parquet"
            self._write_parquet(source, "2026-06-24")

            stage_export(input_root, output_root, move=True, overwrite=False)

            self.assertFalse(source.exists())
            self.assertTrue((output_root / "dt=2026-06-24" / "part-000000.parquet").exists())

    def test_stage_export_renames_on_collision_when_not_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_root = tmp / "input"
            output_root = tmp / "output"
            input_root.mkdir()
            (output_root / "dt=2026-06-24").mkdir(parents=True)
            existing = output_root / "dt=2026-06-24" / "part-000000.parquet"
            self._write_parquet(existing, "2026-06-24")
            source = input_root / "part-000000.parquet"
            self._write_parquet(source, "2026-06-24")

            payload = stage_export(input_root, output_root, move=False, overwrite=False)

            targets = [Path(item["target"]).name for item in payload["staged_files"]]
            self.assertIn("part-000000-000001.parquet", targets)


if __name__ == "__main__":
    unittest.main()
