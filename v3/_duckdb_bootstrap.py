"""Runtime bootstrap for standalone DuckDB-backed v3 scripts."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_SENTINEL = "NEWS_NARRATIVE_V3_UV_BOOTSTRAPPED"
_UV_CANDIDATES = (
    "uv",
    "/opt/homebrew/bin/uv",
    "/usr/local/bin/uv",
)


def _find_uv() -> str | None:
    for candidate in _UV_CANDIDATES:
        resolved = shutil.which(candidate) if "/" not in candidate else candidate
        if resolved and Path(resolved).exists():
            return resolved
    return None


def ensure_duckdb(script_path: str) -> None:
    try:
        import duckdb  # noqa: F401
    except ModuleNotFoundError:
        if os.environ.get(_SENTINEL) == "1":
            raise
        uv = _find_uv()
        if uv is None:
            raise RuntimeError(
                "duckdb is not installed for this Python, and `uv` is not available "
                "to bootstrap it automatically"
            ) from None
        os.environ[_SENTINEL] = "1"
        os.execvp(uv, [uv, "run", "--with", "duckdb>=1.0", str(Path(script_path).resolve()), *sys.argv[1:]])
