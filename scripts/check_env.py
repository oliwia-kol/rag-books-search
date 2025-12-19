#!/usr/bin/env python
"""Environment validation for CPU-only RAG Books Search.

Exit codes
----------
0: ok
2: python version mismatch
3: missing required package
4: missing data artifacts
5: other error
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

REQUIRED_PACKAGES = [
    "streamlit",
    "numpy",
    "pandas",
    "faiss",
    "sentence_transformers",
]

DEFAULT_PUBS = ["OReilly", "Manning", "Pearson"]


def _ok(msg: str):
    print(f"[OK] {msg}")


def _fail(msg: str):
    print(f"[FAIL] {msg}")


def check_python(min_version=(3, 9)) -> bool:
    if sys.version_info < min_version:
        _fail(f"Python>={'%s.%s' % min_version} required; found {sys.version.split()[0]}")
        return False
    _ok(f"Python {sys.version.split()[0]}")
    return True


def check_packages() -> bool:
    ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            _ok(f"package {pkg}")
        except Exception as ex:  # pragma: no cover - defensive
            ok = False
            _fail(f"package {pkg} missing: {type(ex).__name__}: {ex}")
    return ok


def check_data(publishers=None) -> bool:
    pubs = publishers or DEFAULT_PUBS
    base = ROOT / "data"
    ok = True
    for pub in pubs:
        p = base / pub
        faiss_p = p / "index.faiss"
        db_p = p / "meta.sqlite"
        manifest_p = p / "manifest.json"
        exists = p.exists()
        missing = [name for name, flag in [
            ("index.faiss", faiss_p.exists()),
            ("meta.sqlite", db_p.exists()),
            ("manifest.json", manifest_p.exists()),
        ] if not flag]
        if exists and not missing:
            _ok(f"data/{pub}")
        else:
            ok = False
            if not exists:
                _fail(f"data/{pub}: directory missing")
            else:
                _fail(f"data/{pub}: missing {', '.join(missing)}")
    return ok


def load_publishers_from_manifest(manifest_path: Path) -> list[str]:
    with manifest_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    pubs = data.get("publishers")
    if isinstance(pubs, list) and pubs:
        return [str(p) for p in pubs]
    raise ValueError("manifest publishers must be a non-empty list")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Check environment for rag-books-search (CPU)")
    ap.add_argument(
        "--publishers-manifest",
        type=Path,
        help="Optional JSON file containing {'publishers': ['OReilly', ...]} to override defaults.",
    )
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        pubs = DEFAULT_PUBS
        if args.publishers_manifest:
            try:
                pubs = load_publishers_from_manifest(args.publishers_manifest)
            except Exception as ex:
                _fail(f"failed to read publishers manifest: {ex}")
                return 5

        py_ok = check_python()
        pkg_ok = check_packages()
        data_ok = check_data(pubs)

        if not py_ok:
            return 2
        if not pkg_ok:
            return 3
        if not data_ok:
            return 4
        _ok("environment ready")
        return 0
    except KeyboardInterrupt:  # pragma: no cover - convenience
        return 5
    except Exception as ex:  # pragma: no cover - defensive
        _fail(f"unexpected error: {type(ex).__name__}: {ex}")
        return 5


if __name__ == "__main__":
    sys.exit(main())
