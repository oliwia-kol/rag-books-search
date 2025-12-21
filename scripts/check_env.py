"""Environment checker for RAG Books Search.

Validates Python version, required dependencies, and presence of per-corpus
artifacts. Intended for local pre-flight checks before running the app.

Run:
    python scripts/check_env.py
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Dict, List

import importlib

ROOT = Path(__file__).resolve().parents[1]


REQUIRED_MODULES = [
    "streamlit",
    "numpy",
    "faiss",
    "sentence_transformers",
    "torch",
]

DATA_LAYOUT = {
    "OReilly": ["index.faiss", "meta.sqlite", "manifest.json"],
    "Manning": ["index.faiss", "meta.sqlite", "manifest.json"],
    "Pearson": ["index.faiss", "meta.sqlite", "manifest.json"],
}


class CheckResult:
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.data: Dict[str, Dict[str, bool]] = {}

    def ok(self) -> bool:
        return not self.errors


def _check_python(res: CheckResult):
    ver = sys.version_info
    ver_str = platform.python_version()
    if ver < (3, 9):
        res.errors.append(f"Python >=3.9 required, found {ver_str}")
    else:
        res.info.append(f"Python version OK: {ver_str}")


def _check_deps(res: CheckResult):
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
            res.info.append(f"Dependency ok: {mod}")
        except Exception as exc:
            res.errors.append(f"Missing dependency: {mod} ({exc})")


def _check_data(res: CheckResult):
    data_root = ROOT / "data"
    for corp, files in DATA_LAYOUT.items():
        corp_path = data_root / corp
        status = {"exists": corp_path.exists()}
        for f in files:
            status[f] = (corp_path / f).exists()
        status["ok"] = status["exists"] and all(status[f] for f in files)
        if not status["exists"]:
            res.warnings.append(f"Missing corpus folder: {corp_path}")
        elif not status["ok"]:
            missing = [f for f in files if not status[f]]
            res.warnings.append(f"Corpus {corp} incomplete: missing {', '.join(missing)}")
        res.data[corp] = status


def main():
    res = CheckResult()
    _check_python(res)
    _check_deps(res)
    _check_data(res)

    print("=== Environment check ===")
    print(json.dumps({"info": res.info, "warnings": res.warnings, "errors": res.errors, "data": res.data}, indent=2))

    if res.errors:
        sys.exit(1)
    # Missing corpora are warnings, not fatal for CI/local checks.
    sys.exit(0)


if __name__ == "__main__":
    main()
