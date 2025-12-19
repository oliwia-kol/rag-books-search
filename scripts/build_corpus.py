#!/usr/bin/env python
"""Offline corpus builder (CPU-only).

Pipeline (deterministic)
-----------------------
1. ingest plain-text files from --src/<publisher> (sorted)
2. chunk text with fixed --chunk-size
3. embed with SentenceTransformer (seeded)
4. build FAISS index (IndexFlatIP, normalized vectors)
5. write meta.sqlite (chunks + FTS)
6. emit manifest.json and validation report

Use --validate-only to skip embedding and only validate existing artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data"
DEFAULT_SRC = ROOT / "raw"
DEFAULT_PUBS = ["OReilly", "Manning", "Pearson"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def _chunk_text(txt: str, chunk_size: int) -> List[str]:
    words = txt.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        part = " ".join(words[i : i + chunk_size]).strip()
        if part:
            chunks.append(part)
    return chunks


def _read_sources(pub_dir: Path) -> List[Tuple[str, str]]:
    files = sorted([p for p in pub_dir.glob("**/*.txt") if p.is_file()])
    data = []
    for f in files:
        try:
            data.append((str(f), f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return data


def _ensure_out(pub_out: Path):
    pub_out.mkdir(parents=True, exist_ok=True)


def _build_sqlite(pub_out: Path, rows: List[Dict[str, str]]):
    db_p = pub_out / "meta.sqlite"
    con = sqlite3.connect(str(db_p))
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS chunks")
    cur.execute(
        """
        CREATE TABLE chunks (
            cid TEXT,
            fp TEXT,
            sec TEXT,
            cidx INTEGER,
            tx TEXT,
            i64 INTEGER PRIMARY KEY AUTOINCREMENT
        )
        """
    )
    cur.execute("DROP TABLE IF EXISTS chunks_fts")
    cur.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(cid, fp, sec, tx)")
    cur.executemany(
        "INSERT INTO chunks(cid, fp, sec, cidx, tx) VALUES (?, ?, ?, ?, ?)",
        [(r["cid"], r["fp"], r["sec"], r["cidx"], r["tx"]) for r in rows],
    )
    cur.executemany(
        "INSERT INTO chunks_fts(cid, fp, sec, tx) VALUES (?, ?, ?, ?)",
        [(r["cid"], r["fp"], r["sec"], r["tx"]) for r in rows],
    )
    con.commit()
    con.close()
    return db_p


def _build_faiss(pub_out: Path, emb: np.ndarray, dim: int):
    import faiss  # imported lazily

    index = faiss.IndexFlatIP(dim)
    # embeddings already normalized
    index.add(emb.astype("float32"))
    faiss.write_index(index, str(pub_out / "index.faiss"))
    return pub_out / "index.faiss"


def _write_manifest(pub_out: Path, manifest: Dict[str, object]):
    with (pub_out / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


def _validate(pub_out: Path, manifest_expected: Dict[str, object]) -> Tuple[bool, str]:
    try:
        import faiss
    except Exception as ex:  # pragma: no cover
        return False, f"faiss import failed: {ex}"
    ix_p = pub_out / "index.faiss"
    db_p = pub_out / "meta.sqlite"
    mf_p = pub_out / "manifest.json"
    if not ix_p.exists() or not db_p.exists() or not mf_p.exists():
        return False, "missing required artifacts"
    try:
        ix = faiss.read_index(str(ix_p))
        dim = int(ix.d)
    except Exception as ex:  # pragma: no cover
        return False, f"faiss load failed: {ex}"
    try:
        con = sqlite3.connect(str(db_p))
        cnt = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        con.close()
    except Exception as ex:  # pragma: no cover
        return False, f"sqlite load failed: {ex}"

    try:
        manifest = json.loads(mf_p.read_text())
        dim_ok = manifest.get("embed_dim") == dim
        cnt_ok = manifest.get("total_chunks") == cnt
        if not dim_ok:
            return False, f"dim mismatch {dim} vs {manifest.get('embed_dim')}"
        if not cnt_ok:
            return False, f"chunk count mismatch {cnt} vs {manifest.get('total_chunks')}"
    except Exception as ex:  # pragma: no cover
        return False, f"manifest load failed: {ex}"
    return True, "ok"


def build_one(pub: str, args) -> Dict[str, object]:
    pub_src = args.src / pub
    pub_out = args.out / pub
    _ensure_out(pub_out)

    report = {"publisher": pub, "ok": False, "reason": None}

    if args.validate_only:
        ok, reason = _validate(pub_out, {})
        report.update({"ok": ok, "reason": reason})
        return report

    if not pub_src.exists():
        report["reason"] = f"missing source dir {pub_src}"
        return report

    set_seed(args.seed)
    data = _read_sources(pub_src)
    if not data:
        report["reason"] = "no source .txt files"
        return report

    rows = []
    cid = 0
    for fp, txt in data:
        chunks = _chunk_text(txt, args.chunk_size)
        for idx, ch in enumerate(chunks):
            rows.append({"cid": f"{cid}", "fp": fp, "sec": Path(fp).stem, "cidx": idx, "tx": ch})
        cid += 1

    # embed
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.model)
        embs = np.asarray(model.encode([r["tx"] for r in rows], convert_to_numpy=True), dtype="float32")
        # normalize for IP
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        embs = embs / norms
        dim = embs.shape[1]
    except Exception as ex:  # pragma: no cover - build-time failures
        report["reason"] = f"embedding failed: {ex}"
        return report

    _build_faiss(pub_out, embs, dim)
    _build_sqlite(pub_out, rows)
    manifest = {
        "publisher": pub,
        "total_chunks": len(rows),
        "embed_dim": dim,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "source_files": [fp for fp, _ in data],
    }
    _write_manifest(pub_out, manifest)
    report.update({"ok": True, "reason": "ok"})
    return report


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Build FAISS+SQLite corpora (CPU-only)")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Source directory containing <publisher>/*.txt")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory (default: data/)")
    ap.add_argument("--publisher", action="append", help="Publisher(s) to build; default: all known")
    ap.add_argument("--chunk-size", type=int, default=480, help="Words per chunk")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--validate-only", action="store_true", help="Only validate artifacts (no rebuild)")
    ap.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    pubs = args.publisher or DEFAULT_PUBS

    report = {"corpora": [], "ok": True}
    for pub in pubs:
        r = build_one(pub, args)
        report["corpora"].append(r)
        if not r.get("ok"):
            report["ok"] = False

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

    for r in report["corpora"]:
        status = "OK" if r["ok"] else "FAIL"
        print(f"[{status}] {r['publisher']}: {r.get('reason')}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
