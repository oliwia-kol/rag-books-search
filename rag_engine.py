from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import logging
import math
import os
import re
import sqlite3
import time
import json
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from statistics import pstdev

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

# -----------------------------
# -----------------------------
# Paths (repo-local)
# -----------------------------
ROOT = Path(__file__).parent


def _resolve_data_root() -> Path:
    env_data_root = os.environ.get("RAG_DATA_ROOT")
    if env_data_root:
        try:
            return Path(env_data_root).expanduser()
        except Exception:
            pass
    hidden = ROOT / ".data"
    visible = ROOT / "data"
    if hidden.exists():
        return hidden
    return visible


BASE_OUT = _resolve_data_root()

logger = logging.getLogger(__name__)
_LOGGER_CONFIGURED = False
LOG_PATH = Path(os.environ.get("RAG_LOG_PATH", ROOT / "logs" / "query.log"))
LOG_MAX_BYTES = int(os.environ.get("RAG_LOG_MAX_BYTES", 1_000_000))
LOG_BACKUP_COUNT = int(os.environ.get("RAG_LOG_BACKUP_COUNT", 3))
RECENT_QUERY_LOG = Path(os.environ.get("RAG_RECENT_QUERY_LOG", LOG_PATH.parent / "recent_queries.json"))

CTX_CLAMP_MARKER = "... [ctx-clamped]"
LLM_CLAMP_MARKER = "... [llm-clamped]"


def _config_logger():
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    fmt = logging.Formatter("%(message)s")
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    try:
        file_handler = RotatingFileHandler(LOG_PATH, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except Exception:
        # best-effort; stdout/stderr logging still works
        pass
    _LOGGER_CONFIGURED = True


def _scope_from_hits(hits) -> List[str]:
    pubs = []
    for h in hits or []:
        for k in ("corp", "publisher", "pub"):
            v = (h or {}).get(k)
            if v:
                pubs.append(str(v))
                break
    return sorted(set(pubs))


def _clamp_text(txt: str, char_budget: Optional[int], tok_budget: Optional[int], marker: Optional[str]) -> tuple[str, Dict[str, bool]]:
    t = (txt or "").strip()
    char_clamped = False
    tok_clamped = False
    if char_budget is not None and char_budget > 0 and len(t) > char_budget:
        t = t[:char_budget]
        char_clamped = True
    toks = t.split()
    if tok_budget is not None and tok_budget > 0 and len(toks) > tok_budget:
        t = " ".join(toks[:tok_budget])
        tok_clamped = True
    if (char_clamped or tok_clamped) and marker:
        marker_tokens = marker.strip().split()
        marker_txt = " ".join(marker_tokens)
        base_tokens = t.split()
        if tok_budget is not None and tok_budget > 0:
            keep = max(0, tok_budget - len(marker_tokens))
            base_tokens = base_tokens[:keep]
        combined_tokens = base_tokens + marker_tokens
        combined = " ".join(combined_tokens).strip()
        if char_budget is not None and char_budget > 0 and len(combined) > char_budget:
            # trim base tokens until the marker fits
            while len(combined) > char_budget and base_tokens:
                base_tokens = base_tokens[:-1]
                combined_tokens = base_tokens + marker_tokens
                combined = " ".join(combined_tokens).strip()
            if len(combined) > char_budget:
                combined = marker_txt[:char_budget]
        t = combined
    return t, {"char_clamped": char_clamped, "token_clamped": tok_clamped}

CORP = {
    "OReilly": BASE_OUT / "OReilly",
    "Manning": BASE_OUT / "Manning",
    "Pearson": BASE_OUT / "Pearson",
}


def chk(p: Path) -> Dict[str, Any]:
    """Soft-check corpus folder; never raise. Returns a detailed report."""
    exists = p.exists()
    faiss_p = p / "index.faiss"
    db_p = p / "meta.sqlite"
    manifest_p = p / "manifest.json"
    rep = {
        "path": str(p),
        "exists": exists,
        "faiss": faiss_p.exists() if exists else False,
        "db": db_p.exists() if exists else False,
        "manifest": manifest_p.exists() if exists else False,
    }
    rep["ok"] = all([rep["exists"], rep["faiss"], rep["db"], rep["manifest"]])
    reasons = []
    if not exists:
        reasons.append("missing_corpus_dir")
    if exists:
        if not rep["faiss"]:
            reasons.append("missing_faiss")
        if not rep["db"]:
            reasons.append("missing_db")
        if not rep["manifest"]:
            reasons.append("missing_manifest")
    missing = []
    for k, flag in [("faiss", rep["faiss"]), ("db", rep["db"]), ("manifest", rep["manifest"])]:
        if not flag:
            missing.append(k)
    rep["missing"] = missing
    rep["reasons"] = reasons
    return rep


# -----------------------------
# -----------------------------
# Retrieval + judge config
# -----------------------------
NM_MIN = 0.28
NM_MAX = 6
SQLITE_TEXT_MAX = 4000
FAISS_FALLBACK_RETRY_MAX = 8

HCFG = {
    "faiss_fetch_k": 60,
    "fts_fetch_k": 60,
    "final_k": 10,  # default (per-mode overrides below)
    "mmr_k": 20,
    "mmr_lambda": 0.55,
    "min_faiss_score": 0.18,
    "dense_k": 30,
    "lex_k": 30,
    "fusion_dense_w": 0.65,
    "fusion_lex_w": 0.35,
    "max_snippet_chars": 850,
    "max_ctx_chars": 1400,
    "max_prompt_chars": 2000,
    "budgets": {
        "quick": {
            "ctx_chars": 1400,
            "ctx_tokens": 380,
            "prompt_chars": 2000,
            "prompt_tokens": 260,
        },
        "exact": {
            "ctx_chars": 2000,
            "ctx_tokens": 520,
            "prompt_chars": 2800,
            "prompt_tokens": 360,
        },
    },
    "jdg_cache_ttl": 600,
    "jdg_cache_size": 256,
    "modes": {
        "quick": {
            "label": "Quick",
            "description": "Faster answers with tighter retrieval and context budgets.",
            "final_k": 8,
            "mmr_k": 16,
            "dense_k": 24,
            "lex_k": 24,
            "use_jdg": True,
        },
        "exact": {
            "label": "Find Exact Quote",
            "description": "Deeper search for citations with larger budgets and k.",
            "final_k": 12,
            "mmr_k": 28,
            "dense_k": 40,
            "lex_k": 40,
            "use_jdg": True,
        },
    },
}

FTS_CFG = {"batch": 4000, "use_porter": False}

USE_JDG_DEFAULT = True
K_SHOW = 18
MNK = 4
ABS_MN = 0.30

K_JDG = 12
J_DISP_MIN = 0.45
J_STRONG_MIN = 0.60
J_MIN_KEEP = 0.35
J_WEAK_MIN = 0.45
J_MIN_CNT = 1
DIRECT_MIN = 1

# contract constants
RET_KEYS = ["ok", "no_evidence", "answer", "hits", "near_miss", "coverage", "meta"]
META_T_KEYS = ["total", "embed", "dense", "lex", "fuse", "cut", "rerank", "disp_flt", "direct", "near_miss", "llm", "judge_cache", "judge_pred"]
META_N_KEYS = [
    "pubs_req",
    "pubs_used",
    "dense_hits",
    "lex_hits",
    "fetched_dense",
    "fetched_lex",
    "dense_fallback",
    "dense_fallback_fail",
    "cands",
    "after_cut",
    "after_disp",
    "direct_hits",
    "near_miss",
    "uniq_books",
    "uniq_sections",
    "fallback_retries",
    "fallback_failed",
]
META_CAP_KEYS = ["has_emb", "dense_ok", "lex_ok", "judge_requested", "judge_ok", "judge_kind"]
META_FLAG_KEYS = [
    "dense_used",
    "lex_used",
    "veto_applied",
    "veto_disabled",
    "veto_disabled_when_proxy",
    "judge_proxy",
    "llm_used",
    "llm_bypassed",
    "dense_clamped",
    "lex_clamped",
    "near_miss_skipped",
]
STAGES = META_T_KEYS


def _budget_for_mode(mode: str, overrides: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    budgets = HCFG.get("budgets") or {}
    default = budgets.get("quick", {})
    b = budgets.get((mode or "quick").lower(), {}) or {}
    overrides = overrides or {}
    return {
        "ctx_chars": overrides.get("ctx_chars", b.get("ctx_chars", default.get("ctx_chars", HCFG["max_ctx_chars"]))),
        "ctx_tokens": overrides.get("ctx_tokens", b.get("ctx_tokens", default.get("ctx_tokens", HCFG["max_ctx_chars"]))),
        "prompt_chars": overrides.get("prompt_chars", b.get("prompt_chars", default.get("prompt_chars", HCFG["max_prompt_chars"]))),
        "prompt_tokens": overrides.get("prompt_tokens", b.get("prompt_tokens", default.get("prompt_tokens", HCFG["max_prompt_chars"]))),
    }


def _mode_cfg(mode: Optional[str]) -> Dict[str, Any]:
    mode_key = (mode or "quick").lower()
    modes = HCFG.get("modes") or {}
    default = modes.get("quick", {}) or {}
    cfg = modes.get(mode_key, {}) or {}
    merged = {**default, **cfg}
    return {
        "name": mode_key,
        "label": merged.get("label", mode_key.title()),
        "description": merged.get("description", ""),
        "final_k": int(merged.get("final_k", HCFG["final_k"])),
        "mmr_k": int(merged.get("mmr_k", HCFG["mmr_k"])),
        "dense_k": int(merged.get("dense_k", HCFG["dense_k"])),
        "lex_k": int(merged.get("lex_k", HCFG["lex_k"])),
        "use_jdg": bool(merged.get("use_jdg", True)),
    }


def get_mode_cfg(mode: Optional[str] = None) -> Dict[str, Any]:
    return _mode_cfg(mode)


def mode_options() -> List[Dict[str, Any]]:
    modes = HCFG.get("modes") or {}
    out = []
    for name, cfg in modes.items():
        out.append(
            {
                "name": name,
                "label": cfg.get("label", name.title()),
                "description": cfg.get("description", ""),
            }
        )
    if not out:
        out.append({"name": "quick", "label": "Quick", "description": "Faster answers."})
    return out

# -----------------------------
# FTS helpers
# -----------------------------
def fts_query_escape(q: str) -> str:
    # identical intent to 02: tokenise, keep alnum, join with OR
    xs = re.findall(r"[A-Za-z0-9]+", (q or "").strip())
    xs = [x for x in xs if len(x) >= 2]
    if not xs:
        return ""
    # basic FTS5 safe query: token1 OR token2 OR ...
    return " OR ".join(xs)


def fts_ready(con: sqlite3.Connection) -> bool:
    cur = con.cursor()
    r = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'").fetchone()
    return r is not None


def get_by_i64(con: sqlite3.Connection, i64: int):
    cur = con.cursor()
    return cur.execute(
        "SELECT cid, fp, sec, cidx, tx FROM chunks WHERE i64=? LIMIT 1",
        (int(i64),),
    ).fetchone()


def norm_scores(xs, key):
    vals = [x.get(key, 0.0) for x in xs]
    if not vals:
        return xs
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9:
        for x in xs:
            x[key + "_n"] = 0.0
        return xs
    for x in xs:
        x[key + "_n"] = (x.get(key, 0.0) - mn) / (mx - mn)
    return xs


def _cap_tx(tx: Any, max_len: int = SQLITE_TEXT_MAX) -> str:
    try:
        s = str(tx or "")
    except Exception:
        s = ""
    if max_len is None or max_len <= 0:
        return s
    return s[: max_len + 1][:max_len]


# -----------------------------
# Engine
# -----------------------------
@dataclass
class Eng:
    emb: Optional[SentenceTransformer]
    ix: Dict[str, Any]
    dbp: Dict[str, Path]
    corp: Dict[str, Path]
    ix_dim: Dict[str, int]
    corp_report: Dict[str, Dict[str, Any]]
    corp_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def _mk_eng(base_out: Path = BASE_OUT, emb_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Eng:
    # validate corp folders (softly to avoid start-up crashes)
    c2p = {k: (base_out / k) for k in CORP.keys()}
    reports = {k: chk(p) for k, p in c2p.items()}
    ready: Dict[str, Path] = {k: p for k, p in c2p.items() if reports.get(k, {}).get("ok")}

    try:
        emb: Optional[SentenceTransformer] = SentenceTransformer(emb_model)
    except Exception:
        emb = None
    emb_dim = None
    try:
        emb_dim = int(emb.get_sentence_embedding_dimension()) if emb is not None else None
    except Exception:
        emb_dim = None

    ix: Dict[str, Any] = {}
    dbp: Dict[str, Path] = {}
    loaded: Dict[str, Path] = {}
    dims: Dict[str, int] = {}
    status: Dict[str, Dict[str, Any]] = {}

    for k, p in c2p.items():
        ok_dense = False
        dim_ok = True
        failure_reason = None
        rep = reports.get(k, {})
        rep.setdefault("dense_loaded", False)
        rep.setdefault("db_loaded", False)
        rep.setdefault("dim_ok", False)
        rep.setdefault("embed_dim", emb_dim)
        rep.setdefault("ix_dim", None)
        rep.setdefault("failure_reason", None)

        if k in ready:
            try:
                rd = faiss.read_index(str(p / "index.faiss"))
                dims[k] = int(rd.d)
                rep["ix_dim"] = dims.get(k)
                if emb_dim is not None and dims[k] != emb_dim:
                    dim_ok = False
                    failure_reason = f"dim mismatch: emb {emb_dim} vs ix {dims[k]}"
                else:
                    ix[k] = rd
                    ok_dense = True
            except Exception as e:
                dim_ok = False
                failure_reason = f"index load error: {type(e).__name__}"

            db_path = p / "meta.sqlite"
            db_exists = db_path.exists()
            if db_exists:
                dbp[k] = db_path

            if ok_dense or db_exists:
                loaded[k] = p

            rep["dense_loaded"] = ok_dense
            rep["db_loaded"] = db_exists
            rep["dim_ok"] = dim_ok
            rep["embed_dim"] = emb_dim

        # annotate failure_reason for non-ready corpora
        ready_condition = all(
            [
                rep.get("exists"),
                rep.get("faiss"),
                rep.get("db"),
                rep.get("manifest"),
                rep.get("dense_loaded"),
                rep.get("db_loaded"),
                rep.get("dim_ok"),
            ]
        )
        if ready_condition:
            rep["failure_reason"] = None
        else:
            if not rep.get("exists"):
                failure_reason = "corpus folder missing"
            elif rep.get("missing"):
                failure_reason = f"missing: {', '.join(rep['missing'])}"
            elif failure_reason is None and (not rep.get("dense_loaded") or not rep.get("dim_ok")):
                if rep.get("ix_dim") and emb_dim and rep.get("ix_dim") != emb_dim:
                    failure_reason = f"dim mismatch: emb {emb_dim} vs ix {rep.get('ix_dim')}"
                elif failure_reason is None:
                    failure_reason = "dense index unavailable"
            elif failure_reason is None and not rep.get("db_loaded"):
                failure_reason = "metadata db unavailable"
            rep["failure_reason"] = failure_reason
        rep["ready"] = ready_condition
        rep["ok"] = ready_condition
        reasons = list(rep.get("reasons", []))
        if failure_reason and failure_reason not in reasons:
            reasons.append(failure_reason)
        rep["reasons"] = reasons

        status[k] = {
            "publisher": k,
            "ready": ready_condition,
            "ok": ready_condition,
            "loaded": bool(rep.get("dense_loaded") and rep.get("db_loaded") and rep.get("dim_ok")),
            "exists": bool(rep.get("exists")),
            "manifest": bool(rep.get("manifest")),
            "dense_loaded": bool(rep.get("dense_loaded")),
            "db_loaded": bool(rep.get("db_loaded")),
            "dim_ok": bool(rep.get("dim_ok")),
            "missing": list(rep.get("missing", [])),
            "reasons": reasons,
            "failure_reason": rep.get("failure_reason"),
        }
        reports[k] = rep

    return Eng(emb=emb, ix=ix, dbp=dbp, corp=loaded, ix_dim=dims, corp_report=reports, corp_status=status)


def _db(con_p: Path) -> sqlite3.Connection:
    # Streamlit can run multiple threads; allow cross-thread connections.
    return sqlite3.connect(str(con_p), check_same_thread=False)


def embed_query(e: Eng, q: str) -> np.ndarray:
    if e.emb is None:
        return np.array([], dtype="float32")
    try:
        return np.asarray(e.emb.encode(q, convert_to_numpy=True), dtype="float32")
    except Exception:
        return np.array([], dtype="float32")


def _normalize_query(qv: np.ndarray) -> np.ndarray:
    try:
        v = np.asarray(qv, dtype="float32")
        faiss.normalize_L2(v.reshape(1, -1))
        return v
    except Exception:
        return qv


def faiss_search(e: Eng, corp: str, qv: np.ndarray, k: int):
    meta = {
        "fallback_used": 0,
        "fallback_failed": 0,
        "fallback_retries": 0,
        "clamped_k": False,
        "k_requested": k,
    }
    if qv.size == 0:
        return [], meta
    if corp not in e.ix or corp not in e.dbp:
        return [], meta
    ix = e.ix[corp]
    try:
        k_applied = max(1, min(int(k), HCFG["faiss_fetch_k"]))
    except Exception:
        k_applied = max(1, HCFG["faiss_fetch_k"])
    meta["k_applied"] = k_applied
    meta["k_clamped"] = meta["k_applied"] != meta["k_requested"]
    try:
        if qv.dtype != np.float32:
            qv = np.asarray(qv, dtype="float32")
        else:
            qv = np.asarray(qv)
    except Exception:
        return [], meta
    try:
        if qv.shape[0] != ix.d:
            return [], meta
    except Exception:
        return [], meta
    metric_type = None
    try:
        metric_type = getattr(ix, "metric_type", None)
        meta["metric_type"] = metric_type
    except Exception:
        metric_type = None
    try:
        if metric_type == getattr(faiss, "METRIC_INNER_PRODUCT", None):
            qv = qv.reshape(1, -1).copy()
            faiss.normalize_L2(qv)
        qv_search = qv.reshape(1, -1)
    except Exception:
        return [], meta
    try:
        D, I = ix.search(qv_search, k_applied)
    except Exception:
        return [], meta
    ids = I[0].tolist()
    scs = D[0].tolist()

    con = _db(e.dbp[corp])
    out = []
    try:
        pairs = []
        seen_ids = set()
        for s, i64 in zip(scs, ids):
            if i64 == -1 or float(s) < HCFG["min_faiss_score"]:
                continue
            if int(i64) in seen_ids:
                continue
            seen_ids.add(int(i64))
            pairs.append((float(s), int(i64)))
        if not pairs:
            return [], meta
        wanted_ids = [i for _, i in pairs]
        ph = ",".join(["?"] * len(wanted_ids))
        cur = con.cursor()
        missing_ids = []
        try:
            rows = cur.execute(f"SELECT cid, fp, sec, cidx, tx, i64 FROM chunks WHERE i64 IN ({ph})", wanted_ids).fetchall()
            i64_to_row = {int(r[5]): r for r in rows}
            missing_ids = [i for i in wanted_ids if int(i) not in i64_to_row]
        except Exception:
            i64_to_row = {}
            missing_ids = wanted_ids

        fallback_budget = max(0, int(HCFG.get("fallback_retry_max", FAISS_FALLBACK_RETRY_MAX)))
        for i64 in missing_ids[:fallback_budget]:
            meta["fallback_retries"] += 1
            row = get_by_i64(con, int(i64))
            if row:
                cid, fp, sec, cidx, tx = row
                i64_to_row[int(i64)] = (cid, fp, sec, cidx, tx, i64)
            else:
                meta["fallback_failed"] += 1
        if len(missing_ids) > fallback_budget:
            meta["fallback_failed"] += len(missing_ids) - fallback_budget
        for s, i64 in pairs:
            row = i64_to_row.get(int(i64))
            if not row:
                continue
            cid, fp, sec, cidx, tx, _ = row
            tx = _cap_tx(tx)
            out.append(
                {
                    "corp": corp,
                    "cid": cid,
                    "fp": fp,
                    "sec": sec,
                    "cidx": int(cidx),
                    "tx": _cap_tx(tx),
                    # aliases
                    "section": sec,
                    "book": Path(fp).stem if fp else None,
                    "publisher": corp,
                    "text": _cap_tx(tx),
                    "sem_score": float(s),
                }
            )
    finally:
        con.close()
    return out, meta


def fts_search(e: Eng, corp: str, q: str, k: int):
    con = _db(e.dbp[corp])
    try:
        if not fts_ready(con):
            # If chunks_fts doesn't exist, return empty.
            return []
        cur = con.cursor()
        qq = fts_query_escape(q)
        if not qq:
            return []
        k = max(1, min(int(k), HCFG["fts_fetch_k"]))

        rows = cur.execute(
            """
            SELECT cid, fp, sec, tx, bm25(chunks_fts) as b
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY b
            LIMIT ?
            """,
            (qq, k),
        ).fetchall()

        out = []
        # bm25 smaller is better; convert to a "lex_score" where higher is better
        for cid, fp, sec, tx, b in rows:
            out.append(
                {
                    "corp": corp,
                    "cid": cid,
                    "fp": fp,
                    "sec": sec,
                    "cidx": -1,
                    "tx": _cap_tx(tx),
                    # aliases
                    "section": sec,
                    "book": Path(fp).stem if fp else None,
                    "publisher": corp,
                    "text": _cap_tx(tx),
                    "lex_score": float(-b),
                }
            )
        return out
    finally:
        con.close()


def dense_retrieve(e: Eng, corp: str, qv: np.ndarray, k: int | None = None):
    k_req = k or HCFG["faiss_fetch_k"]
    if corp not in e.ix:
        return [], {"fallback_retries": 0, "fallback_failed": 0, "k_requested": k_req, "k_applied": k_req, "k_clamped": False}
    if qv.size == 0:
        return [], {"fallback_retries": 0, "fallback_failed": 0, "k_requested": k_req, "k_applied": k_req, "k_clamped": False}
    rows, meta = faiss_search(e, corp, qv, k_req)
    norm_scores(rows, "sem_score")
    for r in rows:
        r.setdefault("score", r.get("sem_score_n", 0.0))
    return rows, meta


def lex_retrieve(e: Eng, corp: str, q: str, k: int | None = None):
    if corp not in e.dbp:
        return [], {}
    clamp_flag = False
    if k is not None and k > HCFG["fts_fetch_k"]:
        clamp_flag = True
    k_use = k or HCFG["fts_fetch_k"]
    if k_use > HCFG["fts_fetch_k"]:
        clamp_flag = True
        k_use = HCFG["fts_fetch_k"]
    rows = fts_search(e, corp, q, k_use)
    norm_scores(rows, "lex_score")
    for r in rows:
        r.setdefault("score", r.get("lex_score_n", 0.0))
    return rows, {"clamped_k": clamp_flag}


def hybrid_retrieve(
    e: Eng,
    q: str,
    k: int = HCFG["final_k"],
    pubs=None,
    qv: np.ndarray | None = None,
    mmr_k: Optional[int] = None,
    dense_k: Optional[int] = None,
    lex_k: Optional[int] = None,
):
    """Dense+lexical hybrid retrieval across selected publishers."""
    if qv is None:
        qv = embed_query(e, q)
    use_dense = qv.size > 0
    try:
        k_requested = int(k)
    except Exception:
        k_requested = HCFG["final_k"]
    mmr_cap = int(mmr_k) if mmr_k is not None else HCFG["mmr_k"]
    k_applied = max(1, min(k_requested, mmr_cap))

    pubs = pubs or list(e.corp.keys())
    cands = []
    meta = {
        "dense_hits": 0,
        "lex_hits": 0,
        "pubs_used": 0,
        "t_dense": 0.0,
        "t_lex": 0.0,
        "fetched_dense": 0,
        "fetched_lex": 0,
        "k_requested": k_requested,
        "k_applied": k_applied,
        "k_clamped": k_applied != k_requested,
        "mmr_cap": mmr_cap,
        "fallback_retries": 0,
        "fallback_failed": 0,
        "dense_k": None,
        "lex_k": None,
        "dense_fallback": 0,
        "dense_fallback_fail": 0,
        "dense_clamped": False,
        "lex_clamped": False,
    }

    for corp in pubs:
        if corp not in e.corp:
            continue

        t_d0 = _t0()
        if use_dense:
            dense_req = dense_k or HCFG["dense_k"]
            d, d_meta = dense_retrieve(e, corp, qv, k=dense_req)
            meta["dense_k"] = dense_req
            meta["fallback_retries"] += d_meta.get("fallback_retries", 0)
            meta["fallback_failed"] += d_meta.get("fallback_failed", 0)
            meta["k_clamped"] = meta["k_clamped"] or bool(d_meta.get("k_clamped"))
        else:
            d, d_meta = [], {"k_clamped": False}
        meta["t_dense"] += _dt(t_d0)
        t_l0 = _t0()
        lex_req = lex_k or HCFG["lex_k"]
        l, lmeta = lex_retrieve(e, corp, q, k=lex_req)
        meta["lex_k"] = lex_req
        meta["t_lex"] += _dt(t_l0)

        meta["dense_hits"] += len(d)
        meta["lex_hits"] += len(l)
        meta["fetched_dense"] += len(d)
        meta["fetched_lex"] += len(l)
        meta["pubs_used"] += 1
        meta["dense_fallback"] += d_meta.get("fallback_used", 0)
        meta["dense_fallback_fail"] += d_meta.get("fallback_failed", 0)
        meta["dense_clamped"] = meta["dense_clamped"] or d_meta.get("clamped_k", False)
        meta["lex_clamped"] = meta["lex_clamped"] or lmeta.get("clamped_k", False)

        dd = {x["cid"]: x for x in d}
        ll = {x["cid"]: x for x in l}
        cids = set(dd) | set(ll)

        for cid in cids:
            a = dd.get(cid, {})
            b = ll.get(cid, {})
            row = {
                "cid": cid,
                "cidx": a.get("cidx") if "cidx" in a else b.get("cidx"),
                "fp": a.get("fp") or b.get("fp"),
                "tx": a.get("tx") or b.get("tx"),
                "sec": a.get("sec") or b.get("sec"),
                "corp": a.get("corp") or b.get("corp") or corp,
                # aliases for UI convenience
                "text": a.get("text") or b.get("text") or a.get("tx") or b.get("tx"),
                "section": a.get("section") or b.get("section") or a.get("sec") or b.get("sec"),
                "book": a.get("book") or b.get("book") or (Path(a.get("fp") or b.get("fp")).stem if (a.get("fp") or b.get("fp")) else None),
                "publisher": a.get("publisher") or b.get("publisher") or corp,
            }

            ss = float(a.get("sem_score_n", 0.0))
            ls = float(b.get("lex_score_n", 0.0))
            row["sem_score_n"] = ss
            row["lex_score_n"] = ls
            row["score"] = HCFG["fusion_dense_w"] * ss + HCFG["fusion_lex_w"] * ls
            cands.append(row)

    cands.sort(key=lambda z: float(z.get("score", 0.0)), reverse=True)

    # lightweight de-dupe and diversity budget per publisher
    seen = set()
    out = []
    per_pub = {}
    div_cap = max(2, int(k_applied))
    for h in cands:
        key = (h.get("book"), h.get("sec"))
        key_fb = (h.get("fp"), h.get("sec"))
        text_sig = hashlib.sha1(((h.get("text") or h.get("tx") or "")[:200]).encode("utf-8", "ignore")).hexdigest()
        sigs = [key, key_fb, text_sig]
        if any(s in seen for s in sigs):
            continue
        pub = h.get("corp") or h.get("publisher")
        per_pub.setdefault(pub, 0)
        if per_pub[pub] >= max(2, k // 2):
            continue
        per_pub[pub] += 1
        for s in sigs:
            seen.add(s)
        out.append(h)
        if len(out) >= k_applied:
            break

    meta["cands"] = len(cands)
    return out[:k_applied], meta


def _sig(x: float) -> float:
    try:
        return 1 / (1 + math.exp(-x))
    except Exception:
        return 0.0


def _t0():
    return time.time()


def _dt(t0):
    try:
        return time.time() - float(t0)
    except Exception:
        return 0.0


def _safe_msg(ex, max_len: int = 200) -> str:
    msg = f"{type(ex).__name__}: {ex}"
    msg = msg.replace("\n", " ")[:max_len]
    return msg


def _err_id(seed: Optional[str] = None) -> str:
    basis = seed if seed is not None else "err-seed"
    try:
        h = hashlib.sha1(str(basis).encode("utf-8", "ignore")).hexdigest()[:10]
    except Exception:
        h = f"{int(time.time() * 1000)}"
    return f"err-{h}"


def _blank_meta():
    cap = {k: False for k in META_CAP_KEYS}
    cap["judge_kind"] = "none"
    cap["corp_available"] = []
    cap["dense_reason"] = None
    cap["k_requested"] = None
    cap["k_applied"] = None
    cap["k_clamped"] = False
    flags = {k: False for k in META_FLAG_KEYS}
    flags["judge_proxy"] = False
    return {
        "t": {k: 0.0 for k in META_T_KEYS},
        "n": {k: 0 for k in META_N_KEYS},
        "cap": cap,
        "flags": flags,
        "err": None,
        "err_llm": None,
        "clamp": {},
        "log": {},
    }


def get_startup_report(eng: Eng) -> Dict[str, Any]:
    rep = getattr(eng, "corp_report", {}) or {}
    rows = []
    ok = []
    fail = []
    by_corpus = {}
    for pub in CORP.keys():
        r = rep.get(pub, {})
        ready = all(
            [
                r.get("exists"),
                r.get("faiss"),
                r.get("db"),
                r.get("manifest"),
                r.get("dense_loaded"),
                r.get("db_loaded"),
                r.get("dim_ok"),
            ]
        )
        reason = "" if ready else (r.get("failure_reason") or "unavailable")
        row = {
            "publisher": pub,
            "ok": ready,
            "loaded": bool(r.get("dense_loaded") and r.get("db_loaded") and r.get("dim_ok")),
            "loaded_dense": bool(r.get("dense_loaded")),
            "loaded_db": bool(r.get("db_loaded")),
            "ready": bool(ready),
            "reason": reason,
            "exists": bool(r.get("exists")),
            "manifest": bool(r.get("manifest")),
            "dim_ok": bool(r.get("dim_ok")),
            "missing": list(r.get("missing", [])),
            "reasons": list(r.get("reasons", [])),
        }
        rows.append(row)
        by_corpus[pub] = row
        if ready:
            ok.append(pub)
        else:
            fail.append(pub)
    return {"rows": rows, "ok": ok, "fail": fail, "by_corpus": by_corpus}


def _mk_ret(ok: bool = False, no_ev: bool = True, hits=None, nm_hits=None, cov: str = "WEAK", ans: str = "", meta=None):
    meta = meta or _blank_meta()
    return {
        "ok": bool(ok),
        "no_evidence": bool(no_ev),
        "answer": ans if ans is not None else "",
        "confidence": meta.get("conf") if isinstance(meta, dict) else None,
        "hits": list(hits or []),
        "near_miss": list(nm_hits or []),
        "coverage": cov if cov else "WEAK",
        "meta": meta,
    }


def _cut(hs, k=K_SHOW, mnk=MNK, abs_min: float = ABS_MN):
    if not hs:
        return hs, {"kept": 0, "all": 0, "rule": "empty"}
    hs = sorted(hs, key=lambda x: (-float(x.get("score", 0.0)), str(x.get("fp", "")), str(x.get("sec", "")), str(x.get("cid", ""))))
    tp = hs[: max(k, mnk)]
    out = []
    for h in tp:
        if float(h.get("score", 0.0)) >= abs_min or len(out) < mnk:
            out.append(h)
    return out, {
        "kept": len(out),
        "all": len(tp),
        "rule": f"top_k_with_min_keep_abs_min={abs_min}",
    }


@lru_cache(maxsize=1)
def _get_jdg():
    """Load the real cross-encoder judge if available.

    The model path can be overridden via ``RAG_JUDGE_MODEL``. Failures are
    tolerated to keep the system usable in CPU-only environments.
    """

    model_name = os.environ.get("RAG_JUDGE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    try:
        return CrossEncoder(model_name)
    except Exception:
        return None


_JDG_CACHE: Dict[Tuple[str, str], Dict[str, float]] = {}
_JDG_CACHE_MAX = int(HCFG.get("jdg_cache_size", 256))
_JDG_CACHE_TTL = float(HCFG.get("jdg_cache_ttl", 300.0))  # seconds


def _chunk_hash(h: Dict[str, Any]) -> str:
    raw = h.get("chunk_hash")
    if raw:
        return str(raw)
    txt = (h.get("text") or h.get("tx") or "")[:800]
    payload = "|".join(
        [str(h.get("cid") or ""), str(h.get("fp") or ""), str(h.get("cidx") or ""), hashlib.sha1(txt.encode("utf-8", "ignore")).hexdigest()]
    )
    digest = hashlib.sha1(payload.encode("utf-8", "ignore")).hexdigest()
    h["chunk_hash"] = digest
    return digest


def _jdg_cache_prune(now: Optional[float] = None):
    now = now or time.time()
    expire_keys = [k for k, v in _JDG_CACHE.items() if now - v.get("ts", 0.0) > _JDG_CACHE_TTL]
    for k in expire_keys:
        _JDG_CACHE.pop(k, None)
    if len(_JDG_CACHE) > _JDG_CACHE_MAX:
        # drop oldest entries first
        over = max(0, len(_JDG_CACHE) - _JDG_CACHE_MAX)
        for k, _ in sorted(_JDG_CACHE.items(), key=lambda kv: kv[1].get("ts", 0.0))[:over]:
            _JDG_CACHE.pop(k, None)


def _jdg_cache_get(key: Tuple[str, str]) -> Optional[float]:
    now = time.time()
    entry = _JDG_CACHE.get(key)
    if not entry:
        return None
    ts = entry.get("ts", 0.0)
    if now - ts > _JDG_CACHE_TTL:
        _JDG_CACHE.pop(key, None)
        return None
    return entry.get("score")


def _jdg_cache_set(key: Tuple[str, str], score: float):
    _JDG_CACHE[key] = {"score": float(score), "ts": time.time()}
    _jdg_cache_prune()


def _jdg_rerank(q, hs, mode: str = "proxy"):
    mode = (mode or "proxy").lower()
    if mode not in {"real", "proxy", "off"}:
        mode = "proxy"

    def _proxy(label: str, proxy_flag: bool):
        for h in hs:
            js = float(h.get("score", 0.0))
            h["_jdg"] = js
            h["judge01"] = js
        hs.sort(key=lambda x: float(x.get("judge01", 0.0)), reverse=True)
        return hs, {
            "ok": False,
            "t": 0.0,
            "n": len(hs),
            "kind": label,
            "cache_hits": 0,
            "cache_misses": len(hs),
            "t_cache": 0.0,
            "t_pred": 0.0,
            "proxy": proxy_flag,
        }

    if mode == "off":
        return _proxy("off", False)

    use_real = mode == "real"
    j = _get_jdg() if use_real else None
    if j is None:
        return _proxy("proxy_score", True)

    tp = hs[: min(K_JDG, len(hs))]
    pairs = [(q, (h.get("text") or "")[:1200]) for h in tp]
    t0 = time.time()
    sc = []
    cache_hits = 0
    cache_misses = 0
    t_cache = 0.0
    t_pred = 0.0
    _jdg_cache_prune()
    try:
        for h, pr in zip(tp, pairs):
            t_loop = time.time()
            key = (pr[0], _chunk_hash(h))
            cached = _jdg_cache_get(key)
            if cached is not None:
                sc.append(float(cached))
                cache_hits += 1
                t_cache += time.time() - t_loop
                continue
            cache_misses += 1
            t_pred0 = time.time()
            scr = float(j.predict([pr])[0])
            t_pred += time.time() - t_pred0
            sc.append(scr)
            _jdg_cache_set(key, scr)
    except Exception:
        return _proxy("proxy_score", True)
    t1 = time.time()
    for h, s in zip(tp, sc):
        h["_jdg"] = float(s)
        h["judge01"] = _sig(h["_jdg"])
    tp.sort(key=lambda x: float(x.get("_jdg", -1e9)), reverse=True)
    meta = {
        "ok": True,
        "t": t1 - t0,
        "n": len(tp),
        "kind": "cross_encoder",
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "t_cache": t_cache,
        "t_pred": t_pred,
        "proxy": False,
    }
    return tp + hs[len(tp) :], meta


def _disp_flt(hs, min_keep=MNK, jmin=J_DISP_MIN, use_jdg=USE_JDG_DEFAULT):
    if not use_jdg:
        return hs, {"kept": len(hs), "all": len(hs), "rule": "disp:none"}
    a, b = [], []
    for h in hs:
        j01 = h.get("judge01", None)
        if j01 is not None and float(j01) >= jmin:
            a.append(h)
        else:
            b.append(h)
    out = a
    if len(out) < min_keep:
        out = out + b[: (min_keep - len(out))]
    return out, {"kept": len(out), "all": len(hs), "rule": f"disp:judge01>={jmin} (min_keep={min_keep})"}


def _ov_ok(q, h, qs=None):
    txt = (h or {}).get("tx") or (h or {}).get("text") or ""
    qs = qs if qs is not None else set([w.lower() for w in re.findall(r"[A-Za-z0-9]+", q) if len(w) >= 3])
    hs = h.get("_tok")
    if hs is None:
        hs = set([w.lower() for w in re.findall(r"[A-Za-z0-9]+", txt) if len(w) >= 3])
        h["_tok"] = hs
    inter = qs & hs
    return (len(inter) >= 1), {"overlap": len(inter), "qs": len(qs), "hs": len(hs)}


def _direct(hs, q, qs=None, use_jdg=USE_JDG_DEFAULT):
    out = []
    for h in hs:
        ok, ov_meta = _ov_ok(q, h, qs=qs)
        if not ok:
            continue
        h["overlap"] = ov_meta.get("overlap", 0)
        if use_jdg:
            j01 = h.get("judge01", None)
            if j01 is None:
                continue
            if float(j01) >= J_STRONG_MIN:
                out.append(h)
        else:
            if float(h.get("score", 0.0)) >= ABS_MN:
                out.append(h)
    return out


def _noev_jdg(hs):
    js = [float(h.get("judge01")) for h in hs if h.get("judge01") is not None]
    if not js:
        return False, {"mx": None, "mn": None, "cnt": 0}
    jmx = max(js)
    jmn = sum(js) / len(js)
    jcnt = sum(1 for v in js if v >= J_STRONG_MIN)
    veto = (jmx < J_MIN_KEEP) and (jmn < J_WEAK_MIN) and (jcnt < J_MIN_CNT)
    return veto, {"mx": jmx, "mn": jmn, "cnt": jcnt}


def _near_miss(hs, q, qs=None, use_jdg=USE_JDG_DEFAULT):
    c = []
    for h in hs:
        ok, ov_meta = _ov_ok(q, h, qs=qs)
        if not ok:
            continue
        h["overlap"] = ov_meta.get("overlap", 0)
        if use_jdg:
            j01 = h.get("judge01", None)
            if j01 is None:
                continue
            if float(j01) >= NM_MIN:
                c.append(h)
        else:
            if float(h.get("score", 0.0)) >= ABS_MN:
                c.append(h)
    if use_jdg:
        c.sort(key=lambda x: float(x.get("_jdg", -1e9)), reverse=True)
    else:
        c.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return c[:NM_MAX], {"threshold": NM_MIN, "used_judge": bool(use_jdg)}


def _nm_with_metadata(hs, nm_meta, explanation: Optional[str] = None):
    thr = nm_meta.get("threshold", NM_MIN) if isinstance(nm_meta, dict) else NM_MIN
    used_judge = bool(nm_meta.get("used_judge")) if isinstance(nm_meta, dict) else False
    out = []
    for h in hs or []:
        h2 = dict(h)
        h2["near_miss_threshold"] = thr
        h2["used_judge"] = used_judge
        if explanation:
            h2.setdefault("explanation", explanation)
        out.append(h2)
    return out


def _ensure_nm_candidates(nm_hits, hs_pool, q, qs=None, nm_meta=None, min_k: int = 3, max_k: int = NM_MAX):
    out = list(nm_hits or [])
    seen = {(h.get("cid"), h.get("cidx"), h.get("fp")) for h in out}
    qs = qs if qs is not None else set([w.lower() for w in re.findall(r"[A-Za-z0-9]+", q) if len(w) >= 3])
    extras = []
    for h in hs_pool or []:
        key = (h.get("cid"), h.get("cidx"), h.get("fp"))
        if key in seen:
            continue
        ok, ov_meta = _ov_ok(q, h, qs=qs)
        if not ok:
            continue
        h2 = dict(h)
        h2["overlap"] = ov_meta.get("overlap", 0)
        extras.append(h2)
    extras.sort(key=lambda x: (-float(x.get("overlap", 0)), -float(x.get("judge01") or x.get("score", 0.0))))
    for h in extras:
        if len(out) >= max_k:
            break
        out.append(h)
        seen.add((h.get("cid"), h.get("cidx"), h.get("fp")))
    if len(out) < min_k:
        for h in hs_pool or []:
            if len(out) >= max_k or len(out) >= min_k:
                break
            key = (h.get("cid"), h.get("cidx"), h.get("fp"))
            if key in seen:
                continue
            h2 = dict(h)
            h2.setdefault("overlap", 0)
            out.append(h2)
            seen.add(key)
    return out[:max_k]


def _strip_internal(hs):
    out = []
    for h in hs:
        if not isinstance(h, dict):
            continue
        h2 = dict(h)
        h2.pop("_tok", None)
        h2.pop("_jdg", None)
        out.append(h2)
    return out


def _pub_hit(h):
    if not isinstance(h, dict):
        return None
    h2 = dict(h)
    h2.pop("_tok", None)
    h2.pop("_jdg", None)
    # required UI keys with sane defaults
    h2.setdefault("tx", h2.get("text") or "")
    h2.setdefault("sec", h2.get("section") or "")
    fp = h2.get("file") or h2.get("fp") or ""
    h2["fp"] = fp
    h2.setdefault("file", fp)
    h2.setdefault("src", fp)
    h2.setdefault("cid", h2.get("cid") or "")
    h2.setdefault("cidx", h2.get("cidx") or 0)
    h2.setdefault("score", float(h2.get("score", 0.0)))
    h2.setdefault("judge01", h2.get("judge01", None))
    h2["_jdg01"] = h2.get("judge01", None)
    h2.setdefault("corp", h2.get("corp") or h2.get("publisher") or "")
    h2.setdefault("publisher", h2.get("publisher") or h2.get("corp") or "")
    h2.setdefault("book", h2.get("book") or (Path(fp).stem if fp else ""))
    return h2


def _pub_hits(hs):
    out = []
    for h in hs or []:
        ph = _pub_hit(h)
        if ph is not None:
            out.append(ph)
    return out


def _cov(dr, topn=8):
    jss = [float(h.get("judge01", 0.0)) for h in dr[:topn]]
    if not jss:
        return {"mx": 0.0, "mn": 0.0, "std": 0.0, "uc": 0}
    mx = max(jss)
    mn = min(jss)
    std = pstdev(jss) if len(jss) > 1 else 0.0
    uc = len([x for x in jss if x >= J_STRONG_MIN])
    return {"mx": mx, "mn": mn, "std": std, "uc": uc}


def coverage_label(dr, q):
    # helper for UI
    if len(dr) < DIRECT_MIN:
        return "WEAK"
    cv = _cov(dr, topn=8)
    if cv["mx"] >= 0.80 and cv["std"] < 0.06:
        return "HIGH"
    if cv["uc"] >= 2 and cv["std"] < 0.12:
        return "DISTRIBUTED"
    return "OK"


def _calc_confidence(dr):
    if not dr:
        return 0.1
    jss = [float(h.get("judge01", 0.0)) for h in dr[:4]]
    base = sum(jss) / max(1, len(jss))
    return max(0.0, min(1.0, base))


def _assemble_context(dr, budget_chars: int = 1400, budget_tokens: int = 380) -> tuple[str, Dict[str, bool]]:
    if not dr:
        return "", {"char_clamped": False, "token_clamped": False}
    # deterministic by judge01/score
    dr_sorted = sorted(dr, key=lambda h: (-float(h.get("judge01", 0.0)), -float(h.get("score", 0.0)), str(h.get("cid", ""))))
    out = []
    used = 0
    char_clamped = False
    for h in dr_sorted:
        tx = (h.get("tx") or h.get("text") or "")[:800]
        if not tx:
            continue
        next_len = used + len(tx) + (2 if out else 0)
        if next_len > budget_chars:
            # clamp final piece to remaining budget
            remaining = max(0, budget_chars - used)
            if remaining <= 0:
                char_clamped = True
                break
            tx = tx[:remaining]
            char_clamped = True
        out.append(tx)
        used = min(budget_chars, next_len)
        if used >= budget_chars:
            char_clamped = True
            break
    ctx = "\n\n".join(out)
    # soft token budget (approx by whitespace tokens)
    ctx, tok_meta = _clamp_text(ctx, budget_chars, budget_tokens, CTX_CLAMP_MARKER)
    return ctx, {"char_clamped": char_clamped or tok_meta.get("char_clamped"), "token_clamped": tok_meta.get("token_clamped")}


def llm_call(prompt: str, cfg: Optional[dict] = None) -> str:
    # placeholder CPU-friendly stub; can be replaced with real model later
    if not prompt:
        return ""
    cfg = cfg or {}
    mode = (cfg.get("mode") or "quick") if isinstance(cfg, dict) else "quick"
    b = _budget_for_mode(mode, {})
    char_budget = int(cfg.get("char_budget", b.get("prompt_chars", 900)))
    tok_budget = int(cfg.get("tok_budget", b.get("prompt_tokens", 220)))
    p, _ = _clamp_text(prompt, char_budget, tok_budget, LLM_CLAMP_MARKER)
    return p


def get_reader_chunk(e: Eng, fp: str, cidx: int, window: int = 2):
    """Fetch context window for reading mode."""
    try:
        chunks = []
        for corp, dbp in getattr(e, "dbp", {}).items():
            try:
                con = _db(dbp)
                cur = con.cursor()
                rows = cur.execute(
                    """
                    SELECT cid, fp, sec, cidx, tx FROM chunks
                    WHERE fp=? AND cidx BETWEEN ? AND ?
                    ORDER BY cidx
                    """,
                    (fp, int(cidx) - int(window), int(cidx) + int(window)),
                ).fetchall()
                con.close()
                if rows:
                    chunks = [
                        {
                            "cid": r[0],
                            "fp": r[1],
                            "sec": r[2],
                            "cidx": int(r[3]),
                            "tx": (r[4] or "")[: HCFG["max_snippet_chars"]],
                            "corp": corp,
                        }
                        for r in rows
                    ]
                    break
            except Exception:
                continue
        return {"ok": bool(chunks), "chunks": chunks, "err": None if chunks else "not_found"}
    except Exception as ex:
        return {"ok": False, "chunks": [], "err": _safe_msg(ex)}


def _log_event(meta, mode, pubs_requested, qlen, hits=None):
    _config_logger()
    try:
        base = dict(meta.get("log", {}) or {})
        scope_used = _scope_from_hits(hits)
        payload = {
            "ts": time.time(),
            "mode": mode,
            "mode_cfg": meta.get("mode_cfg", {}),
            "scope": {"requested": list(pubs_requested or []), "used": scope_used},
            "qlen": int(qlen or 0),
            "counts": dict(meta.get("n", {})),
            "flags": dict(meta.get("flags", {})),
            "judge_ok": meta.get("cap", {}).get("judge_ok", False),
            "judge_kind": meta.get("cap", {}).get("judge_kind", "none"),
            "judge_mode": base.get("judge_mode"),
            "no_evidence": meta.get("err") is None and meta.get("n", {}).get("direct_hits", 0) == 0,
            "durations": dict(meta.get("t", {})),
            "error_id": (meta.get("err") or {}).get("id"),
            "llm_err": meta.get("err_llm"),
            "llm_dur": meta.get("t", {}).get("llm"),
            "clamp": dict(meta.get("clamp", {})),
            "judge_cache_hits": base.get("judge_cache_hits", 0),
            "judge_cache_misses": base.get("judge_cache_misses", 0),
            "judge": {
                "ok": meta.get("cap", {}).get("judge_ok", False),
                "kind": meta.get("cap", {}).get("judge_kind", "none"),
                "mode": base.get("judge_mode", "none"),
                "cache": {
                    "hits": int(base.get("judge_cache_hits", 0)),
                    "misses": int(base.get("judge_cache_misses", 0)),
                },
                "flags": {
                    "veto_applied": meta.get("flags", {}).get("veto_applied", False),
                    "veto_disabled": meta.get("flags", {}).get("veto_disabled", False),
                    "veto_disabled_when_proxy": meta.get("flags", {}).get("veto_disabled_when_proxy", False),
                },
            },
        }
        meta["log"] = payload
        try:
            logger.info(json.dumps(payload))
        except Exception:
            pass
    except Exception:
        pass


def _record_recent_query(q: str, pubs: List[str] | None = None, limit: int = 12):
    q = (q or "").strip()
    if not q:
        return
    try:
        RECENT_QUERY_LOG.parent.mkdir(parents=True, exist_ok=True)
        recs = []
        if RECENT_QUERY_LOG.exists():
            try:
                with RECENT_QUERY_LOG.open("r", encoding="utf-8") as f:
                    recs = json.load(f) or []
            except Exception:
                recs = []
        pubs_norm = sorted(set(pubs or []))
        now = time.time()
        recs.append({"q": q, "ts": now, "pubs": pubs_norm})
        # dedupe by query text, keep most recent first
        seen = set()
        deduped = []
        for r in sorted(recs, key=lambda x: float(x.get("ts", 0.0)), reverse=True):
            key = (r.get("q") or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append({"q": key, "ts": float(r.get("ts", now)), "pubs": r.get("pubs", [])})
            if len(deduped) >= limit:
                break
        with RECENT_QUERY_LOG.open("w", encoding="utf-8") as f:
            json.dump(deduped, f)
    except Exception:
        pass


def get_recent_queries(limit: int = 5) -> List[str]:
    try:
        if not RECENT_QUERY_LOG.exists():
            return []
        with RECENT_QUERY_LOG.open("r", encoding="utf-8") as f:
            recs = json.load(f) or []
        recs = sorted(recs, key=lambda x: float((x or {}).get("ts", 0.0)), reverse=True)
        out = []
        for r in recs:
            q = (r or {}).get("q")
            if not q:
                continue
            out.append(str(q))
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []


def run_query(
    e: Eng,
    q: str,
    pubs=None,
    sort: str = "Best evidence",
    nm: bool | None = None,
    show_nm: bool | None = None,
    compute_near_miss: bool | None = None,
    jmin: float | None = None,
    use_jdg: bool | None = None,
    jdg_mode: str | None = None,
    judge_mode: str | None = None,
    scope=None,
    use_llm: bool | None = None,
    use_vector_mmr: bool | None = None,
    mode: str | None = None,
    ctx_char_budget: Optional[int] = None,
    ctx_tok_budget: Optional[int] = None,
    prompt_char_budget: Optional[int] = None,
    prompt_tok_budget: Optional[int] = None,
):
    t_total = _t0()
    meta = _blank_meta()
    mode_cfg = _mode_cfg(mode)
    mode_name = mode_cfg["name"]
    meta["mode"] = mode_cfg["name"]
    meta["mode_cfg"] = mode_cfg
    meta["cap"]["mode"] = mode_cfg["name"]
    meta["cap"]["mode_label"] = mode_cfg["label"]
    meta["cap"]["mode_desc"] = mode_cfg["description"]
    meta.setdefault("log", {})["mode"] = mode_cfg["name"]
    meta["log"]["mode_label"] = mode_cfg["label"]
    meta["log"]["mode_description"] = mode_cfg["description"]
    meta["log"]["mode_cfg"] = mode_cfg
    use_jdg_flag = (mode_cfg.get("use_jdg", True) if use_jdg is None else bool(use_jdg)) and USE_JDG_DEFAULT
    jdg_mode = (judge_mode or jdg_mode or "real").lower()
    if jdg_mode not in {"real", "proxy", "off"}:
        jdg_mode = "proxy"
    if (not use_jdg_flag) or jdg_mode == "off":
        jdg_mode = "off"
        use_jdg_flag = False
    compute_nm_flag = True if compute_near_miss is None else bool(compute_near_miss)
    nm_flag = compute_nm_flag and (nm if nm is not None else (show_nm if show_nm is not None else True))
    budget_overrides = {
        "ctx_chars": ctx_char_budget,
        "ctx_tokens": ctx_tok_budget,
        "prompt_chars": prompt_char_budget,
        "prompt_tokens": prompt_tok_budget,
    }
    budget = _budget_for_mode(mode_name, {k: v for k, v in budget_overrides.items() if v is not None})
    mode_cfg["budget"] = budget
    meta["cap"]["has_emb"] = e.emb is not None
    meta["cap"]["dense_ok"] = bool(getattr(e, "ix", {}))
    if not meta["cap"]["has_emb"]:
        meta["cap"]["dense_reason"] = "no_embed_model"
    if not meta["cap"]["dense_ok"]:
        meta["cap"]["dense_reason"] = meta["cap"]["dense_reason"] or "no_dense_index"
    meta["cap"]["lex_ok"] = bool(getattr(e, "dbp", {}))
    meta["cap"]["judge_requested"] = bool(use_jdg_flag)
    meta["cap"]["judge_ok"] = False
    meta["cap"]["judge_kind"] = "none"
    meta["cap"]["corp_available"] = list(getattr(e, "corp", {}).keys())
    meta.setdefault("log", {})["judge_mode"] = jdg_mode

    try:
        q_clean = (q or "").strip()
        if not q_clean:
            msg = "Query is empty. Please enter a question."
            err_id = _err_id("empty_query")
            meta["err"] = {"where": "run_query", "msg": msg, "id": err_id, "error_id": err_id}
            meta["t"]["total"] = _dt(t_total)
            meta["flags"]["llm_bypassed"] = True
            meta["flags"]["llm_used"] = False
            meta["err_llm"] = None
            _log_event(meta, mode_name, pubs or list(getattr(e, "corp", {}).keys()), len(q_clean or q or ""))
            return _mk_ret(ok=False, no_ev=True, hits=[], nm_hits=[], cov="WEAK", ans="", meta=meta)

        qs = set([w.lower() for w in re.findall(r"[A-Za-z0-9]+", q) if len(w) >= 3])
        meta_jdg = {"ok": False, "kind": "none"}

        corp_available = set((getattr(e, "corp", {}) or {}).keys())
        pubs_requested = set(pubs or corp_available)
        missing_corpora = sorted(pubs_requested - corp_available)
        if missing_corpora:
            msg = "Requested corpora are unavailable. Adjust publisher filters or load data."
            err_id = _err_id(f"missing_corpora:{','.join(missing_corpora)}")
            meta["err"] = {
                "where": "run_query",
                "msg": msg,
                "id": err_id,
                "error_id": err_id,
                "missing": missing_corpora,
            }
            meta["t"]["total"] = _dt(t_total)
            meta["flags"]["llm_bypassed"] = True
            meta["flags"]["llm_used"] = False
            meta["err_llm"] = None
            _log_event(meta, mode_name, pubs or list(corp_available), len(q_clean or q or ""))
            return _mk_ret(ok=False, no_ev=True, hits=[], nm_hits=[], cov="WEAK", ans=msg, meta=meta)

        if not getattr(e, "corp", None) or (not getattr(e, "ix", None) and not getattr(e, "dbp", None)):
            msg = "No corpus indexes available. Add .data/ or data/ indexes, or set RAG_DATA_ROOT."
            err_id = _err_id("no_corpus_indexes")
            meta["err"] = {"where": "run_query", "msg": msg, "id": err_id, "error_id": err_id}
            meta["t"]["total"] = _dt(t_total)
            meta["flags"]["llm_bypassed"] = True
            meta["flags"]["llm_used"] = False
            meta["err_llm"] = None
            _log_event(meta, mode_name, pubs or list(getattr(e, "corp", {}).keys()), len(q_clean or q or ""))
            return _mk_ret(
                ok=False,
                no_ev=True,
                hits=[],
                nm_hits=[],
                cov="WEAK",
                ans=msg,
                meta=meta,
            )

        _record_recent_query(q_clean, pubs=pubs_requested)

        # embed
        t_emb = _t0()
        qv = embed_query(e, q)
        meta["t"]["embed"] = _dt(t_emb)
        expected_dim = None
        if getattr(e, "ix_dim", None):
            try:
                expected_dim = next(iter(e.ix_dim.values()))
            except Exception:
                expected_dim = None
        if qv.size == 0:
            meta["cap"]["dense_ok"] = False
        if qv.size > 0 and expected_dim is not None and qv.shape[0] != expected_dim:
            # dimension mismatch: disable dense path
            meta["cap"]["dense_ok"] = False
            meta["cap"]["dense_reason"] = f"embed_dim_mismatch:{qv.shape[0]}!={expected_dim}"
            qv = np.array([], dtype="float32")

        if qv.size > 0:
            meta["flags"]["dense_used"] = True

        # retrieve
        t_ret_dense_lex = _t0()
        mmr_k_mode = mode_cfg.get("mmr_k", HCFG["mmr_k"])
        dense_k_mode = mode_cfg.get("dense_k", HCFG["dense_k"])
        lex_k_mode = mode_cfg.get("lex_k", HCFG["lex_k"])
        hs, rmeta = hybrid_retrieve(
            e,
            q,
            pubs=pubs,
            qv=qv,
            k=mmr_k_mode,
            mmr_k=mmr_k_mode,
            dense_k=dense_k_mode,
            lex_k=lex_k_mode,
        )
        meta["t"]["dense"] = rmeta.get("t_dense", 0.0)
        meta["t"]["lex"] = rmeta.get("t_lex", 0.0)
        meta["t"]["fuse"] = _dt(t_ret_dense_lex) - meta["t"]["dense"] - meta["t"]["lex"]
        meta["n"]["dense_hits"] = rmeta.get("dense_hits", 0)
        meta["n"]["lex_hits"] = rmeta.get("lex_hits", 0)
        meta["n"]["fetched_dense"] = rmeta.get("fetched_dense", 0)
        meta["n"]["fetched_lex"] = rmeta.get("fetched_lex", 0)
        meta["n"]["dense_fallback"] = rmeta.get("dense_fallback", 0)
        meta["n"]["dense_fallback_fail"] = rmeta.get("dense_fallback_fail", 0)
        meta["n"]["cands"] = rmeta.get("cands", 0)
        meta["n"]["pubs_used"] = rmeta.get("pubs_used", 0)
        meta["n"]["pubs_req"] = len(pubs or list(e.corp.keys()))
        meta["n"]["fallback_retries"] = rmeta.get("fallback_retries", 0)
        meta["n"]["fallback_failed"] = rmeta.get("fallback_failed", 0)
        meta["flags"]["lex_used"] = meta["n"]["lex_hits"] > 0 or bool(getattr(e, "dbp", {}))
        meta["cap"]["k_requested"] = rmeta.get("k_requested")
        meta["cap"]["k_applied"] = rmeta.get("k_applied")
        meta["cap"]["k_clamped"] = bool(rmeta.get("k_clamped"))
        meta["flags"]["dense_clamped"] = bool(rmeta.get("dense_clamped"))
        meta["flags"]["lex_clamped"] = bool(rmeta.get("lex_clamped"))
        meta["clamp"]["retrieval"] = {
            "k_requested": rmeta.get("k_requested"),
            "k_applied": rmeta.get("k_applied"),
            "k_clamped": rmeta.get("k_clamped"),
            "mmr_cap": rmeta.get("mmr_cap"),
            "dense_k": rmeta.get("dense_k"),
            "lex_k": rmeta.get("lex_k"),
            "dense_clamped": rmeta.get("dense_clamped"),
            "lex_clamped": rmeta.get("lex_clamped"),
        }
        hs = hs[: mode_cfg.get("final_k", HCFG["final_k"])]

        # sort preference (UI-level)
        s = (sort or "").strip().lower()
        t_sort = _t0()
        if s.startswith("semantic"):
            hs.sort(key=lambda z: float(z.get("sem_score_n", 0.0)), reverse=True)
        elif s.startswith("lex"):
            hs.sort(key=lambda z: float(z.get("lex_score_n", 0.0)), reverse=True)
        else:
            hs.sort(key=lambda z: float(z.get("score", 0.0)), reverse=True)
        meta["t"]["fuse"] += _dt(t_sort)  # include sorting in fuse

        # cutoff
        t_cut = _t0()
        hs2, cut_meta = _cut(hs, k=K_SHOW, mnk=MNK)
        meta["t"]["cut"] = _dt(t_cut)
        meta["n"]["after_cut"] = len(hs2)
        meta["cut_rule"] = cut_meta.get("rule")

        if not use_jdg_flag:
            for h in hs2:
                h.setdefault("judge01", float(h.get("score", 0.0)))
            meta["cap"]["judge_kind"] = "off"
            meta["cap"]["judge_ok"] = False
            meta["flags"]["veto_disabled"] = True
            meta["flags"]["judge_proxy"] = False
            meta["flags"]["veto_disabled_when_proxy"] = jdg_mode in {"proxy", "off"}
            meta.setdefault("log", {})["judge_cache_hits"] = 0
            meta.setdefault("log", {})["judge_cache_misses"] = 0

        # judge rerank (display order)
        hs3 = None
        disp_use_jdg = False
        t_rerank = _t0()
        if use_jdg_flag:
            hs2, meta_jdg = _jdg_rerank(q, hs2, mode=jdg_mode)
            meta["t"]["rerank"] = _dt(t_rerank)
            disp_use_jdg = bool(meta_jdg.get("ok"))
            meta["cap"]["judge_ok"] = disp_use_jdg
            meta["cap"]["judge_kind"] = meta_jdg.get("kind", jdg_mode)
            meta["t"]["judge_cache"] = float(meta_jdg.get("t_cache", 0.0))
            meta["t"]["judge_pred"] = float(meta_jdg.get("t_pred", meta_jdg.get("t", 0.0)))
            meta.setdefault("log", {})["judge_cache_hits"] = int(meta_jdg.get("cache_hits", 0))
            meta.setdefault("log", {})["judge_cache_misses"] = int(
                meta_jdg.get("cache_misses", max(0, meta_jdg.get("n", 0) - meta_jdg.get("cache_hits", 0)))
            )
            meta.setdefault("log", {})["judge_mode"] = jdg_mode
            meta["flags"]["judge_proxy"] = bool(meta_jdg.get("proxy"))
            if meta_jdg.get("kind") != "cross_encoder":
                meta["flags"]["veto_disabled"] = True
            if jdg_mode == "proxy" or meta["flags"]["judge_proxy"]:
                meta["flags"]["veto_disabled_when_proxy"] = True
            veto = False
            if disp_use_jdg:
                veto, _ = _noev_jdg(hs2)
                meta["flags"]["veto_applied"] = bool(veto)
        else:
            meta["t"]["rerank"] = _dt(t_rerank)
            meta["t"]["judge_cache"] = 0.0
            meta["t"]["judge_pred"] = 0.0
            meta.setdefault("log", {})["judge_mode"] = jdg_mode
            meta["flags"]["veto_disabled"] = True
            meta["flags"]["veto_disabled_when_proxy"] = jdg_mode in {"proxy", "off"}

        t_disp = _t0()
        hs3, _ = _disp_flt(
            hs2,
            min_keep=MNK,
            jmin=(J_DISP_MIN if jmin is None else float(jmin)),
            use_jdg=disp_use_jdg,
        )
        meta["t"]["disp_flt"] = _dt(t_disp)
        meta["n"]["after_disp"] = len(hs3)
        meta["n"]["uniq_books"] = len({h.get("book") for h in hs3 if h.get("book")})
        meta["n"]["uniq_sections"] = len({(h.get("book"), h.get("sec")) for h in hs3 if h.get("book") or h.get("sec")})
        meta["n"]["uniq_publishers"] = len({h.get("publisher") or h.get("corp") for h in hs3 if h.get("publisher") or h.get("corp")})

        if not use_jdg_flag:
            meta["cap"]["judge_kind"] = jdg_mode or "off"
            meta["cap"]["judge_ok"] = False

        # direct evidence
        t_direct = _t0()
        use_jdg_direct = use_jdg_flag and bool(meta_jdg.get("ok"))
        dr = _direct(hs3, q, qs=qs, use_jdg=use_jdg_direct)
        meta["t"]["direct"] = _dt(t_direct)
        meta["n"]["direct_hits"] = len(dr)
        cov = coverage_label(dr, q)
        meta["cov"] = _cov(dr if dr else hs3, topn=8)
        meta["coverage"] = cov

        if dr:
            meta["t"]["near_miss"] = 0.0
            meta["n"]["near_miss"] = 0
            meta["conf"] = _calc_confidence(dr)
            # LLM assembly (optional)
            t_llm = _t0()
            ans_txt = ""
            if use_llm:
                ctx, ctx_meta = _assemble_context(dr, budget_chars=budget["ctx_chars"], budget_tokens=budget["ctx_tokens"])
                meta["clamp"]["context"] = {
                    "char_clamped": bool(ctx_meta.get("char_clamped")),
                    "token_clamped": bool(ctx_meta.get("token_clamped")),
                    "budget_chars": budget["ctx_chars"],
                    "budget_tokens": budget["ctx_tokens"],
                }
                meta["flags"]["ctx_clamped"] = bool(ctx_meta.get("char_clamped") or ctx_meta.get("token_clamped"))
                prompt = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer concisely without quotes."
                prompt_clamped, prompt_meta = _clamp_text(
                    prompt, budget["prompt_chars"], budget["prompt_tokens"], LLM_CLAMP_MARKER
                )
                meta["clamp"]["prompt"] = {
                    "char_clamped": bool(prompt_meta.get("char_clamped")),
                    "token_clamped": bool(prompt_meta.get("token_clamped")),
                    "budget_chars": budget["prompt_chars"],
                    "budget_tokens": budget["prompt_tokens"],
                }
                meta["flags"]["prompt_clamped"] = bool(prompt_meta.get("char_clamped") or prompt_meta.get("token_clamped"))
                try:
                    ans_txt = llm_call(
                        prompt_clamped,
                        cfg={
                            "mode": mode_name,
                            "char_budget": budget["prompt_chars"],
                            "tok_budget": budget["prompt_tokens"],
                        },
                    )
                    meta["flags"]["llm_used"] = True
                    meta["err_llm"] = None
                except Exception as ex:
                    ans_txt = ""
                    meta["flags"]["llm_used"] = False
                    meta["flags"]["llm_bypassed"] = True
                    msg = _safe_msg(ex)
                    meta["err_llm"] = msg
                    err_id = _err_id(msg)
                    meta["err"] = {"where": "llm_call", "msg": msg, "id": err_id, "error_id": err_id}
            else:
                meta["flags"]["llm_bypassed"] = True
                meta["err_llm"] = None
            meta["t"]["llm"] = _dt(t_llm)
            meta["t"]["total"] = _dt(t_total)
            _log_event(meta, mode_name, pubs or list(getattr(e, "corp", {}).keys()), len(q), hits=dr)
            return _mk_ret(ok=True, no_ev=False, hits=_pub_hits(dr), nm_hits=[], cov=cov, ans=ans_txt, meta=meta)

        t_nm = _t0()
        nm_hits = []
        nm_meta = {"threshold": NM_MIN, "used_judge": disp_use_jdg}
        if nm_flag:
            nm_hits, nm_meta = _near_miss(hs2, q, qs=qs, use_jdg=disp_use_jdg)
            nm_hits = _nm_with_metadata(nm_hits, nm_meta)
            if not dr:
                nm_hits = _ensure_nm_candidates(nm_hits, hs2, q, qs=qs, nm_meta=nm_meta)
                nm_hits = _nm_with_metadata(nm_hits, nm_meta, explanation="Close but below judge/overlap threshold")
        else:
            if not compute_nm_flag:
                nm_meta["reason"] = "compute_near_miss_disabled"
                meta["flags"]["near_miss_skipped"] = True
            else:
                nm_meta["reason"] = "near_miss_disabled"
        meta["t"]["near_miss"] = _dt(t_nm)
        meta["n"]["near_miss"] = len(nm_hits)
        nm_meta["count"] = len(nm_hits)
        meta["meta_nm"] = nm_meta
        meta["t"]["total"] = _dt(t_total)
        meta["conf"] = _calc_confidence(dr if dr else hs3)
        meta["flags"]["llm_bypassed"] = True
        meta["flags"]["llm_used"] = False
        meta["err_llm"] = None
        _log_event(meta, mode_name, pubs or list(getattr(e, "corp", {}).keys()), len(q), hits=(hs3 or []) + (nm_hits or []))
        # LLM path for soft no-evidence is intentionally disabled; return empty answer
        return _mk_ret(ok=True, no_ev=True, hits=_pub_hits(hs3), nm_hits=_pub_hits(nm_hits), cov=cov, ans="", meta=meta)
    except Exception as ex:
        msg = _safe_msg(ex)
        err_id = _err_id(msg)
        meta["err"] = {"where": "run_query", "msg": msg, "id": err_id, "error_id": err_id}
        meta["t"]["total"] = _dt(t_total)
        _log_event(meta, mode_name, pubs or list(getattr(e, "corp", {}).keys()), len(q))
        return _mk_ret(ok=False, no_ev=True, hits=[], nm_hits=[], cov="WEAK", ans="", meta=meta)


# ---------------------------------------------------------------------------
# Chat answer composition
# ---------------------------------------------------------------------------

def generate_answer(query: str, hits: list, answer: str | None = None) -> str:
    """
    Summarise search hits into a natural language reply for the chat.

    Args:
        query: Original user question.
        hits: List of search result dictionaries returned by ``run_query``.
        answer: Optional answer string already produced by the engine.

    Returns:
        A human-friendly answer.  If ``answer`` is supplied, it is returned
        directly.  Otherwise a simple listing of the top three hit titles and
        sections is returned as a placeholder.
    """
    # Use existing answer if provided
    if answer:
        return answer
    # Fallback: build a simple summary from top hits
    if not hits:
        return "No relevant passages found."
    snippets = []
    for h in hits[:3]:
        title = h.get("book_title_pretty") or h.get("book_title") or h.get("file") or "Unknown book"
        section = h.get("section") or h.get("sec") or ""
        part = f"{title} – {section}" if section else title
        snippets.append(part)
    return "Sources: " + "; ".join(snippets)
