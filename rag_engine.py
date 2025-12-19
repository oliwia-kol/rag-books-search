from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import math
import re
import sqlite3
import time
from statistics import pstdev

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------
# Paths (repo-local)
# -----------------------------
ROOT = Path(__file__).parent
BASE_OUT = ROOT / "data"

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
    missing = []
    for k, flag in [("faiss", rep["faiss"]), ("db", rep["db"]), ("manifest", rep["manifest"])]:
        if not flag:
            missing.append(k)
    rep["missing"] = missing
    return rep


# -----------------------------
# Retrieval + judge config
# -----------------------------
HCFG = {
    "faiss_fetch_k": 60,
    "fts_fetch_k": 60,
    "final_k": 10,  # practical for work
    "mmr_k": 20,
    "mmr_lambda": 0.55,
    "min_faiss_score": 0.18,
    "dense_k": 30,
    "lex_k": 30,
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

NM_MIN = 0.28
NM_MAX = 6

# contract constants
RET_KEYS = ["ok", "no_evidence", "answer", "hits", "near_miss", "coverage", "meta"]
META_T_KEYS = ["total", "embed", "dense", "lex", "fuse", "cut", "rerank", "disp_flt", "direct", "near_miss", "llm"]
META_N_KEYS = [
    "pubs_req",
    "pubs_used",
    "dense_hits",
    "lex_hits",
    "fetched_dense",
    "fetched_lex",
    "cands",
    "after_cut",
    "after_disp",
    "direct_hits",
    "near_miss",
    "uniq_books",
    "uniq_sections",
]
META_CAP_KEYS = ["has_emb", "dense_ok", "lex_ok", "judge_requested", "judge_ok", "judge_kind"]
META_FLAG_KEYS = ["dense_used", "lex_used", "veto_applied", "llm_used"]
STAGES = META_T_KEYS

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

    for k, p in ready.items():
        ok_dense = False
        dim_ok = True
        try:
            rd = faiss.read_index(str(p / "index.faiss"))
            dims[k] = int(rd.d)
            if emb_dim is not None and dims[k] != emb_dim:
                dim_ok = False
            else:
                ix[k] = rd
                ok_dense = True
        except Exception:
            dim_ok = False
            pass

        db_path = p / "meta.sqlite"
        db_exists = db_path.exists()
        if db_exists:
            dbp[k] = db_path

        if ok_dense or db_exists:
            loaded[k] = p

        # enrich report
        rep = reports.get(k, {})
        rep["dense_loaded"] = ok_dense
        rep["db_loaded"] = db_exists
        rep["dim_ok"] = dim_ok
        rep["embed_dim"] = emb_dim
        rep["ix_dim"] = dims.get(k)
        reports[k] = rep

    return Eng(emb=emb, ix=ix, dbp=dbp, corp=loaded, ix_dim=dims, corp_report=reports)


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


def faiss_search(e: Eng, corp: str, qv: np.ndarray, k: int):
    if qv.size == 0:
        return []
    if corp not in e.ix or corp not in e.dbp:
        return []
    ix = e.ix[corp]
    k = max(1, min(int(k), HCFG["faiss_fetch_k"]))
    try:
        if qv.dtype != np.float32:
            qv = np.asarray(qv, dtype="float32")
    except Exception:
        return []
    try:
        if qv.shape[0] != ix.d:
            return []
    except Exception:
        return []
    try:
        D, I = ix.search(qv.reshape(1, -1), k)
    except Exception:
        return []
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
            return []
        wanted_ids = [i for _, i in pairs]
        ph = ",".join(["?"] * len(wanted_ids))
        cur = con.cursor()
        try:
            rows = cur.execute(f"SELECT cid, fp, sec, cidx, tx, i64 FROM chunks WHERE i64 IN ({ph})", wanted_ids).fetchall()
            i64_to_row = {int(r[5]): r for r in rows}
        except Exception:
            i64_to_row = {}
            for _, i64 in pairs:
                row = get_by_i64(con, int(i64))
                if row:
                    cid, fp, sec, cidx, tx = row
                    i64_to_row[int(i64)] = (cid, fp, sec, cidx, tx, i64)
        for s, i64 in pairs:
            row = i64_to_row.get(int(i64))
            if not row:
                continue
            cid, fp, sec, cidx, tx, _ = row
            out.append(
                {
                    "corp": corp,
                    "cid": cid,
                    "fp": fp,
                    "sec": sec,
                    "cidx": int(cidx),
                    "tx": tx,
                    # aliases
                    "section": sec,
                    "book": Path(fp).stem if fp else None,
                    "publisher": corp,
                    "text": tx,
                    "sem_score": float(s),
                }
            )
    finally:
        con.close()
    return out


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
                    "tx": tx,
                    # aliases
                    "section": sec,
                    "book": Path(fp).stem if fp else None,
                    "publisher": corp,
                    "text": tx,
                    "lex_score": float(-b),
                }
            )
        return out
    finally:
        con.close()


def dense_retrieve(e: Eng, corp: str, qv: np.ndarray, k: int | None = None):
    if corp not in e.ix:
        return []
    if qv.size == 0:
        return []
    rows = faiss_search(e, corp, qv, k or HCFG["faiss_fetch_k"])
    norm_scores(rows, "sem_score")
    for r in rows:
        r.setdefault("score", r.get("sem_score_n", 0.0))
    return rows


def lex_retrieve(e: Eng, corp: str, q: str, k: int | None = None):
    if corp not in e.dbp:
        return []
    rows = fts_search(e, corp, q, k or HCFG["fts_fetch_k"])
    norm_scores(rows, "lex_score")
    for r in rows:
        r.setdefault("score", r.get("lex_score_n", 0.0))
    return rows


def hybrid_retrieve(e: Eng, q: str, k: int = HCFG["final_k"], pubs=None, qv: np.ndarray | None = None):
    """Dense+lexical hybrid retrieval across selected publishers."""
    if qv is None:
        qv = embed_query(e, q)
    use_dense = qv.size > 0

    pubs = pubs or list(e.corp.keys())
    cands = []
    meta = {"dense_hits": 0, "lex_hits": 0, "pubs_used": 0, "t_dense": 0.0, "t_lex": 0.0, "fetched_dense": 0, "fetched_lex": 0}

    for corp in pubs:
        if corp not in e.corp:
            continue

        t_d0 = _t0()
        d = dense_retrieve(e, corp, qv, k=HCFG["dense_k"]) if use_dense else []
        meta["t_dense"] += _dt(t_d0)
        t_l0 = _t0()
        l = lex_retrieve(e, corp, q, k=HCFG["lex_k"])
        meta["t_lex"] += _dt(t_l0)

        meta["dense_hits"] += len(d)
        meta["lex_hits"] += len(l)
        meta["fetched_dense"] += len(d)
        meta["fetched_lex"] += len(l)
        meta["pubs_used"] += 1

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
            row["score"] = 0.65 * ss + 0.35 * ls
            cands.append(row)

    cands.sort(key=lambda z: float(z.get("score", 0.0)), reverse=True)

    # lightweight de-dupe and diversity budget per publisher
    seen = set()
    out = []
    per_pub = {}
    div_cap = max(2, int(k))
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
        if len(out) >= HCFG["mmr_k"]:
            break

    meta["cands"] = len(cands)
    return out[:k], meta


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


def _blank_meta():
    cap = {k: False for k in META_CAP_KEYS}
    cap["judge_kind"] = "none"
    cap["corp_available"] = []
    cap["dense_reason"] = None
    flags = {k: False for k in META_FLAG_KEYS}
    return {"t": {k: 0.0 for k in META_T_KEYS}, "n": {k: 0 for k in META_N_KEYS}, "cap": cap, "flags": flags, "err": None, "log": {}}


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


def _get_jdg():
    # Cross-encoder judge intentionally deferred for CPU-only MVP.
    return None


def _jdg_rerank(q, hs):
    j = _get_jdg()
    if j is None:
        # fallback: use normalized score as judge proxy
        for h in hs:
            js = float(h.get("score", 0.0))
            h["_jdg"] = js
            h["judge01"] = js
        hs.sort(key=lambda x: float(x.get("judge01", 0.0)), reverse=True)
        return hs, {"ok": False, "t": 0.0, "n": len(hs), "kind": "proxy_score"}
    tp = hs[: min(K_JDG, len(hs))]
    pairs = [(q, (h.get("text") or "")[:1200]) for h in tp]
    t0 = time.time()
    try:
        sc = j.predict(pairs)
    except Exception:
        return hs, {"ok": False, "t": time.time() - t0, "n": len(tp), "kind": "error"}
    t1 = time.time()
    for h, s in zip(tp, sc):
        h["_jdg"] = float(s)
        h["judge01"] = _sig(h["_jdg"])
    tp.sort(key=lambda x: float(x.get("_jdg", -1e9)), reverse=True)
    return tp + hs[len(tp) :], {"ok": True, "t": t1 - t0, "n": len(tp), "kind": "cross_encoder"}


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
    return c[:NM_MAX]


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


def _assemble_context(dr, budget_chars: int = 1400):
    if not dr:
        return ""
    # deterministic by judge01/score
    dr_sorted = sorted(dr, key=lambda h: (-float(h.get("judge01", 0.0)), -float(h.get("score", 0.0)), str(h.get("cid", ""))))
    out = []
    used = 0
    for h in dr_sorted:
        tx = (h.get("tx") or h.get("text") or "")[:800]
        if not tx:
            continue
        if used + len(tx) > budget_chars and used > 0:
            break
        out.append(tx)
        used += len(tx) + 1
    return "\n\n".join(out)[:budget_chars]


def llm_call(prompt: str, cfg: Optional[dict] = None) -> str:
    # placeholder CPU-friendly stub; can be replaced with real model later
    if not prompt:
        return ""
    return prompt.strip()[:800]


def get_reader_chunk(e: Eng, fp: str, cidx: int, window: int = 2):
    """Fetch context window for reading mode."""
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
                return [
                    {"cid": r[0], "fp": r[1], "sec": r[2], "cidx": int(r[3]), "tx": r[4], "corp": corp}
                    for r in rows
                ]
        except Exception:
            continue
    return []


def _log_event(meta, mode, pubs, qlen):
    try:
        meta["log"] = {
            "ts": time.time(),
            "mode": mode,
            "scope": pubs,
            "qlen": int(qlen or 0),
            "counts": dict(meta.get("n", {})),
            "flags": dict(meta.get("flags", {})),
            "judge_ok": meta.get("cap", {}).get("judge_ok", False),
            "no_evidence": meta.get("err") is None and meta.get("n", {}).get("direct_hits", 0) == 0,
        }
    except Exception:
        pass


def run_query(
    e: Eng,
    q: str,
    pubs=None,
    sort: str = "Best evidence",
    nm: bool | None = None,
    show_nm: bool | None = None,
    jmin: float | None = None,
    use_jdg: bool | None = None,
    scope=None,
    use_llm: bool | None = None,
    use_vector_mmr: bool | None = None,
    mode: str | None = None,
):
    t_total = _t0()
    meta = _blank_meta()
    use_jdg_flag = USE_JDG_DEFAULT if use_jdg is None else bool(use_jdg)
    nm_flag = nm if nm is not None else (show_nm if show_nm is not None else True)
    mode = (mode or "quick").lower()
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

    try:
        qs = set([w.lower() for w in re.findall(r"[A-Za-z0-9]+", q) if len(w) >= 3])
        meta_jdg = {"ok": False, "kind": "none"}

        if not getattr(e, "corp", None) or (not getattr(e, "ix", None) and not getattr(e, "dbp", None)):
            meta["t"]["total"] = _dt(t_total)
            return _mk_ret(
                ok=False,
                no_ev=True,
                hits=[],
                nm_hits=[],
                cov="WEAK",
                ans="No corpus indexes available. Add data/ or configure paths.",
                meta=meta,
            )

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
        hs, rmeta = hybrid_retrieve(e, q, pubs=pubs, qv=qv)
        meta["t"]["dense"] = rmeta.get("t_dense", 0.0)
        meta["t"]["lex"] = rmeta.get("t_lex", 0.0)
        meta["t"]["fuse"] = _dt(t_ret_dense_lex) - meta["t"]["dense"] - meta["t"]["lex"]
        meta["n"]["dense_hits"] = rmeta.get("dense_hits", 0)
        meta["n"]["lex_hits"] = rmeta.get("lex_hits", 0)
        meta["n"]["fetched_dense"] = rmeta.get("fetched_dense", 0)
        meta["n"]["fetched_lex"] = rmeta.get("fetched_lex", 0)
        meta["n"]["cands"] = rmeta.get("cands", 0)
        meta["n"]["pubs_used"] = rmeta.get("pubs_used", 0)
        meta["n"]["pubs_req"] = len(pubs or list(e.corp.keys()))
        meta["flags"]["lex_used"] = meta["n"]["lex_hits"] > 0 or bool(getattr(e, "dbp", {}))
        # mode-aware clamp for quick vs deep
        if mode.startswith("deep"):
            pass
        else:
            hs = hs[: HCFG["final_k"]]

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
        hs2, _ = _cut(hs, k=K_SHOW, mnk=MNK)
        meta["t"]["cut"] = _dt(t_cut)
        meta["n"]["after_cut"] = len(hs2)

        if not use_jdg_flag:
            for h in hs2:
                h.setdefault("judge01", float(h.get("score", 0.0)))

        # judge rerank (display order)
        hs3 = None
        disp_use_jdg = False
        t_rerank = _t0()
        if use_jdg_flag:
            hs2, meta_jdg = _jdg_rerank(q, hs2)
            meta["t"]["rerank"] = _dt(t_rerank)
            disp_use_jdg = bool(meta_jdg.get("ok"))
            meta["cap"]["judge_ok"] = disp_use_jdg
            meta["cap"]["judge_kind"] = meta_jdg.get("kind", "none")
            veto = False
            if disp_use_jdg:
                veto, _ = _noev_jdg(hs2)
                meta["flags"]["veto_applied"] = bool(veto)
        else:
            meta["t"]["rerank"] = _dt(t_rerank)

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

        if not use_jdg_flag:
            meta["cap"]["judge_kind"] = "none"

        # direct evidence
        t_direct = _t0()
        use_jdg_direct = use_jdg_flag and bool(meta_jdg.get("ok"))
        dr = _direct(hs3, q, qs=qs, use_jdg=use_jdg_direct)
        meta["t"]["direct"] = _dt(t_direct)
        meta["n"]["direct_hits"] = len(dr)
        cov = coverage_label(dr, q)

        if dr:
            meta["t"]["near_miss"] = 0.0
            meta["n"]["near_miss"] = 0
            meta["conf"] = _calc_confidence(dr)
            # LLM assembly (optional)
            t_llm = _t0()
            ans_txt = ""
            if use_llm:
                ctx = _assemble_context(dr)
                prompt = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer concisely without quotes."
                try:
                    ans_txt = llm_call(prompt, cfg={"mode": "quick"})
                    meta["flags"]["llm_used"] = True
                except Exception:
                    ans_txt = ""
            meta["t"]["llm"] = _dt(t_llm)
            meta["t"]["total"] = _dt(t_total)
            _log_event(meta, mode, pubs or list(getattr(e, "corp", {}).keys()), len(q))
            return _mk_ret(ok=True, no_ev=False, hits=_pub_hits(dr), nm_hits=[], cov=cov, ans=ans_txt, meta=meta)

        t_nm = _t0()
        nm_hits = _near_miss(hs2, q, qs=qs, use_jdg=disp_use_jdg) if nm_flag else []
        meta["t"]["near_miss"] = _dt(t_nm)
        meta["n"]["near_miss"] = len(nm_hits)
        meta["t"]["total"] = _dt(t_total)
        meta["conf"] = _calc_confidence(dr if dr else hs3)
        _log_event(meta, mode, pubs or list(getattr(e, "corp", {}).keys()), len(q))
        # LLM path for soft no-evidence is intentionally disabled; return empty answer
        return _mk_ret(ok=True, no_ev=True, hits=_pub_hits(hs3), nm_hits=_pub_hits(nm_hits), cov=cov, ans="", meta=meta)
    except Exception as ex:
        meta["err"] = {"where": "run_query", "msg": _safe_msg(ex)}
        meta["t"]["total"] = _dt(t_total)
        _log_event(meta, mode, pubs or list(getattr(e, "corp", {}).keys()), len(q))
        return _mk_ret(ok=False, no_ev=True, hits=[], nm_hits=[], cov="WEAK", ans="", meta=meta)
