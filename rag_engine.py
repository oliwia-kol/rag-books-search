from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math
import re
import sqlite3
import time

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


def chk(p: Path):
    assert p.exists(), p
    assert (p / "index.faiss").exists(), p / "index.faiss"
    assert (p / "meta.sqlite").exists(), p / "meta.sqlite"
    assert (p / "manifest.json").exists(), p / "manifest.json"


# -----------------------------
# Retrieval config (from 02)
# -----------------------------
HCFG = {
    "faiss_fetch_k": 60,
    "fts_fetch_k": 60,
    "final_k": 10,          # practical for work
    "mmr_k": 20,
    "mmr_lambda": 0.55,
    "min_faiss_score": 0.18,
}

FTS_CFG = {"batch": 4000, "use_porter": False}


def fts_query_escape(q: str) -> str:
    # identical intent to 02: tokenise, keep alnum, join with OR
    xs = re.findall(r"[A-Za-z0-9]+", (q or "").strip())
    xs = [x for x in xs if len(x) >= 2]
    if not xs:
        return ""
    # basic FTS5 safe query: token1 OR token2 OR ...
    return " OR ".join(xs)


def fts_ready(con: sqlite3.Connection):
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
    emb: SentenceTransformer
    ix: Dict[str, Any]
    dbp: Dict[str, Path]


def _mk_eng(base_out: Path = BASE_OUT, emb_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Eng:
    # validate corp folders
    c2p = {k: (base_out / k) for k in CORP.keys()}
    for _, p in c2p.items():
        chk(p)

    emb = SentenceTransformer(emb_model)

    ix: Dict[str, Any] = {}
    dbp: Dict[str, Path] = {}

    for k, p in c2p.items():
        ix[k] = faiss.read_index(str(p / "index.faiss"))
        dbp[k] = p / "meta.sqlite"

    return Eng(emb=emb, ix=ix, dbp=dbp)


def _db(con_p: Path) -> sqlite3.Connection:
    # Streamlit can run multiple threads; allow cross-thread connections.
    return sqlite3.connect(str(con_p), check_same_thread=False)


def faiss_search(e: Eng, corp: str, qv: np.ndarray, k: int):
    ix = e.ix[corp]
    D, I = ix.search(qv.reshape(1, -1), k)
    ids = I[0].tolist()
    scs = D[0].tolist()

    out = []
    con = _db(e.dbp[corp])
    try:
        for s, i64 in zip(scs, ids):
            if i64 == -1:
                continue
            if float(s) < HCFG["min_faiss_score"]:
                continue
            row = get_by_i64(con, int(i64))
            if not row:
                continue
            cid, fp, sec, cidx, tx = row
            out.append(
                {
                    "corp": corp,
                    "cid": cid,
                    "fp": fp,
                    "sec": sec,
                    "cidx": int(cidx),
                    "tx": tx,
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

        rows = cur.execute(
            '''
            SELECT cid, fp, sec, tx, bm25(chunks_fts) as b
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY b
            LIMIT ?
            ''',
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
                    "lex_score": float(-b),
                }
            )
        return out
    finally:
        con.close()


def hybrid_retrieve(e: Eng, q: str, k: int = HCFG["final_k"], pubs=None):
    """Dense+lexical hybrid retrieval across selected publishers."""
    qv = embed_query(e, q)

    pubs = pubs or list(e.corp.keys())
    cands = []

    for corp in pubs:
        if corp not in e.corp:
            continue

        d = dense_retrieve(e, corp, qv, k=HCFG["dense_k"])
        l = lex_retrieve(e, corp, q, k=HCFG["lex_k"])

        dd = {x["cid"]: x for x in d}
        ll = {x["cid"]: x for x in l}
        cids = set(dd) | set(ll)

        for cid in cids:
            a = dd.get(cid, {})
            b = ll.get(cid, {})
            row = {
                "cid": cid,
                "cidx": a.get("cidx") if "cidx" in a else b.get("cidx"),
                "txt": a.get("txt") or b.get("txt"),
                "section": a.get("section") or b.get("section"),
                "book": a.get("book") or b.get("book"),
                "publisher": corp,
            }

            ss = float(a.get("sem_score_n", 0.0))
            ls = float(b.get("lex_score_n", 0.0))
            row["sem_score_n"] = ss
            row["lex_score_n"] = ls
            row["score"] = 0.65 * ss + 0.35 * ls
            cands.append(row)

    cands.sort(key=lambda z: float(z.get("score", 0.0)), reverse=True)

    # lightweight de-dupe by (book, section) to improve diversity
    seen = set()
    out = []
    for x in cands:
        k2 = (x.get("book"), x.get("section"))
        if k2 in seen:
            continue
        seen.add(k2)
        out.append(x)
        if len(out) >= k:
            break

    return out

def _cut(hs, k=K_SHOW, mnk=MNK, mg=MG, abs_mn=ABS_MN):
    if not hs:
        return hs, {"kept": 0, "all": 0, "rule": "empty"}
    hs = sorted(hs, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    tp = hs[: max(k, mnk)]
    top1 = float(tp[0].get("score", 0.0))
    thr = max(abs_mn, top1 - mg)
    out = [h for h in tp if float(h.get("score", 0.0)) >= thr]
    if len(out) < mnk:
        out = tp[:mnk]
    out = sorted(out, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return out, {"kept": len(out), "all": len(tp), "top1": top1, "thr": thr, "rule": "score>=max(abs_min, top1-margin) + sort(score)"}


def _jdg_rerank(q, hs):
    j = _get_jdg()
    if j is None:
        return hs, {"ok": False, "t": 0.0, "n": 0}
    tp = hs[: min(K_JDG, len(hs))]
    pairs = [(q, (h.get("tx") or "")[:1200]) for h in tp]
    t0 = time.time()
    try:
        sc = j.predict(pairs)
    except Exception:
        return hs, {"ok": False, "t": time.time() - t0, "n": len(tp)}
    t1 = time.time()
    for h, s in zip(tp, sc):
        h["_jdg"] = float(s)
        h["judge01"] = _sig(h["_jdg"])
    tp.sort(key=lambda x: float(x.get("_jdg", -1e9)), reverse=True)
    return tp + hs[len(tp) :], {"ok": True, "t": t1 - t0, "n": len(tp)}


def _disp_flt(hs, mnk=MNK, jmin=J_DISP_MIN):
    if not USE_JDG:
        return hs, {"kept": len(hs), "all": len(hs), "rule": "disp:none"}
    a, b = [], []
    for h in hs:
        j01 = h.get("judge01", None)
        if j01 is not None and float(j01) >= jmin:
            a.append(h)
        else:
            b.append(h)
    out = a
    if len(out) < mnk:
        out = out + b[: (mnk - len(out))]
    return out, {"kept": len(out), "all": len(hs), "rule": f"disp:judge01>={jmin} (min_keep={mnk})"}


def _direct(hs, q):
    out = []
    for h in hs:
        ok, _ = _ov_ok(q, h)
        if not ok:
            continue
        if USE_JDG:
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


def _near_miss(hs, q):
    c = []
    for h in hs:
        ok, _ = _ov_ok(q, h)
        if not ok:
            continue
        if USE_JDG:
            j01 = h.get("judge01", None)
            if j01 is None:
                continue
            if float(j01) >= NM_MIN:
                c.append(h)
        else:
            if float(h.get("score", 0.0)) >= ABS_MN:
                c.append(h)
    if USE_JDG:
        c.sort(key=lambda x: float(x.get("_jdg", -1e9)), reverse=True)
    else:
        c.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return c[:NM_MAX]


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


def run_query(
    e: Eng,
    q: str,
    pubs=None,
    sort: str = "Best evidence",
    nm: bool = True,
    jmin: float | None = None,
    use_jdg: bool | None = None,
):
    global USE_JDG

    if use_jdg is not None:
        USE_JDG = bool(use_jdg)

    # retrieve
    hs = hybrid_retrieve(e, q, pubs=pubs)

    # sort preference (UI-level)
    s = (sort or "").strip().lower()
    if s.startswith("semantic"):
        hs.sort(key=lambda z: float(z.get("sem_score_n", 0.0)), reverse=True)
    elif s.startswith("lex"):
        hs.sort(key=lambda z: float(z.get("lex_score_n", 0.0)), reverse=True)
    else:
        hs.sort(key=lambda z: float(z.get("score", 0.0)), reverse=True)

    # cutoff on score (stable thresholds)
    K_SHOW = 18
    MNK = 4
    ABS_MN = 3
    MG = 0.0

    hs2, _ = _cut(hs, k=K_SHOW, mnk=MNK, mg=MG, abs_mn=ABS_MN)

    # judge rerank (display order)
    if USE_JDG:
        hs2, _ = _jdg_rerank(q, hs2)
        veto, _ = _noev_jdg(hs2)
        if veto:
            nm2 = _near_miss(hs2, q)
            return {"ok": False, "no_evidence": True, "hits": [], "near_miss": nm2, "coverage": "WEAK"}

    hs3, _ = _disp_flt(hs2, show_nm=bool(nm), min_keep=MNK, jmin=(J_DISP_MIN if jmin is None else float(jmin)))

    dr = _direct(hs3, q)
    cov = coverage_label(dr, q)

    if dr:
        return {"ok": True, "no_evidence": False, "hits": dr, "near_miss": [], "coverage": cov}
    return {"ok": True, "no_evidence": False, "hits": hs3, "near_miss": [], "coverage": cov}
