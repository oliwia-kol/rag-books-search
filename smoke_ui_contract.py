"""Fast contract check for the UI modules.

Run:
  python smoke_ui_contract.py

Goal: catch app/ui API drift in ~1s.
"""

import importlib
from pathlib import Path

import numpy as np

from rag_engine import RET_KEYS, STAGES


def _has(m, ns):
    miss = [n for n in ns if not hasattr(m, n)]
    if miss:
        raise AttributeError(f"{m.__name__} missing: {miss}")


def main():
    us = importlib.import_module("ui_shell")
    ua = importlib.import_module("ui_adapter")
    ut = importlib.import_module("ui_theme")
    re_mod = importlib.import_module("rag_engine")

    _has(us, ["init_state", "sidebar", "global_error_box", "toast_flush", "qp_get", "qp_set", "cb_clear"])
    _has(ua, ["render_answer", "render_conf", "render_context_panel", "render_evidence_list", "render_power_panel"])
    _has(ut, ["apply_theme"])
    _has(re_mod, ["_mk_eng", "run_query", "Eng", "get_startup_report"])

    # chk() soft-report
    rep = re_mod.chk(Path("/tmp/does-not-exist"))
    if rep["ok"] is not False or "faiss" not in rep:
        raise AssertionError("chk should return structured report with ok/faiss/db/manifest")

    # minimal engine contract: should not crash when empty and should flag no corp
    eng = re_mod.Eng(emb=None, ix={}, dbp={}, corp={}, ix_dim={}, corp_report={})
    rq_empty = re_mod.run_query(eng, "smoke test")
    if rq_empty["ok"] is not False or rq_empty["no_evidence"] is not True:
        raise AssertionError("empty engine should set ok=False, no_evidence=True")
    if rq_empty["meta"]["cap"]["dense_ok"] is not False or rq_empty["meta"]["cap"]["lex_ok"] is not False:
        raise AssertionError("empty engine should disable dense/lex in meta.cap")

    # minimal hit schema contract (field names must stay stable)
    sample_hit = {
        "corp": "OReilly",
        "fp": "book/ch01",
        "sec": "Intro",
        "tx": "hello world",
        "cid": "cid-1",
        "cidx": 1,
        "score": 0.5,
    }
    hits, _ = re_mod._jdg_rerank("q", [sample_hit])
    required = {"corp", "fp", "sec", "tx", "cid", "cidx", "score"}
    missing = [k for k in required if k not in hits[0]]
    if missing:
        raise AssertionError(f"hit schema missing keys: {missing}")

    # run_query contract: required keys always present
    rq = re_mod.run_query(eng, "smoke test")
    for k in ["ok", "no_evidence", "hits", "near_miss", "coverage"]:
        if k not in rq:
            raise AssertionError(f"run_query missing key: {k}")
    for k in RET_KEYS:
        if k not in rq:
            raise AssertionError(f"run_query missing contract key: {k}")
    for st in STAGES:
        if st not in rq["meta"]["t"]:
            raise AssertionError(f"meta.t missing stage: {st}")
    if rq["meta"]["t"]["total"] < 0:
        raise AssertionError("meta.t.total should be populated")
    # meta flags/caps should be present and stable
    for flag in ["dense_used", "lex_used", "veto_applied", "llm_used", "llm_bypassed", "dense_clamped", "lex_clamped"]:
        if flag not in rq["meta"]["flags"]:
            raise AssertionError(f"meta.flags missing: {flag}")
    for cap in ["has_emb", "dense_ok", "lex_ok", "judge_requested", "judge_ok", "judge_kind", "corp_available"]:
        if cap not in rq["meta"]["cap"]:
            raise AssertionError(f"meta.cap missing: {cap}")
    for n_key in ["fetched_dense", "fetched_lex", "uniq_books", "uniq_sections", "fallback_retries", "fallback_failed"]:
        if n_key not in rq["meta"]["n"]:
            raise AssertionError(f"meta.n missing: {n_key}")

    # forced exception path
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    orig_hybrid = re_mod.hybrid_retrieve
    re_mod.hybrid_retrieve = boom
    eng_err = re_mod.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})
    rq_err = re_mod.run_query(eng_err, "smoke test")
    re_mod.hybrid_retrieve = orig_hybrid
    if rq_err["ok"] is not False or rq_err["no_evidence"] is not True:
        raise AssertionError("error path should set ok=False, no_evidence=True")
    if not rq_err.get("meta", {}).get("err"):
        raise AssertionError("error path should populate meta.err")
    if rq_err["meta"]["err"]["where"] != "run_query":
        raise AssertionError("error path should annotate err.where=run_query")

    # no direct evidence path
    def fake_retrieve(e, q, pubs=None, qv=None, **kwargs):
        return [
            {"cid": "c", "fp": "f", "sec": "s", "tx": "t", "cidx": 0, "score": 0.5, "sem_score_n": 0.5, "lex_score_n": 0.5}
        ], {
            "dense_hits": 0,
            "lex_hits": 1,
            "cands": 1,
            "pubs_used": 1,
            "t_dense": 0.0,
            "t_lex": 0.0,
            "k_requested": kwargs.get("k", 10),
            "k_applied": kwargs.get("k", 10),
            "k_clamped": False,
            "dense_clamped": False,
            "lex_clamped": False,
            "fallback_retries": 0,
            "fallback_failed": 0,
        }

    orig_retrieve = re_mod.hybrid_retrieve
    re_mod.hybrid_retrieve = fake_retrieve
    eng_nd = re_mod.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})
    rq_nd = re_mod.run_query(eng_nd, "smoke test")
    re_mod.hybrid_retrieve = orig_retrieve
    if rq_nd["ok"] is not True or rq_nd["no_evidence"] is not True:
        raise AssertionError("no-direct-evidence path should be ok=True, no_evidence=True")
    if not rq_nd["hits"]:
        raise AssertionError("hits should contain display-filtered results even when no direct evidence")
    if any("_tok" in h or "_jdg" in h for h in rq_nd["hits"]):
        raise AssertionError("sanitization failed: internal keys leaked")
    if rq_nd["meta"]["cap"]["judge_kind"] != "proxy_score":
        raise AssertionError("proxy judge should be reported as proxy_score")
    if rq_nd["answer"] != "":
        raise AssertionError("no-direct-evidence path should return empty answer string")

    # dimension mismatch disables dense
    orig_embed = re_mod.embed_query
    re_mod.embed_query = lambda *args, **kwargs: np.zeros(5, dtype="float32")
    eng_dim = re_mod.Eng(emb=None, ix={}, dbp={}, corp={"c": None}, ix_dim={"c": 7}, corp_report={})
    rq_dim = re_mod.run_query(eng_dim, "smoke test")
    re_mod.embed_query = orig_embed
    if rq_dim["meta"]["cap"]["dense_ok"] is not False:
        raise AssertionError("dimension mismatch should disable dense_ok")

    print("OK: UI contract satisfied")


if __name__ == "__main__":
    main()
