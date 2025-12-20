import rag_engine as re


sample_hit = {
    "cid": "c1",
    "fp": "f",
    "sec": "s",
    "tx": "query context with evidence",
    "cidx": 0,
    "score": 0.6,
    "corp": "c",
}


def _fake_retrieve(e, q, pubs=None, qv=None, **kwargs):
    meta = {
        "dense_hits": 0,
        "lex_hits": 1,
        "fetched_dense": 0,
        "fetched_lex": 1,
        "cands": 1,
        "pubs_used": 1,
        "t_dense": 0.0,
        "t_lex": 0.0,
        "k_requested": kwargs.get("k"),
        "k_applied": kwargs.get("k"),
        "k_clamped": False,
        "mmr_cap": kwargs.get("mmr_k"),
        "fallback_retries": 0,
        "fallback_failed": 0,
        "dense_k": kwargs.get("dense_k"),
        "lex_k": kwargs.get("lex_k"),
        "dense_fallback": 0,
        "dense_fallback_fail": 0,
        "dense_clamped": False,
        "lex_clamped": False,
    }
    return [dict(sample_hit)], meta


def _mk_eng():
    return re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})


def test_modes_apply_retrieval_budgets(monkeypatch):
    calls = []

    def _spy_hybrid(e, q, pubs=None, qv=None, **kwargs):
        calls.append(kwargs)
        return _fake_retrieve(e, q, pubs=pubs, qv=qv, **kwargs)

    monkeypatch.setattr(re, "hybrid_retrieve", _spy_hybrid)
    eng = _mk_eng()
    res_quick = re.run_query(eng, "query context", pubs=["c"], use_llm=False, use_jdg=False, mode="quick")
    res_exact = re.run_query(eng, "query context", pubs=["c"], use_llm=False, use_jdg=False, mode="exact")

    quick_cfg = re.HCFG["modes"]["quick"]
    exact_cfg = re.HCFG["modes"]["exact"]

    assert calls[0]["k"] == quick_cfg["mmr_k"]
    assert calls[0]["dense_k"] == quick_cfg["dense_k"]
    assert calls[0]["lex_k"] == quick_cfg["lex_k"]
    assert res_quick["meta"]["clamp"]["retrieval"]["k_requested"] == quick_cfg["mmr_k"]
    assert res_quick["meta"]["mode"] == "quick"

    assert calls[1]["k"] == exact_cfg["mmr_k"]
    assert calls[1]["dense_k"] == exact_cfg["dense_k"]
    assert calls[1]["lex_k"] == exact_cfg["lex_k"]
    assert res_exact["meta"]["clamp"]["retrieval"]["k_requested"] == exact_cfg["mmr_k"]
    assert res_exact["meta"]["cap"]["mode_label"] == exact_cfg["label"]


def test_mode_logged_with_parameters(monkeypatch):
    monkeypatch.setattr(re, "hybrid_retrieve", lambda *a, **k: _fake_retrieve(*a, **k))
    eng = _mk_eng()
    res = re.run_query(eng, "query context", pubs=["c"], use_llm=False, use_jdg=False, mode="exact")
    log = res["meta"].get("log", {})
    mode_cfg = log.get("mode_cfg") or {}

    assert log.get("mode") == "exact"
    assert mode_cfg.get("final_k") == re.HCFG["modes"]["exact"]["final_k"]
    assert mode_cfg.get("mmr_k") == re.HCFG["modes"]["exact"]["mmr_k"]
    assert res["meta"]["mode_cfg"]["budget"]["ctx_chars"] == re.HCFG["budgets"]["exact"]["ctx_chars"]
