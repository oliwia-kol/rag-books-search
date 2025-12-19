import rag_engine as re


sample_hit = {
    "cid": "c1",
    "fp": "f",
    "sec": "s",
    "tx": "model quality metrics and monitoring steps",
    "cidx": 0,
    "score": 0.7,
    "sem_score_n": 0.7,
    "lex_score_n": 0.6,
}


def _fake_retrieve(*args, **kwargs):
    meta = {
        "dense_hits": 1,
        "lex_hits": 1,
        "fetched_dense": 1,
        "fetched_lex": 1,
        "cands": 1,
        "pubs_used": 1,
        "t_dense": 0.0,
        "t_lex": 0.0,
        "k_requested": kwargs.get("k", 5),
        "k_applied": kwargs.get("k", 5),
        "k_clamped": False,
        "dense_clamped": False,
        "lex_clamped": False,
        "fallback_retries": 0,
        "fallback_failed": 0,
    }
    return [dict(sample_hit)], meta


def test_mode_parameters_change(monkeypatch):
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    eng = re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})

    quick = re.run_query(eng, "model quality metrics", pubs=["c"], use_llm=False, mode="quick")
    quote = re.run_query(eng, "model quality metrics", pubs=["c"], use_llm=False, mode="find exact quote")

    assert quick["meta"]["mode"] == "quick"
    assert quote["meta"]["mode"] == "quote"
    assert quick["meta"]["mode_params"]["final_k"] != quote["meta"]["mode_params"]["final_k"]
    assert quick["meta"]["log"]["mode_params"]["mmr_k"] == quick["meta"]["mode_params"]["mmr_k"]
    assert quote["meta"]["log"]["mode"] == "quote"


def test_power_panel_fields_populated(monkeypatch):
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    eng = re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})

    res = re.run_query(eng, "model quality metrics", pubs=["c"], use_llm=False, mode="quick")

    hit = res["hits"][0]
    assert "sem_score_n" in hit and "lex_score_n" in hit and "judge01" in hit
    assert "cut_rule" in res["meta"]
    assert "mode_params" in res["meta"]
    assert res["meta"]["t"]["total"] >= 0
    assert res["meta"]["log"]["mode"] == res["meta"]["mode"]
