import rag_engine as re


sample_hit = {"cid": "c1", "fp": "f", "sec": "s", "tx": "query context with evidence", "cidx": 0, "score": 0.6}


def _fake_retrieve(*args, **kwargs):
    return [dict(sample_hit)], {
        "dense_hits": 0,
        "lex_hits": 1,
        "fetched_dense": 0,
        "fetched_lex": 1,
        "cands": 1,
        "pubs_used": 1,
        "t_dense": 0.0,
        "t_lex": 0.0,
        "fallback_retries": 0,
        "fallback_failed": 0,
        "dense_fallback": 0,
        "dense_fallback_fail": 0,
        "k_requested": re.HCFG["mmr_k"],
        "k_applied": re.HCFG["mmr_k"],
        "k_clamped": False,
        "dense_clamped": False,
        "lex_clamped": False,
    }


def test_log_payload_has_required_keys(monkeypatch):
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    eng = re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})
    res = re.run_query(eng, "q", pubs=["c"], use_llm=False)
    log = res["meta"].get("log") or {}
    required = ["ts", "mode", "scope", "counts", "durations", "judge", "no_evidence", "clamp", "error_id", "llm_err"]
    for key in required:
        assert key in log
    assert log["scope"]["requested"] == ["c"]
    assert log["scope"]["used"] == 1
    assert set(re.STAGES).issubset(set(log["durations"].keys()))
    assert set(["k_requested", "k_applied", "k_clamped"]).issubset(set(log["clamp"].keys()))
    assert isinstance(log["no_evidence"], bool)


def test_error_id_propagates_and_logs_llm(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("llm boom")

    monkeypatch.setattr(re, "llm_call", boom)
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    eng = re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})
    res = re.run_query(eng, "query", pubs=["c"], use_llm=True)
    err_id = res["meta"].get("err", {}).get("id")
    assert err_id
    assert res["meta"].get("log", {}).get("error_id") == err_id
    assert res["meta"].get("log", {}).get("llm_err")
    assert res["meta"]["log"]["llm_err"]["msg"]
    assert "llm" in res["meta"]["log"]["durations"]


def test_error_id_is_stable(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("repeatable boom")

    monkeypatch.setattr(re, "llm_call", boom)
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    eng = re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})
    res1 = re.run_query(eng, "query", pubs=["c"], use_llm=True)
    res2 = re.run_query(eng, "query", pubs=["c"], use_llm=True)
    assert res1["meta"].get("err", {}).get("id") == res2["meta"].get("err", {}).get("id")
