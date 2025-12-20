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
    }


def test_log_payload_has_required_keys(monkeypatch):
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    eng = re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})
    res = re.run_query(eng, "q", pubs=["c"], use_llm=False)
    log = res["meta"].get("log") or {}
    for k in ["ts", "mode", "scope", "counts", "durations", "judge_ok", "no_evidence", "clamp"]:
        assert k in log
    assert "judge_kind" in log
    assert "llm_dur" in log


def test_error_id_propagates(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("llm boom")

    monkeypatch.setattr(re, "llm_call", boom)
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    eng = re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})
    res = re.run_query(eng, "query", pubs=["c"], use_llm=True)
    assert res["meta"].get("err", {}).get("id")
    assert res["meta"].get("log", {}).get("error_id") == res["meta"].get("err", {}).get("id")
    assert res["meta"]["err"].get("id", "").startswith("err-")
