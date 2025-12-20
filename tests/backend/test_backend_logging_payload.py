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


def test_log_payload_includes_scope_and_flags(monkeypatch):
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    eng = re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})
    res = re.run_query(eng, "query context", pubs=["c"], use_llm=False)
    log = res["meta"].get("log") or {}

    assert log.get("scope", {}).get("requested") == ["c"]
    assert log.get("scope", {}).get("used") == ["c"]
    assert isinstance(log.get("durations"), dict)
    assert isinstance(log.get("counts"), dict)
    assert log.get("judge", {}).get("mode") == log.get("judge_mode")
    assert "clamp" in log
    assert "llm_err" in log
