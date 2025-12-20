import rag_engine as re


sample_hit = {"cid": "c1", "fp": "f", "sec": "s", "tx": "text about python", "cidx": 0, "score": 0.7}


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


def _fake_rerank(q, hs, mode=None):
    for h in hs:
        h["judge01"] = 0.4
        h["_jdg"] = 0.1
    return hs, {"ok": True, "kind": "cross_encoder"}


def test_near_miss_returns_results_when_no_direct_hits(monkeypatch):
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    monkeypatch.setattr(re, "_jdg_rerank", _fake_rerank)
    eng = re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})
    res = re.run_query(eng, "python", pubs=["c"], use_llm=False, jdg_mode="real")
    assert res["no_evidence"] is True
    assert res["near_miss"]
    assert res["meta"].get("meta_nm", {}).get("threshold") is not None
    first = res["near_miss"][0]
    assert "near_miss_threshold" in first
    assert "used_judge" in first
