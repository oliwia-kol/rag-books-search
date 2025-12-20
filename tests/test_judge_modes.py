import rag_engine as re


sample_hit = {"cid": "c1", "fp": "f", "sec": "s", "tx": "text", "cidx": 0, "score": 0.7}


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


def test_proxy_judge_sets_flags(monkeypatch):
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    eng = re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})
    res = re.run_query(eng, "q", pubs=["c"], use_llm=False, jdg_mode="proxy")
    assert res["meta"]["cap"]["judge_kind"] in {"proxy_score", "none", "off"}
    assert res["meta"]["flags"]["veto_disabled_when_proxy"] is True
    assert res["meta"]["flags"]["judge_proxy"] is True


def test_real_judge_path_sets_cross_encoder(monkeypatch):
    class FakeCE:
        def predict(self, pairs):
            return [0.9 for _ in pairs]

    monkeypatch.setattr(re, "_get_jdg", lambda: FakeCE())
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    eng = re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})
    res = re.run_query(eng, "q", pubs=["c"], use_llm=False, jdg_mode="real")
    assert res["meta"]["cap"]["judge_kind"] == "cross_encoder"
    assert res["meta"]["cap"]["judge_ok"] is True
    assert res["meta"]["flags"]["judge_proxy"] is False


def test_judge_cache_hits(monkeypatch):
    class FakeCE:
        def __init__(self):
            self.calls = 0

        def predict(self, pairs):
            self.calls += 1
            return [0.9 for _ in pairs]

    fake_ce = FakeCE()

    monkeypatch.setattr(re, "_get_jdg", lambda: fake_ce)
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    re._JDG_CACHE.clear()
    eng = re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})
    res1 = re.run_query(eng, "q", pubs=["c"], use_llm=False, jdg_mode="real")
    res2 = re.run_query(eng, "q", pubs=["c"], use_llm=False, jdg_mode="real")
    assert res2["meta"]["cap"]["judge_kind"] == "cross_encoder"
    assert res2["meta"]["cap"]["judge_ok"] is True
    assert res2["meta"]["log"]["flags"]["judge_proxy"] is False
    # cache hit expected on second call (predict should not be called again)
    assert fake_ce.calls == 1
