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


def test_judge_mode_off_disables_veto(monkeypatch):
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    eng = re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})
    res = re.run_query(eng, "query", pubs=["c"], use_llm=False, judge_mode="off")
    assert res["meta"]["cap"]["judge_kind"] == "off"
    assert res["meta"]["cap"]["judge_ok"] is False
    assert res["meta"]["flags"]["veto_applied"] is False
    assert res["meta"]["flags"]["veto_disabled"] is True


def test_real_mode_enables_veto(monkeypatch):
    class FakeCE:
        calls = 0

        def predict(self, pairs):
            FakeCE.calls += 1
            # return a low score to trigger veto
            return [-2.0 for _ in pairs]

    monkeypatch.setattr(re, "_get_jdg", lambda: FakeCE())
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    re._JDG_CACHE.clear()
    eng = re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})
    res = re.run_query(eng, "query", pubs=["c"], use_llm=False, judge_mode="real")
    assert res["meta"]["cap"]["judge_kind"] == "cross_encoder"
    assert res["meta"]["flags"]["veto_applied"] is True
    assert res["meta"]["flags"]["veto_disabled_when_proxy"] is False
    assert FakeCE.calls == 1


def test_cross_encoder_cache_hits(monkeypatch):
    class FakeCE:
        def __init__(self):
            self.calls = 0

        def predict(self, pairs):
            self.calls += 1
            return [0.8 for _ in pairs]

    fake_ce = FakeCE()
    monkeypatch.setattr(re, "_get_jdg", lambda: fake_ce)
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    re._JDG_CACHE.clear()
    eng = re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})
    first = re.run_query(eng, "query", pubs=["c"], use_llm=False, judge_mode="real")
    second = re.run_query(eng, "query", pubs=["c"], use_llm=False, judge_mode="real")

    assert fake_ce.calls == 1  # second call should hit cache
    assert second["meta"]["log"]["judge_cache_hits"] >= 1
    assert second["meta"]["t"]["judge_pred"] >= 0.0
    assert second["meta"]["t"]["judge_cache"] >= 0.0
