import rag_engine as re


sample_hit = {"cid": "c1", "fp": "f", "sec": "s", "tx": "query content text", "cidx": 0, "score": 0.7}


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
        "dense_fallback": 0,
        "dense_fallback_fail": 0,
        "fallback_retries": 0,
        "fallback_failed": 0,
    }


def _mk_eng():
    return re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})


class _FakeCE:
    def __init__(self, val: float):
        self.val = val
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return [self.val for _ in pairs]


def test_real_mode_uses_cross_encoder(monkeypatch):
    ce = _FakeCE(2.0)
    monkeypatch.setattr(re, "_get_jdg", lambda: ce)
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    re._JDG_CACHE.clear()
    eng = _mk_eng()
    res = re.run_query(eng, "query", pubs=["c"], use_llm=False, judge_mode="real")

    assert res["meta"]["cap"]["judge_kind"] == "cross_encoder"
    assert res["meta"]["cap"]["judge_ok"] is True
    assert res["meta"]["flags"]["judge_proxy"] is False
    assert res["meta"]["flags"]["veto_disabled"] is False
    assert ce.calls == 1


def test_real_mode_applies_veto_on_low_scores(monkeypatch):
    ce = _FakeCE(-5.0)
    monkeypatch.setattr(re, "_get_jdg", lambda: ce)
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    re._JDG_CACHE.clear()
    eng = _mk_eng()

    res = re.run_query(eng, "query", pubs=["c"], use_llm=False, judge_mode="real")
    assert res["meta"]["cap"]["judge_ok"] is True
    assert res["meta"]["flags"]["veto_applied"] is True
    assert res["meta"]["flags"]["veto_disabled"] is False


def test_proxy_mode_disables_veto_and_marks_proxy(monkeypatch):
    ce = _FakeCE(-3.0)
    monkeypatch.setattr(re, "_get_jdg", lambda: ce)
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    re._JDG_CACHE.clear()
    eng = _mk_eng()

    res = re.run_query(eng, "query", pubs=["c"], use_llm=False, judge_mode="proxy")
    assert res["meta"]["cap"]["judge_kind"] == "proxy_score"
    assert res["meta"]["cap"]["judge_ok"] is False
    assert res["meta"]["flags"]["veto_applied"] is False
    assert res["meta"]["flags"]["veto_disabled"] is True
    assert res["meta"]["flags"]["veto_disabled_when_proxy"] is True
    assert res["meta"]["flags"]["judge_proxy"] is True


def test_off_mode_disables_judge(monkeypatch):
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    re._JDG_CACHE.clear()
    eng = _mk_eng()

    res = re.run_query(eng, "query", pubs=["c"], use_llm=False, judge_mode="off")
    assert res["meta"]["cap"]["judge_kind"] == "off"
    assert res["meta"]["cap"]["judge_ok"] is False
    assert res["meta"]["flags"]["veto_disabled"] is True
    assert res["meta"]["flags"]["veto_disabled_when_proxy"] is True
    assert res["meta"]["flags"]["judge_proxy"] is False


def test_real_mode_falls_back_to_proxy_when_judge_unavailable(monkeypatch):
    monkeypatch.setattr(re, "_get_jdg", lambda: None)
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    re._JDG_CACHE.clear()
    eng = _mk_eng()

    res = re.run_query(eng, "query", pubs=["c"], use_llm=False, judge_mode="real")
    assert res["meta"]["cap"]["judge_kind"] == "proxy_score"
    assert res["meta"]["cap"]["judge_ok"] is False
    assert res["meta"]["flags"]["judge_proxy"] is True
    assert res["meta"]["flags"]["veto_disabled"] is True
    assert res["meta"]["flags"]["veto_disabled_when_proxy"] is True


def test_cache_hits_are_tracked(monkeypatch):
    ce = _FakeCE(1.0)
    monkeypatch.setattr(re, "_get_jdg", lambda: ce)
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve)
    re._JDG_CACHE.clear()
    eng = _mk_eng()

    first = re.run_query(eng, "query", pubs=["c"], use_llm=False, judge_mode="real")
    second = re.run_query(eng, "query", pubs=["c"], use_llm=False, judge_mode="real")

    assert ce.calls == 1  # second call should hit cache
    assert second["meta"]["cap"]["judge_kind"] == "cross_encoder"
    assert second["meta"]["log"]["judge_cache_hits"] >= 1
    assert second["meta"]["t"]["judge_cache"] >= 0.0
    assert first["meta"]["log"]["judge_cache_misses"] >= 0
