import rag_engine as re


def _fake_retrieve_quality(e, q, pubs=None, qv=None):
    q_lower = (q or "").lower()
    meta = {
        "dense_hits": 0,
        "lex_hits": 0,
        "fetched_dense": 0,
        "fetched_lex": 0,
        "cands": 0,
        "pubs_used": len(pubs or []),
        "t_dense": 0.0,
        "t_lex": 0.0,
    }
    hits = []
    if "easy" in q_lower:
        hits = [
            {"cid": "c1", "fp": "f", "sec": "s", "tx": "easy answer for test", "cidx": 0, "score": 0.82}
        ]
    elif "hard" in q_lower:
        hits = [
            {"cid": "c2", "fp": "f2", "sec": "s2", "tx": "hard case context", "cidx": 0, "score": 0.45}
        ]
    meta["dense_hits"] = len(hits)
    meta["lex_hits"] = len(hits)
    meta["fetched_dense"] = len(hits)
    meta["fetched_lex"] = len(hits)
    meta["cands"] = len(hits)
    return hits, meta


def test_easy_query_returns_ok(monkeypatch):
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve_quality)
    eng = re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})
    res = re.run_query(eng, "easy question", pubs=["c"], use_llm=False)
    assert res["ok"] is True
    assert res["no_evidence"] is False


def test_hard_query_still_returns_hits(monkeypatch):
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve_quality)
    eng = re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})
    res = re.run_query(eng, "hard topic", pubs=["c"], use_llm=False)
    assert res["hits"]
    assert res["meta"]["n"]["cands"] >= 1


def test_noise_query_handles_missing(monkeypatch):
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve_quality)
    eng = re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})
    res = re.run_query(eng, "noise", pubs=["c"], use_llm=False)
    assert res["ok"] in {True, False}  # should not crash
    assert res["meta"]["n"]["cands"] in {0, 1}


def test_missing_corpus_reports_cleanly():
    eng = re.Eng(emb=None, ix={}, dbp={}, corp={}, ix_dim={}, corp_report={})
    res = re.run_query(eng, "anything")
    assert res["ok"] is False
    assert res["no_evidence"] is True
    assert "No corpus" in res["answer"]
