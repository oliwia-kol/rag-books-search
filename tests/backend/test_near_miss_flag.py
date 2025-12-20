import rag_engine as re


sample_hit = {
    "cid": "c1",
    "fp": "f",
    "sec": "s",
    "tx": "text about python and streaming",
    "cidx": 0,
    "score": 0.55,
    "corp": "c",
}


def _fake_retrieve_multi(*args, **kwargs):
    hits = []
    for i, j01 in enumerate([0.45, 0.38, 0.32, 0.22, 0.18, 0.12]):
        h = dict(sample_hit)
        h["cidx"] = i
        h["cid"] = f"cid-{i}"
        h["score"] = sample_hit["score"] - (i * 0.05)
        h["tx"] = f"{sample_hit['tx']} #{i}"
        h["judge01"] = j01
        hits.append(h)
    return hits, {
        "dense_hits": 0,
        "lex_hits": len(hits),
        "fetched_dense": 0,
        "fetched_lex": len(hits),
        "cands": len(hits),
        "pubs_used": 1,
        "t_dense": 0.0,
        "t_lex": 0.0,
    }


def _fake_rerank(q, hs, mode=None):
    for h in hs:
        if h.get("judge01") is None:
            h["judge01"] = 0.4
        h["_jdg"] = h["judge01"]
    return hs, {"ok": True, "kind": "cross_encoder"}


def _mk_eng():
    return re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})


def test_compute_near_miss_can_be_disabled(monkeypatch):
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve_multi)
    monkeypatch.setattr(re, "_jdg_rerank", _fake_rerank)

    eng = _mk_eng()
    res = re.run_query(eng, "python streaming", pubs=["c"], use_llm=False, jdg_mode="real", compute_near_miss=False)

    assert res["no_evidence"] is True
    assert res["near_miss"] == []
    assert res["meta"]["flags"].get("near_miss_skipped") is True
    assert res["meta"]["meta_nm"].get("reason") == "compute_near_miss_disabled"


def test_near_miss_metadata_and_counts(monkeypatch):
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve_multi)
    monkeypatch.setattr(re, "_jdg_rerank", _fake_rerank)

    eng = _mk_eng()
    res = re.run_query(eng, "python streaming", pubs=["c"], use_llm=False, jdg_mode="real")

    nm = res["near_miss"]
    assert 3 <= len(nm) <= 6
    assert res["meta"]["meta_nm"].get("threshold") == re.NM_MIN
    assert res["meta"]["meta_nm"].get("used_judge") is True
    assert res["meta"]["meta_nm"].get("count") == len(nm)
    for h in nm:
        assert "overlap" in h
        assert h.get("near_miss_threshold") == re.NM_MIN
        assert h.get("used_judge") is True
        assert "explanation" in h
