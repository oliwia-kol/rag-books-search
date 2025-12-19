import rag_engine as re
import ui_adapter


sample_hit = {"cid": "c1", "fp": "f", "sec": "s", "tx": "text about python", "cidx": 0, "score": 0.7}


def _fake_retrieve_multi(*args, **kwargs):
    hs = []
    for i in range(5):
        hs.append({"cid": f"c{i}", "fp": f"f{i}", "sec": "s", "tx": f"text about python {i}", "cidx": i, "score": 0.5 - i * 0.01})
    return hs, {
        "dense_hits": 0,
        "lex_hits": 5,
        "fetched_dense": 0,
        "fetched_lex": 5,
        "cands": 5,
        "pubs_used": 1,
        "t_dense": 0.0,
        "t_lex": 0.0,
    }


def _fake_rerank(q, hs, mode=None):
    for i, h in enumerate(hs):
        h["judge01"] = 0.4 + (i * 0.01)
        h["_jdg"] = h["judge01"]
    return hs, {"ok": True, "kind": "cross_encoder"}


def _fake_eng():
    return re.Eng(emb=None, ix={}, dbp={"c": None}, corp={"c": None}, ix_dim={}, corp_report={})


def test_compute_near_miss_flag_skips(monkeypatch):
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve_multi)
    monkeypatch.setattr(re, "_jdg_rerank", _fake_rerank)
    eng = _fake_eng()
    res = re.run_query(eng, "python packages", pubs=["c"], use_llm=False, jdg_mode="real", compute_near_miss=False)
    assert res["no_evidence"] is True
    assert res["near_miss"] == []
    assert res["meta"]["n"]["near_miss"] == 0
    assert res["meta"].get("meta_nm", {}).get("skipped") is True


def test_near_miss_metadata_and_explanation(monkeypatch):
    monkeypatch.setattr(re, "hybrid_retrieve", _fake_retrieve_multi)
    monkeypatch.setattr(re, "_jdg_rerank", _fake_rerank)
    eng = _fake_eng()
    res = re.run_query(eng, "python", pubs=["c"], use_llm=False, jdg_mode="real")
    assert res["no_evidence"] is True
    assert res["near_miss"]
    nm_hit = res["near_miss"][0]
    assert "overlap_count" in nm_hit
    assert nm_hit.get("near_miss_explanation") == re.NEAR_MISS_EXPLANATION
    assert res["meta"].get("near_miss_threshold") == re.NM_MIN
    assert res["meta"].get("meta_nm", {}).get("count") == len(res["near_miss"])


class _DummyCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_render_near_miss_shows_explanation(monkeypatch):
    class _StubStreamlit:
        def __init__(self):
            self.calls = []
            self.session_state = {}

        def subheader(self, msg):
            self.calls.append(("subheader", msg))

        def caption(self, msg):
            self.calls.append(("caption", msg))

        def container(self, border=False):
            self.calls.append(("container", border))
            return _DummyCtx()

        def markdown(self, msg, **kwargs):
            self.calls.append(("markdown", msg))

        def columns(self, n):
            self.calls.append(("columns", n))
            return [_DummyCtx() for _ in range(n)]

        def button(self, *args, **kwargs):
            self.calls.append(("button", kwargs.get("key")))
            return False

        def expander(self, label):
            self.calls.append(("expander", label))
            return _DummyCtx()

        def write(self, msg):
            self.calls.append(("write", msg))

    stub = _StubStreamlit()
    monkeypatch.setattr(ui_adapter, "st", stub)
    rr = {
        "no_evidence": True,
        "near_miss": [
            {
                "cid": "c1",
                "cidx": 0,
                "fp": "f",
                "sec": "s",
                "tx": "text about python",
                "score": 0.4,
                "judge01": 0.4,
                "overlap_count": 2,
                "near_miss_explanation": re.NEAR_MISS_EXPLANATION,
            }
        ],
        "meta": {"meta_nm": {"threshold": re.NM_MIN, "used_judge": True, "explanation": re.NEAR_MISS_EXPLANATION}},
    }
    ui_adapter.render_near_miss(rr, q="python")
    captions = [msg for name, msg in stub.calls if name == "caption"]
    assert any(re.NEAR_MISS_EXPLANATION in str(msg) for msg in captions)
    assert any("Near miss" in str(msg) and "overlap" in str(msg) for msg in captions)
