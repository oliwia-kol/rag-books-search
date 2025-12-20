import time
from pathlib import Path

import numpy as np
import pytest

import rag_engine as re
import smoke_ui_contract


def test_smoke_ui_contract_runs(monkeypatch):
    def _fake_rerank(q, hs, mode=None):
        for h in hs:
            h["_jdg"] = float(h.get("score", 0.0))
            h.setdefault("judge01", h["_jdg"])
        return hs, {"ok": True, "kind": "proxy_score", "cache_hits": 0, "cache_misses": len(hs)}

    monkeypatch.setattr(re, "_jdg_rerank", _fake_rerank)
    monkeypatch.setattr(re, "embed_query", lambda e, q: np.ones(4, dtype="float32"))
    smoke_ui_contract.main()


@pytest.mark.slow
@pytest.mark.skipif(not Path("data").exists(), reason="corpus data missing; perf check is optional")
def test_latency_sanity_with_stubbed_retrieval(monkeypatch):
    eng = re.Eng(
        emb=None,
        ix={"Test": object()},
        dbp={"Test": object()},
        corp={"Test": Path("data/Test")},
        ix_dim={"Test": 4},
        corp_report={},
    )
    eng.corp_status = {"Test": {"loaded": True, "dim_ok": True}}

    monkeypatch.setattr(re, "embed_query", lambda e, q: np.ones(4, dtype="float32"))

    def fake_retrieve(*args, **kwargs):
        hits = [
            {
                "cid": "c1",
                "cidx": 0,
                "fp": "f.txt",
                "sec": "intro",
                "corp": "Test",
                "text": "latency sanity check snippet intro",
                "score": 0.52,
                "sem_score_n": 0.52,
            }
        ]
        return hits, {
            "dense_hits": 1,
            "lex_hits": 0,
            "fetched_dense": 1,
            "fetched_lex": 0,
            "cands": 1,
            "pubs_used": 1,
            "t_dense": 0.0,
            "t_lex": 0.0,
        }

    monkeypatch.setattr(re, "hybrid_retrieve", fake_retrieve)
    start = time.perf_counter()
    out = re.run_query(eng, "latency sanity check", use_jdg=False)
    elapsed = time.perf_counter() - start

    assert out["ok"] is True
    assert out["hits"]
    assert elapsed < 1.0
    assert out["meta"]["t"]["total"] <= elapsed + 0.05
