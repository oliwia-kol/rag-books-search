import os
import time

import numpy as np
import pytest

import rag_engine as re


@pytest.mark.skipif(os.environ.get("SKIP_PERF_CHECK") == "1", reason="Perf check disabled by environment")
def test_run_query_finishes_within_budget(monkeypatch):
    eng = re.Eng(emb=None, ix={"OReilly": object()}, dbp={"OReilly": object()}, corp={"OReilly": object()}, ix_dim={"OReilly": 4}, corp_report={})
    monkeypatch.setattr(re, "embed_query", lambda e, q: np.ones(4, dtype="float32"))

    hits = [
        {"cid": "p1", "cidx": 0, "fp": "p.txt", "sec": "intro", "corp": "OReilly", "text": "quick sanity check", "score": 0.8, "sem_score_n": 0.8},
        {"cid": "p2", "cidx": 1, "fp": "p.txt", "sec": "body", "corp": "OReilly", "text": "quick sanity follow-up", "score": 0.7, "sem_score_n": 0.7},
    ]
    meta = {
        "dense_hits": len(hits),
        "lex_hits": 0,
        "fetched_dense": len(hits),
        "fetched_lex": 0,
        "cands": len(hits),
        "pubs_used": 1,
        "t_dense": 0.0,
        "t_lex": 0.0,
    }
    monkeypatch.setattr(re, "hybrid_retrieve", lambda e, q, pubs=None, qv=None: (hits, meta))

    start = time.perf_counter()
    out = re.run_query(eng, "quick sanity", use_jdg=False)
    elapsed = time.perf_counter() - start

    assert out["ok"] is True
    assert elapsed < 1.5, f"run_query took too long: {elapsed}"
    assert out["meta"]["t"]["total"] <= elapsed
