import types

import numpy as np
import pytest

import rag_engine as re


class FakeEng:
    def __init__(self):
        self.emb = types.SimpleNamespace(get_sentence_embedding_dimension=lambda: 4, encode=lambda q, convert_to_numpy=True: np.ones(4))
        self.ix = {"OReilly": object()}
        self.dbp = {"OReilly": object()}
        self.corp = {"OReilly": object()}
        self.ix_dim = {"OReilly": 4}
        self.corp_status = {"OReilly": {"loaded": True, "dim_ok": True}}


def _mk_hit(cid: str, score: float, *, lex_score: float | None = None, text: str = "easy question text", section: str = "intro"):
    base = {
        "cid": cid,
        "cidx": int(cid.replace("cid-", "").replace("h", "") or 0),
        "fp": "book.txt",
        "sec": section,
        "corp": "OReilly",
        "book": "book",
        "publisher": "OReilly",
        "text": text,
        "score": score,
        "sem_score_n": score,
        "lex_score_n": lex_score or 0.0,
        "judge01": score,
    }
    return base


def _mk_meta(dense_hits: int, lex_hits: int, *, k_requested=None, k_applied=None, k_clamped=False):
    return {
        "dense_hits": dense_hits,
        "lex_hits": lex_hits,
        "fetched_dense": dense_hits,
        "fetched_lex": lex_hits,
        "cands": dense_hits + lex_hits,
        "pubs_used": 1 if dense_hits or lex_hits else 0,
        "t_dense": 0.0,
        "t_lex": 0.0,
        "k_requested": re.HCFG["final_k"] if k_requested is None else k_requested,
        "k_applied": re.HCFG["final_k"] if k_applied is None else k_applied,
        "k_clamped": k_clamped,
    }


@pytest.fixture(autouse=True)
def patch_embed(monkeypatch):
    monkeypatch.setattr(re, "embed_query", lambda e, q: np.ones(4, dtype="float32"))
    return


def test_dense_only_easy_query_prefers_high_coverage(monkeypatch):
    eng = FakeEng()
    hits = [
        _mk_hit("1", 0.91, text="easy question context and answer"),
        _mk_hit("2", 0.89, text="easy question explanation detail"),
        _mk_hit("3", 0.9, text="easy question walkthrough"),
    ]
    monkeypatch.setattr(re, "hybrid_retrieve", lambda e, q, pubs=None, qv=None: (hits, _mk_meta(dense_hits=len(hits), lex_hits=0)))

    out = re.run_query(eng, "easy question", use_jdg=False)

    assert out["ok"] is True
    assert len(out["hits"]) == len(hits)
    assert out["coverage"] == "HIGH"
    assert out["meta"]["n"]["dense_hits"] == len(hits)
    assert out["meta"]["n"]["lex_hits"] == 0
    assert out["meta"]["cap"]["k_clamped"] is False


def test_lex_only_noise_query_reports_k_clamp(monkeypatch):
    eng = FakeEng()
    hits = [_mk_hit("10", 0.72, lex_score=0.72, text="keyword noise example with overlap")]
    requested_k = re.HCFG["mmr_k"] + 5
    meta = _mk_meta(dense_hits=0, lex_hits=len(hits), k_requested=requested_k, k_applied=re.HCFG["mmr_k"], k_clamped=True)
    monkeypatch.setattr(re, "hybrid_retrieve", lambda e, q, pubs=None, qv=None: (hits, meta))

    out = re.run_query(eng, "keyword noise", use_jdg=False)

    assert out["ok"] is True
    assert out["hits"]
    assert out["coverage"] == "OK"
    assert out["meta"]["flags"]["lex_used"] is True
    assert out["meta"]["cap"]["k_requested"] == requested_k
    assert out["meta"]["cap"]["k_applied"] == re.HCFG["mmr_k"]
    assert out["meta"]["cap"]["k_clamped"] is True


def test_hard_query_hybrid_path_is_distributed(monkeypatch):
    eng = FakeEng()
    hits = [
        _mk_hit("h1", 0.66, text="hard question overlap strong"),
        _mk_hit("h2", 0.65, lex_score=0.51, text="hard question deeper discussion"),
        _mk_hit("h3", 0.42, lex_score=0.6, text="hard question evidence spread"),
    ]
    monkeypatch.setattr(re, "hybrid_retrieve", lambda e, q, pubs=None, qv=None: (hits, _mk_meta(dense_hits=2, lex_hits=3)))

    out = re.run_query(eng, "hard question", use_jdg=False)

    assert out["ok"] is True
    assert out["coverage"] == "DISTRIBUTED"
    assert out["meta"]["n"]["after_cut"] >= 3
    assert out["meta"]["n"]["direct_hits"] >= 2
    assert out["meta"]["n"]["uniq_books"] >= 1


def test_missing_corpus_reports_failure(monkeypatch):
    eng = FakeEng()
    eng.corp = {}
    eng.ix = {}
    eng.dbp = {}
    monkeypatch.setattr(re, "hybrid_retrieve", lambda *a, **k: ([], _mk_meta(dense_hits=0, lex_hits=0)))

    out = re.run_query(eng, "noise", use_jdg=False)

    assert out["ok"] is False
    assert out["coverage"] == "WEAK"
    assert out["hits"] == []
    assert out["meta"]["cap"]["dense_reason"] in {"no_dense_index", "no_embed_model"}
