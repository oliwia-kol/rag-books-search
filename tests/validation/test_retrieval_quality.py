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


def _mk_hits(kind="dense"):
    if kind == "dense":
        return [{"cid": "1", "cidx": 0, "fp": "b.txt", "sec": "intro", "corp": "OReilly", "text": "abc", "sem_score_n": 0.9}]
    if kind == "lex":
        return [{"cid": "2", "cidx": 1, "fp": "b.txt", "sec": "intro", "corp": "OReilly", "text": "abc", "lex_score_n": 0.8}]
    return [
        {"cid": "3", "cidx": 2, "fp": "b.txt", "sec": "body", "corp": "OReilly", "text": "abc", "sem_score_n": 0.7},
        {"cid": "4", "cidx": 3, "fp": "b.txt", "sec": "body", "corp": "OReilly", "text": "abc", "lex_score_n": 0.6},
    ]


@pytest.fixture(autouse=True)
def patch_embed(monkeypatch):
    monkeypatch.setattr(re, "embed_query", lambda e, q: np.ones(4, dtype="float32"))
    return


def test_dense_only(monkeypatch):
    eng = FakeEng()
    monkeypatch.setattr(re, "hybrid_retrieve", lambda e, q, pubs=None, qv=None: (_mk_hits("dense"), {"dense_hits": 1, "lex_hits": 0, "fetched_dense": 1, "fetched_lex": 0, "cands": 1, "pubs_used": 1, "t_dense": 0, "t_lex": 0}))
    out = re.run_query(eng, "easy question", use_jdg=False)
    assert out["ok"] is True
    assert out["hits"]
    assert out["meta"]["n"]["dense_hits"] == 1
    assert out["meta"]["n"]["lex_hits"] == 0


def test_lex_only(monkeypatch):
    eng = FakeEng()
    monkeypatch.setattr(re, "hybrid_retrieve", lambda e, q, pubs=None, qv=None: (_mk_hits("lex"), {"dense_hits": 0, "lex_hits": 1, "fetched_dense": 0, "fetched_lex": 1, "cands": 1, "pubs_used": 1, "t_dense": 0, "t_lex": 0}))
    out = re.run_query(eng, "keyword", use_jdg=False)
    assert out["ok"]
    assert out["hits"]
    assert out["meta"]["n"]["lex_hits"] == 1


def test_hybrid_and_near_miss(monkeypatch):
    eng = FakeEng()
    monkeypatch.setattr(re, "hybrid_retrieve", lambda e, q, pubs=None, qv=None: (_mk_hits("hybrid"), {"dense_hits": 1, "lex_hits": 1, "fetched_dense": 1, "fetched_lex": 1, "cands": 2, "pubs_used": 1, "t_dense": 0, "t_lex": 0}))
    out = re.run_query(eng, "hard question", use_jdg=False, nm=True)
    assert out["ok"]
    assert out["hits"]
    assert out["meta"]["cut_rule"]
    assert out["meta"]["n"]["after_cut"] >= 1


def test_missing_corpus(monkeypatch):
    eng = FakeEng()
    eng.corp = {}
    eng.ix = {}
    eng.dbp = {}
    monkeypatch.setattr(re, "hybrid_retrieve", lambda *a, **k: ([], {"dense_hits": 0, "lex_hits": 0, "fetched_dense": 0, "fetched_lex": 0, "cands": 0, "pubs_used": 0, "t_dense": 0, "t_lex": 0}))
    out = re.run_query(eng, "noise", use_jdg=False)
    assert out["ok"] is False
    assert "No corpus indexes" in out["answer"] or out["meta"]["cap"]["dense_reason"] == "no_dense_index"
