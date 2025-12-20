import types

import numpy as np
import pytest

import rag_engine as re


class FakeEng:
    def __init__(self):
        self.emb = types.SimpleNamespace(
            get_sentence_embedding_dimension=lambda: 4, encode=lambda q, convert_to_numpy=True: np.ones(4)
        )
        self.ix = {"OReilly": object()}
        self.dbp = {"OReilly": object()}
        self.corp = {"OReilly": object()}
        self.ix_dim = {"OReilly": 4}
        self.corp_report = {}
        self.corp_status = {"OReilly": {"loaded": True, "dim_ok": True}}


@pytest.fixture(autouse=True)
def patch_embed(monkeypatch):
    monkeypatch.setattr(re, "embed_query", lambda e, q: np.ones(4, dtype="float32"))


@pytest.fixture
def fake_judge(monkeypatch):
    def _fake_rerank(q, hs, mode=None):
        for h in hs:
            if h.get("judge01") is None:
                h["judge01"] = float(h.get("score", 0.0))
            h["_jdg"] = h["judge01"]
        return hs, {"ok": True, "kind": "cross_encoder", "n": len(hs), "cache_hits": 0, "cache_misses": len(hs)}

    monkeypatch.setattr(re, "_jdg_rerank", _fake_rerank)
    return _fake_rerank


def test_dense_only_easy_query_tracks_clamping_and_coverage(monkeypatch):
    eng = FakeEng()

    def fake_dense(*args, **kwargs):
        hits = [
            {
                "cid": "1",
                "cidx": 0,
                "fp": "b.txt",
                "sec": "intro",
                "corp": "OReilly",
                "text": "easy question about retrieval quality intro",
                "score": 0.92,
                "sem_score_n": 0.92,
            },
            {
                "cid": "2",
                "cidx": 1,
                "fp": "b.txt",
                "sec": "body",
                "corp": "OReilly",
                "text": "easy question secondary evidence body",
                "score": 0.91,
                "sem_score_n": 0.91,
            },
        ]
        return hits, {
            "dense_hits": len(hits),
            "lex_hits": 0,
            "fetched_dense": len(hits),
            "fetched_lex": 0,
            "cands": len(hits),
            "pubs_used": 1,
            "t_dense": 0.0,
            "t_lex": 0.0,
            "k_requested": re.HCFG["mmr_k"] + 5,
            "k_applied": re.HCFG["mmr_k"],
            "k_clamped": True,
            "dense_k": re.HCFG["dense_k"],
            "lex_k": re.HCFG["lex_k"],
            "mmr_cap": re.HCFG["mmr_k"],
            "dense_clamped": True,
            "lex_clamped": False,
        }

    monkeypatch.setattr(re, "hybrid_retrieve", fake_dense)
    out = re.run_query(eng, "easy question", use_jdg=False)

    assert out["ok"] is True
    assert out["hits"]
    assert out["coverage"] == "HIGH"
    assert out["meta"]["cov"]["mx"] >= 0.9
    assert out["meta"]["n"]["dense_hits"] == 2
    assert out["meta"]["cap"]["k_clamped"] is True
    assert out["meta"]["cap"]["k_applied"] == re.HCFG["mmr_k"]
    assert out["meta"]["flags"]["dense_clamped"] is True


def test_lex_only_hard_query_reports_judge_distribution(monkeypatch, fake_judge):
    eng = FakeEng()

    def fake_lex(*args, **kwargs):
        hits = [
            {
                "cid": "10",
                "cidx": 0,
                "fp": "b.txt",
                "sec": "intro",
                "corp": "OReilly",
                "text": "hard keyword lookup snippet intro",
                "score": 0.62,
                "lex_score_n": 0.82,
                "judge01": 0.82,
            },
            {
                "cid": "11",
                "cidx": 1,
                "fp": "b.txt",
                "sec": "body",
                "corp": "OReilly",
                "text": "hard keyword lookup contextual body",
                "score": 0.58,
                "lex_score_n": 0.75,
                "judge01": 0.71,
            },
            {
                "cid": "12",
                "cidx": 2,
                "fp": "b.txt",
                "sec": "appendix",
                "corp": "OReilly",
                "text": "hard keyword lookup appendix section",
                "score": 0.55,
                "lex_score_n": 0.66,
                "judge01": 0.66,
            },
        ]
        return hits, {
            "dense_hits": 0,
            "lex_hits": len(hits),
            "fetched_dense": 0,
            "fetched_lex": len(hits),
            "cands": len(hits),
            "pubs_used": 1,
            "t_dense": 0.0,
            "t_lex": 0.0,
            "k_requested": re.HCFG["mmr_k"],
            "k_applied": re.HCFG["mmr_k"],
            "k_clamped": False,
        }

    monkeypatch.setattr(re, "hybrid_retrieve", fake_lex)
    out = re.run_query(eng, "hard keyword lookup", use_llm=False, jdg_mode="real")

    assert out["ok"] is True
    assert out["hits"]
    assert out["coverage"] == "DISTRIBUTED"
    assert out["meta"]["cov"]["uc"] >= 2
    assert out["meta"]["n"]["lex_hits"] == 3
    assert out["meta"]["cap"]["judge_ok"] is True
    assert out["meta"]["cap"]["judge_kind"] == "cross_encoder"
    assert all("judge01" in h for h in out["hits"])


def test_hybrid_near_miss_thresholds_are_included(monkeypatch, fake_judge):
    eng = FakeEng()

    def fake_hybrid(*args, **kwargs):
        hits = [
            {
                "cid": "21",
                "cidx": 0,
                "fp": "b.txt",
                "sec": "intro",
                "corp": "OReilly",
                "text": "hybrid noisy question intro",
                "score": 0.52,
                "sem_score_n": 0.52,
                "judge01": 0.52,
            },
            {
                "cid": "22",
                "cidx": 1,
                "fp": "b.txt",
                "sec": "body",
                "corp": "OReilly",
                "text": "hybrid noisy question body",
                "score": 0.41,
                "lex_score_n": 0.41,
                "judge01": 0.41,
            },
        ]
        return hits, {
            "dense_hits": 1,
            "lex_hits": 1,
            "fetched_dense": 1,
            "fetched_lex": 1,
            "cands": len(hits),
            "pubs_used": 1,
            "t_dense": 0.0,
            "t_lex": 0.0,
            "k_requested": re.HCFG["mmr_k"],
            "k_applied": re.HCFG["mmr_k"],
            "k_clamped": False,
        }

    monkeypatch.setattr(re, "hybrid_retrieve", fake_hybrid)
    out = re.run_query(eng, "hybrid noisy question", use_llm=False, nm=True)

    assert out["ok"] is True
    assert out["no_evidence"] is True  # near-miss only path
    assert out["near_miss"]
    assert out["meta"]["meta_nm"]["threshold"] == re.NM_MIN
    assert out["meta"]["meta_nm"]["used_judge"] is True
    assert all(h.get("near_miss_threshold") == re.NM_MIN for h in out["near_miss"])


def test_missing_corpus(monkeypatch):
    eng = FakeEng()
    eng.corp = {}
    eng.ix = {}
    eng.dbp = {}
    monkeypatch.setattr(
        re,
        "hybrid_retrieve",
        lambda *a, **k: (
            [],
            {"dense_hits": 0, "lex_hits": 0, "fetched_dense": 0, "fetched_lex": 0, "cands": 0, "pubs_used": 0, "t_dense": 0, "t_lex": 0},
        ),
    )
    out = re.run_query(eng, "noise", use_jdg=False)
    assert out["ok"] is False
    assert "No corpus indexes" in out["answer"] or out["meta"]["cap"]["dense_reason"] == "no_dense_index"
