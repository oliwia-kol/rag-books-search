import html
import re

import streamlit as st

import app
import ui_adapter as ua
import ui_shell as us


def test_answer_clamps_after_five_sentences():
    ans = "One. Two. Three. Four. Five. Six. Seven."
    limited = ua._limit_answer_sentences(ans, max_sents=5)
    assert limited["truncated"] is True
    assert len(ua._split_sents(limited["text"])) == 5


def test_empty_answer_preview_stitches_hits():
    hits = [
        {"text": "Alpha evidence snippet."},
        {"text": "Beta follows with more context."},
    ]
    preview = ua._stitch_hits_preview(hits, q="")
    assert "Alpha" in preview and "Beta" in preview
    assert "•" in preview


def test_confidence_state_mapping():
    cov_meta = {"mx": 0.82, "std": 0.02, "uc": 3}
    state = ua._confidence_state(confidence=0.78, coverage="HIGH", cov_meta=cov_meta)
    assert state["state"] == "HIGH"
    assert "mx=0.82" in state["tooltip"]


def test_coverage_counts_single_source_warning():
    hits = [{"book": "b1", "sec": "s1", "publisher": "p1"} for _ in range(2)]
    counts = ua._coverage_counts({"n": {}}, hits)
    assert counts["books"] == 1
    assert counts["publishers"] == 1
    assert counts["single_source"] is True


def test_snippet_highlights_and_clamps():
    long_text = " ".join(["highlight me"] * 80)
    snip = ua._snippet({"text": long_text}, q="highlight")
    plain = re.sub("<.*?>", "", snip)
    assert len(html.unescape(plain)) <= ua.SNIPPET_MAX_CHARS
    assert 'class="hl"' in snip


def test_jmin_default_matches_backend_and_display(monkeypatch):
    st.session_state.clear()
    us.init_state()

    captured: dict = {}

    def fake_run_query(eng, q, **kwargs):
        captured["jmin"] = kwargs.get("jmin")
        return {"meta": {}}

    monkeypatch.setattr(app.re, "run_query", fake_run_query)

    app._run(eng=None, q="hi")

    assert st.session_state["jmin"] == us.JMIN_DEFAULT
    assert captured["jmin"] == us.JMIN_DEFAULT
    assert ua.JMIN_DEFAULT == us.JMIN_DEFAULT
    st.session_state.clear()
