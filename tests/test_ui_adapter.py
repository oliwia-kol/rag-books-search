import html
import re as regex

import streamlit as st

import app_custom
import rag_engine as re
import ui_adapter_custom as ua
import ui_shell_custom as us


class DummyEvidenceStreamlit:
    def __init__(self):
        self.calls = []
        self.session_state = {}
        self.load_more_pressed = False

    def markdown(self, text, unsafe_allow_html=False):
        self.calls.append(("markdown", text, unsafe_allow_html))
        return text

    def caption(self, text):
        self.calls.append(("caption", text))
        return text

    def columns(self, n):
        self.calls.append(("columns", n))
        return [self for _ in range(n)]

    def button(self, label, key=None, on_click=None, args=(), **kwargs):
        self.calls.append(("button", label, key))
        pressed = self.load_more_pressed and label == "Load more"
        if pressed and on_click:
            on_click(*args)
        return pressed

    def write(self, *args, **kwargs):
        self.calls.append(("write", args, kwargs))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


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
    plain = regex.sub("<.*?>", "", snip)
    assert len(html.unescape(plain)) <= ua.SNIPPET_MAX_CHARS
    assert 'class="hl"' in snip


def test_jmin_default_matches_backend_and_display(monkeypatch):
    st.session_state.clear()
    us.init_state()

    captured: dict = {}

    def fake_run_query(eng, q, **kwargs):
        captured["jmin"] = kwargs.get("jmin")
        return {"meta": {}}

    monkeypatch.setattr(app_custom.re, "run_query", fake_run_query)

    app_custom._run(eng=None, q="hi")

    assert st.session_state["jmin"] == us.DEFAULT_JMIN
    assert captured["jmin"] == us.DEFAULT_JMIN
    assert ua.JMIN_DEFAULT == us.DEFAULT_JMIN
    assert re.J_DISP_MIN == us.DEFAULT_JMIN
    st.session_state.clear()


def test_context_resets_on_new_search(monkeypatch):
    st.session_state.clear()
    us.init_state()

    hits_first = [{"cid": "cid-1", "cidx": 0, "text": "First", "judge01": 0.9}]
    hits_second = [{"cid": "cid-2", "cidx": 0, "text": "Second", "judge01": 0.8}]

    def fake_run_query(eng, q, **kwargs):
        if q == "first":
            return {"hits": hits_first, "meta": {}}
        return {"hits": hits_second, "meta": {}}

    monkeypatch.setattr(app_custom.re, "run_query", fake_run_query)

    app_custom._run(eng=None, q="first")
    ua._ctx_open(hits_first[0])
    assert st.session_state["act_hit"]["cid"] == "cid-1"

    app_custom._run(eng=None, q="second")

    assert st.session_state["act_hit"] is None
    assert st.session_state["_scroll_ctx"] is False
    assert st.session_state["_ctx_ts"] is None

    ua.render_context_panel()
    st.session_state.clear()


def test_on_search_resets_selection_before_results(monkeypatch):
    st.session_state.clear()
    us.init_state()
    st.session_state["eng"] = object()
    st.session_state["act_hit"] = {"cid": "stale", "cidx": 7}
    st.session_state["q_inp"] = "refresh"

    captured: dict = {}

    def fake_run_query(eng, q, **kwargs):
        captured["act_hit_before_run"] = st.session_state.get("act_hit")
        return {"hits": [{"cid": "cid-new", "cidx": 0, "text": "New", "judge01": 0.77}], "meta": {}}

    monkeypatch.setattr(app_custom.re, "run_query", fake_run_query)

    app_custom._on_search()

    assert captured["act_hit_before_run"] is None
    assert st.session_state["act_hit"] is None
    assert st.session_state["res"]["hits"][0]["cid"] == "cid-new"
    st.session_state.clear()


def test_context_panel_ignores_stale_selection():
    st.session_state.clear()
    us.init_state()

    hits_first = [{"cid": "cid-1", "cidx": 0, "text": "First", "judge01": 0.9}]
    hits_second = [{"cid": "cid-2", "cidx": 0, "text": "Second", "judge01": 0.8}]

    st.session_state["res"] = {"hits": hits_second, "meta": {}}
    st.session_state["act_hit"] = hits_first[0]

    ua.render_context_panel()

    assert st.session_state["act_hit"] is None
    st.session_state.clear()


def test_evidence_list_lazy_loads_batches(monkeypatch):
    dummy = DummyEvidenceStreamlit()
    monkeypatch.setattr(ua, "st", dummy)
    monkeypatch.setattr(us, "st", dummy)
    us.init_state()

    hits = [
        {"cid": f"cid-{i}", "cidx": 0, "text": f"Hit {i}", "judge01": 0.9 - 0.01 * i}
        for i in range(12)
    ]

    ua.render_evidence_list({"hits": hits}, q="python")
    cards_first = sum(1 for c in dummy.calls if c[0] == "markdown" and "evidence-card" in c[1])
    assert cards_first == min(len(hits), ua.EVIDENCE_BATCH_SIZE)

    dummy.load_more_pressed = True
    dummy.calls.clear()
    ua.render_evidence_list({"hits": hits}, q="python")
    assert dummy.session_state["ev_offset"] == ua.EVIDENCE_BATCH_SIZE

    dummy.load_more_pressed = False
    dummy.calls.clear()
    ua.render_evidence_list({"hits": hits}, q="python")
    cards_after = sum(1 for c in dummy.calls if c[0] == "markdown" and "evidence-card" in c[1])
    assert cards_after == min(len(hits), ua.EVIDENCE_BATCH_SIZE * 2)


def test_evidence_list_batches_multiple_loads(monkeypatch):
    dummy = DummyEvidenceStreamlit()
    monkeypatch.setattr(ua, "st", dummy)
    monkeypatch.setattr(us, "st", dummy)
    us.init_state()

    hits = [
        {"cid": f"cid-{i}", "cidx": 0, "text": f"Hit {i}", "judge01": 0.9 - 0.01 * i}
        for i in range(20)
    ]

    # first render
    ua.render_evidence_list({"hits": hits}, q="python")
    assert dummy.session_state["ev_offset"] == 0

    # first load more
    dummy.load_more_pressed = True
    dummy.calls.clear()
    ua.render_evidence_list({"hits": hits}, q="python")
    assert dummy.session_state["ev_offset"] == ua.EVIDENCE_BATCH_SIZE

    # second load more
    dummy.load_more_pressed = False
    dummy.calls.clear()
    ua.render_evidence_list({"hits": hits}, q="python")
    assert dummy.session_state["ev_offset"] == ua.EVIDENCE_BATCH_SIZE
    dummy.load_more_pressed = True
    dummy.calls.clear()
    ua.render_evidence_list({"hits": hits}, q="python")
    assert dummy.session_state["ev_offset"] == ua.EVIDENCE_BATCH_SIZE * 2


def test_no_results_renders_suggestions(monkeypatch):
    dummy = DummyEvidenceStreamlit()
    monkeypatch.setattr(ua, "st", dummy)
    ua.render_evidence_list({"hits": []}, q="python")
    assert any("No evidence yet" in c[1] for c in dummy.calls if c[0] == "markdown")


def test_badge_tiering_labels():
    assert ua._judge_tier(0.9) == ("Strong", "success")
    assert ua._judge_tier(0.7) == ("Solid", "primary")
    assert ua._judge_tier(0.45) == ("Weak", "warning")
    assert ua._judge_tier(0.1) == ("Poor", "neutral")
