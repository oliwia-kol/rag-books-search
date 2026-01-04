"""UI shell (topbar + sidebar + global UI helpers).

Contract constraints (smoke_ui_contract.py):
- Must expose: init_state, topbar, sidebar, render_hero,
  global_error_box, toast_flush, qp_get, qp_set, cb_clear.
- Must keep DEFAULT_JMIN.

Design goals:
- Sidebar becomes a calm "control rail" (progressive disclosure).
- No noisy debug metrics in the primary surfaces.
"""

from __future__ import annotations

import html
from typing import Any, Dict, Optional

import streamlit as st


# ---- UI/behavior defaults

DEFAULT_JMIN = 0.55
SORT_OPTIONS = ["Best evidence", "Semantic", "Lexical"]


def qp_get(k: str, default: str | None = None) -> str | None:
    try:
        return st.query_params.get(k, default)
    except Exception:
        return default


def qp_set(**kwargs: str) -> None:
    try:
        for k, v in kwargs.items():
            if v is None:
                st.query_params.pop(k, None)
            else:
                st.query_params[k] = str(v)
    except Exception:
        return


def format_ui_error(err_id: Optional[str], msg: str) -> str:
    if err_id:
        return f"{msg} (error_id={err_id})"
    return msg


def _ss_setdefault(ss: Dict[str, Any], k: str, v: Any) -> None:
    if k not in ss:
        ss[k] = v


def init_state() -> None:
    ss = st.session_state

    _ss_setdefault(ss, "q_inp", "")
    _ss_setdefault(ss, "q_history", [])
    _ss_setdefault(ss, "pubs", [])
    _ss_setdefault(ss, "pins", [])
    _ss_setdefault(ss, "clip", "")

    # Query engine controls (kept for compatibility; UI may hide them)
    _ss_setdefault(ss, "mode", "quick")
    _ss_setdefault(ss, "srt", "Best evidence")
    _ss_setdefault(ss, "nm", True)
    _ss_setdefault(ss, "nm_skip", False)
    _ss_setdefault(ss, "jmin", DEFAULT_JMIN)
    _ss_setdefault(ss, "judge_mode", "proxy")
    _ss_setdefault(ss, "show_debug", False)

    # Runtime flags
    _ss_setdefault(ss, "_loading", False)
    _ss_setdefault(ss, "_ui_err", None)
    _ss_setdefault(ss, "_ui_err_id", None)
    _ss_setdefault(ss, "_toast", None)
    _ss_setdefault(ss, "ev_offset", 0)
    _ss_setdefault(ss, "act_hit", None)
    _ss_setdefault(ss, "_ctx_ts", None)
    _ss_setdefault(ss, "_scroll_ctx", False)


def topbar(startup_report: Optional[Dict[str, Any]] = None) -> None:
    """Sticky top bar with brand + subtle status."""

    ready = True
    if isinstance(startup_report, dict):
        rows = startup_report.get("rows") or []
        if rows:
            ready = all(bool(r.get("ready")) for r in rows)

    dot = "<span class='dot'></span>" if ready else "<span class='dot' style='background:rgba(255,143,177,.75); box-shadow:0 0 0 3px rgba(255,143,177,.12);'></span>"
    st.markdown(
        f"""
<div class='topbar'>
  <div class='brand'>
    <div class='brandmark'></div>
    <div>
      <div class='brandtitle'>RAG Books</div>
      <div class='brandtag'>chat-first research with sources</div>
    </div>
  </div>
  <div class='top-actions'>
    <div class='pill' title='Index readiness'>{dot}<span style='font-size:.92rem;'>Indexes</span></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Debug lives in a popover (never front-and-center)
    try:
        with st.popover("⚙︎", use_container_width=False):
            st.checkbox("Diagnostics", key="show_debug")
    except Exception:
        with st.expander("Diagnostics", expanded=False):
            st.checkbox("Diagnostics", key="show_debug")


def render_hero() -> None:
    st.markdown(
        """
<div class='card answer'>
  <h2>Ask a question about your books</h2>
  <div class='text muted'>
    This assistant searches your local library and replies with sources.
    Keep it natural — you can ask for definitions, comparisons, or "find the quote" style requests.
  </div>
  <div style='margin-top:.85rem; display:flex; gap:.45rem; flex-wrap:wrap;'>
    <span class='chip'>Try: “Explain RAG in 4 sentences”</span>
    <span class='chip'>Try: “Vector search vs keyword search”</span>
    <span class='chip'>Try: “Give me the exact quote about monitoring”</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def global_error_box() -> None:
    ss = st.session_state
    msg = ss.get("_ui_err")
    if not msg:
        return
    st.markdown(
        f"""
<div class='card' style='border-color: rgba(255,143,177,.22); background: rgba(255,143,177,.06);'>
  <div style='font-weight:650; margin-bottom:.25rem;'>Something broke</div>
  <div style='color: rgba(238,242,255,.9); line-height:1.55;'>{html.escape(str(msg))}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def toast_flush() -> None:
    ss = st.session_state
    msg = ss.get("_toast")
    if not msg:
        return
    try:
        st.toast(str(msg))
    except Exception:
        # fallback (older streamlit)
        st.info(str(msg))
    ss["_toast"] = None


def cb_clear() -> None:
    st.session_state["clip"] = ""
    st.session_state["_toast"] = "Clipboard cleared"


def _pin_del(i: int) -> None:
    ss = st.session_state
    pins = ss.get("pins", []) or []
    if 0 <= i < len(pins):
        pins.pop(i)
        ss["pins"] = pins
        ss["_toast"] = "Unpinned"


def pins_clear() -> None:
    st.session_state["pins"] = []
    st.session_state["_toast"] = "Pins cleared"


def _use_suggestion(q: str) -> None:
    st.session_state["q_inp"] = q
    qp_set(q=q)


def sidebar(eng, startup_report: Optional[Dict[str, Any]] = None, mount=None) -> bool:
    """Render the left rail. Returns True when Search submitted."""

    ss = st.session_state
    host = mount if mount is not None else st.sidebar
    submitted = False

    with host:
        st.markdown("<div class='rail'>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Search</div>", unsafe_allow_html=True)
        with st.form("q_form", clear_on_submit=False):
            st.text_input("", key="q_inp", placeholder="Ask anything about the books…", label_visibility="collapsed")
            c1, c2 = st.columns([0.58, 0.42], gap="small")
            with c1:
                st.markdown("<div class='cta'>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Search", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.button("Clear", on_click=lambda: _use_suggestion(""), use_container_width=True)

        # Suggestions (chips)
        st.markdown("<div class='section-title' style='margin-top:.8rem;'>Suggestions</div>", unsafe_allow_html=True)
        sug = [
            "What is retrieval-augmented generation?",
            "Vector search vs keyword search",
            "How do you evaluate LLM systems?",
        ]
        cols = st.columns(3, gap="small")
        for i, s in enumerate(sug):
            with cols[i % 3]:
                st.button(s, key=f"sug_{i}", on_click=_use_suggestion, args=(s,), use_container_width=True)

        # Filters
        st.markdown("<div class='section-title' style='margin-top:.95rem;'>Publishers</div>", unsafe_allow_html=True)
        pubs = []
        try:
            pubs = sorted(list(getattr(eng, "corp", {}).keys()))
        except Exception:
            pubs = ["OReilly", "Manning", "Pearson"]
        ss["pubs"] = st.multiselect("", pubs, default=ss.get("pubs", []), label_visibility="collapsed")

        # Advanced (progressive disclosure)
        with st.expander("Advanced", expanded=False):
            st.toggle("Include near-miss", key="nm")
            st.slider("Evidence threshold", 0.0, 1.0, float(ss.get("jmin", DEFAULT_JMIN)), 0.01, key="jmin")
            st.selectbox("Judge mode", ["proxy", "real", "off"], key="judge_mode")

        # Recent
        hist = ss.get("q_history", []) or []
        if hist:
            st.markdown("<div class='section-title' style='margin-top:.95rem;'>Recent</div>", unsafe_allow_html=True)
            for i, q in enumerate(hist[:6]):
                st.button(q, key=f"hist_{i}", on_click=_use_suggestion, args=(q,), use_container_width=True)

        # Pins
        pins = ss.get("pins", []) or []
        if pins:
            st.markdown("<div class='section-title' style='margin-top:.95rem;'>Pinned</div>", unsafe_allow_html=True)
            for i, p in enumerate(pins[:12]):
                t = p.get("t") or "Untitled"
                sec = p.get("sec") or ""
                lab = f"{t} — {sec}" if sec else t
                c1, c2 = st.columns([0.78, 0.22], gap="small")
                with c1:
                    st.caption(lab)
                with c2:
                    st.button("Unpin", key=f"unpin_{i}", on_click=_pin_del, args=(i,), use_container_width=True)
            st.button("Clear pins", key="pins_clear", on_click=pins_clear, use_container_width=True)

        # Clipboard
        st.markdown("<div class='section-title' style='margin-top:.95rem;'>Clipboard</div>", unsafe_allow_html=True)
        clip = ss.get("clip", "")
        if clip:
            st.code(clip, language=None)
            st.button("Clear clipboard", on_click=cb_clear, use_container_width=True)
        else:
            st.caption("Use “Copy” on a source to collect citations here.")

        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        qp_set(q=st.session_state.get("q_inp", ""))
    return submitted
