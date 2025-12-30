"""
Custom UI shell components for the RAG Books Search application.

This module defines the top bar and sidebar for the RAG Books Search UI.  It
aims to strike a balance between functionality and aesthetics: the dark
theme is enforced globally, controls are grouped logically, and the
interface provides visual breathing room.  Suggestions and recent
queries are displayed as concise buttons in a single row to avoid the
stacked, vertical lists that cluttered earlier iterations.  Session
state keys are managed carefully to avoid collisions with widget keys.

Key changes from the stock implementation:

* **Permanent dark mode** – the user can no longer toggle between light
  and dark themes.  The dark palette defined in ``ui_theme.py``
  is always applied.
* **Concise suggestions** – suggestions are rendered as short labels to
  fit comfortably in a single row.  Clicking a suggestion prefills the
  search box with the full text via a callback.
* **Grouped filters** – publishers, mode selector and sort order appear
  together in a clearly separated section.  Advanced options are tucked
  away in an expander.
* **Fixed session state keys** – widgets use the same keys that
  ``st.session_state`` relies on.  We avoid direct assignment to those
  keys, allowing Streamlit to manage state updates automatically and
  preventing the ``StreamlitAPIException`` encountered previously.

This file is designed to be imported by ``app.py``.  See that
module for how the page is assembled.
"""

from __future__ import annotations

import html
from typing import Iterable, Optional

import streamlit as st
import streamlit.components.v1 as components

import rag_engine as re


# -----------------------------------------------------------------------------
# Helpers for query parameters and session initialisation
# -----------------------------------------------------------------------------

DEFAULT_JMIN = re.J_DISP_MIN
SORT_OPTIONS = ["Best evidence", "Semantic"]


def qp_get(k: str, default: Optional[str] = None) -> Optional[str]:
    """Return the value of a query parameter if present, else ``default``.

    This helper wraps access to ``st.query_params`` with a try/except to
    avoid exceptions in unsupported contexts.
    """
    try:
        return st.query_params.get(k, default)
    except Exception:
        return default


def qp_set(**params: Optional[str]) -> None:
    """Update the current query parameters with the provided mapping.

    Only non-``None`` values are used.  Failures are silently ignored to
    prevent crashes when running in environments that do not support URL
    updates (e.g. some embedded contexts).
    """
    try:
        st.query_params.update({k: v for k, v in params.items() if v is not None})
    except Exception:
        pass


def init_state() -> None:
    """Populate ``st.session_state`` with sensible defaults.

    This function should be called early in your app to ensure all
    expected keys exist in session state.  We avoid creating any keys
    that would conflict with widget keys (e.g. we don't set ``ss['srt']``
    because the ``selectbox`` will manage that entry itself).
    """
    ss = st.session_state
    ss.setdefault("theme_mode", "dark")
    ss.setdefault("show_debug", False)
    ss.setdefault("adv", False)
    ss.setdefault("mode", "quick")
    ss.setdefault("pubs", ["OReilly", "Manning", "Pearson"])
    ss.setdefault("nm", True)
    ss.setdefault("nm_skip", False)
    ss.setdefault("jmin", DEFAULT_JMIN)
    ss.setdefault("judge_mode", "real")
    # ensure pinned items and clipboard exist
    ss.setdefault("pins", [])
    ss.setdefault("clip", "")
    ss.setdefault("act_hit", None)
    ss.setdefault("q_history", [])
    ss.setdefault("_loading", False)
    ss.setdefault("_toast", None)
    ss.setdefault("_toast_last", None)
    ss.setdefault("_ui_err", None)
    ss.setdefault("_ui_err_id", None)
    ss.setdefault("_scroll_ctx", False)
    ss.setdefault("_ctx_ts", None)
    ss.setdefault("ev_offset", 0)


# -----------------------------------------------------------------------------
# Utility callbacks
# -----------------------------------------------------------------------------

def toast_flush() -> None:
    """Show a toast message if one is pending, then clear it."""
    ss = st.session_state
    msg = ss.get("_toast")
    if msg:
        try:
            st.toast(msg)
        except Exception:
            st.info(msg)
        ss["_toast_last"] = msg
        ss["_toast"] = None


def global_error_box(renderer=None) -> None:
    """Render an error box when a UI-level error is present."""
    ss = st.session_state
    err = ss.get("_ui_err")
    if not err:
        return
    error_id = ss.get("_ui_err_id")
    hint = "If this persists, reload the page and retry the query."
    if renderer is not None:
        renderer(
            {
                "message": err,
                "error_id": error_id,
                "hint": f"{hint} Error IDs help with debugging.",
            }
        )
        return
    with st.container(border=True):
        st.error(err)
        c1, c2 = st.columns([0.75, 0.25])
        with c2:
            if st.button("Dismiss", key="err_dismiss"):
                ss["_ui_err"] = None
                ss["_ui_err_id"] = None
                st.rerun()
        with c1:
            if error_id:
                st.caption(f"{hint} Error ID: {error_id}")
            else:
                st.caption(f"{hint} Error IDs help with debugging.")


def format_ui_error(err_id: Optional[str], msg: Optional[str]) -> str:
    """Return a human‑friendly error message for display in the UI.

    This helper mirrors the behaviour of the upstream ``ui_shell.format_ui_error``
    function.  It prepends an error ID when available and appends
    guidance text instructing the user to retry.  Without this helper,
    ``app._on_search`` would raise an ``AttributeError`` when
    reporting exceptions, because ``ui_shell`` previously lacked
    ``format_ui_error``.

    Args:
        err_id: Optional unique identifier for the error (may be ``None``).
        msg: A short description of the error.

    Returns:
        A formatted string suitable for ``st.error()``.
    """
    base = msg or "Unknown error"
    if err_id:
        return f"Error ({err_id}): {base}. Please retry the query."
    return f"Error: {base}. Please retry the query."


def cb_clear() -> None:
    """Clear the clipboard and trigger a toast."""
    ss = st.session_state
    ss["clip"] = ""
    ss["_toast"] = "Clipboard cleared"


def pins_clear() -> None:
    """Remove all pinned items and trigger a toast."""
    ss = st.session_state
    ss["pins"] = []
    ss["_toast"] = "Pins cleared"


def _pin_del(i: int) -> None:
    """Delete a pinned entry by index."""
    ss = st.session_state
    try:
        ss["pins"].pop(i)
        ss["_toast"] = "Unpinned"
    except Exception:
        pass


def _pin_lbl(p: dict) -> str:
    """Return a human-readable label for a pinned item."""
    title = (p or {}).get("t", "")
    section = (p or {}).get("sec", "")
    return f"{title} | {section}" if section else title


# -----------------------------------------------------------------------------
# UI components
# -----------------------------------------------------------------------------

def mode_selector() -> str:
    """Render the radio selector for search mode (quick vs exact)."""
    ss = st.session_state
    opts = re.mode_options()
    names = [o["name"] for o in opts]
    labels = {o["name"]: o["label"] for o in opts}
    descs = {o["name"]: o.get("description", "") for o in opts}
    try:
        idx = names.index(ss.get("mode", names[0]))
    except ValueError:
        idx = 0
    choice = st.radio(
        "Mode",
        options=names,
        index=idx,
        format_func=lambda v: labels.get(v, v.title()),
        key="mode",
        help="Quick = speed, Exact = deeper search for precise citations.",
    )
    st.caption(f"{labels.get(choice, choice.title())}: {descs.get(choice, '')}")
    return choice


def topbar() -> None:
    """Render the top navigation bar.

    The bar contains the app name, a few navigation pills and a debug
    checkbox.  Dark mode is enforced; there is no theme toggle.
    """
    ss = st.session_state
    st.markdown(
        """
<div class="topbar">
  <div class="brand">
    <span class="logo"><i class="ph ph-books"></i></span>
    <div class="name">
      <span class="title">RAG Books Search</span>
      <span class="muted">Evidence‑first search</span>
    </div>
  </div>
  <div class="icon-nav">
    <span class="pill"><i class="ph ph-compass"></i> Explore</span>
    <span class="pill"><i class="ph ph-stack-simple"></i> Context</span>
    <span class="pill"><i class="ph ph-sparkle"></i> Judge on</span>
    <span class="pill"><i class="ph ph-chat-teardrop-text"></i> Chat</span>
    <span class="pill"><i class="ph ph-gear"></i> Settings</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    ctrl_left, ctrl_right = st.columns([0.7, 0.3])
    with ctrl_left:
        st.markdown(
            """
<div class="slim-actions">
  <span class="btn"><i class="ph ph-magnifying-glass"></i> Search books</span>
  <span class="btn"><i class="ph ph-bookmarks-simple"></i> Pins</span>
  <span class="btn"><i class="ph ph-git-branch"></i> Modes</span>
</div>
""",
            unsafe_allow_html=True,
        )
    with ctrl_right:
        st.checkbox("Debug", key="show_debug", help="Show timings & capabilities")
    ss["theme_mode"] = "dark"


def render_hero() -> None:
    """Display a hero section when no search has been submitted."""
    st.markdown(
        """
<div class="hero">
  <div>
    <div class="hero-icon"><i class="ph ph-lightning"></i></div>
    <h2>Dark, contrast‑safe evidence search</h2>
    <div class="lede">Start with a prompt or a suggested topic to see citations, context and near‑miss passages without leaving the page.</div>
    <div class="slim-actions">
      <span class="btn"><i class="ph ph-flag-checkered"></i> Evidence‑first</span>
      <span class="btn"><i class="ph ph-eye"></i> Judge rerank on</span>
      <span class="btn"><i class="ph ph-waves"></i> Reduced motion ready</span>
    </div>
  </div>
  <div>
    <div class="bullets">
      <span class="item"><i class="ph ph-books"></i> Filter by publisher quickly</span>
      <span class="item"><i class="ph ph-chats-circle"></i> Compact, icon‑led navigation</span>
      <span class="item"><i class="ph ph-device-mobile"></i> Context panel overlays on small screens</span>
    </div>
    <div class="stats" style="margin-top: 10px;">
      <span class="stat"><i class="ph ph-magnifying-glass"></i> Ask anything</span>
      <span class="stat"><i class="ph ph-bounding-box"></i> Gradient accent</span>
      <span class="stat"><i class="ph ph-shield-checkered"></i> Focus rings</span>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _apply_query_prefill(val: str) -> None:
    """Prefill the search input with the provided value."""
    if val:
        st.session_state["q_inp"] = val


def sidebar(eng=None, startup_report=None, mount=None) -> bool:
    """Render the sidebar and return ``True`` if the form was submitted."""
    ss = st.session_state
    host = mount or st.sidebar
    submitted = False
    with host:
        st.markdown("<div class='rail'>", unsafe_allow_html=True)
        # Search form
        st.markdown("<div class='section-title'>Search</div>", unsafe_allow_html=True)
        with st.form("q_form", clear_on_submit=False):
            st.text_input(
                "Search books",
                key="q_inp",
                placeholder="Ask a question about our books…",
                label_visibility="collapsed",
                help="Press Enter or click Search to submit.",
            )
            submitted = st.form_submit_button("Search", use_container_width=True)
        # Suggestions
        st.caption("Suggestions")
        suggestions: Iterable[tuple[str, str]] = [
            ("What is RAG?", "What is retrieval‑augmented generation?"),
            ("Chunking text", "How do I chunk text for better recall?"),
            ("Vector vs keyword", "Compare vector search vs keyword search in this corpus."),
        ]
        cols = st.columns(len(list(suggestions)))
        for i, (label, full) in enumerate(suggestions):
            with cols[i]:
                # Use on_click to prefill the query; direct assignment after button click can
                # trigger a StreamlitAPIException in some versions.  See:
                # https://docs.streamlit.io/library/advanced-features/session-state#callbacks
                st.button(
                    label,
                    key=f"q_sug_{i}",
                    help=full,
                    use_container_width=True,
                    on_click=_apply_query_prefill,
                    args=(full,),
                )
        # Recent searches
        persisted = re.get_recent_queries(limit=5) or []
        history = ss.get("q_history", []) or []
        merged: list[str] = []
        for v in history + [p for p in persisted if p not in history]:
            if v not in merged:
                merged.append(v)
        if merged:
            st.caption("Recent searches")
            cols_hist = st.columns(min(len(merged), 3))
            for i, val in enumerate(merged[:3]):
                with cols_hist[i % len(cols_hist)]:
                    # As with suggestions, use on_click for history entries to avoid direct state
                    # assignment errors when modifying q_inp
                    st.button(
                        val,
                        key=f"q_hist_{i}",
                        use_container_width=True,
                        on_click=_apply_query_prefill,
                        args=(val,),
                    )
        # Keyboard shortcuts
        components.html(
            """
<script>
(function(){
  const doc = window.parent.document;
  function focusSearch(){
    const el = doc.querySelector('input[aria-label="Search books"]');
    if (el){ el.focus(); el.select(); }
  }
  function submitForm(){
    const btn = Array.from(doc.querySelectorAll('button')).find(
      (b) => b.innerText.trim() === 'Search'
    );
    if (btn){ btn.click(); }
  }
  doc.addEventListener('keydown', function(ev){
    const tag = ev.target && ev.target.tagName ? ev.target.tagName.toLowerCase() : '';
    const typing = ['input','textarea'].includes(tag);
    if(!typing && ev.key === '/' && !ev.metaKey && !ev.ctrlKey && !ev.altKey){
      ev.preventDefault(); focusSearch();
    }
    if(ev.key === 'Enter' && ev.ctrlKey){
      ev.preventDefault(); submitForm();
    }
  }, {passive:false});
})();
</script>
""",
            height=0,
        )
        # Filters
        st.divider()
        st.markdown("<div class='section-title'>Filters</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Publishers</div>", unsafe_allow_html=True)
        try:
            st.multiselect(
                "Publishers",
                options=["OReilly", "Manning", "Pearson"],
                default=ss.get("pubs", []),
                key="pubs",
                label_visibility="collapsed",
            )
        except TypeError:
            ss["pubs"] = st.multiselect(
                "Publishers",
                options=["OReilly", "Manning", "Pearson"],
                default=ss.get("pubs", []),
                label_visibility="collapsed",
            )
        st.markdown("<div class='section-title'>Mode</div>", unsafe_allow_html=True)
        mode_selector()
        st.caption("Fast vs depth presets.")
        st.markdown("<div class='section-title'>Sort order</div>", unsafe_allow_html=True)
        try:
            default_sort = SORT_OPTIONS.index(ss.get("srt", SORT_OPTIONS[0]))
        except ValueError:
            default_sort = 0
        st.selectbox(
            "Sort order",
            SORT_OPTIONS,
            index=default_sort,
            key="sort_widget",
            help="Best evidence emphasises citation richness; Semantic emphasises embedding similarity.",
            label_visibility="collapsed",
        )
        ss["srt"] = st.session_state.get("sort_widget", SORT_OPTIONS[0])
        # Advanced options
        st.markdown("<div class='section-title'>Advanced options</div>", unsafe_allow_html=True)
        with st.expander("Advanced settings", expanded=False):
            st.toggle(
                "Near‑miss",
                key="nm",
                help="Show weak overlaps when no direct evidence is available.",
            )
            st.selectbox(
                "Judge mode",
                options=["proxy", "real", "off"],
                index=["proxy", "real", "off"].index(ss.get("judge_mode", "proxy")),
                key="judge_mode",
                help="proxy = score based, real = cross‑encoder (CPU), off = bypass (for debugging only).",
            )
            try:
                st.slider(
                    "Min judge01 (display)",
                    0.0,
                    0.95,
                    float(ss.get("jmin", DEFAULT_JMIN)),
                    0.05,
                    key="jmin",
                    help="Hide evidence below this judge score while keeping at least a handful of results.",
                    label_visibility="collapsed",
                )
            except TypeError:
                ss["jmin"] = st.slider(
                    "Min judge01 (display)",
                    0.0,
                    0.95,
                    float(ss.get("jmin", DEFAULT_JMIN)),
                    0.05,
                    help="Hide evidence below this judge score while keeping at least a handful of results.",
                    label_visibility="collapsed",
                )
            st.toggle(
                "Skip near‑miss computation (faster)",
                key="nm_skip",
                help="Disable the additional near‑miss pass to speed up queries when exact matches are enough.",
            )
        # Pinned items
        st.divider()
        st.markdown("<div class='section-title'>Pinned</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='scroll-area' style='max-height:200px; overflow-y:auto; padding-right:6px;'>",
            unsafe_allow_html=True,
        )
        pins = ss.get("pins", [])
        deduped = []
        seen = set()
        for p in pins:
            key = (p.get("cid"), p.get("cidx"))
            if key not in seen:
                deduped.append(p)
                seen.add(key)
        if len(deduped) != len(pins):
            ss["pins"] = deduped
            pins = deduped
        if not pins:
            st.caption("Pin evidence cards to keep them here.")
        else:
            for i, p in enumerate(pins):
                c1, c2 = st.columns([0.8, 0.2])
                with c1:
                    st.markdown(
                        f"<div class='pin-entry'><span class='chip muted pin-idx'>#{i + 1}</span>"
                        f"<span class='pin-label'>{html.escape(_pin_lbl(p))}</span></div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.button(
                        "Unpin",
                        key=f"unpin_{i}",
                        on_click=_pin_del,
                        args=(i,),
                        help="Unpin",
                    )
        st.markdown("</div>", unsafe_allow_html=True)
        st.button("Clear pins", key="pins_clear", on_click=pins_clear, use_container_width=True)
        # Clipboard
        st.markdown("<div class='section-title'>Clipboard</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='scroll-area' style='max-height:180px; overflow-y:auto; padding-right:6px;'>",
            unsafe_allow_html=True,
        )
        clip_text = ss.get("clip", "")
        if clip_text:
            st.code(clip_text, language=None)
        else:
            st.caption("Use Copy on a card to put a citation here.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.button("Clear clipboard", key="clip_clear", on_click=cb_clear, use_container_width=True)
        # Startup status
        st.divider()
        st.markdown("<div class='section-title'>Startup</div>", unsafe_allow_html=True)
        summary = startup_report or {}
        rows = summary.get("rows", []) if isinstance(summary, dict) else summary
        if not rows:
            st.caption("No corpus status available.")
        else:
            for row in rows:
                ready = bool(row.get("ready"))
                color = "#2aa865" if ready else "#d23030"
                reason = "Ready" if ready else (row.get("reason") or "Unavailable")
                st.markdown(
                    f"<div style='display:flex; gap:8px; align-items:center; color:{color}; font-size:0.9em;'>"
                    f"<span style='font-size:1.1em;'>●</span>"
                    f"<span><strong>{row.get('publisher', 'Unknown')}</strong> — {reason}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)
    return submitted
