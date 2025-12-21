"""
Custom UI shell components for the RAG Books Search application.

This module wraps Streamlit primitives to construct the sidebar, topbar and
supporting UI utilities.  It is derived from the original ``ui_shell.py``
but introduces a number of improvements aimed at simplifying the user
experience:

* The top bar no longer offers a dark/light toggle.  Dark mode is
  enforced globally.  A debug checkbox remains for development.
* The sidebar groups core filters together under a "Filters" section and
  hides advanced options in an expander.  The search input is larger
  and more inviting, and suggestions are shown beneath it.
* Pinned items and clipboard areas have been kept but with improved
  spacing.
* Default state initialisation and error handling have been left largely
  unchanged.
"""

import html
import streamlit as st
import streamlit.components.v1 as components

import rag_engine as re


DEFAULT_JMIN = re.J_DISP_MIN
JMIN_DEFAULT = DEFAULT_JMIN
SORT_OPTIONS = ["Best evidence", "Semantic"]


def qp_get(k: str, d=None):
    """Retrieve a query parameter from the URL if available."""
    try:
        return st.query_params.get(k, d)
    except Exception:
        return d


def qp_set(**kw) -> None:
    """Update the query parameters in the URL."""
    try:
        st.query_params.update({k: v for k, v in kw.items() if v is not None})
    except Exception:
        pass


def init_state() -> None:
    """Set up default values in ``st.session_state`` for UI state variables."""
    ss = st.session_state
    ss.setdefault("theme_mode", "dark")
    ss.setdefault("show_debug", False)
    ss.setdefault("adv", False)
    ss.setdefault("mode", "quick")
    ss.setdefault("pubs", ["OReilly", "Manning", "Pearson"])
    ss.setdefault("srt", SORT_OPTIONS[0])
    ss.setdefault("nm", True)           # show near‑miss when ok=True
    ss.setdefault("nm_skip", False)     # skip near‑miss computation to save cost
    ss.setdefault("jmin", DEFAULT_JMIN)  # display min judge01
    ss.setdefault("judge_mode", "real")  # judge mode: real / proxy / off
    if "jdg_mode" in ss:
        ss.setdefault("judge_mode", ss.get("jdg_mode"))
    ss["use_jdg"] = True                # judge must be ON by default
    ss.setdefault("pins", [])           # list[dict]
    ss.setdefault("clip", "")
    ss.setdefault("act_hit", None)      # active hit for context panel
    ss.setdefault("q_history", [])
    ss.setdefault("_loading", False)
    ss.setdefault("_toast", None)
    ss.setdefault("_toast_last", None)
    ss.setdefault("_ui_err", None)
    ss.setdefault("_ui_err_id", None)
    ss.setdefault("_scroll_ctx", False)
    ss.setdefault("_ctx_ts", None)
    ss.setdefault("ev_offset", 0)


def toast_flush() -> None:
    """Display any pending toast message and clear it afterwards."""
    ss = st.session_state
    msg = ss.get("_toast")
    if msg:
        try:
            st.toast(msg)
        except Exception:
            st.info(msg)
        ss["_toast_last"] = msg
        ss["_toast"] = None


def global_error_box(renderer=None):
    """Render a global error message if present in the session state."""
    ss = st.session_state
    err = ss.get("_ui_err")
    if not err:
        return
    payload = {
        "message": err,
        "error_id": ss.get("_ui_err_id"),
        "hint": "If this persists, reload the page and retry the query. Error IDs help with debugging.",
    }
    if renderer:
        renderer(payload)
        return payload
    with st.container(border=True):
        st.error(err)
        c1, c2 = st.columns([0.75, 0.25])
        with c2:
            if st.button("Dismiss", key="err_dismiss"):
                ss["_ui_err"] = None
                ss["_ui_err_id"] = None
                st.rerun()
        with c1:
            if ss.get("_ui_err_id"):
                st.caption(f"{payload['hint']} Error ID: {ss['_ui_err_id']}")
            else:
                st.caption(payload["hint"])


def format_ui_error(err_id: str | None, msg: str | None) -> str:
    """Format an error message for display."""
    base = msg or "Unknown error"
    if err_id:
        return f"Error ({err_id}): {base}. Please retry the query."
    return f"Error: {base}. Please retry the query."


def cb_clear() -> None:
    """Clear the clipboard and show a toast."""
    ss = st.session_state
    ss["clip"] = ""
    ss["_toast"] = "Clipboard cleared"


def pins_clear() -> None:
    """Clear the pins list and show a toast."""
    ss = st.session_state
    ss["pins"] = []
    ss["_toast"] = "Pins cleared"


def _pin_del(i: int) -> None:
    """Remove a pin at the given index."""
    ss = st.session_state
    try:
        ss["pins"].pop(i)
        ss["_toast"] = "Unpinned"
    except Exception:
        pass


def _pin_lbl(p: dict) -> str:
    """Format a pin label from its metadata."""
    t = (p or {}).get("t", "")
    s = (p or {}).get("sec", "")
    if s:
        return f"{t} | {s}"
    return t


def mode_selector() -> str:
    """Render a radio selector for query modes (quick/exact)."""
    ss = st.session_state
    opts = re.mode_options()
    names = [o["name"] for o in opts]
    labels = {o["name"]: f"{o['label']}" for o in opts}
    desc = {o["name"]: o.get("description", "") for o in opts}
    try:
        idx = names.index(ss.get("mode", "quick"))
    except ValueError:
        idx = 0
    choice = st.radio(
        "Mode",
        options=names,
        index=idx,
        format_func=lambda v: f"{labels.get(v, v.title())}",
        help="Quick = speed, Exact = deeper search for precise citations.",
        key="mode",
    )
    st.caption(f"{labels.get(choice, choice.title())}: {desc.get(choice, '')}")
    return choice


def topbar() -> str:
    """Render the application top bar.

    The top bar contains the brand, icon navigation and a debug toggle.
    Dark mode is enforced globally; no user‑facing toggle is presented.
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
        # Only a debug checkbox remains; dark mode is always on
        st.checkbox("Debug", key="show_debug", help="Show timings & capabilities")
    # Persist the enforced theme
    ss["theme_mode"] = "dark"
    return "dark"


def render_hero() -> None:
    """Render a hero section when no results are available."""
    st.markdown(
        """
<div class="hero">
  <div>
    <div class="hero-icon"><i class="ph ph-lightning"></i></div>
    <h2>Dark, contrast‑safe evidence search</h2>
    <div class="lede">Start with a prompt or suggested topic to see citations, context, and near‑miss passages without leaving the page.</div>
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
    """Helper to prefill the query input when clicking a suggestion or history item."""
    if not val:
        return
    st.session_state["q_inp"] = val


def sidebar(eng=None, startup_report=None, mount=None) -> bool:
    """Render the search sidebar.

    The sidebar contains the search form, suggestions, recent searches,
    filter controls and advanced settings.  Pinned evidence cards and the
    clipboard are shown below the filters.  Returns ``True`` if the
    search form was submitted.

    Args:
        eng: The retrieval engine (unused in this function but kept for
             compatibility with the original signature).
        startup_report: A report of the corpus load status.
        mount: An optional container (column) to mount the sidebar into.

    Returns:
        bool: Whether the form was submitted.
    """
    ss = st.session_state
    host = mount or st.sidebar
    submitted = False
    with host:
        st.markdown("<div class='rail'>", unsafe_allow_html=True)
        # Search form
        st.markdown("<div class='section-title'>Search</div>", unsafe_allow_html=True)
        with st.form("q_form", clear_on_submit=False):
            # Single full‑width text input with a descriptive placeholder
            st.text_input(
                "Search books",
                key="q_inp",
                placeholder="Ask a question about our books…",
                label_visibility="collapsed",
                help="Press Enter or click Search to submit.",
            )
            submitted = st.form_submit_button("Search", use_container_width=True)
        # Suggestions and history
        st.caption("Suggestions")
        suggestions = [
            "What is retrieval‑augmented generation?",
            "How do I chunk text for better recall?",
            "Compare vector search vs keyword search in this corpus.",
        ]
        sug_cols = st.columns(len(suggestions))
        for i, sug in enumerate(suggestions):
            with sug_cols[i]:
                if st.button(sug, key=f"q_suggestion_{i}", use_container_width=True):
                    _apply_query_prefill(sug)
        # Recent searches from session and engine
        if ss.get("q_history") or re.get_recent_queries():
            st.caption("Recent searches")
            hist = ss.get("q_history", [])
            persisted = re.get_recent_queries(limit=5)
            merged = []
            for val in hist + [v for v in persisted if v not in hist]:
                if val not in merged:
                    merged.append(val)
            cols = st.columns(min(len(merged), 3) or 1)
            for i, val in enumerate(merged[:3]):
                with cols[i % len(cols)]:
                    if st.button(val, key=f"q_history_{i}", use_container_width=True):
                        _apply_query_prefill(val)
        # Keyboard shortcut support to focus the search input and submit on Ctrl+Enter
        components.html(
            """
<script>
(function(){
  const doc = window.parent.document;
  function focusSearch() {
    const el = doc.querySelector('input[aria-label="Search books"]');
    if (el) { el.focus(); el.select(); }
  }
  function submitForm() {
    const btn = Array.from(doc.querySelectorAll('button')).find(
      (b) => b.innerText.trim() === 'Search'
    );
    if (btn) { btn.click(); }
  }
  doc.addEventListener('keydown', function(ev){
    const tag = ev.target && ev.target.tagName ? ev.target.tagName.toLowerCase() : '';
    const typing = ['input', 'textarea'].includes(tag);
    if (!typing && ev.key === '/' && !ev.metaKey && !ev.ctrlKey && !ev.altKey) {
      ev.preventDefault();
      focusSearch();
    }
    if (ev.key === 'Enter' && ev.ctrlKey) {
      ev.preventDefault();
      submitForm();
    }
  }, {passive:false});
})();
</script>
""",
            height=0,
        )
        # Filters section
        st.divider()
        st.markdown("<div class='section-title'>Filters</div>", unsafe_allow_html=True)
        # Publishers
        st.markdown("<div class='section-title'>Publishers</div>", unsafe_allow_html=True)
        ss["pubs"] = st.multiselect(
            "Publishers",
            options=["OReilly", "Manning", "Pearson"],
            default=ss.get("pubs", []),
            label_visibility="collapsed",
        )
        # Mode selector
        st.markdown("<div class='section-title'>Mode</div>", unsafe_allow_html=True)
        mode_selector()
        st.caption("Fast vs depth presets.")
        # Sort order
        st.markdown("<div class='section-title'>Sort</div>", unsafe_allow_html=True)
        try:
            srt_idx = SORT_OPTIONS.index(ss.get("srt", SORT_OPTIONS[0]))
        except ValueError:
            srt_idx = 0
        ss["srt"] = st.selectbox(
            "Sort order",
            SORT_OPTIONS,
            index=srt_idx,
            key="srt",
            help="Best evidence favours citations; Semantic favours embedding similarity.",
            label_visibility="collapsed",
        )
        # Advanced settings
        st.markdown("<div class='section-title'>Advanced options</div>", unsafe_allow_html=True)
        with st.expander("Advanced settings", expanded=False):
            st.toggle(
                "Near‑miss",
                key="nm",
                value=ss.get("nm", True),
                help="Show weak overlaps when no direct evidence is available.",
            )
            st.selectbox(
                "Judge mode",
                options=["proxy", "real", "off"],
                index=["proxy", "real", "off"].index(ss.get("judge_mode", "proxy")),
                help="proxy = score‑based, real = cross‑encoder (CPU), off = bypass (for debugging only).",
                key="judge_mode",
            )
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
            "<div class='scroll-area' style='max-height: 200px; overflow-y:auto; padding-right:6px;'>",
            unsafe_allow_html=True,
        )
        ps = ss.get("pins", [])
        # Deduplicate pins by (cid, cidx)
        if ps:
            seen = set()
            deduped = []
            for p in ps:
                key = (p.get("cid"), p.get("cidx"))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(p)
            if len(deduped) != len(ps):
                ss["pins"] = deduped
                ps = deduped
        if not ps:
            st.caption("Pin evidence cards to keep them here.")
        else:
            for i, p in enumerate(ps):
                c1, c2 = st.columns([0.8, 0.2])
                with c1:
                    st.markdown(
                        """
<div class='pin-entry'>
  <span class='chip muted pin-idx'>#{idx}</span>
  <span class='pin-label'>{label}</span>
</div>
""".format(idx=i + 1, label=html.escape(_pin_lbl(p))),
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.button("Unpin", key=f"unpin_{i}", on_click=_pin_del, args=(i,), help="Unpin")
        st.markdown("</div>", unsafe_allow_html=True)
        st.button("Clear pins", key="pins_clear", on_click=pins_clear, use_container_width=True)
        # Clipboard
        st.markdown("<div class='section-title'>Clipboard</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='scroll-area' style='max-height: 180px; overflow-y:auto; padding-right:6px;'>",
            unsafe_allow_html=True,
        )
        if ss.get("clip"):
            st.code(ss["clip"], language=None)
        else:
            st.caption("Use Copy on a card to put a citation here.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.button("Clear clipboard", key="clip_clear", on_click=cb_clear, use_container_width=True)
        # Startup report
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
                    f"<span style='font-size:1.1em;'>{'●'}</span>"
                    f"<span><strong>{row.get('publisher', 'Unknown')}</strong> — {reason}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)
        # Close rail
        st.markdown("</div>", unsafe_allow_html=True)
    return submitted