import streamlit as st
import streamlit.components.v1 as components

import rag_engine as re

DEFAULT_JMIN = re.J_DISP_MIN
JMIN_DEFAULT = DEFAULT_JMIN
SORT_OPTIONS = ["Best evidence", "Semantic"]


def qp_get(k: str, d=None):
    try:
        return st.query_params.get(k, d)
    except Exception:
        return d


def qp_set(**kw):
    try:
        st.query_params.update({k: v for k, v in kw.items() if v is not None})
    except Exception:
        pass


def init_state():
    ss = st.session_state

    ss.setdefault("theme_mode", "dark")
    ss.setdefault("show_debug", False)
    ss.setdefault("adv", False)
    ss.setdefault("mode", "quick")
    ss.setdefault("pubs", ["OReilly", "Manning", "Pearson"])
    ss.setdefault("srt", SORT_OPTIONS[0])
    ss.setdefault("nm", True)           # show near-miss when ok=True
    ss.setdefault("nm_skip", False)     # skip near-miss computation to save cost
    ss.setdefault("jmin", DEFAULT_JMIN)  # display min judge01
    ss.setdefault("judge_mode", "real")  # judge mode: real / proxy / off
    if "jdg_mode" in ss:
        ss.setdefault("judge_mode", ss.get("jdg_mode"))

    ss["use_jdg"] = True                # judge must be ON by default (and stay on)

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


def toast_flush():
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
    base = msg or "Unknown error"
    if err_id:
        return f"Error ({err_id}): {base}. Please retry the query."
    return f"Error: {base}. Please retry the query."


def cb_clear():
    ss = st.session_state
    ss["clip"] = ""
    ss["_toast"] = "Clipboard cleared"


def pins_clear():
    ss = st.session_state
    ss["pins"] = []
    ss["_toast"] = "Pins cleared"


def _pin_del(i: int):
    ss = st.session_state
    try:
        ss["pins"].pop(i)
        ss["_toast"] = "Unpinned"
    except Exception:
        pass


def _pin_lbl(p: dict) -> str:
    t = (p or {}).get("t", "")
    s = (p or {}).get("sec", "")
    if s:
        return f"{t} | {s}"
    return t


def mode_selector():
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


def topbar():
    ss = st.session_state
    st.markdown(
        """
<div class="topbar">
  <div class="brand">
    <span class="logo"><i class="ph ph-books"></i></span>
    <div class="name">
      <span class="title">RAG Books Search</span>
      <span class="muted">Evidence-first search</span>
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
    ctrl_left, ctrl_right = st.columns([0.6, 0.4])
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
        c1, c2 = st.columns(2)
        with c1:
            dark_pref = ss.get("theme_mode", "dark") == "dark"
            new_pref = st.toggle("Dark mode", value=dark_pref, key="dark_mode_toggle")
            ss["theme_mode"] = "dark" if new_pref else "light"
        with c2:
            st.checkbox("Debug", key="show_debug", help="Show timings & capabilities")

    return ss.get("theme_mode", "dark")


def render_hero():
    st.markdown(
        """
<div class="hero">
  <div>
    <div class="hero-icon"><i class="ph ph-lightning"></i></div>
    <h2>Dark, contrast-safe evidence search</h2>
    <div class="lede">Start with a prompt or suggested topic to see citations, context, and near-miss passages without leaving the page.</div>
    <div class="slim-actions">
      <span class="btn"><i class="ph ph-flag-checkered"></i> Evidence-first</span>
      <span class="btn"><i class="ph ph-eye"></i> Judge rerank on</span>
      <span class="btn"><i class="ph ph-waves"></i> Reduced motion ready</span>
    </div>
  </div>
  <div>
    <div class="bullets">
      <span class="item"><i class="ph ph-books"></i> Filter by publisher quickly</span>
      <span class="item"><i class="ph ph-chats-circle"></i> Compact, icon-led navigation</span>
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


def _apply_query_prefill(val: str):
    if not val:
        return
    st.session_state["q_inp"] = val


def sidebar(eng=None, startup_report=None, mount=None):
    ss = st.session_state
    host = mount or st.sidebar
    submitted = False
    with host:
        st.markdown("<div class='rail'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Query</div>", unsafe_allow_html=True)
        with st.form("q_form", clear_on_submit=False):
            icon, field = st.columns([0.14, 0.86])
            with icon:
                st.markdown(
                    "<div style='margin-top:10px; text-align:center; color: var(--muted-2); font-size:1.1rem;'>"
                    "<i class='ph ph-magnifying-glass'></i>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with field:
                st.text_input(
                    "Search books",
                    key="q_inp",
                    placeholder="Search the books library…",
                    label_visibility="collapsed",
                    help="Press Enter or click Search to submit. Press / or Ctrl+K to focus.",
                )
            submitted = st.form_submit_button("Search", use_container_width=True)

        st.caption("Suggestions")
        suggestions = [
            "What is retrieval-augmented generation?",
            "How do I chunk text for better recall?",
            "Compare vector search vs keyword search in this corpus.",
        ]
        sug_cols = st.columns(len(suggestions))
        for i, sug in enumerate(suggestions):
            with sug_cols[i]:
                if st.button(sug, key=f"q_suggestion_{i}", use_container_width=True):
                    _apply_query_prefill(sug)

        if ss.get("q_history"):
            st.caption("Recent searches")
            hist = ss.get("q_history", [])
            cols = st.columns(min(len(hist), 3))
            for i, val in enumerate(hist[:3]):
                with cols[i % len(cols)]:
                    if st.button(val, key=f"q_history_{i}", use_container_width=True):
                        _apply_query_prefill(val)

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
    if (ev.key.toLowerCase() === 'k' && ev.ctrlKey) {
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

        st.markdown("<div class='section-title'>Publishers</div>", unsafe_allow_html=True)
        ss["pubs"] = st.multiselect(
            "Publishers",
            options=["OReilly", "Manning", "Pearson"],
            default=ss.get("pubs", []),
            label_visibility="collapsed",
        )

        st.markdown("<div class='section-title'>Mode</div>", unsafe_allow_html=True)
        mode_selector()
        st.caption("Fast vs depth presets.")

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
            help="Best evidence favors citations; Semantic favors embedding similarity.",
            label_visibility="collapsed",
        )

        st.markdown("<div class='section-title'>Advanced options</div>", unsafe_allow_html=True)
        with st.expander("Judge & near-miss controls", expanded=False):
            st.toggle(
                "Near-miss",
                key="nm",
                value=ss.get("nm", True),
                help="Show weak overlaps when no direct evidence is available.",
            )
            st.toggle(
                "Judge (forced ON)",
                key="_use_jdg_view",
                value=True,
                disabled=True,
                help="Cross-encoder rerank stays enabled to keep evidence quality high.",
            )
            st.selectbox(
                "Judge mode",
                options=["proxy", "real", "off"],
                index=["proxy", "real", "off"].index(ss.get("judge_mode", "proxy")),
                help="proxy = score-based, real = cross-encoder (CPU), off = bypass (for debugging only).",
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
                "Skip near-miss computation (faster)",
                key="nm_skip",
                help="Disable the additional near-miss pass to speed up queries when exact matches are enough.",
            )

        st.divider()

        st.markdown("<div class='section-title'>Pinned</div>", unsafe_allow_html=True)
        ps = ss.get("pins", [])
        if not ps:
            st.caption("Pin evidence cards to keep them here.")
        else:
            st.markdown(
                "<div class='scroll-area' style='max-height: 220px; overflow-y:auto; padding-right:6px;'>",
                unsafe_allow_html=True,
            )
            for i, p in enumerate(ps):
                c1, c2 = st.columns([0.82, 0.18])
                with c1:
                    st.write(_pin_lbl(p))
                with c2:
                    st.button("Unpin", key=f"unpin_{i}", on_click=_pin_del, args=(i,), help="Unpin")
            st.markdown("</div>", unsafe_allow_html=True)
            st.button("Clear pins", key="pins_clear", on_click=pins_clear, use_container_width=True)

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

    return submitted
