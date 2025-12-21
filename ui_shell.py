import streamlit as st

import rag_engine as re

JMIN_DEFAULT = 0.45


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

    ss.setdefault("theme_mode", "light")
    ss.setdefault("show_debug", False)
    ss.setdefault("adv", False)
    ss.setdefault("mode", "quick")
    ss.setdefault("pubs", ["OReilly", "Manning", "Pearson"])
    ss.setdefault("srt", "Best evidence")
    ss.setdefault("nm", True)           # show near-miss when ok=True
    ss.setdefault("nm_skip", False)     # skip near-miss computation to save cost
    ss.setdefault("jmin", JMIN_DEFAULT)  # display min judge01
    ss.setdefault("judge_mode", "real")  # judge mode: real / proxy / off
    if "jdg_mode" in ss:
        ss.setdefault("judge_mode", ss.get("jdg_mode"))

    ss["use_jdg"] = True                # judge must be ON by default (and stay on)

    ss.setdefault("pins", [])           # list[dict]
    ss.setdefault("clip", "")
    ss.setdefault("act_hit", None)      # active hit for context panel

    ss.setdefault("_toast", None)
    ss.setdefault("_ui_err", None)
    ss.setdefault("_ui_err_id", None)
    ss.setdefault("_scroll_ctx", False)
    ss.setdefault("_ctx_ts", None)


def toast_flush():
    ss = st.session_state
    msg = ss.get("_toast")
    if msg:
        try:
            st.toast(msg)
        except Exception:
            st.info(msg)
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
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        st.markdown("<div class='topbar'><span class='brand'>RAG Books Search</span></div>", unsafe_allow_html=True)
    with col2:
        c1, c2 = st.columns(2)
        with c1:
            dark_pref = ss.get("theme_mode", "light") == "dark"
            new_pref = st.toggle("Dark mode", value=dark_pref, key="dark_mode_toggle")
            ss["theme_mode"] = "dark" if new_pref else "light"
        with c2:
            st.checkbox("Debug", key="show_debug", help="Show timings & capabilities")


def sidebar(eng=None, startup_report=None, mount=None):
    ss = st.session_state
    host = mount or st.sidebar
    submitted = False
    with host:
        st.markdown("<div class='section-title'>Query</div>", unsafe_allow_html=True)
        with st.form("q_form", clear_on_submit=False):
            st.text_area(
                "",
                key="q_inp",
                placeholder="Ask about the books…",
                label_visibility="collapsed",
                height=88,
            )
            submitted = st.form_submit_button("Search", use_container_width=True)

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

        st.markdown("<div class='section-title'>Toggles</div>", unsafe_allow_html=True)
        st.toggle("Near-miss", key="nm", value=ss.get("nm", True), help="Show weak overlaps when no direct evidence.")
        st.toggle("Judge (forced ON)", key="_use_jdg_view", value=True, disabled=True, help="Cross-encoder rerank")
        st.selectbox(
            "Judge mode",
            options=["proxy", "real", "off"],
            index=["proxy", "real", "off"].index(ss.get("judge_mode", "proxy")),
            help="proxy = score-based, real = cross-encoder (CPU), off = bypass",
            key="judge_mode",
        )

        with st.expander("Advanced", expanded=False):
            st.caption("Sort")
            srt_opts = ["Best evidence", "Semantic"]
            try:
                srt_idx = srt_opts.index(ss.get("srt", srt_opts[0]))
            except ValueError:
                srt_idx = 0
            ss["srt"] = st.selectbox("", srt_opts, index=srt_idx, key="srt", label_visibility="collapsed")
            st.caption("Min judge01 (display)")
            ss["jmin"] = st.slider("", 0.0, 0.95, float(ss.get("jmin", JMIN_DEFAULT)), 0.05, label_visibility="collapsed")
            st.toggle("Skip near-miss computation (faster)", key="nm_skip")

        st.divider()

        st.markdown("<div class='section-title'>Pinned</div>", unsafe_allow_html=True)
        ps = ss.get("pins", [])
        if not ps:
            st.caption("Pin evidence cards to keep them here.")
        else:
            for i, p in enumerate(ps):
                c1, c2 = st.columns([0.82, 0.18])
                with c1:
                    st.write(_pin_lbl(p))
                with c2:
                    st.button("Unpin", key=f"unpin_{i}", on_click=_pin_del, args=(i,), help="Unpin")
            st.button("Clear pins", key="pins_clear", on_click=pins_clear, use_container_width=True)

        st.markdown("<div class='section-title'>Clipboard</div>", unsafe_allow_html=True)
        if ss.get("clip"):
            st.code(ss["clip"], language=None)
        else:
            st.caption("Use Copy on a card to put a citation here.")
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

    return submitted
