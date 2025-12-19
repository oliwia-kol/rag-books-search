import streamlit as st


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

    # stable defaults
    ss.setdefault("adv", False)
    ss.setdefault("pubs", ["OReilly", "Manning", "Pearson"])
    ss.setdefault("srt", "Best evidence")
    ss.setdefault("nm", True)          # show near-miss when ok=True
    ss.setdefault("jmin", 0.45)        # display min judge01

    # HARD REQUIREMENTS
    ss["use_jdg"] = True               # judge must be ON by default (and stay on)

    ss.setdefault("pins", [])          # list[dict]
    ss.setdefault("clip", "")
    ss.setdefault("act_hit", None)     # active hit for context panel

    ss.setdefault("_toast", None)
    ss.setdefault("_ui_err", None)
    ss.setdefault("_scroll_ctx", False)


def toast_flush():
    ss = st.session_state
    msg = ss.get("_toast")
    if msg:
        try:
            st.toast(msg)
        except Exception:
            st.info(msg)
        ss["_toast"] = None


def global_error_box():
    ss = st.session_state
    err = ss.get("_ui_err")
    if not err:
        return
    with st.container(border=True):
        st.error(err)
        c1, c2 = st.columns([0.75, 0.25])
        with c2:
            if st.button("Dismiss", key="err_dismiss"):
                ss["_ui_err"] = None
                st.rerun()


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


def sidebar(eng=None, startup_report=None):
    ss = st.session_state
    with st.sidebar:
        st.toggle("Advanced", key="adv")

        st.caption("Publisher scope")
        # multi-select pills-ish
        ss["pubs"] = st.multiselect(
            "",
            options=["OReilly", "Manning", "Pearson"],
            default=ss.get("pubs", []),
            label_visibility="collapsed",
        )

        # judge is always ON. show status only.
        st.caption("Judge")
        st.checkbox("USE_JDG (rerank)", value=True, disabled=True, help="Forced ON", key="_use_jdg_view")

        if ss.get("adv"):
            st.caption("Sort")
            ss["srt"] = st.selectbox("", ["Best evidence", "Semantic"], index=0, label_visibility="collapsed")

            st.caption("Min judge01 (display)")
            ss["jmin"] = st.slider("", 0.0, 0.95, float(ss.get("jmin", 0.35)), 0.05, label_visibility="collapsed")

            st.toggle("Show near-miss even when ok=True", key="nm")

        st.divider()

        st.subheader("Pinned")
        ps = ss.get("pins", [])
        if not ps:
            st.caption("Pin evidence cards to keep them here.")
        else:
            for i, p in enumerate(ps):
                c1, c2 = st.columns([0.88, 0.12])
                with c1:
                    st.write(_pin_lbl(p))
                with c2:
                    st.button("×", key=f"unpin_{i}", on_click=_pin_del, args=(i,), help="Unpin")
            st.button("Clear pins", key="pins_clear", on_click=pins_clear)

        st.divider()

        st.subheader("Clipboard")
        if ss.get("clip"):
            st.code(ss["clip"], language=None)
        else:
            st.caption("Use Copy on a card to put a citation here.")
        st.button("Clear clipboard", key="clip_clear", on_click=cb_clear)

        st.divider()

        st.subheader("Startup status")
        if not startup_report:
            st.caption("No corpus status available.")
        else:
            for row in startup_report:
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
