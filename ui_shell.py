import streamlit as st

import rag_engine as re


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
    ss.setdefault("nm_skip", False)    # skip near-miss computation to save cost
    ss.setdefault("jmin", 0.45)        # display min judge01
    ss.setdefault("jdg_mode", "proxy") # judge mode: proxy / real / off

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
        with c1:
            st.caption("If this persists, reload the page and retry the query. Error IDs help with debugging.")


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


def _badge(txt: str, tone: str = "ghost") -> str:
    tone_cls = {
        "ghost": "rag-badge-ghost",
        "strong": "rag-badge-strong",
        "warn": "rag-badge-warn",
        "good": "rag-badge-good",
    }.get(tone, "rag-badge-ghost")
    return f"<span class='rag-badge {tone_cls}'>{txt}</span>"


def mode_panel():
    ss = st.session_state
    nm_state = "Showing near-miss" if ss.get("nm", True) else "Hidden when ok=True"
    nm_tone = "good" if ss.get("nm", True) else "ghost"
    if ss.get("nm_skip", False):
        nm_state = "Near-miss skipped (faster)"
        nm_tone = "warn"

    with st.container(border=True):
        st.markdown('<div class="rag-card-top rag-card-mk"></div>', unsafe_allow_html=True)
        st.markdown('<div class="rag-mode-panel">', unsafe_allow_html=True)
        st.markdown("### Modes & visibility")
        c1, c2, c3 = st.columns([0.34, 0.33, 0.33])
        with c1:
            st.caption("Judge selector")
            st.markdown(_badge(f"Mode: {ss.get('jdg_mode', 'proxy').upper()}", tone="strong"), unsafe_allow_html=True)
            st.caption("Reranker locked ON.")
        with c2:
            st.caption("Power panel")
            st.markdown(_badge("Context + pins ready", tone="ghost"), unsafe_allow_html=True)
            st.caption("Use Expand on a card to open Context.")
        with c3:
            st.caption("Near-miss")
            st.markdown(_badge(nm_state, tone=nm_tone), unsafe_allow_html=True)
            st.caption("Tweak near-miss in Advanced > Sidebar.")
        st.markdown("</div>", unsafe_allow_html=True)


def sidebar(eng=None, startup_report=None):
    ss = st.session_state
    with st.sidebar:
        st.toggle("Advanced", key="adv")

        if eng is not None:
            rep = re.get_startup_report(eng)
            ok = rep.get("ok", [])
            fail = rep.get("fail", [])
            st.caption(f"Startup: loaded {len(ok)} / {len(ok)+len(fail)} corpora")
            if fail:
                with st.expander("Corpus status", expanded=False):
                    for k, v in rep.get("by_corpus", {}).items():
                        emoji = "✅" if v.get("ok") else "⚠️"
                        reasons = ", ".join(v.get("reasons", []))
                        st.write(f"{emoji} {k}: dense_loaded={v.get('dense_loaded')} db_loaded={v.get('db_loaded')} dim_ok={v.get('dim_ok')} {reasons}")

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
        st.selectbox(
            "Judge mode",
            options=["proxy", "real", "off"],
            index=["proxy", "real", "off"].index(ss.get("jdg_mode", "proxy")),
            help="proxy = score-based, real = cross-encoder (CPU), off = bypass",
            key="jdg_mode",
        )

        if eng is not None:
            st.caption("Startup status")
            try:
                rep = getattr(eng, "corp_status", {}) or {}
                for name, r in rep.items():
                    ok = r.get("loaded") and r.get("dim_ok")
                    lbl = "loaded" if ok else "failed"
                    reason = ", ".join(r.get("reasons", []) or [])
                    st.write(f"{name}: {lbl}" + (f" ({reason})" if reason else ""))
            except Exception:
                st.write("status unavailable")

        if ss.get("adv"):
            st.caption("Sort")
            ss["srt"] = st.selectbox("", ["Best evidence", "Semantic"], index=0, label_visibility="collapsed")

            st.caption("Min judge01 (display)")
            ss["jmin"] = st.slider("", 0.0, 0.95, float(ss.get("jmin", 0.35)), 0.05, label_visibility="collapsed")

            st.toggle("Show near-miss even when ok=True", key="nm")
            st.toggle("Skip near-miss computation (faster)", key="nm_skip")

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
