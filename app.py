# app.py
"""Streamlit entrypoint for RAG Books Search.

UI NORTH STAR (from LOVEABLE_PROMPT_UI.md): calm, chat-first, evidence as support.

Contract constraints:
- Must stay compatible with rag_engine.py
- Must keep ui_shell/ui_adapter/ui_theme APIs (see smoke_ui_contract.py)
"""

from __future__ import annotations

import streamlit as st

import rag_engine as re
import ui_adapter as ua
import ui_chat as uc
import ui_shell as us
import ui_theme as ut


st.set_page_config(page_title="RAG Books Search", layout="wide")

try:
    st.set_option("client.showErrorDetails", False)
except Exception:
    pass


@st.cache_resource
def _mk_eng():
    return re._mk_eng()


def _run(eng, q: str) -> None:
    ss = st.session_state
    pubs = ss.get("pubs", [])

    ss["_loading"] = True
    ss["act_hit"] = None
    ss["ev_offset"] = 0
    ss["res"] = None
    ss["last_q"] = q

    try:
        rr = re.run_query(
            eng,
            q,
            pubs=pubs,
            use_jdg=True,
            judge_mode=ss.get("judge_mode", ss.get("jdg_mode", "proxy")),
            sort=ss.get("srt", us.SORT_OPTIONS[0]),
            show_nm=bool(ss.get("nm", True)),
            nm=not bool(ss.get("nm_skip", False)),
            jmin=float(ss.get("jmin", us.DEFAULT_JMIN)),
            mode=ss.get("mode", "quick"),
        )
        ss["res"] = rr

        err = (rr or {}).get("meta", {}).get("err")
        if err:
            ss["_ui_err"] = us.format_ui_error(err.get("id"), err.get("msg"))
            ss["_ui_err_id"] = err.get("id")
        else:
            ss["_ui_err"] = None
            ss["_ui_err_id"] = None
    finally:
        ss["_loading"] = False


def _on_search() -> None:
    ss = st.session_state
    q = (ss.get("q_inp") or "").strip()
    if not q:
        return

    try:
        _run(ss["eng"], q)
        hist = ss.get("q_history", []) or []
        if q in hist:
            hist.remove(q)
        hist.insert(0, q)
        ss["q_history"] = hist[:8]
        us.qp_set(q=q)
    except Exception as e:
        ss["_ui_err"] = us.format_ui_error(None, f"{type(e).__name__}: {e}")
        ss["_ui_err_id"] = None


def main() -> None:
    us.init_state()
    uc.init_chat_state()
    ss = st.session_state

    if "eng" not in ss:
        ss["eng"] = _mk_eng()
    if "startup_report" not in ss:
        ss["startup_report"] = re.get_startup_report(ss["eng"])

    if "_qp_loaded" not in ss:
        q0 = us.qp_get("q", "")
        if q0:
            ss["q_inp"] = q0
        ss["_qp_loaded"] = True

    ut.apply_theme("dark")

    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    us.topbar()

    submitted = us.sidebar(ss["eng"], startup_report=ss.get("startup_report"))
    if submitted:
        _on_search()
    us.toast_flush()

    # MAIN STAGE
    st.markdown("<div class='stage'>", unsafe_allow_html=True)

    # Chat is the visual center (even before we ship a full chatbot).
    uc.render_chat(ss["eng"])

    us.global_error_box()

    rr = ss.get("res")
    if not rr:
        us.render_hero()
        st.markdown("</div></div>", unsafe_allow_html=True)
        return

    ua.render_status_strip(rr)
    ua.render_answer(rr)

    # Progressive disclosure: sources/context live behind tabs.
    t1, t2, t3 = st.tabs(["Sources", "Context", "Near-miss"])
    with t1:
        ua.render_evidence_list(rr, q=ss.get("last_q", ""))
    with t2:
        ua.render_context_panel()
    with t3:
        ua.render_near_miss(rr, q=ss.get("last_q", ""))

    # Optional debug (still behind a single toggle).
    if ss.get("show_debug"):
        ua.render_power_panel(rr)

    st.markdown("</div></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
