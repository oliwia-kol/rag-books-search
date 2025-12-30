import streamlit as st

import rag_engine as re
import ui_shell as us
import ui_adapter as ua
import ui_theme as ut


st.set_page_config(page_title="RAG Books Search", layout="wide")

try:
    st.set_option("client.showErrorDetails", False)
except Exception:
    pass


@st.cache_resource
def _mk_eng():
    return re._mk_eng()


def _run(eng, q: str):
    ss = st.session_state
    pubs = ss.get("pubs", [])
    ss["_loading"] = True
    ss["act_hit"] = None
    ss["_scroll_ctx"] = False
    ss["_ctx_ts"] = None
    ss["ev_offset"] = 0
    ss["res"] = None
    # judge is forced ON
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
        ss["last_q"] = q
        if rr.get("meta", {}).get("err"):
            err = rr["meta"]["err"]
            ss["_ui_err"] = us.format_ui_error(err.get("id"), err.get("msg"))
            ss["_ui_err_id"] = err.get("id")
        else:
            ss["_ui_err"] = None
            ss["_ui_err_id"] = None
    finally:
        ss["_loading"] = False


def _on_search():
    ss = st.session_state
    q = (ss.get("q_inp") or "").strip()
    ss["_loading"] = False
    ss["act_hit"] = None
    ss["_scroll_ctx"] = False
    ss["_ctx_ts"] = None
    if not q:
        return
    try:
        ss["_loading"] = True
        _run(ss["eng"], q)
        us.qp_set(q=q)
        hist = ss.get("q_history", [])
        if q in hist:
            hist.remove(q)
        hist.insert(0, q)
        ss["q_history"] = hist[:5]
    except Exception as e:
        ss["_ui_err"] = us.format_ui_error(None, f"{type(e).__name__}: {e}")
        ss["_ui_err_id"] = None
    finally:
        ss["_loading"] = False


def main():
    us.init_state()
    ss = st.session_state
    if "eng" not in ss:
        ss["eng"] = _mk_eng()
    if "startup_report" not in ss:
        ss["startup_report"] = re.get_startup_report(ss["eng"])

    # one-time load from URL
    if "_qp_loaded" not in ss:
        q0 = us.qp_get("q", "")
        if q0:
            ss["q_inp"] = q0
        ss["_qp_loaded"] = True
    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    theme_mode = us.topbar()
    ut.apply_theme(theme_mode)
    st.markdown("<div class='layout-grid'>", unsafe_allow_html=True)
    left, main_col, detail = st.columns([0.28, 0.48, 0.24], gap="large")

    submitted = us.sidebar(ss["eng"], startup_report=ss.get("startup_report"), mount=left)
    if submitted:
        _on_search()

    us.toast_flush()

    with main_col:
        mode_cfg = re.get_mode_cfg(ss.get("mode", "quick"))
        st.markdown(
            f"<div class='section-title'>Mode</div><div class='chip muted'>{mode_cfg.get('label', 'Quick')}</div>",
            unsafe_allow_html=True,
        )
        st.caption(mode_cfg.get("description", "Speed vs depth"))
        us.global_error_box()

        rr = ss.get("res")
        if not rr:
            us.render_hero()
        else:
            ua.render_status_strip(rr)
            ua.render_answer(rr)
            ua.render_evidence_list(rr, q=ss.get("last_q", ""))
            ua.render_near_miss(rr, q=ss.get("last_q", ""))

    with detail:
        st.markdown("<div class='context-pane'>", unsafe_allow_html=True)
        with st.expander("Context / Details", expanded=True):
            ua.render_context_panel()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    if ss.get("show_debug") and ss.get("res"):
        ua.render_power_panel(ss["res"])
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
