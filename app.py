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
    # judge is forced ON
    rr = re.run_query(
        eng,
        q,
        pubs=pubs,
        use_jdg=True,
        judge_mode=ss.get("judge_mode", ss.get("jdg_mode", "proxy")),
        sort=ss.get("srt", "Best evidence"),
        show_nm=bool(ss.get("nm", True)),
        nm=not bool(ss.get("nm_skip", False)),
        jmin=float(ss.get("jmin", 0.45)),
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


def _on_search():
    ss = st.session_state
    q = (ss.get("q_inp") or "").strip()
    if not q:
        return
    try:
        _run(ss["eng"], q)
        us.qp_set(q=q)
    except Exception as e:
        ss["_ui_err"] = us.format_ui_error(None, f"{type(e).__name__}: {e}")
        ss["_ui_err_id"] = None


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

    us.topbar()
    ut.apply_theme(ss.get("theme_mode", "light"))
    left, main_col, detail = st.columns([0.3, 0.46, 0.24], gap="large")

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
            st.markdown(
                "<div class='empty-state'>Enter a query on the left to see evidence-first results. "
                "Near-miss results will appear separately when no direct evidence exists.</div>",
                unsafe_allow_html=True,
            )
        else:
            ua.render_status_strip(rr)
            ua.render_answer(rr)
            ua.render_evidence_list(rr, q=ss.get("last_q", ""))
            ua.render_near_miss(rr, q=ss.get("last_q", ""))

    with detail:
        with st.expander("Context / Details", expanded=True):
            ua.render_context_panel()

    if ss.get("show_debug") and ss.get("res"):
        ua.render_power_panel(ss["res"])


if __name__ == "__main__":
    main()
