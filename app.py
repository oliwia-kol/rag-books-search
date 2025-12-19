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
        sort=ss.get("srt", "Best evidence"),
        show_nm=bool(ss.get("nm", True)),
        jmin=float(ss.get("jmin", 0.45)),
    )
    ss["res"] = rr
    ss["last_q"] = q


def _on_search():
    ss = st.session_state
    q = (ss.get("q_inp") or "").strip()
    if not q:
        return
    try:
        _run(ss["eng"], q)
        us.qp_set(q=q)
    except Exception as e:
        ss["_ui_err"] = f"Search failed: {type(e).__name__}: {e}"


def main():
    ut.apply_theme()
    us.init_state()

    ss = st.session_state
    if "eng" not in ss:
        ss["eng"] = _mk_eng()

    # one-time load from URL
    if "_qp_loaded" not in ss:
        q0 = us.qp_get("q", "")
        if q0:
            ss["q_inp"] = q0
        ss["_qp_loaded"] = True

    us.sidebar(ss["eng"])
    us.toast_flush()

    st.title("RAG Books Search")
    st.caption("evidence-first • judge ON • CPU-friendly")

    with st.form("q_form", clear_on_submit=False):
        st.text_input(
            "Query",
            key="q_inp",
            placeholder="e.g. What is model monitoring in production?",
        )
        st.form_submit_button("Search", on_click=_on_search)

    us.global_error_box()

    rr = ss.get("res")
    if not rr:
        return

    ua.render_answer(rr)
    ua.render_conf(rr)
    ua.render_context_panel()
    ua.render_evidence_list(rr, q=ss.get("last_q", ""))


if __name__ == "__main__":
    main()
