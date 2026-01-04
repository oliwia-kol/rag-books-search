"""RAG Books Search — Streamlit app entry.

UI goals (LOVEABLE v2):
- Answer-first, calm, editorial reading experience.
- Sources and Context as intentional layers (progressive disclosure).
- No raw metrics in the main view.
- Debug/diagnostics remain available but never dominate.
"""

from __future__ import annotations

import streamlit as st

import rag_engine as re
import ui_adapter as ua
import ui_chat as uc
import ui_shell as us
import ui_theme as ut

APP_TITLE = "RAG Books — chat-first research with sources"


st.set_page_config(
    page_title="RAG Books",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def _mk_eng():
    return re._mk_eng()


def _safe_run(eng, q: str):
    """Run engine and store rr + q in session state, with defensive error capture."""
    ss = st.session_state
    q = (q or "").strip()
    if not q:
        return None

    # remember history (top, unique)
    hist = ss.get("q_history", []) or []
    if q in hist:
        hist.remove(q)
    hist.insert(0, q)
    ss["q_history"] = hist[:20]

    ss["_loading"] = True
    ss["_ui_err"] = None
    ss["_ui_err_id"] = None
    ss["ev_offset"] = 0
    ss["act_hit"] = None

    try:
        rr = re.run_query(
            eng,
            q,
            pubs=ss.get("pubs", []),
            use_jdg=True,
            judge_mode=ss.get("judge_mode", "proxy"),
            sort=ss.get("srt", "Best evidence"),
            show_nm=bool(ss.get("nm", True)),
            nm=not bool(ss.get("nm_skip", False)),
            jmin=float(ss.get("jmin", re.J_DISP_MIN)),
            mode=ss.get("mode", "quick"),
        )
        ss["rr"] = rr
        ss["q_last"] = q
        return rr
    except Exception as exc:
        ss["_ui_err"] = f"{type(exc).__name__}: {exc}"
        # if engine attached an error_id in meta, capture it
        try:
            ss["_ui_err_id"] = (locals().get("rr") or {}).get("meta", {}).get("error_id")
        except Exception:
            ss["_ui_err_id"] = None
        ss["rr"] = None
        return None
    finally:
        ss["_loading"] = False


def _main():
    ut.apply_theme("dark")
    us.init_state()

    eng = _mk_eng()
    startup_report = None
    try:
        startup_report = getattr(eng, "startup_report", lambda: None)()
    except Exception:
        startup_report = None

    us.topbar(startup_report=startup_report)
    us.toast_flush()

    # Layout: slim rail + main reading column
    rail_col, main_col = st.columns([0.28, 0.72], gap="large")

    with rail_col:
        submitted = us.sidebar(eng, startup_report=startup_report, mount=st.container())
        if submitted:
            _safe_run(eng, st.session_state.get("q_inp", ""))

    with main_col:
        us.global_error_box()

        q_last = st.session_state.get("q_last", "") or ""
        rr = st.session_state.get("rr")

        if not rr:
            # Hero state
            us.render_hero()
            return

        # Status + answer (calm, editorial)
        ua.render_conf(rr)
        st.markdown("<div style='height:.65rem;'></div>", unsafe_allow_html=True)
        ua.render_answer(rr)

        st.markdown("<div style='height:.85rem;'></div>", unsafe_allow_html=True)

        # Layers: Sources | Context | Near-miss
        t_sources, t_context, t_near = st.tabs(["Sources", "Context", "Near-miss"])

        with t_sources:
            ua.render_evidence_list(rr, q=q_last)

        with t_context:
            ua.render_context_panel()

        with t_near:
            ua.render_near_miss(rr, q=q_last)

        # Diagnostics: opt-in only
        if st.session_state.get("show_debug"):
            with st.expander("Diagnostics (raw)", expanded=False):
                ua.render_power_panel(rr)

        # Lightweight “future chatbot” panel (kept subtle)
        with st.expander("Chat (preview)", expanded=False):
            uc.render_chat(eng)


if __name__ == "__main__":
    _main()
