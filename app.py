"""
Custom application entry point for the RAG Books Search UI.

This module replicates the behaviour of the original ``app.py`` from the
`rag‑books‑search` repository while applying a number of usability
improvements:

* The layout proportions have been adjusted to dedicate more space to
  the evidence list and reduce the width of the context pane.  The
  default column widths are 25 % (sidebar), 55 % (main results) and
  20 % (context).
* Dark mode is enforced.  There is no toggle for light mode – the app
  always uses the dark palette defined in ``ui_theme.py``.
* The top bar no longer contains an end‑user dark‑mode toggle.  It still
  exposes a debug checkbox for development purposes.
* The sidebar UI has been completely reworked in ``ui_shell``.

To run this app you can execute ``streamlit run app.py`` within
your project.  It depends on the modules ``rag_engine``,
``ui_shell``, ``ui_adapter`` and ``ui_theme``.
"""

import streamlit as st

import rag_engine as re
import ui_shell as us
import ui_adapter as ua
import ui_theme as ut
import ui_chat as uc


# Configure the Streamlit page.  Wide layout gives us more horizontal
# space for the three‑column design.
st.set_page_config(page_title="RAG Books Search", layout="wide")

# Hide error details in end‑user mode to avoid leaking stack traces.
try:
    st.set_option("client.showErrorDetails", False)
except Exception:
    pass


# Cache the engine instance so it persists across reruns.  The engine is
# expensive to initialise (it loads the vector indexes and cross encoder).
@st.cache_resource
def _mk_eng():
    return re._mk_eng()


def _run(eng, q: str) -> None:
    """Run a query against the retrieval engine and update session state.

    This function mirrors the original ``_run`` implementation but takes
    advantage of the improved state initialisation in ``ui_shell``.
    It writes the results, errors and timing information into
    ``st.session_state`` so that downstream UI code can display them.

    Args:
        eng: The retrieval engine instance.
        q: The user query string.
    """
    ss = st.session_state
    pubs = ss.get("pubs", [])
    # Reset per‑query state
    ss["_loading"] = True
    ss["act_hit"] = None
    ss["_scroll_ctx"] = False
    ss["_ctx_ts"] = None
    ss["ev_offset"] = 0
    ss["res"] = None
    # Judge is forced ON for this product
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


def _on_search() -> None:
    """Callback invoked when the search form is submitted.

    This wrapper simply trims the input, runs the query and persists
    history in session state.  Exceptions are caught and formatted into
    a user‑friendly error message.
    """
    ss = st.session_state
    q = (ss.get("q_inp") or "").strip()
    # Reset UI state prior to search
    ss["_loading"] = False
    ss["act_hit"] = None
    ss["_scroll_ctx"] = False
    ss["_ctx_ts"] = None
    if not q:
        return
    try:
        ss["_loading"] = True
        _run(ss["eng"], q)
        # Persist query in history (most recent first)
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


def main() -> None:
    """Entry point for the Streamlit app.

    Sets up the session state, loads the retrieval engine and renders the
    three primary columns: sidebar, main content and context panel.  The
    dark theme is applied unconditionally via ``ui_theme.apply_theme``.
    """
    # Initialise session state defaults and chat state
    us.init_state()
    uc.init_chat_state()
    ss = st.session_state
    # Ensure the engine and startup report are loaded only once
    if "eng" not in ss:
        ss["eng"] = _mk_eng()
    if "startup_report" not in ss:
        ss["startup_report"] = re.get_startup_report(ss["eng"])
    # One‑time load from URL parameters (prefill query)
    if "_qp_loaded" not in ss:
        q0 = us.qp_get("q", "")
        if q0:
            ss["q_inp"] = q0
        ss["_qp_loaded"] = True
    # Begin building the DOM
    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    # Render the top bar and enforce dark theme
    us.topbar()
    # Apply custom dark theme (no light mode support)
    ut.apply_theme("dark")
    st.markdown("<div class='layout-grid'>", unsafe_allow_html=True)
    # Layout columns with custom proportions (25%, 55%, 20%)
    left_col, main_col, detail_col = st.columns([0.25, 0.55, 0.20], gap="large")
    # Render the sidebar.  If the form is submitted, trigger the query
    submitted = us.sidebar(ss["eng"], startup_report=ss.get("startup_report"), mount=left_col)
    if submitted:
        _on_search()
    # Show any queued toast messages
    us.toast_flush()
    # Main column: mode display, error box, hero or evidence list
    with main_col:
        # render chat interface at the top of the main column
        uc.render_chat(ss["eng"])
        mode_cfg = re.get_mode_cfg(ss.get("mode", "quick"))
        st.markdown(
            f"<div class='section-title'>Mode</div><div class='chip muted'>{mode_cfg.get('label', 'Quick')}</div>",
            unsafe_allow_html=True,
        )
        st.caption(mode_cfg.get("description", "Speed vs depth"))
        us.global_error_box()
        rr = ss.get("res")
        if not rr:
            # Show hero section when no results are available
            us.render_hero()
        else:
            ua.render_status_strip(rr)
            ua.render_answer(rr)
            ua.render_evidence_list(rr, q=ss.get("last_q", ""))
            ua.render_near_miss(rr, q=ss.get("last_q", ""))
    # Context column: details panel
    with detail_col:
        st.markdown("<div class='context-pane'>", unsafe_allow_html=True)
        with st.expander("Context / Details", expanded=True):
            ua.render_context_panel()
        st.markdown("</div>", unsafe_allow_html=True)
    # Close layout grid and app shell
    st.markdown("</div>", unsafe_allow_html=True)
    if ss.get("show_debug") and ss.get("res"):
        ua.render_power_panel(ss["res"])
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
