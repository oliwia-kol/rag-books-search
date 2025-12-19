import rag_engine as re
from streamlit.testing.v1 import AppTest


def _render_mode_and_panel():
    import streamlit as st
    import ui_shell as us
    import ui_adapter as ua

    sample = {
        "hits": [
            {"cid": "c1", "sem_score_n": 0.9, "lex_score_n": 0.5, "judge01": 0.82, "score": 0.74},
            {"cid": "c2", "sem_score_n": 0.4, "lex_score_n": 0.7, "judge01": 0.61, "score": 0.55},
        ],
        "meta": {
            "mode": "quick",
            "mode_params": {"label": "Quick", "final_k": 8, "mmr_k": 16, "ctx_chars": 900, "ctx_tokens": 220},
            "cut_rule": "top_k_with_min_keep_abs_min=0.3",
            "t": {
                "total": 0.01,
                "embed": 0.0,
                "dense": 0.0,
                "lex": 0.0,
                "fuse": 0.0,
                "cut": 0.0,
                "rerank": 0.0,
                "disp_flt": 0.0,
                "direct": 0.0,
                "near_miss": 0.0,
                "llm": 0.0,
            },
            "flags": {
                "dense_used": True,
                "lex_used": True,
                "veto_applied": False,
                "veto_disabled": False,
                "llm_used": False,
                "llm_bypassed": True,
                "dense_clamped": False,
                "lex_clamped": False,
            },
            "cap": {"k_requested": 8, "k_applied": 8, "k_clamped": False, "judge_kind": "proxy_score", "dense_reason": None, "corp_available": ["OReilly"]},
            "meta_nm": {"threshold": 0.28, "used_judge": True},
        },
    }
    startup = {"rows": [{"publisher": "OReilly", "ready": True, "loaded_dense": True, "loaded_db": True, "reason": ""}], "ok": ["OReilly"], "fail": [], "by_corpus": {}}

    st.session_state.clear()
    us.init_state()
    us.sidebar(startup_report=startup)
    ua.render_power_panel(sample)


def test_mode_selector_and_power_panel_rendering():
    at = AppTest.from_function(_render_mode_and_panel)
    at.run()

    assert at.session_state["mode"] == re.MODE_DEFAULT
    assert any(exp.label == "Power panel" for exp in at.expander)
    assert at.dataframe, "power panel should render tables"
