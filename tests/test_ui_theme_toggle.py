import streamlit as st

import ui_theme as ut


def test_apply_theme_uses_current_session_mode(monkeypatch):
    st.session_state.clear()
    st.session_state["theme_mode"] = "dark"
    rendered = []

    def fake_markdown(content, *_, **__):
        rendered.append(content)

    monkeypatch.setattr(st, "markdown", fake_markdown)

    ut.apply_theme()

    assert st.session_state["theme_mode"] == "dark"
    assert any("data-theme', 'dark'" in block for block in rendered)


def test_apply_theme_respects_runtime_toggle(monkeypatch):
    st.session_state.clear()
    st.session_state["theme_mode"] = "light"
    rendered = []

    def fake_markdown(content, *_, **__):
        rendered.append(content)

    monkeypatch.setattr(st, "markdown", fake_markdown)

    # emulate toggle mutation in the same run loop
    st.session_state["theme_mode"] = "dark"
    ut.apply_theme()

    assert st.session_state["theme_mode"] == "dark"
    assert any("data-theme', 'dark'" in block for block in rendered)
