import ui_shell as us


def test_format_ui_error_includes_id_and_hint():
    msg = us.format_ui_error("err-abc123", "boom")
    assert "err-abc123" in msg
    assert "retry" in msg.lower()


def test_format_ui_error_handles_missing_id():
    msg = us.format_ui_error(None, "oops")
    assert "oops" in msg
    assert "retry" in msg.lower()


def test_global_error_box_includes_id_and_hint(monkeypatch):
    import streamlit as st

    st.session_state.clear()
    us.init_state()
    st.session_state["_ui_err"] = us.format_ui_error("err-test123", "boom")
    st.session_state["_ui_err_id"] = "err-test123"
    rendered = []

    us.global_error_box(renderer=lambda payload: rendered.append(payload))

    assert rendered, "renderer should capture payload"
    payload = rendered[0]
    assert payload["error_id"] == "err-test123"
    assert "err-test123" in payload["message"]
    assert "retry" in payload["hint"].lower()
