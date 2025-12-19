import ui_shell as us


def test_format_ui_error_includes_id_and_hint():
    msg = us.format_ui_error("err-abc123", "boom")
    assert "err-abc123" in msg
    assert "retry" in msg.lower()


def test_format_ui_error_handles_missing_id():
    msg = us.format_ui_error(None, "oops")
    assert "oops" in msg
    assert "retry" in msg.lower()
