import ui_shell as us


class DummySidebarStreamlit:
    def __init__(self):
        self.calls = []
        self.session_state = {}
        self.query_params = {}

    def markdown(self, text, unsafe_allow_html=False):
        self.calls.append(("markdown", text, unsafe_allow_html))
        return text

    def caption(self, text):
        self.calls.append(("caption", text))
        return text

    def divider(self):
        self.calls.append(("divider",))

    def form(self, name, clear_on_submit=False):
        self.calls.append(("form", name, clear_on_submit))
        return DummyForm(self)

    def text_input(self, label, key=None, placeholder=None, label_visibility=None, help=None):
        self.calls.append(("text_input", label, key))
        self.session_state[key] = self.session_state.get(key, "")
        return self.session_state[key]

    def selectbox(self, label, options, index=0, format_func=None, key=None, help=None, label_visibility=None):
        self.calls.append(("selectbox", label, options, index, key))
        choice = options[index] if options else None
        if key:
            self.session_state[key] = choice
        return choice

    def form_submit_button(self, label, use_container_width=None):
        self.calls.append(("form_submit_button", label))
        return False

    def multiselect(self, label, options, default=None, label_visibility=None):
        self.calls.append(("multiselect", label, options))
        value = default or []
        self.session_state["pubs"] = value
        return value

    def toggle(self, label, key=None, value=False, help=None, disabled=False):
        self.calls.append(("toggle", label, key, value))
        if key:
            self.session_state[key] = value
        return value

    def slider(self, label, min_value, max_value, value, step, help=None, label_visibility=None):
        self.calls.append(("slider", label, value))
        self.session_state["jmin"] = value
        return value

    def expander(self, label, expanded=False):
        self.calls.append(("expander", label, expanded))
        return self

    def checkbox(self, label, key=None, help=None):
        self.calls.append(("checkbox", label, key))
        return False

    def columns(self, n):
        self.calls.append(("columns", n))
        count = len(n) if isinstance(n, (list, tuple)) else n
        return [self for _ in range(count)]

    def radio(self, label, options, index=0, format_func=None, help=None, key=None):
        self.calls.append(("radio", label, options, index))
        choice = options[index] if options else None
        if key:
            self.session_state[key] = choice
        return choice

    def write(self, *args, **kwargs):
        self.calls.append(("write", args, kwargs))
        return args

    def button(self, label, key=None, on_click=None, args=(), use_container_width=None, help=None, disabled=False):
        self.calls.append(("button", label, key))
        return False

    def code(self, body, language=None):
        self.calls.append(("code", body, language))
        return body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class DummyForm:
    def __init__(self, parent: DummySidebarStreamlit):
        self.parent = parent

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def text_input(self, *args, **kwargs):
        return self.parent.text_input(*args, **kwargs)

    def selectbox(self, *args, **kwargs):
        return self.parent.selectbox(*args, **kwargs)

    def form_submit_button(self, *args, **kwargs):
        return self.parent.form_submit_button(*args, **kwargs)


def test_sidebar_wraps_pins_and_clipboard(monkeypatch):
    dummy = DummySidebarStreamlit()
    monkeypatch.setattr(us, "st", dummy)
    us.init_state()
    dummy.session_state["pins"] = [
        {"t": "Book 1", "sec": "Ch 1", "pub": "P1", "cid": "c1", "cidx": 0},
        {"t": "Book 2", "sec": "Ch 2", "pub": "P2", "cid": "c2", "cidx": 1},
    ]
    dummy.session_state["clip"] = "Example citation\nLine 2"

    us.sidebar(mount=dummy)

    scroll_sections = [c for c in dummy.calls if c[0] == "markdown" and "scroll-area" in c[1]]
    assert len(scroll_sections) >= 2
    assert all("overflow-y:auto" in c[1] for c in scroll_sections)
    assert any(c[0] == "code" for c in dummy.calls)


def test_toast_flush_retains_last_message(monkeypatch):
    import streamlit as st

    monkeypatch.setattr(us, "st", st)
    st.session_state.clear()
    us.init_state()
    st.session_state["_toast"] = "Pins cleared"
    captured = []
    monkeypatch.setattr(st, "toast", lambda msg: captured.append(msg))

    us.toast_flush()

    assert captured == ["Pins cleared"]
    assert st.session_state["_toast"] is None
    assert st.session_state["_toast_last"] == "Pins cleared"
    st.session_state.clear()
