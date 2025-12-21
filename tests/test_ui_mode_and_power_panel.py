import ui_adapter_custom as ua
import ui_shell_custom as us


class DummyModeStreamlit:
    def __init__(self, choice="exact"):
        self.choice = choice
        self.calls = []
        self.session_state = {"mode": "quick"}

    def radio(self, label, options, index=0, format_func=None, help=None, key=None):
        self.calls.append(("radio", label, options, index))
        selected = self.choice or options[index]
        if format_func:
            format_func(selected)
        self.session_state[key] = selected
        return selected

    def caption(self, text):
        self.calls.append(("caption", text))
        return text


class DummyPowerStreamlit:
    def __init__(self):
        self.calls = []
        self.session_state = {}

    def markdown(self, text, unsafe_allow_html=False):
        self.calls.append(("markdown", text, unsafe_allow_html))
        return text

    def expander(self, label, expanded=False):
        self.calls.append(("expander", label, expanded))
        return self

    def caption(self, text):
        self.calls.append(("caption", text))
        return text

    def columns(self, n):
        self.calls.append(("columns", n))
        return [self for _ in range(n)]

    def write(self, payload):
        self.calls.append(("write", payload))
        return payload

    def json(self, payload):
        self.calls.append(("json", payload))
        return payload

    def dataframe(self, data, hide_index=None, use_container_width=None):
        self.calls.append(("dataframe", data))
        return data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_mode_selector_updates_state(monkeypatch):
    dummy = DummyModeStreamlit(choice="exact")
    monkeypatch.setattr(us, "st", dummy)

    choice = us.mode_selector()

    assert choice == "exact"
    assert dummy.session_state["mode"] == "exact"
    assert any("caption" == c[0] and "Exact" in c[1] for c in dummy.calls)


def test_power_panel_renders_mode_and_table(monkeypatch):
    dummy = DummyPowerStreamlit()
    monkeypatch.setattr(ua, "st", dummy)

    rr = {
        "hits": [
            {
                "cid": "cid-1",
                "cidx": 0,
                "text": "Example evidence snippet.",
                "score": 0.9,
                "sem_score_n": 0.8,
                "lex_score_n": 0.6,
                "judge01": 0.7,
                "book": "b1",
                "sec": "s1",
                "publisher": "p1",
            }
        ],
        "meta": {
            "mode": "exact",
            "mode_cfg": {"label": "Find Exact Quote", "description": "Deeper search", "final_k": 12, "mmr_k": 28},
            "log": {"judge_mode": "proxy"},
            "cap": {"judge_kind": "proxy", "judge_ok": True},
            "flags": {"judge_proxy": True},
            "t": {"judge_cache": 0.0, "judge_pred": 0.0},
            "clamp": {
                "retrieval": {"k_requested": 28, "k_applied": 28},
                "context": {"char_clamped": False},
                "prompt": {"char_clamped": False},
            },
            "cut_rule": "top_k_with_min_keep_abs_min=0.3",
            "meta_nm": {"threshold": 0.2},
        },
    }

    ua.render_power_panel(rr)

    assert any(c[0] == "expander" for c in dummy.calls)
    assert any(c[0] == "write" and "cut_rule" in c[1] for c in dummy.calls)
    assert any(c[0] == "dataframe" for c in dummy.calls)
