import ui_adapter_custom as ua


class DummyStreamlit:
    def __init__(self):
        self.calls = []
        self.session_state = {}

    def _record(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))

    def subheader(self, text):
        self._record("subheader", text)

    def caption(self, text):
        self._record("caption", text)

    def info(self, text):
        self._record("info", text)

    def markdown(self, text, unsafe_allow_html=False):
        self._record("markdown", text, unsafe_allow_html)

    def container(self, border=False):
        self._record("container", border)
        return self

    def expander(self, label, expanded=False):
        self._record("expander", label, expanded)
        return self

    def columns(self, n):
        self._record("columns", n)
        return [self for _ in range(n)]

    def button(self, *args, **kwargs):
        self._record("button", args, kwargs)
        return False

    def write(self, *args, **kwargs):
        self._record("write", args, kwargs)

    def json(self, *args, **kwargs):
        self._record("json", args, kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_near_miss_section_renders_banner_and_cards(monkeypatch):
    dummy = DummyStreamlit()
    monkeypatch.setattr(ua, "st", dummy)

    rr = {
        "no_evidence": True,
        "near_miss": [
            {
                "cid": "cid-1",
                "cidx": 0,
                "text": "Example near miss about python streaming",
                "score": 0.42,
                "judge01": 0.42,
                "overlap": 2,
                "near_miss_threshold": 0.28,
                "used_judge": True,
                "explanation": "Close but below judge/overlap threshold",
            }
        ],
        "meta": {"meta_nm": {"threshold": 0.28, "used_judge": True}},
    }

    ua.render_near_miss(rr, q="python streaming")

    assert any(c[0] == "expander" and "Near misses" in c[1][0] for c in dummy.calls)
    assert any(c[0] == "info" and "Close but below" in c[1][0] for c in dummy.calls)
    assert any(c[0] == "caption" and "overlap" in str(c[1][0]) for c in dummy.calls)


def test_near_miss_suggestions_render_when_empty(monkeypatch):
    dummy = DummyStreamlit()
    monkeypatch.setattr(ua, "st", dummy)

    rr = {"no_evidence": True, "near_miss": [], "meta": {"meta_nm": {"reason": "near_miss_disabled"}}}

    ua.render_near_miss(rr, q="python streaming")

    assert any(c[0] == "markdown" and "No near-miss results" in c[1][0] for c in dummy.calls)
    assert any("Loosen the query" in c[1][0] for c in dummy.calls if c[0] == "markdown")


def test_evidence_list_suggestions_render_when_empty(monkeypatch):
    dummy = DummyStreamlit()
    monkeypatch.setattr(ua, "st", dummy)

    ua.render_evidence_list({"hits": []}, q="biology")

    assert any(c[0] == "markdown" and "No evidence yet" in c[1][0] for c in dummy.calls)
    assert any("Select another publisher" in c[1][0] for c in dummy.calls if c[0] == "markdown")
