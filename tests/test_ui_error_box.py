import ui_shell as us


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeStreamlit:
    def __init__(self):
        self.session_state = {"_ui_err": {"message": "LLM failed", "id": "err-deadbeef", "details": "timeout"}}
        self.errors = []
        self.captions = []
        self.rerun_called = False

    def container(self, border=True):
        return _Ctx()

    def error(self, msg):
        self.errors.append(msg)

    def columns(self, sizes):
        return (_Ctx(), _Ctx())

    def button(self, *args, **kwargs):
        return False

    def caption(self, msg, **kwargs):
        self.captions.append(msg)

    def rerun(self):
        self.rerun_called = True


def test_global_error_box_surfaces_error_id_and_retry(monkeypatch):
    fake = _FakeStreamlit()
    monkeypatch.setattr(us, "st", fake)

    us.global_error_box()

    assert any("err-deadbeef" in e for e in fake.errors)
    assert any("retry" in c.lower() for c in fake.captions)
