"""Fast contract check for the UI modules.

Run:
  python smoke_ui_contract.py

Goal: catch app/ui API drift in ~1s.
"""

import importlib


def _has(m, ns):
    miss = [n for n in ns if not hasattr(m, n)]
    if miss:
        raise AttributeError(f"{m.__name__} missing: {miss}")


def main():
    app = importlib.import_module("app")
    us = importlib.import_module("ui_shell")
    ua = importlib.import_module("ui_adapter")
    ut = importlib.import_module("ui_theme")

    _has(us, ["init_state", "sidebar", "global_error_box", "toast_flush", "qp_get", "qp_set", "cb_clear"])
    _has(ua, ["render_answer", "render_conf", "render_context_panel", "render_evidence_list"])
    _has(ut, ["apply_theme"])

    print("OK: UI contract satisfied")


if __name__ == "__main__":
    main()
