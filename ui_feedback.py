"""Feedback logging utilities for RAG Books Search."""

from __future__ import annotations

import json
import time
from pathlib import Path

FEEDBACK_LOG = Path(__file__).resolve().parent / "feedback_log.json"


def record_feedback(query: str, answer: str, positive: bool) -> None:
    """Record user feedback on an answer to a local log file."""
    payload = {
        "query": query,
        "answer": answer,
        "positive": bool(positive),
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Best-effort logging; feedback should not break the UI.
        return
