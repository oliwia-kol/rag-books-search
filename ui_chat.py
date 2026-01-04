"""
Chat UI component for RAG Books Search.

This module provides functions to render a chat interface in Streamlit and
generate responses using the retrieval engine. The chat is stateful.
"""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

import rag_engine as re


def init_chat_state() -> None:
    ss = st.session_state
    ss.setdefault("chat_history", [])  # list of {"role": "user"/"assistant", "content": str}
    ss.setdefault("chat_input", "")


def _on_chat_submit(engine: Any) -> None:
    ss = st.session_state
    user_input = (ss.get("chat_input") or "").strip()
    if not user_input:
        return
    ss["chat_history"].append({"role": "user", "content": user_input})
    ss["chat_input"] = ""
    try:
        rr = re.run_query(
            engine,
            user_input,
            pubs=ss.get("pubs", []),
            use_jdg=True,
            judge_mode=ss.get("judge_mode", "proxy"),
            sort=ss.get("srt", "Best evidence"),
            show_nm=bool(ss.get("nm", True)),
            nm=not bool(ss.get("nm_skip", False)),
            jmin=float(ss.get("jmin", re.J_DISP_MIN)),
            mode=ss.get("mode", "quick"),
        )
        hits = list((rr or {}).get("hits") or [])
        answer = (rr or {}).get("answer") or ""
        assistant_msg = re.generate_answer(user_input, hits, answer)
    except Exception as exc:
        assistant_msg = f"Error: {type(exc).__name__}: {exc}"
    ss["chat_history"].append({"role": "assistant", "content": assistant_msg})


def render_chat(engine: Any) -> None:
    init_chat_state()
    ss = st.session_state
    st.markdown("<div class='chat-panel'>", unsafe_allow_html=True)
    hist = ss.get("chat_history", []) or []
    for msg in hist[-6:]:
        role = (msg.get("role") or "assistant").strip().lower()
        content = html.escape(str(msg.get("content", "")))
        row = "user" if role == "user" else "assistant"
        st.markdown(
            (
                f"<div class='chat-row {row}'>"
                f"  <div class='chat-bubble {row}'>"
                f"    <div class='role'>{'You' if row=='user' else 'Assistant'}</div>"
                f"    <div>{content}</div>"
                f"  </div>"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.text_input(
        "Ask a question",
        key="chat_input",
        placeholder="Type your question and press Enter…",
        on_change=_on_chat_submit,
        args=(engine,),
        label_visibility="collapsed",
    )
