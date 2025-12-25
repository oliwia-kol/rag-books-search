"""
Chat UI component for RAG Books Search.

This module provides functions to render a chat interface in Streamlit and
generate responses using the retrieval engine.  The chat is stateful: it
remembers conversation history via ``st.session_state``.  The response
generation currently calls ``rag_engine.generate_answer``, which should be
implemented as a stub in ``rag_engine.py`` (see Etap 3).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

import rag_engine as re


def init_chat_state() -> None:
    """Initialise chat-related keys in ``st.session_state``."""
    ss = st.session_state
    ss.setdefault("chat_history", [])  # list of {"role": "user"/"assistant", "content": str}
    ss.setdefault("chat_input", "")


def _on_chat_submit(engine: Any) -> None:
    """
    Callback invoked when the user submits a chat message.

    It appends the user message to history, runs the RAG engine to fetch
    evidence and a preliminary answer, then calls ``generate_answer`` to
    produce a natural-language reply.  Errors are caught and shown inline.
    """
    ss = st.session_state
    user_input = (ss.get("chat_input") or "").strip()
    if not user_input:
        return
    # append user message
    ss["chat_history"].append({"role": "user", "content": user_input})
    ss["chat_input"] = ""
    try:
        # run the retrieval engine; judge is forced on (consistent with app)
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
        # compose assistant reply (stub until LLM integration)
        assistant_msg = re.generate_answer(user_input, hits, answer)
    except Exception as exc:
        assistant_msg = f"Error: {type(exc).__name__}: {exc}"
    # append assistant message
    ss["chat_history"].append({"role": "assistant", "content": assistant_msg})


def render_chat(engine: Any) -> None:
    """
    Render the chat panel with history and input box.

    Args:
        engine: The retrieval engine instance from ``st.session_state["eng"]``.
    """
    init_chat_state()
    ss = st.session_state
    st.markdown("<div class='chat-panel'>", unsafe_allow_html=True)
    # display message history
    for msg in ss.get("chat_history", []):
        role = msg.get("role")
        content = msg.get("content", "")
        cls = "chat-msg-user" if role == "user" else "chat-msg-assistant"
        st.markdown(f"<div class='{cls}'><p>{content}</p></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    # input area; submission triggers the callback
    st.text_input(
        "Ask a question",
        key="chat_input",
        placeholder="Type your question and press Enter…",
        on_change=_on_chat_submit,
        args=(engine,),
        label_visibility="collapsed",
    )
