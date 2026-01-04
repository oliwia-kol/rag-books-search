"""Render RAG results (answer, sources, context).

Contract constraints (smoke_ui_contract.py):
- Must expose: render_answer, render_conf, render_context_panel, render_evidence_list.

Design goals:
- Answer feels like reading, not a dashboard.
- Sources are elegant cards; metadata is supportive.
- Context is explicit and discoverable.
"""

from __future__ import annotations

import html
import re
import time
from typing import Any, Dict

import streamlit as st


def _ws(s: str) -> str:
    return " ".join((s or "").split())


def _sent(s: str, n: int = 2, mx: int = 720) -> str:
    s = _ws(s)
    if not s:
        return ""
    out, cur = [], ""
    for ch in s:
        cur += ch
        if ch in ".!?":
            out.append(cur.strip())
            cur = ""
        if len(" ".join(out)) >= mx:
            break
        if len(out) >= n:
            break
    if not out:
        out = [s[:mx]]
    return " ".join(out).strip()


def _titleize_slug(s: str) -> str:
    s = _ws(s)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:1].upper() + s[1:] if s else ""


def _pub(h: Dict[str, Any]) -> str:
    return _ws((h or {}).get("pub") or (h or {}).get("publisher") or (h or {}).get("corp") or "")


def _book(h: Dict[str, Any]) -> str:
    for k in ("book", "title", "name"):
        v = (h or {}).get(k)
        if v:
            return _titleize_slug(str(v))
    v = (h or {}).get("file")
    if v:
        v = re.sub(r"\.[a-zA-Z0-9]+$", "", str(v))
        return _titleize_slug(v)
    return "Untitled"


def _sec(h: Dict[str, Any]) -> str:
    v = _ws((h or {}).get("section") or (h or {}).get("sec") or "")
    if not v:
        return ""
    v = v.replace("Text/", "").replace("text/", "")
    v = re.sub(r"chapter[-_ ]?(\d+)(\.[a-z]+)?", r"Chapter \1", v, flags=re.I)
    v = re.sub(r"\s+", " ", v).strip()
    return _titleize_slug(v)


def _j01(h: Dict[str, Any]) -> float:
    try:
        return float((h or {}).get("judge01") or 0.0)
    except Exception:
        return 0.0


def _toast(msg: str) -> None:
    st.session_state["_toast"] = msg


def _pin_add(h: Dict[str, Any]) -> None:
    ss = st.session_state
    p = {
        "t": _book(h),
        "sec": _sec(h),
        "pub": _pub(h),
        "cid": str((h or {}).get("cid") or ""),
        "cidx": str((h or {}).get("cidx") or ""),
    }
    key = (p.get("cid"), p.get("cidx"))
    for e in ss.get("pins", []) or []:
        if (e.get("cid"), e.get("cidx")) == key:
            _toast("Already pinned")
            return
    ss.setdefault("pins", []).append(p)
    _toast("Pinned")


def _clip_set(h: Dict[str, Any]) -> None:
    t = _book(h)
    pub = _pub(h)
    sec = _sec(h)
    hdr = f"{t} ({pub})" if pub else t
    if sec:
        hdr += f" — {sec}"
    sn = _sent((h or {}).get("text") or "", n=1)
    r = (h or {}).get("cidx")
    ref = f"Ref: cidx={r}" if r is not None and str(r) != "" else ""
    out = hdr
    if sn:
        out += "\n" + sn
    if ref:
        out += "\n" + ref
    st.session_state["clip"] = out
    _toast("Citation copied")


def _ctx_open(h: Dict[str, Any]) -> None:
    st.session_state["act_hit"] = h
    st.session_state["_ctx_ts"] = time.time()
    _toast("Context opened")


def render_conf(rr: Dict[str, Any]):
    render_status_strip(rr)


def render_status_strip(rr: Dict[str, Any]) -> None:
    meta = (rr or {}).get("meta", {}) or {}
    hits = (rr or {}).get("hits", []) or []

    no_ev = bool((rr or {}).get("no_evidence"))
    direct = "No direct evidence" if no_ev else "Evidence found"

    n_hits = len(hits)
    n_books = meta.get("n", {}).get("uniq_books")
    if n_books is None:
        n_books = len({h.get("book") for h in hits if h.get("book")})
    dur_ms = int((meta.get("t", {}).get("total", 0.0) or 0.0) * 1000)

    js = [_j01(h) for h in hits]
    j = max(js) if js else 0.0
    if j >= 0.85 and not no_ev:
        cls, label = "strong", "Strong"
    elif j >= 0.60 and not no_ev:
        cls, label = "mixed", "Mixed"
    else:
        cls, label = "weak", "Weak"

    st.markdown(
        (
            "<div class='card' style='padding:.75rem 1.0rem;'>"
            "  <div style='display:flex; justify-content:space-between; align-items:center; gap:.75rem; flex-wrap:wrap;'>"
            f"    <div style='display:flex; align-items:center; gap:.55rem;'>"
            f"      <span class='chip {cls}'>{label}</span>"
            f"      <span style='color:rgba(238,242,255,.92); font-weight:600;'>{html.escape(direct)}</span>"
            "    </div>"
            "    <div style='display:flex; gap:.45rem; flex-wrap:wrap;'>"
            f"      <span class='chip'>Sources: {n_hits}</span>"
            f"      <span class='chip'>Books: {n_books}</span>"
            f"      <span class='chip' title='Total time'>{dur_ms} ms</span>"
            "    </div>"
            "  </div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_answer(rr: Dict[str, Any]) -> None:
    ans = (rr or {}).get("answer") or ""
    no_ev = bool((rr or {}).get("no_evidence"))

    if no_ev:
        st.markdown(
            """
<div class='card answer'>
  <h2>Answer</h2>
  <div class='text muted'>I can’t answer confidently from the books for this query. I’m showing the closest matches in <b>Sources</b>.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    if not ans:
        return

    txt = html.escape(ans).replace("\n\n", "</p><p class='text'>").replace("\n", "<br/>")
    st.markdown(
        f"""
<div class='card answer'>
  <h2>Answer</h2>
  <p class='text'>{txt}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def _ev_actions(h: Dict[str, Any], i: int) -> None:
    c1, c2, c3 = st.columns([0.26, 0.36, 0.38], gap="small")
    with c1:
        st.button("Pin", key=f"pin_{i}", on_click=_pin_add, args=(h,), use_container_width=True)
    with c2:
        st.button("Copy", key=f"cpy_{i}", on_click=_clip_set, args=(h,), use_container_width=True)
    with c3:
        st.button("Open context", key=f"ctx_{i}", on_click=_ctx_open, args=(h,), use_container_width=True)


def render_evidence_list(rr: Dict[str, Any], q: str = "") -> None:
    hits = (rr or {}).get("hits", []) or []
    if not hits:
        st.markdown("<div class='card'><div style='color:var(--mut2);'>No sources.</div></div>", unsafe_allow_html=True)
        return

    ss = st.session_state
    off = int(ss.get("ev_offset", 0) or 0)
    page = 8
    shown = hits[: max(page, off + page)]

    for i, h in enumerate(shown):
        pub = _pub(h) or ""
        bk = _book(h)
        sec = _sec(h)
        j = _j01(h)
        tier = "strong" if j >= 0.85 else "mixed" if j >= 0.60 else "weak"

        meta = " • ".join([x for x in [pub, sec] if x])
        sn = (h or {}).get("text") or ""
        sn = _sent(sn, n=3, mx=860)

        try:
            qs = [w for w in re.findall(r"[A-Za-z0-9_]+", q or "") if len(w) >= 4]
            qs = list(dict.fromkeys(qs))[:6]
            for w in qs:
                sn = re.sub(rf"\b({re.escape(w)})\b", r"<span class='hl'>\1</span>", sn, flags=re.I)
        except Exception:
            sn = html.escape(sn)
        else:
            sn = sn if "<span" in sn else html.escape(sn)

        st.markdown(
            f"""
<div class='card'>
  <div class='ev-head'>
    <div>
      <div class='ev-title'>{html.escape(bk)}</div>
      <div class='ev-meta'>{html.escape(meta)}</div>
    </div>
    <div style='display:flex; gap:.4rem; align-items:center; flex-wrap:wrap;'>
      <span class='chip {tier}' title='judge01'>{tier.title()}</span>
      <span class='chip' title='judge01'>{j:.2f}</span>
    </div>
  </div>
  <div class='ev-sn'>{sn}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        _ev_actions(h, i)
        st.markdown("<div style='height:.55rem;'></div>", unsafe_allow_html=True)

    if len(shown) < len(hits):
        if st.button("Load more sources", use_container_width=True):
            ss["ev_offset"] = len(shown)


def render_context_panel() -> None:
    h = st.session_state.get("act_hit")
    if not h:
        st.markdown(
            """
<div class='card'>
  <div style='font-weight:650; margin-bottom:.35rem;'>Context</div>
  <div style='color:var(--mut2); line-height:1.6;'>Open a source to see the surrounding passage here.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    pub = _pub(h) or ""
    bk = _book(h)
    sec = _sec(h)
    meta = " • ".join([x for x in [pub, sec] if x])
    txt = html.escape(_ws((h or {}).get("text") or ""))

    st.markdown(
        f"""
<div class='card'>
  <div style='display:flex; justify-content:space-between; gap:.75rem; align-items:flex-start;'>
    <div>
      <div style='font-weight:700; letter-spacing:.2px;'>{html.escape(bk)}</div>
      <div style='color:var(--mut2); margin-top:.15rem;'>{html.escape(meta)}</div>
    </div>
  </div>
  <div style='margin-top:.75rem; line-height:1.75; color:var(--ink);'>{txt}</div>
  <div style='margin-top:.85rem;'>
""",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([0.5, 0.5], gap="small")
    with c1:
        st.button("Copy", on_click=_clip_set, args=(h,), use_container_width=True)
    with c2:
        st.button("Close", on_click=lambda: st.session_state.update({"act_hit": None}), use_container_width=True)
    st.markdown("</div></div>", unsafe_allow_html=True)


def render_near_miss(rr: Dict[str, Any], q: str = "") -> None:
    nm = (rr or {}).get("near_miss") or []
    if not nm:
        return
    with st.expander("Closest matches (near-miss)", expanded=False):
        tmp = {"hits": nm, "meta": (rr or {}).get("meta", {})}
        render_evidence_list(tmp, q=q)


def render_power_panel(rr: Dict[str, Any]) -> None:
    meta = (rr or {}).get("meta", {}) or {}
    st.caption("Meta")
    st.json(
        {
            "mode": meta.get("mode"),
            "coverage": meta.get("coverage"),
            "conf": meta.get("conf"),
            "cov": meta.get("cov"),
            "cut_rule": meta.get("cut_rule"),
            "flags": meta.get("flags"),
            "cap": meta.get("cap"),
            "t": meta.get("t"),
            "n": meta.get("n"),
        }
    )
