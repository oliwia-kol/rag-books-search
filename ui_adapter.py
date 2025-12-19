import re
import time
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components


def _ws(s: str) -> str:
    return " ".join(str(s or "").replace("\u00a0", " ").split())


def _de_mojibake(s: str) -> str:
    # fast cleanup of common RAG artifacts
    s = str(s or "")
    s = s.replace("Â", "").replace("\uFFFD", "")
    return s


def _split_sents(txt: str) -> List[str]:
    t = _ws(_de_mojibake(txt))
    if not t:
        return []
    # naive sentence split with punctuation kept
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.strip() for p in parts if p.strip()]


def _sent(txt: str, n: int = 1, mx: int = 520) -> str:
    ss = _split_sents(txt)
    if not ss:
        return ""
    out: List[str] = []
    for s in ss:
        if len(s) < 28:
            continue
        out.append(s)
        if len(out) >= n:
            break
    if not out:
        out = [ss[0]]
    t = " ".join(out)

    # ensure it looks like a sentence
    if t and t[-1] not in ".!?":
        t += "."
    if t and not (t[0].isupper() or t[0].isdigit()):
        t = "[...] " + t

    if len(t) > mx:
        t = t[: mx - 1].rstrip() + "…"
    return t


def _snippet(h: Dict[str, Any], q: str = "", mx: int = 280) -> str:
    txt = (h or {}).get("text") or (h or {}).get("tx") or ""
    t = _sent(txt, n=1, mx=mx)
    if not q:
        return t
    terms = [re.escape(w) for w in re.findall(r"[A-Za-z0-9]{3,}", q)]
    if not terms:
        return t
    pat = re.compile("(" + "|".join(terms) + ")", flags=re.I)
    return pat.sub(r"**\\1**", t)


def _titleize_slug(s: str) -> str:
    s = _ws(_de_mojibake(s))
    if not s:
        return ""
    # split camelcase-ish and underscores
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # smart titlecase but keep ALLCAPS acronyms
    out = []
    for w in s.split():
        if w.isupper() and len(w) <= 5:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def _pretty_title(h: Dict[str, Any]) -> str:
    for k in ("book_pretty", "book_title_pretty", "book_title"):
        v = (h or {}).get(k)
        if v:
            return _titleize_slug(v)
    v = (h or {}).get("file")
    if v:
        v = re.sub(r"\.[a-zA-Z0-9]+$", "", str(v))
        return _titleize_slug(v)
    return "Untitled"


def _sec_lbl(h: Dict[str, Any]) -> str:
    v = (h or {}).get("section") or (h or {}).get("sec") or ""
    v = _ws(_de_mojibake(v))
    if not v:
        return ""
    # normalize common patterns
    v = v.replace("Text/", "").replace("text/", "")
    v = re.sub(r"chapter[-_ ]?(\d+)(\.[a-z]+)?", r"Chapter \1", v, flags=re.I)
    v = re.sub(r"ch(\d+)$", r"Chapter \1", v, flags=re.I)
    v = re.sub(r"\s+", " ", v).strip()
    return _titleize_slug(v)


def _pub(h: Dict[str, Any]) -> str:
    return _ws((h or {}).get("pub") or (h or {}).get("publisher") or "")


def _cid(h: Dict[str, Any]) -> str:
    return str((h or {}).get("cid") or "")


def _cidx(h: Dict[str, Any]) -> str:
    v = (h or {}).get("cidx")
    if v is None:
        return ""
    return str(v)


def _j01(h: Dict[str, Any]) -> float:
    try:
        return float((h or {}).get("judge01") or 0.0)
    except Exception:
        return 0.0


def _score(h: Dict[str, Any]) -> float:
    try:
        return float((h or {}).get("score") or 0.0)
    except Exception:
        return 0.0


def _toast(msg: str):
    st.session_state["_toast"] = msg


def _pin_add(h: Dict[str, Any]):
    ss = st.session_state
    p = {
        "t": _pretty_title(h),
        "sec": _sec_lbl(h),
        "pub": _pub(h),
        "cid": _cid(h),
        "cidx": _cidx(h),
    }
    # dedupe by (cid,cidx)
    key = (p.get("cid"), p.get("cidx"))
    for e in ss.get("pins", []):
        if (e.get("cid"), e.get("cidx")) == key:
            _toast("Already pinned")
            return
    ss.setdefault("pins", []).append(p)
    _toast("Pinned")


def _clip_set(h: Dict[str, Any]):
    t = _pretty_title(h)
    pub = _pub(h)
    sec = _sec_lbl(h)
    if sec:
        l1 = f"{t} ({pub}), {sec}" if pub else f"{t}, {sec}"
    else:
        l1 = f"{t} ({pub})" if pub else t

    sn = _sent((h or {}).get("text") or "", n=1)
    r = _cidx(h)
    l3 = f"Ref: cidx={r}" if r else ""

    out = l1
    if sn:
        out += "\n" + sn
    if l3:
        out += "\n" + l3

    st.session_state["clip"] = out
    _toast("Citation ready")


def _ctx_open(h: Dict[str, Any]):
    st.session_state["act_hit"] = h
    st.session_state["_scroll_ctx"] = True
    st.session_state["_ctx_ts"] = time.time()
    _toast("Context opened")


def _ctx_close():
    st.session_state["act_hit"] = None
    _toast("Context closed")


def _rank_key(h: Dict[str, Any]) -> float:
    # judge is forced ON in this product
    return _j01(h)


def render_answer(rr: Dict[str, Any]):
    ans = (rr or {}).get("answer") or ""
    with st.container(border=True):
        st.markdown('<div class="rag-card-top"></div>', unsafe_allow_html=True)
        st.subheader("Answer")
        if ans:
            sents = _split_sents(ans)
            limited = " ".join(sents[:5]) if sents else ans
            st.write(limited)
            if len(sents) > 5:
                st.caption("Clamped to 5 sentences for readability.")
        else:
            hits = (rr or {}).get("hits") or []
            if hits:
                st.caption("No LLM answer. Evidence-first preview:")
                st.write(_snippet(hits[0], q=""))
            else:
                st.caption("No answer.")


def render_conf(rr: Dict[str, Any]):
    cf = (rr or {}).get("confidence")
    if cf is None:
        return
    try:
        v = float(cf)
    except Exception:
        return
    v = max(0.0, min(1.0, v))
    cov = (rr or {}).get("coverage") or "WEAK"
    meta = (rr or {}).get("meta") or {}
    n = meta.get("n", {})
    books = n.get("uniq_books", 0)
    secs = n.get("uniq_sections", 0)
    state = "Low" if v < 0.35 else "Medium" if v < 0.65 else "High"
    st.caption("Confidence")
    c1, c2 = st.columns([0.65, 0.35])
    with c1:
        try:
            st.progress(v, text=f"{state} • coverage {cov}")
        except Exception:
            st.progress(v)
            st.caption(f"{state} • coverage {cov}")
    with c2:
        st.caption(f"Books: {books} | Sections: {secs}")
        if books <= 1:
            st.warning("Single-source evidence. Verify carefully.", icon="⚠️")


def render_context_panel():
    ss = st.session_state
    h = ss.get("act_hit")
    with st.container(border=True):
        st.markdown('<div id="ctx_panel" class="rag-card-top"></div>', unsafe_allow_html=True)
        c1, c2 = st.columns([0.78, 0.22])
        with c1:
            st.subheader("Context")
        with c2:
            st.button("×", key="ctx_close", on_click=_ctx_close, help="Close")

        if not h:
            st.caption("Use Expand on an evidence card to inspect context here.")
        else:
            tt = _pretty_title(h)
            sec = _sec_lbl(h)
            st.write(f"{tt}" + (f" | {sec}" if sec else ""))
            st.caption("Evidence window (highlighted). Full reader: Stage 3.")
            st.write(_ws((h or {}).get("ctx") or (h or {}).get("text") or ""))

    if ss.get("_scroll_ctx"):
        ss["_scroll_ctx"] = False
        # smooth scroll + flash
        components.html(
            """
<script>
const el = window.parent.document.getElementById('ctx_panel');
if (el){
  el.scrollIntoView({behavior:'smooth', block:'start'});
  const c = el.closest('[data-testid="stContainer"]');
  if (c){
    c.classList.add('ctx-flash');
    setTimeout(()=>c.classList.remove('ctx-flash'), 900);
  }
}
</script>
""",
            height=0,
        )


def render_evidence_list(rr: Dict[str, Any], q: str = ""):
    ss = st.session_state
    hs = list((rr or {}).get("hits") or [])
    if not hs:
        return

    # sort by judge01 always
    hs.sort(key=_rank_key, reverse=True)

    # display filter by min judge01 (keep some results)
    jmn = float(ss.get("jmin", 0.35))
    mk = 5
    out: List[Dict[str, Any]] = [h for h in hs if _j01(h) >= jmn]
    if len(out) < mk:
        out = hs[:mk]

    st.subheader("Evidence")
    for i, h in enumerate(out):
        render_card(h, q, i)


def render_card(h: Dict[str, Any], q: str, i: int, near_miss: bool = False):
    tt = _pretty_title(h)
    pub = _pub(h)
    sec = _sec_lbl(h)
    j = _j01(h)

    with st.container(border=True):
        st.markdown('<div class="rag-card-top"></div>', unsafe_allow_html=True)
        st.markdown(f"**{tt}**")
        meta = " | ".join([x for x in [pub, sec] if x])
        if meta:
            st.caption(meta)

        # small quality line
        st.caption(f"judge01 {j:.2f} | score {_score(h):.2f}")

        if near_miss:
            st.caption("Near-miss candidate (no direct evidence).")
        st.markdown(_snippet(h, q=q, mx=260))

        b1, b2, b3 = st.columns(3)
        with b1:
            st.button(
                "Pin",
                key=f"pin_{_cid(h)}_{_cidx(h)}_{i}",
                use_container_width=True,
                on_click=_pin_add,
                args=(h,),
            )
        with b2:
            st.button(
                "Copy",
                key=f"copy_{_cid(h)}_{_cidx(h)}_{i}",
                use_container_width=True,
                on_click=_clip_set,
                args=(h,),
            )
        with b3:
            st.button(
                "Expand",
                key=f"exp_{_cid(h)}_{_cidx(h)}_{i}",
                use_container_width=True,
                on_click=_ctx_open,
                args=(h,),
            )

        with st.expander("Why this card?"):
            ov = (h or {}).get("overlap")
            if ov is not None:
                st.write(f"overlap: {ov}")
            st.write(f"judge01: {j:.3f}")
            st.write(f"score: {_score(h):.3f}")


def render_near_miss(rr: Dict[str, Any], q: str = ""):
    if not (rr or {}).get("no_evidence"):
        return
    nm = list((rr or {}).get("near_miss") or [])
    if not nm:
        return
    nm = nm[:6]
    st.subheader("Near-miss evidence (no direct hit)")
    meta_nm = (rr or {}).get("meta", {}).get("meta_nm", {}) or {}
    st.caption(
        f"Showing {len(nm)} candidates • threshold {meta_nm.get('threshold', 0):.2f} • judge_used={meta_nm.get('used_judge', False)}"
    )
    for i, h in enumerate(nm):
        render_card(h, q, i, near_miss=True)
