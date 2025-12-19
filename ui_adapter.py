import html
import re
import time
from statistics import pstdev
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


def _snippet(h: Dict[str, Any], q: str = "", mx: int = 240) -> str:
    return _sent((h or {}).get("text") or (h or {}).get("tx") or "", n=1, mx=mx)


def _highlight_snippet(snippet: str, q: str = "") -> str:
    if not snippet:
        return ""
    terms = [re.escape(w) for w in re.findall(r"[A-Za-z0-9]{3,}", q or "")]
    if not terms:
        return html.escape(snippet)
    pat = re.compile("(" + "|".join(terms) + ")", flags=re.I)
    return pat.sub(
        lambda m: f'<span class="hit-term">{html.escape(m.group(0))}</span>',
        html.escape(snippet),
    )


def _snippet_html(h: Dict[str, Any], q: str = "", mx: int = 240) -> str:
    base = _snippet(h, q="", mx=mx)
    return _highlight_snippet(base, q=q)


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


def _clamp_answer(ans: str, max_sents: int = 5) -> Dict[str, Any]:
    sents = _split_sents(ans)
    if not sents:
        return {"text": "", "truncated": False}
    limited = " ".join(sents[:max_sents])
    return {"text": limited, "truncated": len(sents) > max_sents}


def _stitch_hits_preview(hits: List[Dict[str, Any]], mx_hits: int = 3) -> str:
    if not hits:
        return ""
    parts = []
    for h in hits[:mx_hits]:
        sn = _snippet(h, mx=200)
        if sn:
            parts.append(sn)
    return " / ".join(parts)


def _judge_stats(hs: List[Dict[str, Any]], topn: int = 8) -> Dict[str, float]:
    vals = []
    for h in hs[:topn]:
        try:
            vals.append(float(h.get("judge01", 0.0)))
        except Exception:
            continue
    if not vals:
        return {"mx": 0.0, "mn": 0.0, "std": 0.0, "uc": 0, "cnt": 0}
    mx = max(vals)
    mn = min(vals)
    std = pstdev(vals) if len(vals) > 1 else 0.0
    uc = len([v for v in vals if v >= 0.60])
    return {"mx": mx, "mn": mn, "std": std, "uc": uc, "cnt": len(vals)}


def _confidence_state(cov: str, conf: Optional[float], stats: Dict[str, float]) -> str:
    cov = (cov or "WEAK").upper()
    c = conf if conf is not None else 0.0
    try:
        c = max(0.0, min(1.0, float(c)))
    except Exception:
        c = 0.0
    mx = stats.get("mx", 0.0)
    uc = stats.get("uc", 0)
    std = stats.get("std", 0.0)
    if cov in ("HIGH", "DISTRIBUTED") and c >= 0.7 and mx >= 0.8 and uc >= 2 and std <= 0.15:
        return "HIGH"
    if c >= 0.4 and mx >= 0.55:
        return "MEDIUM"
    return "LOW"


def _coverage_counts(hits: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    nmeta = (meta or {}).get("n", {}) if isinstance(meta, dict) else {}
    books = nmeta.get("uniq_books")
    secs = nmeta.get("uniq_sections")
    book_ids = [h.get("book") or h.get("book_title") or h.get("book_title_pretty") for h in hits]
    pub_ids = [h.get("publisher") or h.get("pub") or h.get("corp") for h in hits]
    if books is None or (books == 0 and hits):
        books = len({b for b in book_ids if b})
    if secs is None or (secs == 0 and hits):
        secs = len(
            {
                (book_ids[i], hits[i].get("sec") or hits[i].get("section"))
                for i in range(len(hits))
                if book_ids[i] or hits[i].get("sec") or hits[i].get("section")
            }
        )
    uniq_books = max(books or 0, 0)
    uniq_pubs = len({p for p in pub_ids if p})
    has_sources = bool(hits) or uniq_books > 0 or uniq_pubs > 0
    return {
        "books": uniq_books,
        "sections": max(secs or 0, 0),
        "publishers": uniq_pubs,
        "single_source": has_sources and (uniq_books <= 1 or uniq_pubs <= 1),
    }


def render_answer(rr: Dict[str, Any]):
    ans = (rr or {}).get("answer") or ""
    with st.container(border=True):
        st.markdown('<div class="rag-card-top"></div>', unsafe_allow_html=True)
        st.subheader("Answer")
        if ans:
            c = _clamp_answer(ans, max_sents=5)
            st.write(c["text"])
            if c["truncated"]:
                st.caption("Truncated to 5 sentences for readability.")
        else:
            hits = (rr or {}).get("hits") or []
            if hits:
                st.caption("No LLM answer. Evidence-first preview (stitched):")
                st.write(_stitch_hits_preview(hits))
            else:
                st.caption("No answer.")


def render_conf(rr: Dict[str, Any]):
    meta = (rr or {}).get("meta") or {}
    cf = (rr or {}).get("confidence", meta.get("conf"))
    coverage = (rr or {}).get("coverage") or meta.get("coverage") or "WEAK"
    hits = list((rr or {}).get("hits") or [])
    stats = _judge_stats(hits)
    state = _confidence_state(coverage, cf, stats)

    tooltip = (
        f"coverage={coverage} • conf={cf if cf is not None else 0.0:.2f} • "
        f"judge mx={stats.get('mx', 0.0):.2f} • std={stats.get('std', 0.0):.2f} • uc={stats.get('uc', 0)}"
    )
    badge_colors = {
        "LOW": ("#fee2e2", "#b91c1c"),
        "MEDIUM": ("#fef9c3", "#92400e"),
        "HIGH": ("#dcfce7", "#065f46"),
    }
    bg, fg = badge_colors.get(state, ("#e5e7eb", "#111827"))
    badge_html = f'<span title="{html.escape(tooltip)}" style="padding:4px 10px;border-radius:14px;font-weight:700;background:{bg};color:{fg};border:1px solid #d1d5db;display:inline-block;">{state}</span>'

    st.caption("Confidence")
    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        st.markdown(badge_html, unsafe_allow_html=True)
        st.caption(f"Coverage: {coverage}")
    with c2:
        counts = _coverage_counts(hits, meta)
        st.caption(f"Books: {counts['books']} | Sections: {counts['sections']}")
        if counts["single_source"]:
            st.warning("Single-source evidence. Verify carefully.", icon="⚠️")
    st.caption("State derived from coverage, judge spread (mx/std/uc), and confidence score.")


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
        st.markdown(_snippet_html(h, q=q, mx=240), unsafe_allow_html=True)

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
