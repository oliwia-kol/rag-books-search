import html
import re
import time
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components


SNIPPET_MAX_CHARS = 240


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


def _highlight_terms(txt: str, terms: List[str]) -> str:
    if not txt or not terms:
        return html.escape(txt)

    pat = re.compile("(" + "|".join([re.escape(t) for t in terms]) + ")", flags=re.I)
    out = []
    last = 0
    for m in pat.finditer(txt):
        start, end = m.span()
        if start > last:
            out.append(html.escape(txt[last:start]))
        out.append(f'<span class="hl">{html.escape(m.group(0))}</span>')
        last = end
    if last < len(txt):
        out.append(html.escape(txt[last:]))
    return "".join(out)


def _snippet(h: Dict[str, Any], q: str = "", mx: int = SNIPPET_MAX_CHARS) -> str:
    txt = (h or {}).get("text") or (h or {}).get("tx") or ""
    t = _sent(txt, n=1, mx=mx)
    terms = re.findall(r"[A-Za-z0-9]{3,}", q or "")
    return _highlight_terms(t, terms)


def _limit_answer_sentences(ans: str, max_sents: int = 5) -> Dict[str, Any]:
    sents = _split_sents(ans)
    limited = " ".join(sents[:max_sents]) if sents else ans
    truncated = len(sents) > max_sents
    return {"text": limited, "truncated": truncated, "sentences": sents}


def _stitch_hits_preview(hits: List[Dict[str, Any]], q: str = "") -> str:
    if not hits:
        return ""
    parts = []
    for h in hits[:3]:
        sn = _snippet(h, q=q)
        if sn:
            parts.append(sn)
    return " • ".join(parts)


def _coverage_counts(meta: Dict[str, Any], hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = (meta or {}).get("n") or {}
    books = int(n.get("uniq_books") or 0)
    secs = int(n.get("uniq_sections") or 0)
    pubs = int(n.get("uniq_publishers") or 0)

    if not (books and secs and pubs):
        books = books or len({h.get("book") for h in hits if h.get("book")})
        secs = secs or len({(h.get("book"), h.get("sec")) for h in hits if h.get("book") or h.get("sec")})
        pubs = pubs or len({h.get("publisher") or h.get("corp") for h in hits if h.get("publisher") or h.get("corp")})

    return {
        "books": books,
        "sections": secs,
        "publishers": pubs,
        "single_source": books <= 1 or pubs <= 1,
    }


def _confidence_state(confidence: float, coverage: str, cov_meta: Dict[str, Any]) -> Dict[str, Any]:
    cov_meta = cov_meta or {}
    mx = float(cov_meta.get("mx") or 0.0)
    std = float(cov_meta.get("std") or 0.0)
    uc = int(cov_meta.get("uc") or 0)

    if confidence >= 0.7 and mx >= 0.7 and uc >= 2:
        state = "HIGH"
    elif confidence >= 0.35 and mx >= 0.45:
        state = "MED"
    else:
        state = "LOW"

    tooltip = f"coverage={coverage or 'n/a'} • conf={confidence:.2f} • mx={mx:.2f} • std={std:.2f} • uc={uc}"
    return {"state": state, "mx": mx, "std": std, "uc": uc, "tooltip": tooltip}


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
            limited = _limit_answer_sentences(ans, max_sents=5)
            st.write(limited["text"])
            if limited["truncated"]:
                st.caption("Clamped to 5 sentences (truncated).")
        else:
            hits = (rr or {}).get("hits") or []
            if hits:
                st.caption("No LLM answer. Evidence-first preview (stitched):")
                st.markdown(_stitch_hits_preview(hits, q=""), unsafe_allow_html=True)
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
    cov_state = _confidence_state(confidence=v, coverage=cov, cov_meta=meta.get("cov"))
    counts = _coverage_counts(meta, rr.get("hits") or [])
    st.caption("Confidence")
    c1, c2 = st.columns([0.65, 0.35])
    with c1:
        badge = f'<span class="conf-badge conf-{cov_state["state"].lower()}" title="{html.escape(cov_state["tooltip"])}">{cov_state["state"]}</span>'
        try:
            st.progress(v, text=f"{cov_state['state'].title()} • coverage {cov}")
        except Exception:
            st.progress(v)
            st.caption(f"{cov_state['state'].title()} • coverage {cov}")
        st.markdown(badge, unsafe_allow_html=True)
    with c2:
        st.caption(f"Books: {counts['books']} | Sections: {counts['sections']} | Publishers: {counts['publishers']}")
        if counts["single_source"]:
            st.warning("Single-source evidence. Verify carefully.", icon="⚠️")
    meta_flags = meta.get("flags", {})
    judge_mode = (meta.get("log", {}) or {}).get("judge_mode") or (meta.get("cap", {}) or {}).get("judge_kind")
    veto_disabled = meta_flags.get("veto_disabled") or meta_flags.get("veto_disabled_when_proxy")
    veto_state = "applied" if meta_flags.get("veto_applied") else ("disabled" if veto_disabled else "ready")
    proxy_lbl = "proxy" if meta_flags.get("judge_proxy") or judge_mode == "proxy" else judge_mode
    st.caption(f"Judge mode: {proxy_lbl or 'unknown'} • veto {veto_state}")
    if meta_flags.get("veto_disabled_when_proxy"):
        st.caption("Veto is disabled when proxy/off paths are active.")


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
    ov = (h or {}).get("overlap")

    with st.container(border=True):
        st.markdown('<div class="rag-card-top"></div>', unsafe_allow_html=True)
        st.markdown(f"**{tt}**")
        meta = " | ".join([x for x in [pub, sec] if x])
        if meta:
            st.caption(meta)

        score_line = f"judge01 {j:.2f} | score {_score(h):.2f}"
        if near_miss:
            ov_txt = f"overlap {int(ov) if ov is not None else 0}"
            st.caption(f"{ov_txt} | {score_line}")
            st.caption((h or {}).get("explanation") or "Near-miss candidate (no direct evidence).")
        else:
            # small quality line
            st.caption(score_line)
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
            if near_miss:
                if (h or {}).get("near_miss_threshold") is not None:
                    st.write(f"near_miss_threshold: {float((h or {}).get('near_miss_threshold')):.2f}")
                st.write(f"used_judge: {bool((h or {}).get('used_judge'))}")


def render_near_miss(rr: Dict[str, Any], q: str = ""):
    if not (rr or {}).get("no_evidence"):
        return
    nm = list((rr or {}).get("near_miss") or [])
    if not nm:
        return
    nm = nm[:6]
    meta_nm = (rr or {}).get("meta", {}).get("meta_nm", {}) or {}
    reason = meta_nm.get("reason") or "Close but below judge/overlap threshold."
    header = f"Near misses ({len(nm)})"
    with st.expander(header, expanded=False):
        st.subheader("Near misses (no direct hit)")
        st.info(
            f"{reason} Threshold {meta_nm.get('threshold', 0):.2f} • judge_used={meta_nm.get('used_judge', False)}"
        )
        st.caption(
            f"Showing {len(nm)} candidates • overlap/threshold metadata included on each card."
        )
        for i, h in enumerate(nm):
            render_card(h, q, i, near_miss=True)


def render_power_panel(rr: Dict[str, Any]):
    meta = (rr or {}).get("meta") or {}
    log = meta.get("log", {}) or {}
    with st.expander("Power panel (debug)", expanded=False):
        st.caption("Judge + cache stats")
        c1, c2 = st.columns(2)
        with c1:
            st.write(
                {
                    "mode": log.get("judge_mode") or meta.get("cap", {}).get("judge_kind"),
                    "kind": meta.get("cap", {}).get("judge_kind"),
                    "judge_ok": meta.get("cap", {}).get("judge_ok"),
                    "judge_proxy": meta.get("flags", {}).get("judge_proxy"),
                }
            )
        with c2:
            st.write(
                {
                    "cache_hits": log.get("judge_cache_hits", 0),
                    "cache_misses": log.get("judge_cache_misses", 0),
                    "t_cache": meta.get("t", {}).get("judge_cache", 0.0),
                    "t_pred": meta.get("t", {}).get("judge_pred", 0.0),
                }
            )
        st.caption("Timings (s)")
        st.json(meta.get("t", {}))
        st.caption("Flags")
        st.json(meta.get("flags", {}))
