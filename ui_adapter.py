import html
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components

from ui_shell import DEFAULT_JMIN


SNIPPET_MAX_CHARS = 240
EVIDENCE_BATCH_SIZE = 8
JMIN_DEFAULT = DEFAULT_JMIN


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


def _render_no_results_suggestions(title: str, reason: str, tips: Optional[List[str]] = None):
    tips = tips or [
        "Try broader keywords or fewer filters.",
        "Switch modes to adjust depth vs. speed.",
        "Verify the selected publishers have data available.",
    ]
    st.markdown(
        """
<div class="empty-state">
  <div style="font-weight:600; margin-bottom:4px;">{title}</div>
  <div style="color: var(--muted); margin-bottom:6px;">{reason}</div>
  <ul style="margin:0 0 0.2rem 1.2rem; padding:0; color: var(--muted); line-height:1.5;">
    {tips}
  </ul>
</div>
""".format(
            title=html.escape(title),
            reason=html.escape(reason),
            tips="".join([f"<li>{html.escape(t)}</li>" for t in tips]),
        ),
        unsafe_allow_html=True,
    )


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


def _ctx_nav(hits: List[Dict[str, Any]], current_idx: Optional[int], delta: int):
    if current_idx is None:
        return
    nxt = current_idx + delta
    if nxt < 0 or nxt >= len(hits):
        return
    _ctx_open(hits[nxt])


def _rank_key(h: Dict[str, Any]) -> float:
    # judge is forced ON in this product
    return _j01(h)


def _judge_tier(j: float) -> Tuple[str, str]:
    if j >= 0.85:
        return ("Strong", "success")
    if j >= 0.65:
        return ("Solid", "primary")
    if j >= 0.4:
        return ("Weak", "warning")
    return ("Poor", "neutral")


def _badge(cls: str, text: str) -> str:
    return f"<span class='chip {cls}'>{html.escape(text)}</span>"


def _load_more_evidence(total: int):
    ss = st.session_state
    cur = int(ss.get("ev_offset", 0) or 0)
    nxt = min(total, cur + EVIDENCE_BATCH_SIZE)
    ss["ev_offset"] = nxt


def render_status_strip(rr: Dict[str, Any]):
    meta = (rr or {}).get("meta", {})
    hits = (rr or {}).get("hits", [])
    pubs_used = meta.get("n", {}).get("uniq_publishers") or len({h.get("publisher") or h.get("corp") for h in hits})
    count = len(hits)
    coverage = (rr or {}).get("coverage") or "WEAK"
    confidence = float((rr or {}).get("confidence") or 0.0)
    cov_state = _confidence_state(confidence, coverage, meta.get("cov"))
    badge_cls = {
        "HIGH": "success",
        "DISTRIBUTED": "primary",
        "OK": "primary",
        "WEAK": "warning",
    }.get(str(coverage).upper(), "neutral")

    evidence_state = "No indexes available" if meta.get("cap", {}).get("corp_available") is False else (
        "Direct evidence found" if not (rr or {}).get("no_evidence") else "Weak evidence, showing closest matches"
    )
    meta_line = f"Hits: {count} | Pubs: {pubs_used or 0} | {int(meta.get('t', {}).get('total', 0)*1000)} ms"
    st.markdown(
        """
<div class="status-strip">
  <div><span class="status-chip {cls}" title="{tooltip}">{cov}</span></div>
  <div style="color: var(--text); font-size: 0.96rem;">{state}</div>
  <div class="status-meta">{meta_line}</div>
</div>
""".format(
            cls=badge_cls,
            tooltip=html.escape(cov_state["tooltip"]),
            cov=coverage,
            state=html.escape(evidence_state),
            meta_line=html.escape(meta_line),
        ),
        unsafe_allow_html=True,
    )


def render_answer(rr: Dict[str, Any]):
    ans = (rr or {}).get("answer") or ""
    if (rr or {}).get("no_evidence"):
        st.markdown(
            """
<div class="panel">
  <div class="section-title">Answer</div>
  <p style="margin:0; color: var(--muted);">Abstain — no direct evidence to answer confidently.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        return
    if not ans:
        return

    limited = _limit_answer_sentences(ans, max_sents=5)
    hits = (rr or {}).get("hits") or []
    citations = ""
    if hits:
        refs = "".join([_badge("primary", f"[{i+1}]") for i, _ in enumerate(hits[:4])])
        citations = f"<div class='chips'>{refs}<span class='chip muted'>Citations</span></div>"
    st.markdown(
        """
<div class="panel">
  <div class="section-title">Answer</div>
  <div style="line-height:1.55; font-size:1rem; color: var(--text);">{text}</div>
  {foot}{citations}
</div>
""".format(
            text=html.escape(limited["text"]),
            foot="<div class='chips'><span class='chip muted'>Clamped to 5 sentences</span></div>" if limited["truncated"] else "",
            citations=citations,
        ),
        unsafe_allow_html=True,
    )


def render_context_panel():
    ss = st.session_state
    h = ss.get("act_hit")
    rr = ss.get("res") or {}
    hits = list((rr or {}).get("hits") or [])
    current_idx: Optional[int] = None
    if h and hits:
        for idx, x in enumerate(hits):
            if _cid(h) == _cid(x) and _cidx(h) == _cidx(x):
                current_idx = idx
                break
        if current_idx is None:
            h = None
            ss["act_hit"] = None
            ss["_scroll_ctx"] = False
            ss["_ctx_ts"] = None
    elif h and not hits:
        h = None
        ss["act_hit"] = None
        ss["_scroll_ctx"] = False
        ss["_ctx_ts"] = None
    with st.container():
        st.markdown('<div id="ctx_panel"></div>', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Details</div>", unsafe_allow_html=True)
        if not h:
            st.markdown("<div class='empty-state'>Select an evidence card to view full context.</div>", unsafe_allow_html=True)
        else:
            tt = _pretty_title(h)
            sec = _sec_lbl(h)
            st.markdown(f"<div class='chips'>{_badge('primary', tt)}{_badge('muted', sec) if sec else ''}</div>", unsafe_allow_html=True)
            nav_close, nav_prev, nav_next = st.columns([0.34, 0.33, 0.33])
            with nav_close:
                st.button("Close", key="ctx_close", on_click=_ctx_close, help="Close details", use_container_width=True)
            with nav_prev:
                st.button(
                    "Previous",
                    key="ctx_prev",
                    on_click=_ctx_nav,
                    args=(hits, current_idx, -1),
                    disabled=current_idx in (None, 0),
                    help="View the previous evidence card",
                    use_container_width=True,
                )
            with nav_next:
                st.button(
                    "Next",
                    key="ctx_next",
                    on_click=_ctx_nav,
                    args=(hits, current_idx, 1),
                    disabled=current_idx is None or current_idx >= len(hits) - 1,
                    help="View the next evidence card",
                    use_container_width=True,
                )
            st.caption("Excerpt")
            st.markdown(
                """
<div class="details-panel">
  <pre style="white-space:pre-wrap; line-height:1.6;">{tx}</pre>
</div>
""".format(tx=html.escape(_ws((h or {}).get("ctx") or (h or {}).get("text") or ""))),
                unsafe_allow_html=True,
            )
            st.caption("Metadata")
            st.markdown(
                """
<div class="chips">
  {chips}
</div>
""".format(
                    chips="".join(
                        [
                            _badge("muted", f"cid { _cid(h)}"),
                            _badge("muted", f"cidx { _cidx(h)}"),
                            _badge("muted", f"corp { _pub(h) or 'n/a'}"),
                        ]
                    )
                ),
                unsafe_allow_html=True,
            )

    if ss.get("_scroll_ctx"):
        ss["_scroll_ctx"] = False
        components.html(
            """
<script>
const el = window.parent.document.getElementById('ctx_panel');
if (el){
  el.scrollIntoView({behavior:'smooth', block:'start'});
}
</script>
""",
            height=0,
        )


def render_evidence_list(rr: Dict[str, Any], q: str = ""):
    ss = st.session_state
    ss.setdefault("ev_offset", 0)
    hs = list((rr or {}).get("hits") or [])
    if not hs:
        _render_no_results_suggestions(
            "No evidence yet",
            "Run a search to see citations, or adjust the query and publisher filters.",
            tips=[
                "Try different keywords or phrasing.",
                "Select another publisher or clear corpus filters.",
                "Switch modes if you need more depth or faster results.",
            ],
        )
        return

    hs.sort(key=_rank_key, reverse=True)
    jmn = float(ss.get("jmin", DEFAULT_JMIN))
    mk = 8
    out: List[Dict[str, Any]] = [h for h in hs if _j01(h) >= jmn]
    if len(out) < mk:
        out = hs

    total = len(out)
    off = int(ss.get("ev_offset", 0) or 0)
    if off < 0 or off >= total:
        off = 0
    ss["ev_offset"] = off

    end = min(total, max(EVIDENCE_BATCH_SIZE, off + EVIDENCE_BATCH_SIZE))
    batch = out[:end]

    st.markdown("<div class='section-title'>Evidence</div>", unsafe_allow_html=True)
    for i, h in enumerate(batch):
        render_card(h, q, i)

    remaining = total - end
    if remaining > 0:
        st.button(
            "Load more",
            key="evidence_load_more",
            on_click=_load_more_evidence,
            args=(total,),
            help=f"Show {min(remaining, EVIDENCE_BATCH_SIZE)} more evidence cards",
            use_container_width=True,
        )
        st.caption(f"{remaining} more evidence cards available")


def _is_selected(h: Dict[str, Any]) -> bool:
    ss = st.session_state
    active = ss.get("act_hit") or {}
    return _cid(active) == _cid(h) and _cidx(active) == _cidx(h)


def render_card(h: Dict[str, Any], q: str, i: int, near_miss: bool = False):
    tt = _pretty_title(h)
    pub = _pub(h)
    sec = _sec_lbl(h)
    j = _j01(h)
    tier_lbl, tier_cls = _judge_tier(j)
    ov = (h or {}).get("overlap")

    classes = ["evidence-card"]
    if near_miss:
        classes.append("near")
    if _is_selected(h):
        classes.append("selected")

    score_chips = [
        _badge("muted", f"#{i+1}"),
        _badge(tier_cls, tier_lbl),
        _badge(tier_cls, f"J: {j:.2f}"),
        _badge("muted", f"S: {_score(h):.2f}"),
    ]
    if near_miss:
        score_chips.append(_badge("secondary", "Near-miss"))

    st.markdown(
        """
<div class="{cls}">
  <div style="display:flex; justify-content:space-between; gap:8px; align-items:flex-start;">
    <div>
      <div class="evidence-title">{title}</div>
      <div class="evidence-meta">{meta}</div>
    </div>
    <div class="score-row">{chips}</div>
  </div>
  <div class="evidence-snippet">{snippet}</div>
  <div style="margin-top:8px; color: var(--muted-2); font-size:0.9rem;">{detail}</div>
</div>
""".format(
            cls=" ".join(classes),
            title=html.escape(tt),
            meta=html.escape(" • ".join([x for x in [pub, sec] if x])),
            chips="".join(score_chips),
            snippet=_snippet(h, q=q, mx=260),
            detail=html.escape(
                (f"overlap {int(ov)} | " if near_miss and ov is not None else "")
                + f"judge01 {j:.2f} • score {_score(h):.2f}"
            ),
        ),
        unsafe_allow_html=True,
    )

    b1, b2, b3 = st.columns(3)
    with b1:
        st.button(
            "Pin",
            key=f"pin_{_cid(h)}_{_cidx(h)}_{i}",
            use_container_width=True,
            on_click=_pin_add,
            args=(h,),
            help="Pin this evidence to the sidebar",
        )
    with b2:
        st.button(
            "Copy snippet",
            key=f"copy_{_cid(h)}_{_cidx(h)}_{i}",
            use_container_width=True,
            on_click=_clip_set,
            args=(h,),
            help="Copy citation to clipboard",
        )
    with b3:
        st.button(
            "Open details",
            key=f"exp_{_cid(h)}_{_cidx(h)}_{i}",
            use_container_width=True,
            on_click=_ctx_open,
            args=(h,),
            help="Open the context panel",
        )


def render_near_miss(rr: Dict[str, Any], q: str = ""):
    if not (rr or {}).get("no_evidence"):
        return
    if not st.session_state.get("nm", True):
        return
    nm = list((rr or {}).get("near_miss") or [])
    if not nm:
        _render_no_results_suggestions(
            "No near-miss results",
            "We couldn’t find closely related passages.",
            tips=[
                "Loosen the query phrasing or add synonyms.",
                "Try enabling additional publishers.",
                "Lower the judge threshold in Settings to broaden matches.",
            ],
        )
        return
    meta_nm = (rr or {}).get("meta", {}).get("meta_nm", {}) or {}
    reason = meta_nm.get("reason") or "Close but below judge/overlap threshold."
    with st.expander("Near misses (weak overlap)", expanded=True):
        st.info(reason)
        st.caption(f"{len(nm)} near-miss results (overlap/judge threshold {meta_nm.get('threshold', 0):.2f})")
        st.markdown(
            """
<div class="panel" style="margin-top:8px;">
  <div style="color: var(--muted); margin-bottom:6px;">These overlap with the query but didn’t meet the evidence threshold.</div>
  <div class="chips">{chips}</div>
</div>
""".format(
                chips="".join(
                    [
                        _badge("secondary", f"{len(nm)} shown"),
                        _badge("muted", f"Threshold {meta_nm.get('threshold', 0):.2f}"),
                        _badge("muted", f"Reason: {reason}"),
                    ]
                )
            ),
            unsafe_allow_html=True,
        )
        for i, h in enumerate(nm[:6]):
            render_card(h, q, i, near_miss=True)


def render_power_panel(rr: Dict[str, Any]):
    meta = (rr or {}).get("meta") or {}
    log = meta.get("log", {}) or {}
    clamp = meta.get("clamp", {}) or {}
    mode_cfg = meta.get("mode_cfg", {}) or {}
    hits = list((rr or {}).get("hits") or [])
    with st.expander("Debug", expanded=False):
        st.markdown("<div class='debug-block'>", unsafe_allow_html=True)
        st.caption(f"Mode: {mode_cfg.get('label', meta.get('mode', 'quick')).title()} — {mode_cfg.get('description', '')}")
        st.write(
            {
                "mode": meta.get("mode"),
                "mode_cfg": {k: v for k, v in mode_cfg.items() if k not in {"description", "label"}},
                "cut_rule": meta.get("cut_rule"),
                "near_miss_threshold": meta.get("meta_nm", {}).get("threshold"),
            }
        )
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
        st.caption("Retrieval + clamp")
        st.write(
            {
                "retrieval": clamp.get("retrieval"),
                "context": clamp.get("context"),
                "prompt": clamp.get("prompt"),
            }
        )
        st.caption("Timings (s)")
        st.json(meta.get("t", {}))
        st.caption("Flags")
        st.json(meta.get("flags", {}))
        st.caption("Capabilities")
        st.json(meta.get("cap", {}))
        st.caption("Hits (sem/lex/judge01/cut_rule)")
        rows = []
        for h in hits:
            rows.append(
                {
                    "id": _cid(h),
                    "book": _pretty_title(h),
                    "sec": _sec_lbl(h),
                    "sem": float(h.get("sem_score_n", h.get("sem_score", 0.0) or 0.0)),
                    "lex": float(h.get("lex_score_n", h.get("lex_score", 0.0) or 0.0)),
                    "judge01": _j01(h),
                    "cut_rule": meta.get("cut_rule"),
                }
            )
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_conf(rr: Dict[str, Any]):
    # Backwards compatibility: render the status strip in place of old confidence block.
    render_status_strip(rr)
