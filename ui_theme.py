# ui_theme.py
"""Theme injector for the RAG Books Search Streamlit UI.

Single dark theme, product-grade feel:
- calm canvas
- typography for reading
- restrained gradients for focus/meaning

Public API (contract): apply_theme(mode)
"""

from __future__ import annotations

import streamlit as st


def _apply_mode_flag(mode: str) -> None:
    st.session_state["theme_mode"] = mode or "dark"
    st.markdown(
        """
<script>
try {
  const root = window.parent.document.documentElement;
  root.setAttribute('data-theme', 'dark');
  root.style.colorScheme = 'dark';
} catch (e) {}
</script>
""",
        unsafe_allow_html=True,
    )


def apply_theme(mode: str | None = None) -> None:
    """Inject CSS. `mode` is kept for compatibility; only dark is supported."""
    _apply_mode_flag(mode or "dark")
    st.markdown(
        r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://cdn.jsdelivr.net/npm/phosphor-icons@1.4.2/src/css/phosphor.css');

:root {
  --bg: #070A12;
  --bg2: #0B1020;
  --surface: rgba(255,255,255,0.04);
  --surface2: rgba(255,255,255,0.06);
  --border: rgba(255,255,255,0.08);
  --border2: rgba(255,255,255,0.12);
  --text: #F4F6FB;
  --muted: rgba(244,246,251,0.72);
  --muted2: rgba(244,246,251,0.52);

  --a1: #8AA8FF; /* calm periwinkle */
  --a2: #7BE0D1; /* mint */
  --a3: #F3C178; /* sand */

  --ok: #7BE0A1;
  --warn: #F3C178;
  --bad: #F08A8A;

  --r12: 12px;
  --r16: 16px;
  --r20: 20px;

  --sh1: 0 10px 30px rgba(0,0,0,0.35);
  --sh2: 0 18px 60px rgba(0,0,0,0.45);

  --g-focus: linear-gradient(135deg, rgba(138,168,255,0.55), rgba(123,224,209,0.35), rgba(243,193,120,0.25));
  --g-brand: radial-gradient(1200px 500px at 20% -10%, rgba(138,168,255,0.22), transparent 55%),
             radial-gradient(900px 500px at 85% -20%, rgba(123,224,209,0.14), transparent 55%),
             radial-gradient(900px 500px at 60% 120%, rgba(243,193,120,0.10), transparent 55%);

  --font: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}

html, body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  -webkit-font-smoothing: antialiased;
}

/* Streamlit base */
.main .block-container{
  max-width: 1160px;
  padding-top: 1.4rem;
  padding-bottom: 2.2rem;
}

/* Sidebar separator */
section[data-testid="stSidebar"]{
  border-right: 1px solid var(--border);
}

/* Topbar */
.topbar{
  position: sticky;
  top: 0;
  z-index: 20;
  padding: 0.7rem 0.1rem 0.6rem;
  margin: -0.6rem 0 0.8rem;
  background: var(--g-brand), var(--bg);
  border-bottom: 1px solid var(--border);
}
.brand{display:flex;align-items:center;gap:12px;}
.brand .logo{
  width: 40px; height: 40px; border-radius: 12px;
  display:flex;align-items:center;justify-content:center;
  background: var(--g-focus);
  color: rgba(7,10,18,0.85);
  box-shadow: var(--sh1);
}
.brand .title{font-weight:700;letter-spacing:0.2px;}
.brand .muted{color: var(--muted2); font-size: 0.9rem; margin-top: 1px;}
.topbar-right{display:flex;align-items:center;justify-content:flex-end;}
.top-hint{color: var(--muted2); font-size: 0.88rem;}
.topbar-tools{display:flex;justify-content:flex-end; margin-top: 0.25rem;}

/* Stage */
.stage{ padding: 0.6rem 0.1rem 0.2rem; }

/* Chat */
.chat{display:flex;flex-direction:column;gap:10px;margin: 0.2rem 0 0.8rem;}
.chat-bubble{
  max-width: 92%;
  padding: 10px 12px;
  border-radius: var(--r16);
  line-height: 1.45;
  font-size: 1.02rem;
  border: 1px solid var(--border);
  background: var(--surface);
}
.chat-bubble.user{
  align-self: flex-end;
  background: rgba(138,168,255,0.10);
  border-color: rgba(138,168,255,0.18);
}
.chat-bubble.asst{
  align-self: flex-start;
  background: rgba(255,255,255,0.04);
}

/* Answer panel */
.panel{
  border: 1px solid var(--border);
  background: var(--surface);
  border-radius: var(--r20);
  padding: 14px 14px;
  box-shadow: var(--sh1);
}
.panel .h{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;}
.kicker{color: var(--muted2); font-size: 0.82rem; letter-spacing: 0.10em; text-transform: uppercase;}
.answer{font-size: 1.05rem;line-height: 1.6;color: var(--text);}

/* Legacy helpers used across ui_* modules */
.section-title{color: var(--muted2); font-size: 0.82rem; letter-spacing: 0.10em; text-transform: uppercase; margin: 0.2rem 0 0.45rem;}
.chips{display:flex;flex-wrap:wrap;gap:6px;align-items:center;}
.chip{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;border:1px solid var(--border);background:rgba(255,255,255,0.04);color:var(--muted);font-size:0.82rem;}
.chip.muted{color: var(--muted2);}
.chip.primary{border-color: rgba(138,168,255,0.28); background: rgba(138,168,255,0.10); color: rgba(240,244,255,0.92);}
.chip.secondary{border-color: rgba(243,193,120,0.28); background: rgba(243,193,120,0.10); color: rgba(255,245,230,0.92);}
.chip.success{border-color: rgba(123,224,161,0.28); background: rgba(123,224,161,0.10); color: rgba(230,255,243,0.92);}

.card-shell{border:1px solid var(--border);background:rgba(255,255,255,0.03);border-radius: var(--r20);padding:12px 12px;box-shadow: var(--sh1);}
.card-shell.selected{border-color: rgba(138,168,255,0.28); box-shadow: 0 0 0 1px rgba(138,168,255,0.12) inset, var(--sh1);}
.card-head{display:flex;justify-content:space-between;gap:10px;align-items:flex-start;}
.pub-pill{display:inline-flex;align-items:center;padding:4px 10px;border-radius:999px;border:1px solid var(--border);background:rgba(255,255,255,0.04);color: var(--muted);font-size:0.82rem;}
.evidence-title{font-weight:650;margin-top:8px;}
.evidence-meta{color: var(--muted2); font-size:0.9rem; margin-top:2px;}
.evidence-snippet{margin-top:8px; line-height:1.55;}
.evidence-foot{color: var(--muted2); font-size:0.88rem; margin-top:8px;}

/* Hero */
.hero{
  border: 1px solid var(--border);
  background: var(--g-brand), rgba(255,255,255,0.03);
  border-radius: 24px;
  padding: 18px 18px;
  box-shadow: var(--sh1);
}
.hero h2{margin: 0.4rem 0 0.25rem;}
.hero .lede{color: var(--muted); line-height: 1.55;}

/* Tabs */
button[data-baseweb="tab"]{ color: var(--muted) !important; }
button[data-baseweb="tab"][aria-selected="true"]{ color: var(--text) !important; }

/* Inputs */
input, textarea {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--border) !important;
}
div[data-baseweb="input"]:has(input:focus-within),
div[data-baseweb="textarea"]:has(textarea:focus-within){
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(138,168,255,0.18), 0 0 0 1px rgba(138,168,255,0.22) inset !important;
  border-radius: 10px;
}

/* Buttons */
button[kind="secondary"], button[kind="primary"], .stButton > button {
  border-radius: 12px !important;
}

/* Reduce visual noise in captions */
.stCaption{color: var(--muted2) !important;}

/* Status + empty states */
.status-strip{border:1px solid var(--border);background:rgba(255,255,255,0.03);border-radius: var(--r20);padding:12px 12px;box-shadow: var(--sh1);margin: 0.7rem 0;}
.empty-state{border:1px dashed var(--border);background:rgba(255,255,255,0.02);border-radius: var(--r20);padding:12px 12px;color: var(--muted);}
.ctx-shell{border:1px solid var(--border);background:rgba(255,255,255,0.03);border-radius: var(--r20);padding:12px 12px;}
.details-panel{border:1px solid var(--border);background:rgba(255,255,255,0.02);border-radius: var(--r20);padding:12px 12px;}
.hl{background: rgba(138,168,255,0.20); border:1px solid rgba(138,168,255,0.16); padding:0 3px; border-radius: 6px;}

</style>
""",
        unsafe_allow_html=True,
    )
