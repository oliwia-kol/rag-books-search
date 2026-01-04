"""Visual theme for RAG Books (LOVEABLE UI).

Design intent:
- Calm dark canvas + intelligent color signals.
- Editorial typography.
- Soft gradients, glow on interaction, and consistent spacing.
- Hide Streamlit-isms as much as possible.

Contract: exposes ``apply_theme``.
"""

from __future__ import annotations

import streamlit as st


def _apply_mode_flag(mode: str) -> None:
    # Keep compatibility with existing code/tests.
    st.session_state["theme_mode"] = mode
    st.session_state["_theme_signal"] = st.session_state.get("_theme_signal", 0) + 1
    st.markdown(
        (
            "<script>\n"
            "try {\n"
            "  const root = window.parent.document.documentElement;\n"
            f"  root.setAttribute('data-theme', '{mode}');\n"
            f"  root.style.colorScheme = '{mode}';\n"
            "} catch(e) {}\n"
            "</script>\n"
        ),
        unsafe_allow_html=True,
    )


def apply_theme(mode: str | None = None) -> None:
    """Inject CSS.

    Note: Streamlit theme config is limited; we override via CSS.
    """
    m = mode or "dark"
    _apply_mode_flag(m)

    css = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Instrument+Serif:ital,opsz,wght@0,8..96,400;1,8..96,400&family=Fira+Code:wght@400;600&display=swap');

:root{
  --bg0:#070A12;
  --bg1:#0B1020;
  --bg2:#0F1630;
  --panel:#0E152B;
  --panel2:#111B35;
  --ink:#EEF2FF;
  --mut:#AAB3CF;
  --mut2:#7E88A8;
  --bdr:rgba(255,255,255,.08);
  --bdr2:rgba(255,255,255,.12);

  /* Accent system (calm, non-neon) */
  --a0:#8AA4FF;
  --a1:#6EE7D8;
  --a2:#F6C177;
  --a3:#FF8FB1;
  --gA:linear-gradient(135deg, rgba(138,164,255,.95), rgba(110,231,216,.85) 48%, rgba(246,193,119,.80));
  --gB:linear-gradient(135deg, rgba(255,143,177,.85), rgba(138,164,255,.70));
  --glow: 0 0 0 1px rgba(138,164,255,.25), 0 12px 40px rgba(0,0,0,.55);
  --shadow: 0 18px 60px rgba(0,0,0,.55);
  --r12:12px;
  --r16:16px;
  --r20:20px;
  --ease:cubic-bezier(.2,.9,.2,1);

  color-scheme: dark;
}

html, body{background: radial-gradient(1200px 700px at 12% -10%, rgba(138,164,255,.18), transparent 55%),
                     radial-gradient(900px 600px at 90% 0%, rgba(110,231,216,.12), transparent 60%),
                     radial-gradient(900px 700px at 70% 110%, rgba(246,193,119,.10), transparent 60%),
                     var(--bg0);
           color:var(--ink);
           font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif;
           -webkit-font-smoothing: antialiased;}

/* Streamlit container spacing */
.main .block-container{max-width: 1240px; padding-top: 1.3rem; padding-bottom: 2.2rem;}

/* Kill a bunch of default Streamlit chrome */
header, footer{visibility:hidden; height:0;}
div[data-testid="stToolbar"]{visibility:hidden; height:0;}
div[data-testid="stDecoration"]{display:none;}

/* App shell */
.app-shell{display:flex; flex-direction:column; gap: 0.85rem;}
.layout-grid{width:100%;}

/* Topbar */
.topbar{
  position: sticky; top:0; z-index: 50;
  display:flex; justify-content:space-between; align-items:center;
  padding: .55rem .35rem;
  backdrop-filter: blur(10px);
  background: linear-gradient(180deg, rgba(7,10,18,.72), rgba(7,10,18,.35));
  border-bottom: 1px solid rgba(255,255,255,.06);
}
.brand{display:flex; align-items:center; gap:.7rem;}
.brandmark{
  width:40px; height:40px; border-radius: 14px;
  background: var(--gA);
  box-shadow: 0 10px 30px rgba(0,0,0,.45);
}
.brandtitle{font-weight: 700; letter-spacing:.2px; font-size: 1.02rem;}
.brandtag{color:var(--mut2); font-size: .92rem; margin-top:-2px;}
.top-actions{display:flex; align-items:center; gap:.55rem;}
.pill{
  display:inline-flex; align-items:center; gap:.45rem;
  padding:.42rem .65rem; border-radius: 999px;
  border: 1px solid rgba(255,255,255,.10);
  background: rgba(255,255,255,.04);
  color: var(--mut);
  transition: transform .18s var(--ease), background .18s var(--ease), border-color .18s var(--ease);
}
.pill:hover{transform: translateY(-1px); background: rgba(255,255,255,.06); border-color: rgba(255,255,255,.14);}
.pill .dot{width:8px; height:8px; border-radius: 99px; background: rgba(110,231,216,.75); box-shadow: 0 0 0 3px rgba(110,231,216,.12);}

/* Sidebar rail */
.rail{
  border: 1px solid rgba(255,255,255,.08);
  border-radius: var(--r20);
  background: linear-gradient(180deg, rgba(17,27,53,.78), rgba(14,21,43,.68));
  box-shadow: var(--shadow);
  padding: 1.0rem;
}
.rail .section-title{margin-top:.35rem;}

.section-title{
  font-size:.72rem; text-transform:uppercase; letter-spacing:.12em;
  color: var(--mut2);
  margin-bottom:.45rem;
}

/* Inputs + buttons */
input, textarea{caret-color: var(--a0) !important;}
div[data-testid="stTextInput"] input{
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,.10) !important;
  background: rgba(255,255,255,.04) !important;
  color: var(--ink) !important;
  padding: .8rem .9rem !important;
  transition: border-color .18s var(--ease), box-shadow .18s var(--ease), transform .18s var(--ease);
}
div[data-testid="stTextInput"] input:focus{
  border-color: rgba(138,164,255,.45) !important;
  box-shadow: 0 0 0 4px rgba(138,164,255,.16) !important;
}

button[kind="primary"], .stButton>button{
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,.12) !important;
  background: rgba(255,255,255,.04) !important;
  color: var(--ink) !important;
  transition: transform .18s var(--ease), background .18s var(--ease), border-color .18s var(--ease), box-shadow .18s var(--ease);
}
.stButton>button:hover{
  transform: translateY(-1px);
  border-color: rgba(138,164,255,.28) !important;
  box-shadow: 0 0 0 3px rgba(138,164,255,.10);
  background: rgba(255,255,255,.06) !important;
}
.stButton>button:active{transform: translateY(0px) scale(.99);}

/* Primary CTA button style via class hook */
.cta button{background: linear-gradient(135deg, rgba(138,164,255,.22), rgba(110,231,216,.18)) !important;
            border-color: rgba(255,255,255,.16) !important;}
.cta button:hover{box-shadow: 0 0 0 3px rgba(110,231,216,.10), 0 0 0 1px rgba(110,231,216,.20) !important;}

/* Chips */
.chip{
  display:inline-flex; align-items:center; gap:.4rem;
  padding:.28rem .55rem;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,.10);
  background: rgba(255,255,255,.04);
  color: var(--mut);
  font-size: .84rem;
}
.chip.strong{border-color: rgba(110,231,216,.22); background: rgba(110,231,216,.10); color: #D8FFFA;}
.chip.mixed{border-color: rgba(246,193,119,.24); background: rgba(246,193,119,.11); color: #FFE9C8;}
.chip.weak{border-color: rgba(170,179,207,.18); background: rgba(255,255,255,.03); color: var(--mut);}

/* Cards */
.card{
  border: 1px solid rgba(255,255,255,.08);
  border-radius: var(--r16);
  background: linear-gradient(180deg, rgba(17,27,53,.62), rgba(14,21,43,.62));
  box-shadow: 0 10px 40px rgba(0,0,0,.42);
  padding: 1.0rem;
  transition: transform .18s var(--ease), border-color .18s var(--ease), box-shadow .18s var(--ease);
}
.card:hover{transform: translateY(-1px); border-color: rgba(138,164,255,.16); box-shadow: 0 16px 55px rgba(0,0,0,.48);} 

/* Answer block */
.answer{
  padding: 1.1rem 1.1rem 1.0rem;
}
.answer h2{margin:0 0 .55rem 0; font-size: 1.05rem; letter-spacing:.2px;}
.answer .text{font-size: 1.02rem; line-height: 1.7; color: var(--ink);} 
.answer .muted{color: var(--mut2);} 

/* Evidence */
.ev-head{display:flex; justify-content:space-between; gap: .75rem; align-items:flex-start; margin-bottom: .7rem;}
.ev-title{font-weight: 650; letter-spacing:.2px;}
.ev-meta{color:var(--mut2); font-size: .9rem; margin-top: .15rem;}
.ev-sn{line-height: 1.65; color: var(--ink);}
.hl{background: rgba(138,164,255,.18); border: 1px solid rgba(138,164,255,.20); padding: 0 .25rem; border-radius: 8px;}

/* Tabs */
div[data-testid="stTabs"] button{
  border-radius: 999px !important;
  padding: .35rem .75rem !important;
}

/* Chat */
.chat-panel{display:flex; flex-direction:column; gap:.55rem; margin: .15rem 0 .75rem;}
.chat-row{display:flex;}
.chat-row.user{justify-content:flex-end;}
.chat-row.assistant{justify-content:flex-start;}
.chat-bubble{
  max-width: 86%;
  padding: .75rem .9rem;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,.10);
  background: rgba(255,255,255,.04);
  line-height: 1.6;
  box-shadow: 0 10px 34px rgba(0,0,0,.35);
}
.chat-bubble.user{background: linear-gradient(135deg, rgba(138,164,255,.16), rgba(255,255,255,.03)); border-color: rgba(138,164,255,.20);}
.chat-bubble.assistant{background: linear-gradient(135deg, rgba(110,231,216,.10), rgba(255,255,255,.03)); border-color: rgba(110,231,216,.16);}
.chat-bubble .role{font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; color: var(--mut2); margin-bottom:.15rem;}
div[data-testid="stTabs"] [aria-selected="true"]{
  background: linear-gradient(135deg, rgba(138,164,255,.14), rgba(110,231,216,.10)) !important;
  border: 1px solid rgba(255,255,255,.14) !important;
}

/* Code blocks (clipboard) */
code, pre{font-family: "Fira Code", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;}

/* Reduce visual clutter */
.stCaption{color: var(--mut2) !important;}
.stDivider{opacity:.35;}

/* Mobile/iPad friendliness */
@media (max-width: 900px){
  .main .block-container{padding-top: 1.0rem;}
}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)
