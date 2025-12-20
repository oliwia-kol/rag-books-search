import streamlit as st


def apply_theme():
    css = r"""
<style>
:root{
  --bg0:#0b0f16;
  --bd0:rgba(255,255,255,.10);
  --bd1:rgba(255,255,255,.16);
  --tx0:rgba(255,255,255,.92);
  --tx1:rgba(255,255,255,.72);
  --tx2:rgba(255,255,255,.58);
  --ac0:rgba(255,215,102,.85);
  --card:rgba(255,255,255,.03);
  --card-hover:rgba(255,255,255,.06);
}

html, body {
  background: var(--bg0);
  color: var(--tx0);
  -webkit-font-smoothing: antialiased;
}

header[data-testid="stHeader"]{
  position: sticky;
  top: 0;
  z-index: 12;
  background: rgba(11,15,22,.96);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--bd0);
}

/* avoid header overlap (mobile/iPad) */
.main .block-container{
  padding-top: 5.6rem;
  padding-bottom: 2.3rem;
  max-width: 1180px;
  margin: 0 auto;
  gap: .75rem;
}

.main .block-container h1, .main .block-container h2, .main .block-container h3{
  letter-spacing: 0.01em;
  line-height: 1.25;
  margin-bottom: .35rem;
}
.main .block-container p{
  line-height: 1.6;
  margin-bottom: .35rem;
}
.stMarkdown p{ margin-bottom: .28rem; }
.main .block-container .stMarkdown em{ color: var(--tx1); }

/* compact sidebar controls */
section[data-testid="stSidebar"] [data-testid="stButton"] button{
  padding: .25rem .5rem;
  min-height: 2.1rem;
  line-height: 1.1;
  white-space: nowrap;
}
section[data-testid="stSidebar"] .stCheckbox, section[data-testid="stSidebar"] .stSelectbox{
  margin-bottom: .35rem;
}

.stProgress > div[data-testid="stProgressBar"]{
  border-radius: 999px;
  height: 10px;
}

/* mode / selector hierarchy */
section[data-testid="stSidebar"] .mode-kicker{
  display: flex;
  gap: .35rem;
  flex-wrap: wrap;
  align-items: center;
  margin: .15rem 0 .35rem;
}
section[data-testid="stSidebar"] .mode-kicker .ui-badge{ background: rgba(255,255,255,.05); }
section[data-testid="stSidebar"] [data-testid="stRadio"] > div{
  border: 1px solid var(--bd0);
  border-radius: 14px;
  padding: .35rem .55rem;
  background: rgba(255,255,255,.03);
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label{
  border-radius: 10px;
  padding: .2rem .35rem;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{
  background: rgba(255,255,255,.05);
}

.badge-row{
  display: flex;
  gap: .35rem;
  flex-wrap: wrap;
  align-items: center;
  margin: .25rem 0;
}
.ui-badge{
  border: 1px solid var(--bd0);
  background: rgba(255,255,255,.06);
  border-radius: 999px;
  padding: .22rem .6rem;
  font-size: .82rem;
  line-height: 1.2;
  letter-spacing: 0.01em;
}
.ui-badge-strong{
  background: rgba(255,215,102,.16);
  border-color: rgba(255,215,102,.45);
  color: var(--tx0);
}
.ui-badge-warn{
  background: rgba(255,126,126,.18);
  border-color: rgba(255,126,126,.5);
  color: #ffdede;
}
.ui-badge-soft{
  background: rgba(255,255,255,.08);
  color: var(--tx1);
}
.ui-badge-muted{
  background: rgba(255,255,255,.04);
  color: var(--tx2);
}

/* card marker styling via :has() (Chrome/Safari 16+) */
div[data-testid="stContainer"]:has(.rag-card-mk){
  border: 1px solid var(--bd0) !important;
  background: var(--card);
  border-radius: 16px;
}
div[data-testid="stContainer"]:has(.rag-card-top){
  border: 1px solid var(--bd0) !important;
  background: linear-gradient(130deg, rgba(255,255,255,.03), rgba(255,255,255,.015)) !important;
  border-radius: 16px;
  transition: border-color .2s ease, box-shadow .2s ease, transform .18s ease;
}
div[data-testid="stContainer"]:has(.rag-card-top):hover{
  border-color: var(--bd1) !important;
  box-shadow: 0 18px 38px rgba(0,0,0,.35);
  transform: translateY(-1px);
}

.rag-card-top{
  border-top-left-radius: 16px;
  border-top-right-radius: 16px;
}

.rag-title{
  font-size: 1.05rem;
  font-weight: 650;
  line-height: 1.3;
  margin-bottom: .15rem;
}
.rag-meta{
  color: var(--tx2);
  font-size: .9rem;
  line-height: 1.35;
}
.rag-sub{
  color: var(--tx1);
  font-size: .92rem;
  line-height: 1.3;
}

/* action bar buttons */
.rag-act [data-testid="stButton"] button{
  width: 100%;
  white-space: nowrap;
}
div[data-testid="stContainer"]:has(.rag-card-top) [data-testid="stButton"] button{
  width: 100%;
  white-space: nowrap;
  border: 1px solid var(--bd0);
  background: rgba(255,255,255,.04);
  color: var(--tx0);
  border-radius: 12px;
  transition: border-color .2s ease, box-shadow .2s ease, transform .12s ease;
}
div[data-testid="stContainer"]:has(.rag-card-top) [data-testid="stButton"] button:hover{
  border-color: var(--ac0);
  box-shadow: 0 0 0 1px rgba(255,215,102,.28);
  transform: translateY(-1px);
}
div[data-testid="stContainer"]:has(.rag-card-top) [data-testid="stButton"] button:focus-visible{
  outline: 2px solid var(--ac0);
  outline-offset: 2px;
}

/* context flash */
@keyframes ctxflash{ 0%{ box-shadow: 0 0 0 0 rgba(255,215,102,.35);} 100%{ box-shadow: 0 0 0 12px rgba(255,215,102,0);} }
.ctx-flash{ animation: ctxflash 0.8s ease-out 1; border-color: var(--ac0) !important; }

/* expander styling */
div[data-testid="stExpander"]{
  border-radius: 14px !important;
  border: 1px solid var(--bd0);
  overflow: hidden;
}
div[data-testid="stExpander"] > details > summary{
  background: rgba(255,255,255,.03);
  padding: .5rem .75rem;
}

/* responsive padding tweaks */
@media (min-width: 1200px){
  .main .block-container{
    max-width: 1240px;
    padding-left: 2.25rem;
    padding-right: 2.25rem;
  }
}
@media (max-width: 1100px){
  .main .block-container{
    padding-left: 1.1rem;
    padding-right: 1.1rem;
  }
}
@media (max-width: 900px){
  .main .block-container{ padding-top: 4.6rem; }
}

/* Safari / iPadOS polish */
@supports (-webkit-touch-callout: none){
  header[data-testid="stHeader"]{ position: sticky; top: 0; }
  .main .block-container{
    padding-top: 5rem;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
  }
  section[data-testid="stSidebar"]{
    padding-top: .65rem;
  }
  section[data-testid="stSidebar"] [data-testid="stButton"] button{
    line-height: 1.15;
  }
}

</style>
"""
    st.markdown(css, unsafe_allow_html=True)
