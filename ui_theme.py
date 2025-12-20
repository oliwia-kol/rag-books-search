import streamlit as st


def apply_theme():
    css = r"""
<style>
:root{
  --bg0:#0c111d;
  --bg1:#10182b;
  --bd0:rgba(255,255,255,.12);
  --bd1:rgba(255,255,255,.18);
  --tx0:rgba(255,255,255,.96);
  --tx1:rgba(255,255,255,.78);
  --tx2:rgba(255,255,255,.60);
  --ac0:#f5c452;
  --ac1:#58b2ff;
  --ac2:#5fe3b9;
  --warn:#ffb26b;
  --err:#ff7b7b;
}

html, body { background: var(--bg0); color: var(--tx0); }
body{
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  -webkit-text-size-adjust: 100%;
}

/* avoid header overlap (mobile/iPad) */
.main .block-container{
  padding-top: 5.5rem;
  padding-bottom: 2.6rem;
  max-width: 1200px;
  row-gap: 1rem;
}

.main .block-container h1, .main .block-container h2, .main .block-container h3{
  line-height: 1.25;
  letter-spacing: 0.01em;
}
.main .block-container p{
  line-height: 1.55;
  color: var(--tx1);
}
.main .block-container .stCaption, .main .block-container .stMarkdown p, .main .block-container .stMarkdown div{
  line-height: 1.45;
}

header[data-testid="stHeader"]{
  background: linear-gradient(180deg, rgba(12,17,29,0.95) 0%, rgba(12,17,29,0.55) 100%);
  border-bottom: 1px solid var(--bd0);
}

/* compact sidebar controls */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(16,24,43,0.94) 0%, rgba(12,17,29,0.94) 100%);
  padding-top: 1.2rem !important;
}
section[data-testid="stSidebar"] [data-testid="stButton"] button{
  padding: .35rem .7rem;
  min-height: 2.35rem;
  line-height: 1.15;
  white-space: nowrap;
  border-radius: 10px;
}
section[data-testid="stSidebar"] .stCheckbox, section[data-testid="stSidebar"] .stSelectbox{
  margin-bottom: .4rem;
}

.stProgress > div[data-testid="stProgressBar"]{
  border-radius: 999px;
  height: 12px;
  background: rgba(255,255,255,.12);
}
.stProgress > div[data-testid="stProgressBar"] > div{
  background: linear-gradient(90deg, var(--ac2), var(--ac1));
}

/* card marker styling via :has() (Chrome/Safari 16+) */
div[data-testid="stContainer"]:has(.rag-card-mk), .rag-card{
  border: 1px solid var(--bd0) !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.02) 100%);
  border-radius: 16px;
  transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
  box-shadow: 0 20px 45px rgba(0,0,0,0.35);
}
div[data-testid="stContainer"]:has(.rag-card-mk):hover{
  border-color: var(--ac1) !important;
  transform: translateY(-2px);
  box-shadow: 0 22px 55px rgba(16, 24, 43, 0.5);
}

.rag-card-top{
  border-top-left-radius: 16px;
  border-top-right-radius: 16px;
  height: 8px;
  background: linear-gradient(90deg, var(--ac0), var(--ac1));
}

.rag-title{ font-size: 1.08rem; font-weight: 680; line-height: 1.28; letter-spacing: 0.01em; }
.rag-meta{ color: var(--tx2); font-size: .92rem; line-height: 1.35; }
.rag-sub{ color: var(--tx1); font-size: .96rem; line-height: 1.45; }
.rag-qual{ color: var(--tx2); font-size: .88rem; }

.rag-badge{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  border: 1px solid var(--bd1);
  background: rgba(255,255,255,0.06);
  font-size: .82rem;
  color: var(--tx0);
  letter-spacing: 0.01em;
}
.rag-badge-ghost{ border-color: var(--bd0); background: rgba(255,255,255,0.04); color: var(--tx1); }
.rag-badge-strong{ border-color: var(--ac1); background: rgba(88,178,255,0.12); color: #cfe6ff; }
.rag-badge-warn{ border-color: var(--warn); background: rgba(255,178,107,0.16); color: #ffe2c7; }
.rag-badge-err{ border-color: var(--err); background: rgba(255,123,123,0.16); color: #ffd6d6; }
.rag-badge-good{ border-color: var(--ac2); background: rgba(95,227,185,0.16); color: #d5fff1; }

.rag-status-row{
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  align-items: center;
}

.rag-mode-panel{
  background: linear-gradient(180deg, rgba(16,24,43,0.6) 0%, rgba(12,17,29,0.85) 100%);
  border: 1px solid var(--bd0);
  border-radius: 14px;
  padding: 0.85rem 1rem 0.6rem;
}
.rag-mode-panel h3{
  font-size: 1rem;
  margin-bottom: .4rem;
}

/* action bar buttons */
.rag-act [data-testid="stButton"] button,
div[data-testid="stContainer"]:has(.rag-card-mk) [data-testid="stButton"] button{
  width: 100%;
  white-space: nowrap;
  border-radius: 12px;
  border: 1px solid var(--bd0);
  background: linear-gradient(180deg, rgba(255,255,255,0.07) 0%, rgba(255,255,255,0.02) 100%);
  color: var(--tx0);
  font-weight: 650;
  transition: border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease;
}
.rag-act [data-testid="stButton"] button:hover,
div[data-testid="stContainer"]:has(.rag-card-mk) [data-testid="stButton"] button:hover{
  border-color: var(--ac0);
  box-shadow: 0 10px 24px rgba(0,0,0,0.35);
}
.rag-act [data-testid="stButton"] button:active,
div[data-testid="stContainer"]:has(.rag-card-mk) [data-testid="stButton"] button:active{
  transform: translateY(1px);
}

/* context flash */
@keyframes ctxflash{ 0%{ box-shadow: 0 0 0 0 rgba(255,215,102,.35);} 100%{ box-shadow: 0 0 0 12px rgba(255,215,102,0);} }
.ctx-flash{ animation: ctxflash 0.8s ease-out 1; border-color: var(--ac0) !important; }

/* evidence snippet clamp */
.stMarkdown p{
  margin-bottom: .25rem;
}

.rag-near-miss{
  border: 1px solid var(--bd1);
  border-radius: 14px;
  padding: .75rem 1rem;
  background: rgba(88,178,255,0.07);
}

/* responsive padding tweak for tablets */
@media (max-width: 900px){
  .main .block-container{ padding-top: 4.6rem; padding-left: 1rem; padding-right: 1rem; }
  .rag-status-row{ gap: 8px; }
}

@media (max-width: 640px){
  .main .block-container{ padding-top: 4.1rem; }
  .rag-act [data-testid="stButton"] button{ font-size: .9rem; }
}

/* iOS Safari rendering quirks */
@supports (-webkit-touch-callout: none){
  .main .block-container{ padding-top: 4.9rem; }
  button, input, select, textarea{ font-size: 16px !important; }
}

</style>
"""
    st.markdown(css, unsafe_allow_html=True)
