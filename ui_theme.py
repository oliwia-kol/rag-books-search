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
}

html, body { background: var(--bg0); color: var(--tx0); }

/* avoid header overlap (mobile/iPad) */
.main .block-container{
  padding-top: 5.25rem;
  padding-bottom: 2rem;
}

.main .block-container h1, .main .block-container h2, .main .block-container h3{
  letter-spacing: 0.01em;
}
.main .block-container p{
  line-height: 1.55;
}

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

/* card marker styling via :has() (Chrome/Safari 16+) */
div[data-testid="stContainer"]:has(.rag-card-mk){
  border: 1px solid var(--bd0) !important;
  background: rgba(255,255,255,.03);
  border-radius: 16px;
}

.rag-card-top{
  border-top-left-radius: 16px;
  border-top-right-radius: 16px;
}

.rag-title{ font-size: 1.05rem; font-weight: 650; }
.rag-meta{ color: var(--tx2); font-size: .9rem; }
.rag-sub{ color: var(--tx1); font-size: .92rem; }

/* action bar buttons */
.rag-act [data-testid="stButton"] button{
  width: 100%;
  white-space: nowrap;
}

/* context flash */
@keyframes ctxflash{ 0%{ box-shadow: 0 0 0 0 rgba(255,215,102,.35);} 100%{ box-shadow: 0 0 0 12px rgba(255,215,102,0);} }
.ctx-flash{ animation: ctxflash 0.8s ease-out 1; border-color: var(--ac0) !important; }

/* evidence snippet clamp */
.stMarkdown p{
  margin-bottom: .25rem;
}

/* responsive padding tweak for tablets */
@media (max-width: 900px){
  .main .block-container{ padding-top: 4.5rem; }
}

</style>
"""
    st.markdown(css, unsafe_allow_html=True)
