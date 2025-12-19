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

/* compact sidebar controls */
section[data-testid="stSidebar"] [data-testid="stButton"] button{
  padding: .25rem .5rem;
  min-height: 2.1rem;
  line-height: 1.1;
  white-space: nowrap;
}

/* card marker styling via :has() (Chrome/Safari 16+) */
div[data-testid="stContainer"]:has(.rag-card-mk){
  border: 1px solid var(--bd0) !important;
  background: rgba(255,255,255,.03);
  border-radius: 16px;
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

</style>
"""
    st.markdown(css, unsafe_allow_html=True)
