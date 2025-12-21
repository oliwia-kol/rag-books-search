import streamlit as st


THEME_TOKENS = {
    "spacing": [4, 8, 12, 16, 24, 32],
    "radius": {"card": 12, "input": 10, "chip": 8},
    "color": {
        "bg": "#f6f4f0",
        "surface": "#ffffff",
        "text": "#131518",
        "muted": "#4a4f58",
        "border": "#e3e0db",
        "primary": "#2d5bd1",  # cobalt accent
        "secondary": "#d97a24",  # amber/rose accent for near-miss
        "success": "#2f8f5b",
        "warning": "#c07a1a",
        "danger": "#c74949",
    },
}


def _apply_mode_flag(mode: str):
    """Set a data-theme attribute on the root document for CSS scoping."""
    st.session_state["theme_mode"] = mode
    st.markdown(
        f"""
<script>
try {{
  const root = window.parent.document.documentElement;
  root.setAttribute('data-theme', '{mode}');
}} catch (e) {{}}
</script>
""",
        unsafe_allow_html=True,
    )


def apply_theme(mode: str | None = None):
    """Inject shared styling tokens and component polish.

    Args:
        mode: optional theme mode ("light" or "dark"). Falls back to session state.
    """

    mode = mode or st.session_state.get("theme_mode", "light")
    _apply_mode_flag(mode)

    css = r"""
<style>
:root {
  --bg: #f6f4f0;
  --surface: #ffffff;
  --surface-2: #f0eeea;
  --text: #151720;
  --muted: #525866;
  --muted-2: #6a707d;
  --border: #e4e1db;
  --border-strong: #cfc8bd;
  --primary: #2d5bd1;
  --primary-soft: rgba(45, 91, 209, 0.12);
  --secondary: #d97a24;
  --secondary-soft: rgba(217, 122, 36, 0.12);
  --success: #2f8f5b;
  --success-soft: rgba(47, 143, 91, 0.12);
  --warning: #c07a1a;
  --warning-soft: rgba(192, 122, 26, 0.14);
  --danger: #c74949;
  --danger-soft: rgba(199, 73, 73, 0.14);
  --radius-card: 12px;
  --radius-input: 10px;
  --radius-chip: 8px;
  --shadow-subtle: 0 8px 24px rgba(0,0,0,0.04);
  --shadow-none: none;
  --font-body: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
}

[data-theme="dark"] {
  --bg: #0f1116;
  --surface: #151924;
  --surface-2: #0d1018;
  --text: #f3f4f7;
  --muted: #c1c4cc;
  --muted-2: #9da3af;
  --border: #1f2430;
  --border-strong: #2d3445;
  --primary: #6ea8ff;
  --primary-soft: rgba(110, 168, 255, 0.16);
  --secondary: #f0a15a;
  --secondary-soft: rgba(240, 161, 90, 0.18);
  --success: #6ccf9a;
  --success-soft: rgba(108, 207, 154, 0.18);
  --warning: #f0bf68;
  --warning-soft: rgba(240, 191, 104, 0.18);
  --danger: #f18b8b;
  --danger-soft: rgba(241, 139, 139, 0.2);
  --shadow-subtle: 0 14px 34px rgba(0,0,0,0.35);
}

html, body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-body);
  -webkit-font-smoothing: antialiased;
}

.main .block-container {
  max-width: 1340px;
  padding-top: 2.25rem;
  padding-bottom: 2.25rem;
  gap: 0.75rem;
}

.app-shell {
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 0.25rem;
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 15;
  background: var(--bg);
}

.topbar .brand {
  font-size: 1rem;
  letter-spacing: 0.02em;
  font-weight: 650;
}

.control-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface);
  color: var(--muted);
  font-size: 0.9rem;
}

.status-strip {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 10px;
  align-items: center;
  padding: 12px 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  background: var(--surface);
  box-shadow: var(--shadow-subtle);
}

.status-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 12px;
  border: 1px solid var(--border);
  font-weight: 600;
  letter-spacing: 0.01em;
}
.status-chip.success { background: var(--success-soft); color: var(--success); border-color: rgba(47,143,91,0.28); }
.status-chip.primary { background: var(--primary-soft); color: var(--primary); border-color: rgba(45,91,209,0.3); }
.status-chip.neutral { background: rgba(0,0,0,0.02); color: var(--muted); }
.status-chip.warning { background: var(--warning-soft); color: var(--warning); border-color: rgba(192,122,26,0.26); }
.status-chip.danger { background: var(--danger-soft); color: var(--danger); border-color: rgba(199,73,73,0.24); }

.status-meta {
  display: flex;
  gap: 10px;
  justify-content: flex-end;
  font-size: 0.9rem;
  color: var(--muted);
}

.section-title { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted-2); margin-bottom: 0.2rem; }

.panel {
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  padding: 1rem;
  background: var(--surface);
  box-shadow: var(--shadow-subtle);
}

.panel h3 {
  margin: 0 0 0.2rem 0;
  font-size: 1.05rem;
  font-weight: 600;
}

.chips { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; }
.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: var(--radius-chip);
  border: 1px solid var(--border);
  background: var(--surface-2);
  font-size: 0.88rem;
  color: var(--muted);
}
.chip.primary { border-color: rgba(45,91,209,0.28); color: var(--primary); background: var(--primary-soft); }
.chip.secondary { border-color: rgba(217,122,36,0.28); color: var(--secondary); background: var(--secondary-soft); }
.chip.muted { color: var(--muted-2); }

.evidence-card {
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  padding: 0.85rem 0.95rem;
  background: var(--surface);
  transition: border-color 0.15s ease, transform 0.12s ease, box-shadow 0.18s ease;
  box-shadow: var(--shadow-none);
}
.evidence-card:hover { border-color: var(--border-strong); transform: translateY(-1px); box-shadow: var(--shadow-subtle); }
.evidence-card.selected { border-left: 3px solid var(--primary); background: var(--primary-soft); }
.evidence-card.near { border-left: 3px solid var(--secondary); background: linear-gradient(90deg, var(--secondary-soft), transparent); }
.evidence-title { font-size: 1rem; font-weight: 600; margin-bottom: 0.1rem; color: var(--text); }
.evidence-meta { color: var(--muted); font-size: 0.9rem; margin-bottom: 0.15rem; }
.evidence-snippet { color: var(--text); line-height: 1.55; margin-top: 0.35rem; }
.score-row { display: flex; gap: 8px; justify-content: flex-end; flex-wrap: wrap; }

.action-row { display: flex; gap: 8px; margin-top: 10px; }
.action-row button { width: 100%; }

.details-panel pre, .debug-block pre { background: var(--surface-2); border-radius: var(--radius-card); padding: 12px; border: 1px solid var(--border); }

.debug-block { border: 1px dashed var(--border-strong); border-radius: var(--radius-card); padding: 12px; background: var(--surface-2); }

.hl { background: rgba(45, 91, 209, 0.16); padding: 1px 2px; border-radius: 6px; }

.empty-state { border: 1px dashed var(--border); border-radius: var(--radius-card); padding: 1rem 1.1rem; background: var(--surface); color: var(--muted); }

.skeleton { height: 110px; border-radius: var(--radius-card); background: linear-gradient(90deg, rgba(0,0,0,0.03), rgba(0,0,0,0.08), rgba(0,0,0,0.03)); animation: pulse 1.2s ease-in-out infinite; border: 1px solid var(--border); }
@keyframes pulse { 0% { background-position: 0% 50%; } 100% { background-position: 100% 50%; } }

@media (max-width: 1100px) {
  .status-strip { grid-template-columns: 1fr; }
  .status-meta { justify-content: flex-start; }
  .topbar { position: relative; }
}

@media (max-width: 900px) {
  .main .block-container { padding-top: 1.2rem; }
  .status-strip { padding: 12px; }
  .action-row { flex-direction: column; }
}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)

