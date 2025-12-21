import streamlit as st


THEME_TOKENS = {
    "spacing": [4, 8, 12, 16, 24, 32],
    "radius": {"card": 12, "input": 10, "chip": 8},
    "color": {
        # dark-first palette for contrast safety
        "bg": "#0b0e14",
        "surface": "#0f1522",
        "text": "#f5f7fb",
        "muted": "#c1c8d6",
        "accent": "#9ad1ff",
        "soft": "rgba(154, 209, 255, 0.14)",
        "shadow": "0 14px 34px rgba(0,0,0,0.4)",
        "focus": "0 0 0 3px rgba(154, 209, 255, 0.35)",
    },
}


def _apply_mode_flag(mode: str | None):
    """Set a data-theme attribute on the root document for CSS scoping."""
    resolved = mode or st.session_state.get("theme_mode", "dark")
    st.session_state["theme_mode"] = resolved
    st.session_state["_theme_signal"] = st.session_state.get("_theme_signal", 0) + 1
    st.markdown(
        f"""
<script>
try {{
  const root = window.parent.document.documentElement;
  const mode = '{resolved}';
  root.setAttribute('data-theme', '{resolved}');
  if (root.getAttribute('data-theme') !== mode) {{
    root.setAttribute('data-theme', mode);
  }}
  root.style.colorScheme = mode === 'dark' ? 'dark' : 'light';
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

    resolved_mode = mode or st.session_state.get("theme_mode", "dark")
    _apply_mode_flag(resolved_mode)

    css = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;600&display=swap');
@import url('https://cdn.jsdelivr.net/npm/phosphor-icons@1.4.2/src/css/phosphor.css');

:root {
  --bg: #0b0e14;
  --surface: #0f1522;
  --surface-2: #121a2b;
  --text: #f5f7fb;
  --muted: #c1c8d6;
  --muted-2: #8d95a9;
  --border: #1e2635;
  --border-strong: #2b3547;
  --accent: #9ad1ff;
  --accent-strong: #7bb9f5;
  --accent-soft: rgba(154, 209, 255, 0.16);
  --soft: rgba(255, 255, 255, 0.03);
  --secondary: #f3b26b;
  --secondary-soft: rgba(243, 178, 107, 0.2);
  --success: #7ad9a6;
  --success-soft: rgba(122, 217, 166, 0.2);
  --warning: #f3c36f;
  --warning-soft: rgba(243, 195, 111, 0.22);
  --danger: #f59a9a;
  --danger-soft: rgba(245, 154, 154, 0.22);
  --radius-card: 12px;
  --radius-input: 10px;
  --radius-chip: 8px;
  --shadow-subtle: 0 12px 32px rgba(0,0,0,0.22);
  --shadow-strong: 0 22px 44px rgba(0,0,0,0.35);
  --shadow-none: none;
  --font-body: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
  --font-code: "Fira Code", "SFMono-Regular", Consolas, ui-monospace, monospace;
  --gradient-iris: linear-gradient(120deg, #8ab6ff, #5b8ef4, #4ec1e5);
  --gradient-amber: linear-gradient(120deg, #f3b26b, #f78e43, #f25f5c);
  --focus-ring: 0 0 0 3px rgba(154, 209, 255, 0.35);
  --focus-outline: 2px solid var(--accent);
  color-scheme: dark;
}

[data-theme="light"] {
  --bg: #f6f4f0;
  --surface: #ffffff;
  --surface-2: #f4f1ec;
  --text: #151720;
  --muted: #4c5361;
  --muted-2: #6a707d;
  --border: #e4e1db;
  --border-strong: #cfc8bd;
  --accent: #2d5bd1;
  --accent-strong: #244bc5;
  --accent-soft: rgba(45, 91, 209, 0.12);
  --soft: rgba(0, 0, 0, 0.02);
  --secondary: #d97a24;
  --secondary-soft: rgba(217, 122, 36, 0.12);
  --success: #2f8f5b;
  --success-soft: rgba(47, 143, 91, 0.12);
  --warning: #c07a1a;
  --warning-soft: rgba(192, 122, 26, 0.14);
  --danger: #c74949;
  --danger-soft: rgba(199, 73, 73, 0.14);
  --shadow-subtle: 0 8px 24px rgba(0,0,0,0.08);
  --shadow-strong: 0 18px 42px rgba(0,0,0,0.16);
  color-scheme: light;
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
  background: radial-gradient(circle at top left, rgba(124, 160, 255, 0.08), transparent 38%), var(--bg);
}

.brand {
  display: flex;
  gap: 10px;
  align-items: center;
}

.brand .logo {
  width: 38px;
  height: 38px;
  border-radius: 10px;
  background: var(--gradient-iris);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #0a0f1a;
  box-shadow: var(--shadow-subtle);
}

.brand .logo i { font-size: 1.25rem; }

.brand .name {
  display: flex;
  flex-direction: column;
  line-height: 1.15;
}

.brand .name .title { font-size: 1rem; letter-spacing: 0.02em; font-weight: 650; }
.brand .name .muted { color: var(--muted-2); font-size: 0.88rem; }

.icon-nav {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2);
  color: var(--muted);
  box-shadow: var(--shadow-none);
}

.icon-nav .pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 999px;
  color: var(--text);
  background: var(--soft);
  border: 1px solid var(--border);
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

.status-gauges {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  align-items: center;
}

.gauge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--muted);
  font-size: 0.9rem;
}
.gauge-label { font-weight: 600; color: var(--muted-2); }
.gauge-bar {
  width: 92px;
  height: 8px;
  background: var(--surface-2);
  border-radius: 999px;
  overflow: hidden;
  border: 1px solid var(--border);
}
.gauge-fill {
  height: 100%;
  background: var(--secondary);
  border-radius: inherit;
}
.gauge-fill.accent { background: var(--accent); }
.gauge-value { font-variant-numeric: tabular-nums; color: var(--text); }

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
.status-chip.primary { background: var(--accent-soft); color: var(--accent-strong); border-color: rgba(122,185,255,0.3); }
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
.chip.primary { border-color: rgba(122,185,255,0.28); color: var(--accent-strong); background: var(--accent-soft); }
.chip.secondary { border-color: rgba(217,122,36,0.28); color: var(--secondary); background: var(--secondary-soft); }
.chip.muted { color: var(--muted-2); }

.card-shell {
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  padding: 0.75rem 0.85rem;
  background: var(--surface);
  transition: border-color 0.15s ease, transform 0.12s ease, box-shadow 0.18s ease;
  box-shadow: var(--shadow-none);
  position: relative;
  overflow: hidden;
}
.card-shell:hover { border-color: var(--border-strong); transform: translateY(-1px); box-shadow: var(--shadow-subtle); }
.card-shell.selected { border-left: 3px solid var(--accent); background: var(--accent-soft); }
.card-shell.near { border-left: 3px solid var(--secondary); background: linear-gradient(90deg, var(--secondary-soft), transparent); }
.card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 6px 8px;
  margin: -4px -6px 8px -6px;
  border-radius: 10px;
  background: linear-gradient(120deg, color-mix(in srgb, var(--pub-color), transparent 40%), var(--surface));
  border: 1px solid color-mix(in srgb, var(--pub-color), var(--border));
}
.card-head-left { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.pub-pill {
  padding: 6px 10px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--pub-color), transparent 80%);
  color: var(--text);
  border: 1px solid color-mix(in srgb, var(--pub-color), var(--border));
  font-weight: 600;
}
.evidence-card { border: none; padding: 0; }
.evidence-title { font-size: 1rem; font-weight: 600; margin-bottom: 0.1rem; color: var(--text); }
.evidence-meta { color: var(--muted); font-size: 0.9rem; margin-bottom: 0.15rem; }
.evidence-snippet { color: var(--text); line-height: 1.55; margin-top: 0.35rem; }
.evidence-foot { margin-top: 8px; color: var(--muted-2); font-size: 0.9rem; }
.score-row { display: flex; gap: 8px; justify-content: flex-end; flex-wrap: wrap; }

.action-row { display: flex; gap: 8px; margin-top: 10px; transition: opacity 0.15s ease, max-height 0.2s ease; }
.action-row button { width: 100%; }
.stContainer:has(.card-shell) .card-actions { opacity: 0; max-height: 0; overflow: hidden; pointer-events: none; }
.stContainer:has(.card-shell):hover .card-actions,
.touch .card-actions { opacity: 1; max-height: 160px; pointer-events: all; }
@media (hover: none) { .card-actions { opacity: 1 !important; max-height: 200px !important; pointer-events: all; } }

.details-panel pre, .debug-block pre { background: var(--surface-2); border-radius: var(--radius-card); padding: 12px; border: 1px solid var(--border); font-family: var(--font-code); }

.debug-block { border: 1px dashed var(--border-strong); border-radius: var(--radius-card); padding: 12px; background: var(--surface-2); }

.hl { background: rgba(122, 185, 255, 0.18); padding: 1px 2px; border-radius: 6px; }

.empty-state { border: 1px dashed var(--border); border-radius: var(--radius-card); padding: 1rem 1.1rem; background: var(--surface); color: var(--muted); }

.skeleton { height: 110px; border-radius: var(--radius-card); background: linear-gradient(90deg, rgba(0,0,0,0.03), rgba(0,0,0,0.08), rgba(0,0,0,0.03)); animation: pulse 1.2s ease-in-out infinite; border: 1px solid var(--border); }
@keyframes pulse { 0% { background-position: 0% 50%; } 100% { background-position: 100% 50%; } }

button:focus-visible,
[role="button"]:focus-visible,
input:focus-visible,
textarea:focus-visible,
select:focus-visible,
.st-key-command:focus-visible {
  outline: var(--focus-outline);
  outline-offset: 2px;
  box-shadow: var(--focus-ring);
  transition: box-shadow 0.12s ease, outline 0.12s ease;
}

.app-shell .layout-grid {
  display: grid;
  grid-template-columns: 0.28fr 0.48fr 0.24fr;
  gap: 1.25rem;
  align-items: start;
}

.rail { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-card); padding: 1rem; box-shadow: var(--shadow-subtle); }

.context-pane { background: var(--surface); border-radius: var(--radius-card); padding: 0.6rem 0.8rem; border: 1px solid var(--border); box-shadow: var(--shadow-subtle); }
.context-pane .stExpander { background: transparent; border: none; }
.context-pane .stExpander > div { background: transparent; }
.ctx-shell { transition: transform 0.18s ease, opacity 0.2s ease; opacity: 0.82; transform: translateX(8px); }
.ctx-shell.active { opacity: 1; transform: translateX(0); }

.hero {
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem 1.25rem;
  background: linear-gradient(135deg, rgba(154,209,255,0.08), rgba(243,178,107,0.06)), var(--surface);
  box-shadow: var(--shadow-subtle);
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: 1rem;
  align-items: center;
}
.hero .hero-icon { width: 52px; height: 52px; border-radius: 14px; display: inline-flex; align-items: center; justify-content: center; background: var(--gradient-iris); color: #0a0f1a; box-shadow: var(--shadow-subtle); }
.hero h2 { margin: 0; font-size: 1.4rem; }
.hero .lede { color: var(--muted); margin-bottom: 0.6rem; line-height: 1.6; }
.hero .bullets { display: grid; gap: 8px; }
.hero .bullets .item { display: inline-flex; align-items: center; gap: 8px; padding: 8px 10px; border: 1px solid var(--border); border-radius: 10px; background: var(--soft); }
.hero .stats { display: flex; gap: 10px; flex-wrap: wrap; }
.hero .stats .stat { padding: 10px 12px; border-radius: 12px; background: var(--accent-soft); color: var(--accent-strong); border: 1px solid rgba(122,185,255,0.26); }

.slim-actions {
  display: inline-flex;
  gap: 6px;
  flex-wrap: wrap;
}
.slim-actions .btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 10px;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: var(--surface-2);
  color: var(--text);
  box-shadow: var(--shadow-none);
}

.pin-entry {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--text);
}
.pin-entry .pin-label { color: var(--text); font-size: 0.95rem; }

@media (max-width: 1100px) {
  .status-strip { grid-template-columns: 1fr; }
  .status-meta { justify-content: flex-start; }
  .topbar { position: relative; }
  .app-shell .layout-grid { grid-template-columns: 0.34fr 0.66fr; }
  .context-pane { grid-column: span 2; }
  .hero { grid-template-columns: 1fr; }
}

@media (max-width: 900px) {
  .main .block-container { padding-top: 1.2rem; }
  .status-strip { padding: 12px; }
  .action-row { flex-direction: column; }
  .app-shell .layout-grid { grid-template-columns: 1fr; }
  .rail { position: sticky; top: 68px; z-index: 12; box-shadow: var(--shadow-strong); }
  .context-pane {
    position: fixed;
    inset: 78px 12px auto 12px;
    max-height: calc(100vh - 110px);
    overflow: auto;
    z-index: 30;
  }
}

@media (prefers-reduced-motion: reduce) {
  * { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; scroll-behavior: auto !important; }
}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)
