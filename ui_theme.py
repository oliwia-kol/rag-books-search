"""
Custom theme definitions for the RAG Books Search UI.

This module defines a single, cohesive dark theme for the application.
Unlike the original ``ui_theme.py``, it intentionally omits the light
palette to reduce complexity and enforce a consistent look across the
entire app.  The colours are chosen to provide high contrast,
eye‑catching accents and smooth gradients while remaining easy on the
eyes.  Focus rings and hover states are carefully tuned for keyboard
accessibility.
"""

import streamlit as st


def _apply_mode_flag(mode: str) -> None:
    """Set the ``data-theme`` attribute on the root element.

    Streamlit apps run within an iframe; this script reaches up to the
    parent document to set a data attribute.  We default to dark but
    still store the requested mode for compatibility with tests.
    """
    st.session_state["theme_mode"] = mode
    st.session_state["_theme_signal"] = st.session_state.get("_theme_signal", 0) + 1
    st.markdown(
        (
            "<script>\n"
            "try {\n"
            "  const root = window.parent.document.documentElement;\n"
            f"  root.setAttribute('data-theme', '{mode}');\n"
            f"  root.style.colorScheme = '{mode}';\n"
            "} catch (e) {}\n"
            "</script>\n"
        ),
        unsafe_allow_html=True,
    )


def apply_theme(mode: str | None = None) -> None:
    """Inject the CSS for the dark theme into the Streamlit app.

    The ``mode`` argument updates session state for compatibility.
    """
    effective_mode = mode or "dark"
    _apply_mode_flag(effective_mode)
    css = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;600&display=swap');
@import url('https://cdn.jsdelivr.net/npm/phosphor-icons@1.4.2/src/css/phosphor.css');

:root {
  /* Base palette */
  --bg: #0b0e16;
  --surface: #121a2c;
  --surface-2: #151f33;
  --text: #f5f7fb;
  --muted: #c1c8d6;
  --muted-2: #8d95a9;
  --border: #1e2638;
  --border-strong: #2b3650;
  /* Accent palette */
  --accent: #9cafff;
  --accent-strong: #7c96f5;
  --accent-soft: rgba(156, 175, 255, 0.16);
  --secondary: #f3b26b;
  --secondary-soft: rgba(243, 178, 107, 0.18);
  --success: #7ad9a6;
  --success-soft: rgba(122, 217, 166, 0.2);
  --warning: #f3c36f;
  --warning-soft: rgba(243, 195, 111, 0.22);
  --danger: #f59a9a;
  --danger-soft: rgba(245, 154, 154, 0.22);

  /* Additional accent gradients for future customization (e.g., golden and cyan variants).
     These variables allow UI designers to switch between accent colour schemes
     without editing multiple CSS rules.  They are unused initially but can be
     referenced in components (e.g. for chat bubbles or highlight borders). */
  --gradient-accent-gold: linear-gradient(120deg, #ffd28e, #f3b26b, #e89c5d);
  --gradient-accent-cyan: linear-gradient(120deg, #80ffd1, #52e3c2, #29d1b1);
  --accent-gold: #f3b26b;
  --accent-cyan: #29d1b1;
  /* Radii */
  --radius-card: 12px;
  --radius-input: 10px;
  --radius-chip: 8px;
  /* Shadows */
  --shadow-subtle: 0 12px 32px rgba(0,0,0,0.32);
  --shadow-strong: 0 24px 48px rgba(0,0,0,0.45);
  --shadow-none: none;
  /* Typography */
  --font-body: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
  --font-code: "Fira Code", "SFMono-Regular", Consolas, ui-monospace, monospace;
  /* Gradients */
  --gradient-primary: linear-gradient(120deg, #8ab6ff, #6e8ef7, #5c7be9);
  --gradient-secondary: linear-gradient(120deg, #f3b26b, #f78e43, #f25f5c);
  /* Focus ring */
  --focus-ring: 0 0 0 3px rgba(156, 175, 255, 0.35);
  --focus-outline: 2px solid var(--accent);
  color-scheme: dark;
}

html, body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-body);
  -webkit-font-smoothing: antialiased;
}

/* Limit page width and provide top/bottom padding */
.main .block-container {
  max-width: 1340px;
  padding-top: 2.5rem;
  padding-bottom: 2.5rem;
  gap: 0.75rem;
}

/* Shell wrappers */
.app-shell {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
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
  background: radial-gradient(circle at top left, rgba(124, 160, 255, 0.10), transparent 38%), var(--bg);
}

.brand {
  display: flex;
  gap: 10px;
  align-items: center;
}
.brand .logo {
  width: 40px;
  height: 40px;
  border-radius: 10px;
  background: var(--gradient-primary);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #0a0f1a;
  box-shadow: var(--shadow-subtle);
}
.brand .logo i { font-size: 1.3rem; }
.brand .name {
  display: flex;
  flex-direction: column;
  line-height: 1.15;
}
.brand .name .title { font-size: 1.05rem; letter-spacing: 0.02em; font-weight: 650; }
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
  background: var(--surface);
  border: 1px solid var(--border);
}

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
  transition: background 0.15s ease;
}
.slim-actions .btn:hover {
  background: var(--accent-soft);
}

/* Section titles and panels */
.section-title {
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--muted-2);
  margin-bottom: 0.35rem;
}
.panel {
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  padding: 1rem;
  background: var(--surface);
  box-shadow: var(--shadow-subtle);
}
.panel h3 {
  margin: 0 0 0.2rem 0;
  font-size: 1.1rem;
  font-weight: 600;
}

/* Cards */
.card-shell {
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  padding: 0.8rem 0.9rem;
  background: var(--surface);
  transition: border-color 0.15s ease, transform 0.12s ease, box-shadow 0.18s ease;
  box-shadow: var(--shadow-none);
  position: relative;
  overflow: hidden;
}
.card-shell:hover {
  border-color: var(--border-strong);
  transform: translateY(-1px);
  box-shadow: var(--shadow-subtle);
}
.card-shell.selected {
  border-left: 3px solid var(--accent);
  background: var(--accent-soft);
}
.card-shell.near {
  border-left: 3px solid var(--secondary);
  background: linear-gradient(90deg, var(--secondary-soft), transparent);
}
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
@media (hover: none) { .card-actions { opacity: 1 !important; max-height: 200px !important; pointer-events: all !important; } }

.details-panel pre, .debug-block pre {
  background: var(--surface-2);
  border-radius: var(--radius-card);
  padding: 12px;
  border: 1px solid var(--border);
  font-family: var(--font-code);
}
.debug-block {
  border: 1px dashed var(--border-strong);
  border-radius: var(--radius-card);
  padding: 12px;
  background: var(--surface-2);
}
.hl {
  background: rgba(156, 175, 255, 0.20);
  padding: 1px 2px;
  border-radius: 6px;
}
.empty-state {
  border: 1px dashed var(--border);
  border-radius: var(--radius-card);
  padding: 1rem 1.1rem;
  background: var(--surface);
  color: var(--muted);
}
.skeleton {
  height: 110px;
  border-radius: var(--radius-card);
  background: linear-gradient(90deg, rgba(0,0,0,0.03), rgba(0,0,0,0.08), rgba(0,0,0,0.03));
  animation: pulse 1.2s ease-in-out infinite;
  border: 1px solid var(--border);
}
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

/* Layout grid for the three columns */
.app-shell .layout-grid {
  display: grid;
  grid-template-columns: 0.25fr 0.55fr 0.20fr;
  gap: 1.25rem;
  align-items: start;
}
.rail {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  padding: 1rem;
  box-shadow: var(--shadow-subtle);
}
.context-pane {
  background: var(--surface);
  border-radius: var(--radius-card);
  padding: 0.6rem 0.8rem;
  border: 1px solid var(--border);
  box-shadow: var(--shadow-subtle);
}
.context-pane .stExpander { background: transparent; border: none; }
.context-pane .stExpander > div { background: transparent; }
.ctx-shell { transition: transform 0.18s ease, opacity 0.2s ease; opacity: 0.85; transform: translateX(8px); }
.ctx-shell.active { opacity: 1; transform: translateX(0); }

/* Responsive tweaks */
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
