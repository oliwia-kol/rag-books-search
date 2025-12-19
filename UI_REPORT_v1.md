# UI Report v1 (validated pack)

## What was fixed (hard blockers)
1. **API mismatch** (app ↔ ui modules)
   - Rebuilt a consistent pack: `app.py`, `ui_shell.py`, `ui_adapter.py`, `ui_theme.py` share one contract.

2. **Expand/context not opening**
   - Expand now sets `session_state["act_hit"]` and renders a **fixed Context panel**.
   - Auto-scroll + flash highlight to make it discoverable.

3. **Judge default ON (forced)**
   - `use_jdg` is **forced True** every session.
   - Ranking uses `judge01` (fallback to `score`).
   - Default display threshold set to `judge01 >= 0.45`.

4. **Publisher scope wired end-to-end**
   - `rag_engine.hybrid_retrieve(..., pubs=...)` restricts retrieval to selected publishers.

## What is intentionally left for next stages
- “Back/Forward UX” hint for query params.
- Typography pass (line-heights, spacing micro-tuning).
- Optional contract smoke test script in repo.

## Manual validation (PASS/FAIL)
- Search -> results render without crash.
- Click Pin once -> sidebar pinned updates immediately.
- Click Copy once -> clipboard updates immediately.
- Click Expand once -> page scrolls to Context panel and it shows content + Close.
- Fresh session -> judge is ON by default and ordering reflects judge.
