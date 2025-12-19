# UI Stage Checklist v1 (validated pack)

Legend: [x] done, [ ] todo

## Stage 1A: State + filters + stability
- [x] One entrypoint: `streamlit run app.py`
- [x] All UI state stored in `st.session_state` (no local copies that desync)
- [x] Publisher scope (OReilly/Manning/Pearson) wired end-to-end into retrieval
- [x] Advanced toggle hides non-critical controls
- [x] URL query param `?q=` loads initial query into the input
- [ ] Back/Forward UX polish (explicit “Loaded from URL” hint)

## Stage 1B: Pin / Copy / Expand (context panel)
- [x] Pin 1-click: adds to sidebar immediately
- [x] Unpin 1-click: removes immediately
- [x] Copy 1-click: updates sidebar clipboard immediately
- [x] Clipboard is product-format (no cid/cidx as content)
- [x] Expand opens fixed Context panel (no expander under cards)
- [x] Expand auto-scrolls to Context panel + flash highlight
- [x] Close clears Context panel

## Stage 1B: Judge defaults (hard requirement)
- [x] Judge is forced ON for every session (toggle disabled)
- [x] Ranking is by `judge01` (fallback to `score`)
- [x] Default display threshold (`judge01`) is set to a sensible value (0.45)

## Stage 1C: Visual polish (quick wins)
- [x] Cards have subtle border + spacing
- [x] Sidebar pinned rows: title + meta + non-overlapping remove button
- [ ] Typography consistency pass (titles/metadata line-height)

## Stage 1D: Regression / contract safety
- [x] No API mismatch between `app.py` and `ui_*.py` (single validated pack)
- [ ] Add tiny contract smoke script in repo (optional)

