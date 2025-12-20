# RAG Books Search (CPU-only)

Streamlit UI for retrieval-augmented search across indexed technical books. The pipeline is CPU-only (FAISS + sqlite + CrossEncoder judge), so no GPU/CUDA is required or supported.

## Quick start (fastest path)

```bash
git clone <REPO_URL>
cd rag-books-search
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Data layout (required corpora)

Place CPU-friendly indexes under `data/` using the same structure for each publisher:

```
data/
├── OReilly/
│   ├── index.faiss
│   ├── meta.sqlite
│   └── manifest.json
├── Manning/
│   ├── index.faiss
│   ├── meta.sqlite
│   └── manifest.json
└── Pearson/
    ├── index.faiss
    ├── meta.sqlite
    └── manifest.json
```

Only the index artifacts are needed; source PDFs/EPUBs are not required.

## Checks and tests

- Quick environment sanity check (Python, deps, corpus files): `python scripts/check_env.py`
- Contract smoke for UI + engine modules: `python smoke_ui_contract.py`
- Compile entrypoints: `python -m py_compile app.py rag_engine.py ui_adapter.py ui_shell.py ui_theme.py smoke_ui_contract.py`
- Full test suite: `pytest`
