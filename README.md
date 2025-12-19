# RAG Books Search (CPU-only)

Minimalna aplikacja do wyszukiwania informacji w zindeksowanych książkach technicznych
z użyciem RAG (FAISS + sqlite + CrossEncoder judge).

Aplikacja:
- zwraca tylko fragmenty poparte danymi z książek,
- pokazuje gdzie znaleźć źródło (książka + sekcja),
- unika halucynacji poprzez twarde reguły evidence,
- działa **wyłącznie na CPU** (bez GPU / CUDA).

> CPU-only wymusza:
> - instalację binariów Pytorch/FAISS z kanału CPU,
> - brak zależności od GPU/CUDA w kodzie (w tym w CrossEncoder),
> - budżety tokenowe/znakowe w LLM i kontekście, aby uniknąć przepełnień.

UI: Streamlit  
Logika: `rag_engine.py`

---

## Struktura danych (wymagana)

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

Repo **nie wymaga** plików EPUB/PDF – tylko artefaktów indeksów.

Szybka weryfikacja środowiska (Python + zależności + artefakty danych):

```bash
python scripts/check_env.py
```

---

## Jak uruchomić lokalnie (copy–paste)

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

Aplikacja będzie dostępna pod:
http://localhost:8501

## Testy i walidacja

- Szybki smoke: `python smoke_ui_contract.py`
- Walidacja jakości retrieval: `pytest tests/validation/test_retrieval_quality.py`
- Pakiet testów jednostkowych: `pytest tests`

CI uruchamia smoke + walidację na gałęzi głównej.
