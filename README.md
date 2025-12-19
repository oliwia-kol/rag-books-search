# RAG Books Search (CPU-only)

Minimalna aplikacja do wyszukiwania informacji w zindeksowanych książkach technicznych
z użyciem RAG (FAISS + sqlite + CrossEncoder judge).

Aplikacja:
- zwraca tylko fragmenty poparte danymi z książek,
- pokazuje gdzie znaleźć źródło (książka + sekcja),
- unika halucynacji poprzez twarde reguły evidence,
- działa **wyłącznie na CPU** (bez GPU / CUDA).

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
