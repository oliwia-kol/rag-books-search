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

## CPU-only i ograniczenia

- brak wsparcia dla GPU/CUDA (pipeline i runtime muszą działać na CPU),
- embeddingi normalizowane (IndexFlatIP) – przy zmianie modelu zachowaj spójny wymiar,
- limity: domyślnie 10 wyników finalnych, kontekst LLM ~1400 znaków, prompt clamp ~2000 znaków.

## Wydawcy i struktura danych (wymagana)

Obsługiwani wydawcy (domyślni): **OReilly**, **Manning**, **Pearson**. Możesz dodać kolejnych, jeżeli zachowasz layout:

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

Szybkie sprawdzenia:
- Smoke UI contract: `python smoke_ui_contract.py`
- Walidacja jakości retrieval: `pytest tests/validation/test_retrieval_quality.py`
- Testy bezpieczeństwa retrieval: `pytest tests/validation/test_retrieval_safety.py`
- Smoke/perf (można wyłączyć perf: `SKIP_PERF_CHECK=1`): `pytest tests/smoke`

Aby lokalnie odwzorować CI (patrz `.github/workflows/ci.yml`):

```bash
python -m py_compile app.py rag_engine.py ui_shell.py ui_adapter.py ui_theme.py smoke_ui_contract.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/validation tests/smoke --maxfail=1 --durations=25
```

CI uruchamia lint (jeśli dostępny), kompilację bajtkodu oraz pakiety `tests/validation` i `tests/smoke` z raportem JUnit.
