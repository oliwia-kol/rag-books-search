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

## Build offline (Phase 1)

1. Przygotuj `raw/<Publisher>/*.txt` (plain text).
2. Uruchom budowę korpusów (deterministyczne chunkowanie + embedding + FAISS + SQLite + manifest):

```bash
python scripts/build_corpus.py --src raw --out data --report build_report.json
```

- flaga `--publisher` pozwala zbudować tylko wybranych,
- `--validate-only` sprawdza spójność (wymiary, liczba chunków) bez ponownego embedowania.

Raport (`build_report.json`) zwraca sukces/fail per korpus z powodem.

## Walidacja środowiska

Skrypt sprawdza wersję Pythona (>=3.9), paczki i komplet plików `data/<pub>/{index.faiss,meta.sqlite,manifest.json}`:

```bash
python scripts/check_env.py
```

## Polityka logowania/błędy (runtime)

- każde zapytanie zapisuje strukturalne metadane w `meta.log` (tryb, scope, liczniki, flagi, no_evidence),
- guardraile: clamp kontekstu i promptu, wymuszony judge ON w UI, densy wyłączane przy mismatch wymiarów,
- błędy runtime zwracane w polu `meta.err` + globalny box w UI; brak korpusów -> komunikat o dodaniu `data/`.
- Luka vs DoD: brak trwałego audytu plikowego/log shipping, judge działa w trybie proxy (brak prawdziwego cross-encodera), brak polityki retencji.

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

## Walidacja retrieval (Phase 2)

Testy pokrywające dense-only, lex-only, hybrid oraz brak korpusu:

```bash
pytest tests/validation/test_retrieval_quality.py
```

## Komendy CI / smoke

- środowisko: `python scripts/check_env.py`
- budowa korpusów offline: `python scripts/build_corpus.py --src raw --out data --report build_report.json`
- testy walidacji retrieval: `pytest tests/validation/test_retrieval_quality.py`
