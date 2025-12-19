from rag_engine import Eng, get_startup_report


def test_startup_report_missing_corpus_has_reason():
    corp_report = {
        "OReilly": {
            "exists": False,
            "faiss": False,
            "db": False,
            "manifest": False,
            "dense_loaded": False,
            "db_loaded": False,
            "dim_ok": False,
            "embed_dim": None,
            "ix_dim": None,
            "failure_reason": "corpus folder missing",
        }
    }
    eng = Eng(emb=None, ix={}, dbp={}, corp={}, ix_dim={}, corp_report=corp_report)
    report = get_startup_report(eng)
    rows = report.get("rows")

    assert rows, "startup report should contain entries"
    first = rows[0]
    for key in ["publisher", "loaded_dense", "loaded_db", "ready", "reason"]:
        assert key in first, f"startup row missing {key}"
    assert first["publisher"] == "OReilly"
    assert first["ready"] is False
    assert first["reason"]
