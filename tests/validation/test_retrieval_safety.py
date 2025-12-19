import sqlite3
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import faiss
import numpy as np

import rag_engine as re_mod


class RetrievalSafetyTest(unittest.TestCase):
    def _build_engine(self, vectors, texts, missing_ids=None):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        base = Path(tmpdir.name)
        db_path = base / "meta.sqlite"
        con = sqlite3.connect(db_path)
        con.execute("CREATE TABLE chunks (cid TEXT, fp TEXT, sec TEXT, cidx INTEGER, tx TEXT, i64 INTEGER PRIMARY KEY)")
        missing = set(missing_ids or set())
        for idx, (vec, tx) in enumerate(zip(vectors, texts)):
            if idx in missing:
                continue
            con.execute(
                "INSERT INTO chunks (cid, fp, sec, cidx, tx, i64) VALUES (?, ?, ?, ?, ?, ?)",
                (f"cid-{idx}", "fp", "sec", idx, tx, idx),
            )
        con.commit()
        con.close()

        dim = len(vectors[0])
        index = faiss.IndexFlatIP(dim)
        index.add(np.asarray(vectors, dtype="float32"))

        return re_mod.Eng(
            emb=None,
            ix={"Test": index},
            dbp={"Test": db_path},
            corp={"Test": base},
            ix_dim={"Test": dim},
            corp_report={},
        )

    def test_query_vectors_are_normalized_for_ip(self):
        eng = self._build_engine([[1.0, 0.0], [0.0, 1.0]], ["a", "b"])
        rows, meta = re_mod.faiss_search(eng, "Test", np.array([10.0, 0.0], dtype="float32"), k=2)
        self.assertTrue(rows)
        self.assertEqual(meta["metric_type"], faiss.METRIC_INNER_PRODUCT)
        self.assertLessEqual(rows[0]["sem_score"], 1.01)

    def test_sqlite_text_is_truncated(self):
        long_text = "x" * (re_mod.SQLITE_TEXT_MAX + 50)
        eng = self._build_engine([[1.0, 0.0]], [long_text])
        rows, _ = re_mod.faiss_search(eng, "Test", np.array([1.0, 0.0], dtype="float32"), k=1)
        self.assertTrue(rows)
        self.assertEqual(len(rows[0]["tx"]), re_mod.SQLITE_TEXT_MAX)
        self.assertEqual(len(rows[0]["text"]), re_mod.SQLITE_TEXT_MAX)

    def test_fallback_counters_increment_and_propagate(self):
        vectors = [[0.7, 0.7], [0.7, 0.7]]
        texts = ["kept", "missing"]
        eng = self._build_engine(vectors, texts, missing_ids={1})
        qv = np.array([0.7, 0.7], dtype="float32")

        rows, meta = re_mod.faiss_search(eng, "Test", qv, k=2)
        self.assertTrue(rows)
        self.assertGreaterEqual(meta["fallback_retries"], 1)
        self.assertGreaterEqual(meta["fallback_failed"], 1)

        with mock.patch("rag_engine.embed_query", return_value=qv):
            rq = re_mod.run_query(eng, "q")
        self.assertEqual(rq["meta"]["n"]["fallback_retries"], meta["fallback_retries"])
        self.assertEqual(rq["meta"]["n"]["fallback_failed"], meta["fallback_failed"])

    def test_hybrid_k_clamping_is_reported(self):
        eng = self._build_engine([[1.0, 0.0]], ["short"])
        qv = np.array([1.0, 0.0], dtype="float32")
        requested_k = re_mod.HCFG["mmr_k"] + 5
        hits, rmeta = re_mod.hybrid_retrieve(eng, "q", k=requested_k, pubs=["Test"], qv=qv)
        self.assertTrue(hits)
        self.assertTrue(rmeta["k_clamped"])
        self.assertEqual(rmeta["k_requested"], requested_k)
        self.assertEqual(rmeta["k_applied"], re_mod.HCFG["mmr_k"])
        self.assertLessEqual(len(hits), re_mod.HCFG["mmr_k"])


if __name__ == "__main__":
    unittest.main()
