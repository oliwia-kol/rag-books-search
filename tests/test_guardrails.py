import rag_engine as re

def test_llm_call_respects_budgets():
    prompt = "word " * 1000
    out = re.llm_call(prompt, cfg={"char_budget": 120, "tok_budget": 12})
    assert len(out) <= 120 + len(re.LLM_CLAMP_MARKER) + 1
    assert len(out.split()) <= 12
    assert re.LLM_CLAMP_MARKER.strip("[] .") in out


def test_assemble_context_clamps_and_marks():
    hits = [{"tx": "x" * 500, "score": 0.9, "judge01": 0.8, "cid": "1"}, {"tx": "y" * 500, "score": 0.8, "judge01": 0.7, "cid": "2"}]
    ctx, meta = re._assemble_context(hits, budget_chars=600, budget_tokens=80)
    assert re.CTX_CLAMP_MARKER in ctx
    assert meta["char_clamped"] is True or meta["token_clamped"] is True


def test_llm_bypassed_when_no_evidence_and_use_llm_requested():
    eng = re.Eng(emb=None, ix={}, dbp={}, corp={}, ix_dim={}, corp_report={})
    res = re.run_query(eng, "test", use_llm=True)
    assert res["no_evidence"] is True
    assert res["meta"]["flags"]["llm_bypassed"] is True
    assert res["meta"]["flags"]["llm_used"] is False
    assert res["meta"]["err_llm"] is None
