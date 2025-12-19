import rag_engine as re

def test_llm_call_respects_budgets():
    prompt = "word " * 1000
    out = re.llm_call(prompt, cfg={"char_budget": 120, "tok_budget": 12})
    assert len(out) <= 120
    assert len(out.split()) <= 12


def test_llm_bypassed_when_no_evidence_and_use_llm_requested():
    eng = re.Eng(emb=None, ix={}, dbp={}, corp={}, ix_dim={}, corp_report={})
    res = re.run_query(eng, "test", use_llm=True)
    assert res["no_evidence"] is True
    assert res["meta"]["flags"]["llm_bypassed"] is True
