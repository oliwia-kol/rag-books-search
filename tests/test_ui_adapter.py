import ui_adapter as ua


def test_clamp_answer_truncates_to_five_sentences():
    ans = "One. Two. Three. Four. Five. Six. Seven."
    res = ua._clamp_answer(ans, max_sents=5)
    assert res["truncated"] is True
    assert res["text"].count(".") == 5
    assert "Six" not in res["text"]


def test_stitch_hits_preview_uses_multiple_hits():
    hits = [{"text": "Alpha text"}, {"text": "Beta text"}, {"text": "Gamma text"}]
    preview = ua._stitch_hits_preview(hits, mx_hits=2)
    assert "Alpha" in preview and "Beta" in preview
    assert "Gamma" not in preview


def test_confidence_state_mapping():
    high = ua._confidence_state("HIGH", 0.82, {"mx": 0.9, "uc": 3, "std": 0.05})
    med = ua._confidence_state("OK", 0.5, {"mx": 0.6, "uc": 1, "std": 0.2})
    low = ua._confidence_state("WEAK", 0.2, {"mx": 0.3, "uc": 0, "std": 0.4})
    assert high == "HIGH"
    assert med == "MEDIUM"
    assert low == "LOW"


def test_coverage_counts_detects_single_source():
    hits = [
        {"book": "book-a", "publisher": "pub-a", "section": "s1"},
        {"book": "book-a", "publisher": "pub-a", "section": "s2"},
    ]
    counts = ua._coverage_counts(hits, meta={"n": {"uniq_books": 1, "uniq_sections": 2}})
    assert counts["books"] == 1
    assert counts["sections"] == 2
    assert counts["single_source"] is True


def test_snippet_html_highlights_terms_and_clamps():
    hit = {"text": "This is a small example of highlighted text with terms sprinkled throughout the snippet for testing."}
    out = ua._snippet_html(hit, q="example terms", mx=40)
    assert '<span class="hit-term">' in out
    assert len(out) <= 300  # includes markup overhead
