from app.retrieve import reciprocal_rank_fusion


def test_rrf_rewards_agreement():
    # id=1 ranks high in both lists -> should come first after fusion.
    vec = [{"id": 1, "doc": "a", "content": "x"}, {"id": 2, "doc": "b", "content": "y"}]
    kw = [{"id": 1, "doc": "a", "content": "x"}, {"id": 3, "doc": "c", "content": "z"}]
    fused = reciprocal_rank_fusion([vec, kw])
    assert fused[0]["id"] == 1


def test_rrf_deduplicates():
    vec = [{"id": 1, "doc": "a", "content": "x"}]
    kw = [{"id": 1, "doc": "a", "content": "x"}]
    fused = reciprocal_rank_fusion([vec, kw])
    assert len(fused) == 1
