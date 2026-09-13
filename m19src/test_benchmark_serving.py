from m19src import benchmark_serving


def test_collapse_enforces_passage_ties_and_exclusion_before_depth():
    indices = [0, 1, 2]
    scores = [1.0, 1.0, 0.5]
    artifacts = ["z-artifact", "a-artifact", "excluded"]
    passage_ids = ["z", "a", "first"]
    passages = [{"text": value} for value in passage_ids]
    result = benchmark_serving._collapse_top(
        indices, scores, artifacts, passage_ids, passages, excluded_artifacts={"excluded"})
    assert [row["passage_id"] for row in result["artifacts"]] == ["a", "z"]


def test_exact_top_resolves_ties_across_registered_depth_by_passage_id():
    scores = [1.0] * 501 + [0.5]
    passage_ids = [f"passage-{index:03d}" for index in reversed(range(502))]
    selected = benchmark_serving._exact_top_indices(
        scores, passage_ids, [False] * len(scores), depth=500)
    assert len(selected) == 500
    assert [passage_ids[index] for index in selected] == sorted(passage_ids[:501])[:500]
