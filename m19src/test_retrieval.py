import numpy as np
import pytest

from m19src import retrieval


def _passage(i, artifact=None, digest=None):
    return {"passage_id": f"p{i:03d}", "artifact_id": artifact or f"a{i:03d}",
            "normalized_text_sha256": digest or f"h{i:03d}", "text": f"text {i}"}


def test_exclusion_precedes_truncation_and_collapse_keeps_best_passage():
    passages = [_passage(i) for i in range(505)]
    scores = np.arange(505, 0, -1, dtype=np.float64)
    # Excluding four head passages must backfill through p503 before the registered 500 cut.
    out = retrieval.collapse_passage_scores(
        scores, passages, excluded_artifacts={f"a{i:03d}" for i in range(4)}
    )
    assert out["exclusion_before_truncation"]
    assert out["scored_passages"] == 500
    assert out["artifacts"][0]["passage_id"] == "p004"
    assert out["unique_artifacts"] == 100

    duplicate = [_passage(0, "same"), _passage(1, "same"), _passage(2, "other")]
    small = retrieval.collapse_passage_scores([1.0, 2.0, 0.5], duplicate)
    assert small["artifacts"][0]["passage_id"] == "p001"
    assert small["shortfall"] == 98


def test_score_and_passage_id_ties_are_deterministic():
    passages = [
        {"passage_id": "p2", "artifact_id": "b"},
        {"passage_id": "p1", "artifact_id": "a"},
    ]
    out = retrieval.collapse_passage_scores([1.0, 1.0], passages)
    assert [row["passage_id"] for row in out["artifacts"]] == ["p1", "p2"]
    with pytest.raises(ValueError, match="registered"):
        retrieval.collapse_passage_scores([1.0, 1.0], passages, passage_depth=100)


def test_dbsf_matches_sample_std_and_preserves_route_passages():
    dense = {"artifacts": [
        {"artifact_id": "a", "score": 3.0, "passage_id": "da", "passage": {"text": "dense a"}},
        {"artifact_id": "b", "score": 1.0, "passage_id": "db", "passage": {"text": "dense b"}},
    ]}
    lexical = {"artifacts": [
        {"artifact_id": "b", "score": 4.0, "passage_id": "lb", "passage": {"text": "lex b"}},
        {"artifact_id": "c", "score": 2.0, "passage_id": "lc", "passage": {"text": "lex c"}},
    ]}
    fused = retrieval.dbsf_artifacts({"dense": dense, "lexical": lexical})
    assert fused[0]["artifact_id"] == "b"
    assert set(fused[0]["route_support"]) == {"dense", "lexical"}
    assert fused[0]["route_support"]["dense"]["passage_id"] == "db"


def test_exact_dense_operates_at_artifact_level():
    documents = np.eye(3, dtype=np.float32)
    passages = [_passage(0, "same"), _passage(1, "same"), _passage(2, "other")]
    out = retrieval.exact_dense_artifacts(np.array([1, 0, 0], np.float32), documents, passages)
    assert [row["artifact_id"] for row in out["artifacts"]] == ["same", "other"]
