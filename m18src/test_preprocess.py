from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m18src"))

import preprocess as P


def test_t1_normalizes_only_ephemeral_values():
    raw = ("S3 EC2 IPv4 SHA256 A100 GPT-4 CUDA 12.6 grpc 500 port 6333 "
           "550e8400-e29b-41d4-a716-446655440000 2026-09-12T20:00:01Z "
           "0x7ffeefbff5c0 0123456789abcdef0123456789abcdef pod-7d9f8c7b6c-abc12")
    got = P.normalize_t1(raw)
    for literal in ("S3", "EC2", "IPv4", "SHA256", "A100", "GPT-4", "12.6", "500", "6333"):
        assert literal in got
    for placeholder in P.PLACEHOLDERS.values():
        assert placeholder in got
    assert P.normalize_t1("Log-Structured-Merge-Database") == "Log-Structured-Merge-Database"


def test_collision_audit_rejects_frequent_cross_group_merge():
    rows = []
    for i in range(5):
        rows.append({"text": f"failure 550e8400-e29b-41d4-a716-44665544000{i}",
                     "relevance_group": "a" if i < 2 else "b"})
    report = P.collision_audit(rows)
    assert not report["pass"]
    assert report["ambiguous_high_frequency"]


def test_collision_audit_ignores_unchanged_shared_terms():
    rows = [{"text": "MMR", "relevance_group": f"thread-{i}"} for i in range(8)]
    report = P.collision_audit(rows)
    assert report["pass"] and not report["collisions"]


def test_t2_masks_only_continuation_digits():
    class Tok:
        def id_to_token(self, i):
            return {1: "##12", 2: "12", 3: "s3", 4: "##x"}[i]
    assert P.t2_mask_ids([1, 2, 3, 4], Tok()) == [2, 3, 4]
