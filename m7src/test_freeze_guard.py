"""A frozen table must refuse to be scored against a different teacher.

Codex gate 2026-08-26, BLOCKER 4: FREEZE.json recorded teacher repo and revision but nothing ever
compared them with the encoder the scoring process had actually selected, and M7_ENCODER selects
that from the environment. So a table distilled against one teacher could be evaluated against
another -- silently, with every byte-level hash guard passing, in the one run that is allowed to
touch the six-set labels.

    ../.venv/bin/python test_freeze_guard.py
"""
import sys

import encoders
import freeze

FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def main():
    live = freeze.encoder_fingerprint()
    check(freeze.encoder_drift(live) == {}, "the active encoder does not drift from itself")

    other = encoders.get("arctic-embed-l" if live["name"] != "arctic-embed-l" else "bge-base-en-v1.5")
    frozen_other = freeze.encoder_fingerprint(other)
    drift = freeze.encoder_drift(frozen_other)
    check("name" in drift and "dim" in drift,
          f"a different encoder drifts on name and dim (got {sorted(drift)})")

    # The subtle case the byte hashes cannot catch: same repo and revision, different read-out.
    same_repo_diff_pooling = dict(live, pooling="mean" if live["pooling"] == "cls" else "cls")
    check(freeze.encoder_drift(same_repo_diff_pooling) == {"pooling": (
              same_repo_diff_pooling["pooling"], live["pooling"])},
          "a pooling change is caught even with repo and revision identical")

    for field, value in (("post_dense", "2_Dense_1024"), ("query_prefix", "something else"),
                         ("max_length", 128), ("tokenizer_id", "other"), ("cls_id", 0),
                         ("config_kwargs", {"unpad_inputs": False})):
        check(field in freeze.encoder_drift(dict(live, **{field: value})),
              f"{field} is part of the frozen identity")

    print(f"\n{len(FAIL)} failure(s)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
