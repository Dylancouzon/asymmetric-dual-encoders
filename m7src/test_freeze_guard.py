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

    # Pick a comparison encoder that genuinely differs in DIM from whatever is active, instead of
    # naming one. This test asserted drift on {name, dim} against a hardcoded arctic-embed-l, and
    # silently broke at the stella swap: stella and arctic-embed-l are both 1024-d, so `dim` no
    # longer drifted and a committed suite had been failing unnoticed ever since. A fixture that
    # depends on which encoder is active is a fixture that expires.
    other = next((encoders.get(n) for n in sorted(encoders.REGISTRY)
                  if encoders.get(n).dim != live["dim"]), None)
    assert other is not None, "no registered encoder differs in dim; widen the registry"
    drift = freeze.encoder_drift(freeze.encoder_fingerprint(other))
    check("name" in drift and "dim" in drift,
          f"a different-dim encoder ({other.name}) drifts on name and dim (got {sorted(drift)})")

    # The subtle case the byte hashes cannot catch: same repo and revision, different read-out.
    same_repo_diff_pooling = dict(live, pooling="mean" if live["pooling"] == "cls" else "cls")
    check(freeze.encoder_drift(same_repo_diff_pooling) == {"pooling": (
              same_repo_diff_pooling["pooling"], live["pooling"])},
          "a pooling change is caught even with repo and revision identical")

    # Sentinels, not plausible values. `post_dense` was probed with "2_Dense_1024", which IS
    # stella's real value -- so after the swap the "change" was a no-op and the check passed
    # vacuously until it started failing. A drift fixture must be a value no Spec can hold.
    for field, value in (("post_dense", "__sentinel_not_a_real_dense_dir__"),
                         ("query_prefix", "__sentinel_prefix__"),
                         ("max_length", -1), ("tokenizer_id", "__sentinel_tokenizer__"),
                         ("cls_id", -1), ("config_kwargs", {"__sentinel__": True})):
        assert live.get(field) != value, f"sentinel for {field} collides with the live Spec"
        check(field in freeze.encoder_drift(dict(live, **{field: value})),
              f"{field} is part of the frozen identity")

    print(f"\n{len(FAIL)} failure(s)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
