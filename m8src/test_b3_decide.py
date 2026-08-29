"""B3's verdict logic, exercised on every branch it can take.

The rule is a four-way gate evaluated in a fixed order, and three of the four outcomes are ones a
session would rather not get -- which is exactly why each is tested here rather than trusted to be
read correctly on the night the numbers land. Fabricated scalars only: nothing here touches a
trained arm, and the point is the DECISION, not the data.
"""
import json
import sys
import types

import numpy as np

import m8base
import b3_decide
import b3_pool
import probe_guard

BAR = 0.0040
SEEDS = list(b3_pool.SEEDS)


def _scalars(gain_vs_025, gain_vs_050, base=0.30, jitter=0.0):
    """Build {run_id: value} so that 1.00 beats 0.25 by `gain_vs_025` and 0.50 by `gain_vs_050`."""
    vals = {}
    for i, s in enumerate(SEEDS):
        j = jitter * (1 if i == 0 else -1)
        vals[b3_pool.arm_id(0.25, s)] = base
        vals[b3_pool.arm_id(0.50, s)] = base + (gain_vs_025 - gain_vs_050)
        vals[b3_pool.arm_id(0.75, s)] = base + gain_vs_025 - gain_vs_050 / 2
        vals[b3_pool.arm_id(1.00, s)] = base + gain_vs_025 + j
    return vals


def _run(dense, fused, counts=None, problems=()):
    """Drive main() with fabricated scalars, returning the verdict payload."""
    counts = counts or {0.25: 85212, 0.5: 170425, 0.75: 255637, 1.0: 340850}
    saved = (b3_decide._dense_scalars, b3_decide._fused_scalars, b3_pool.collect,
             probe_guard.write_result, sys.argv)
    captured = {}
    b3_decide._dense_scalars = lambda p: {r: {b3_decide.DENSE_ENDPOINT: v,
                                              "group_vector_median": v} for r, v in dense.items()}
    b3_decide._fused_scalars = lambda p: dict(fused)
    b3_pool.collect = lambda: {"problems": list(problems),
                               "pair_counts_by_fraction": counts, "arms": {}}
    probe_guard.write_result = lambda path, payload, pid, **kw: captured.update(payload)
    sys.argv = ["b3_decide", "--dump", "x", "--fused", "y"]
    import contextlib, io
    try:
        with contextlib.redirect_stdout(io.StringIO()):     # main() prints its verdict payload
            b3_decide.main()
    finally:
        (b3_decide._dense_scalars, b3_decide._fused_scalars, b3_pool.collect,
         probe_guard.write_result, sys.argv) = saved
    return captured


def t_pass_requires_both_scalars():
    """PASS only when the PRIMARY (1.00 vs 0.50) clears the bar on dense AND fused."""
    big = _scalars(0.030, 0.020)                       # clears manipulation and primary
    assert _run(big, big)["verdict"] == "PASS"
    weak_fused = _scalars(0.030, 0.001)                # fused primary under the bar
    assert _run(big, weak_fused)["verdict"] == "FAIL", "one scalar under the bar must not PASS"
    assert _run(weak_fused, big)["verdict"] == "FAIL", "the conjunction must hold both ways"


def t_manipulation_gate_precedes_the_primary():
    """A 4x dose that moves nothing is UNINFORMATIVE, never FAIL -- the gate runs first."""
    flat = _scalars(0.0005, 0.0002)
    out = _run(flat, flat)
    assert out["verdict"] == "UNINFORMATIVE", out["verdict"]
    # and it must NOT be reported as evidence of no starvation by accident
    assert "uninformative" not in out["headline"].lower() or "deprioritis" in out["headline"]


def t_fail_only_after_the_gate_passes():
    """Manipulation clears, primary does not -> FAIL, worded as the registry allows."""
    d = _scalars(0.030, 0.001)
    out = _run(d, d)
    assert out["verdict"] == "FAIL", out["verdict"]
    h = out["headline"].lower()
    for banned in probe_guard.registry()["probes"]["B3"]["fail_wording_prohibits"]:
        assert banned.lower() not in h, f"FAIL headline says a prohibited thing: {out['headline']}"
    assert "0.0040" in out["headline"] or "half the pool" in h


def t_negative_dose_is_its_own_verdict():
    """A dose that HURTS by at least the bar is a finding, not a FAIL and not UNINFORMATIVE."""
    d = _scalars(-0.030, -0.020)
    out = _run(d, d)
    assert out["verdict"] == "NEGATIVE-DOSE", out["verdict"]
    # checked BEFORE the manipulation gate, which a negative gain would otherwise fail
    assert "worse" in out["headline"].lower()


def t_negative_on_one_scalar_only_is_not_negative_dose():
    """Both scalars must agree before the pool-quality claim is made."""
    neg, pos = _scalars(-0.030, -0.020), _scalars(0.030, 0.020)
    assert _run(neg, pos)["verdict"] != "NEGATIVE-DOSE"


def t_sign_agreement_is_reported_but_never_gates():
    """A seed disagreeing in sign must not change a PASS -- the rule dropped it as a gate."""
    # jitter exceeds the per-seed gain, so seeds 1 and 2 go NEGATIVE while the mean
    # stays comfortably above the bar -- the exact case a sign gate would have vetoed.
    d = _scalars(0.030, 0.020, jitter=0.025)
    out = _run(d, d)
    agree = out["descriptive"]["seed_sign_agreement"]["primary_1.00_vs_0.50"]["dense"]
    assert agree is False, "the fixture was meant to make the seeds disagree"
    assert out["verdict"] == "PASS", "sign disagreement must not veto: it is reported, not a gate"


def t_refuses_arms_that_are_not_a_dose_curve():
    """collect()'s problems must stop scoring, not be scored around."""
    d = _scalars(0.030, 0.020)
    try:
        _run(d, d, problems=["m8b3-p050-s1: NO sidecar"])
    except SystemExit as e:
        assert "not a dose curve" in str(e), str(e)
    else:
        raise AssertionError("a broken arm set was scored anyway")


def t_slope_is_descriptive_and_positive_when_the_curve_rises():
    d = _scalars(0.030, 0.020)
    out = _run(d, d)
    assert out["descriptive"]["slope_per_doubling_of_pairs"]["dense"] > 0


def t_the_1_00_arms_are_the_adopted_floor_arms():
    """The registration says the f=1.00 arms ARE m8nf-seed0/1/2; the code must agree."""
    assert [b3_pool.arm_id(1.00, s) for s in SEEDS] == ["m8nf-seed0", "m8nf-seed1", "m8nf-seed2"]
    assert b3_pool.arm_id(0.50, 1) == "m8b3-p050-s1"


def main():
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("t_") or not isinstance(fn, types.FunctionType):
            continue
        try:
            fn()
            print(f"  PASS  {name[2:]}")
        except Exception as e:                                          # noqa: BLE001
            fails += 1
            print(f"  FAIL  {name[2:]}: {type(e).__name__}: {e}")
    print(f"{fails} FAILURES" if fails else "all B3 verdict checks pass")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
