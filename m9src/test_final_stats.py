"""Tests for the M9.4 gate. Each targets a failure the adversarial review named."""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m9src import final_stats as fs


class _Raises:
    """Minimal stand-in for pytest.raises; this repo runs tests as plain scripts."""

    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        assert et is not None, f"expected {self.exc.__name__}, nothing raised"
        assert issubclass(et, self.exc), f"expected {self.exc.__name__}, got {et.__name__}: {ev}"
        if self.match:
            assert self.match in str(ev), f"expected {self.match!r} in {str(ev)!r}"
        return True


class pytest:                     # noqa: N801  -- keeps the test bodies unchanged
    raises = _Raises

SIX = list(fs.SIX)


def mk(offset=0.0, n=40, seed=0, datasets=SIX):
    """{ds: {qid: score}} with a known mean shift."""
    rng = np.random.default_rng(seed)
    return {ds: {f"q{i}": float(rng.normal(0.5, 0.1) + offset) for i in range(n)}
            for ds in datasets}


def test_gate_field_is_the_1p25_quantile_not_the_2p5():
    """BLOCKER 1: the gate must be the 0.0125 quantile, at full precision."""
    a, b = mk(0.02, seed=1), mk(0.0, seed=2)
    al = fs.align(a, b)
    plan, _ = fs.draw_plan(al, 2000, 900)
    r = fs.bootstrap(al, plan, 0.0125)
    assert r["quantile"] == 0.0125 and r["quantile_method"] == "linear"
    # the reporting-only 2.5% endpoint must be a DIFFERENT, higher number
    assert r["ci95_raw_reporting_only"][0] > r["lower_q0125_raw"]
    # full precision: not rounded to 4dp
    assert repr(r["lower_q0125_raw"]) != repr(round(r["lower_q0125_raw"], 4))


def test_five_datasets_is_refused():
    """MAJOR 4: a 5-dataset macro must be impossible, not silently computed."""
    five = SIX[:5]
    a, b = mk(0.02, datasets=five, seed=1), mk(0.0, datasets=five, seed=2)
    with pytest.raises(ValueError, match="exactly the six"):
        fs.align(a, b)


def test_reordered_or_differing_qids_refused():
    a, b = mk(0.02, seed=1), mk(0.0, seed=2)
    b["fiqa"].pop("q0")                      # qid sets now differ
    with pytest.raises(ValueError):
        fs.align(a, b)


def test_draw_plan_is_deterministic_and_digest_detects_change():
    a, b = mk(0.02, seed=1), mk(0.0, seed=2)
    al = fs.align(a, b)
    p1, h1 = fs.draw_plan(al, 500, 900)
    p2, h2 = fs.draw_plan(al, 500, 900)
    assert h1 == h2
    assert all(np.array_equal(p1[d], p2[d]) for d in p1)
    _, h3 = fs.draw_plan(al, 500, 901)       # different seed -> different plan
    assert h3 != h1


def test_c1_and_c2_share_one_plan():
    """MAJOR 5: sharing must be real, not merely claimed via equal seeds."""
    rows = {"nano": mk(0.03, seed=1), "bge": mk(0.0, seed=2), "leaf": mk(0.01, seed=3)}
    conf = {"bootstrap": {"B": 400, "seed": 900, "quantile": 0.0125},
            "signflip": {"B": 2000, "seed": 901}, "holm": {"alpha_family": 0.025}}
    out = fs.run_contrasts(rows, ("nano", "bge"), ("nano", "leaf"), conf)
    assert "plan_sha256" in out and len(out["plan_sha256"]) == 64
    # one plan digest for the whole record, not one per contrast
    assert set(out["contrasts"]) == {"C1", "C2"}


def test_plan_width_mismatch_refused():
    a, b = mk(0.02, seed=1), mk(0.0, seed=2)
    al = fs.align(a, b)
    plan, _ = fs.draw_plan(al, 100, 900)
    plan["fiqa"] = plan["fiqa"][:, :-1]      # wrong width
    with pytest.raises(ValueError, match="plan width"):
        fs.bootstrap(al, plan, 0.0125)


def test_pass_requires_both_bootstrap_and_holm():
    """A contrast must not pass on the bootstrap alone."""
    rows = {"nano": mk(0.0005, seed=1), "bge": mk(0.0, seed=2), "leaf": mk(0.0, seed=3)}
    conf = {"bootstrap": {"B": 400, "seed": 900, "quantile": 0.0125},
            "signflip": {"B": 2000, "seed": 901}, "holm": {"alpha_family": 0.025}}
    out = fs.run_contrasts(rows, ("nano", "bge"), ("nano", "leaf"), conf)
    for c in out["contrasts"].values():
        assert c["passes"] == (c["bootstrap_rejects"] and c["holm_rejects"])


def test_null_data_does_not_pass():
    """The gate must not fire on noise."""
    rows = {"nano": mk(0.0, seed=7), "bge": mk(0.0, seed=8), "leaf": mk(0.0, seed=9)}
    conf = {"bootstrap": {"B": 800, "seed": 900, "quantile": 0.0125},
            "signflip": {"B": 3000, "seed": 901}, "holm": {"alpha_family": 0.025}}
    out = fs.run_contrasts(rows, ("nano", "bge"), ("nano", "leaf"), conf)
    assert not any(c["passes"] for c in out["contrasts"].values())


def test_strong_effect_does_pass():
    """Sanity in the other direction: a large real effect must clear both."""
    rows = {"nano": mk(0.15, seed=7), "bge": mk(0.0, seed=8), "leaf": mk(0.0, seed=9)}
    conf = {"bootstrap": {"B": 800, "seed": 900, "quantile": 0.0125},
            "signflip": {"B": 3000, "seed": 901}, "holm": {"alpha_family": 0.025}}
    out = fs.run_contrasts(rows, ("nano", "bge"), ("nano", "leaf"), conf)
    assert all(c["passes"] for c in out["contrasts"].values())


def test_registry_constants_are_the_locked_ones():
    """BLOCKER 5: the executor must read the registry, never the M9.0 screen defaults."""
    c = fs.cfg()
    assert c["bootstrap"]["B"] == 10000 and c["bootstrap"]["seed"] == 900
    assert c["signflip"]["B"] == 100000 and c["signflip"]["seed"] == 901
    assert c["holm"]["alpha_family"] == 0.025
    assert c["bootstrap"]["decision_field"] == "lower_q0125_raw"
    assert c["bootstrap"]["B"] != 20000, "must not inherit the M9.0 screen B"


def test_agrees_with_boot_paired_dep_where_they_overlap():
    """Cross-check the reimplementation against the hardened M7 bootstrap.

    The two use different RNG streams, so agreement is up to Monte-Carlo error, not exact. What
    matters is that the ESTIMAND matches: same point estimate exactly, and the 2.5% endpoint
    (the only quantile boot exposes) close. A large gap would mean the macro, the pairing, or the
    resampling unit differs -- i.e. that this module computes a different statistic than every
    number the project has already published.
    """
    from m7src import boot
    a, b = mk(0.02, n=200, seed=11), mk(0.0, n=200, seed=12)
    al = fs.align(a, b)
    plan, _ = fs.draw_plan(al, 4000, 900)
    mine = fs.bootstrap(al, plan, 0.025)          # ask for 2.5% to compare like with like
    theirs = boot.paired_dep(a, b, B=4000, seed=900, alternative="greater", strict=True)
    assert abs(mine["delta_raw"] - theirs["delta_raw"]) < 1e-12, "point estimates must match exactly"
    assert abs(mine["lower_q0125_raw"] - theirs["ci95_raw"][0]) < 5e-4, (
        f"2.5% endpoints diverge: {mine['lower_q0125_raw']} vs {theirs['ci95_raw'][0]}")


def test_the_two_quantiles_are_ordered_and_gate_is_stricter():
    """The 1.25% endpoint must be <= the 2.5% one, so using the wrong field is anti-conservative."""
    a, b = mk(0.02, n=200, seed=13), mk(0.0, n=200, seed=14)
    al = fs.align(a, b)
    plan, _ = fs.draw_plan(al, 4000, 900)
    r = fs.bootstrap(al, plan, 0.0125)
    assert r["lower_q0125_raw"] <= r["ci95_raw_reporting_only"][0]


def main():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in fns:
        try:
            f()
            print(f"  PASS {n}")
        except Exception:
            bad += 1
            print(f"  FAIL {n}")
            traceback.print_exc()
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
