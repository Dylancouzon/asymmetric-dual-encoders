"""Every branch of E14-HEAD's verdict, on SYNTHESIZED input.

`m8/CODEMAP.md` pitfall 17: a test that iterates whatever the data happens to contain asserts
nothing when the data is empty, and passes. So none of these read a real artifact -- each case is
constructed so the branch under test is the branch that runs, and each is paired with an input
that makes it FAIL, because a check whose failing input you cannot name is not a check yet.
"""
import sys

import e14_decide as D

FAILS = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else '  FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def con_of(lin=(0.01, 0.01), mlp=(0.01, 0.01)):
    """A contrasts() blob with the given (dense, fused) mean gains, bar-compared at 0.0040."""
    bar = 0.0040
    out = {}
    for t, (d, f) in (("lin", lin), ("mlp", mlp)):
        out[t] = {"dense": {"mean_gain": d, "meets_bar": d >= bar},
                  "fused": {"mean_gain": f, "meets_bar": f >= bar}}
    return out


def hp_of(lin=True, mlp=True):
    return {"lin": {"rejects_null": lin}, "mlp": {"rejects_null": mlp}}


# ---- the verdict branches -----------------------------------------------------------------
v = D.verdict_of(con_of(lin=(0.01, 0.01)), hp_of(), inadequate=[])
check("both scalars over the bar and Holm rejects -> CLEARS", v["lin"] == "CLEARS", v["lin"])

v = D.verdict_of(con_of(lin=(0.01, 0.001)), hp_of(), inadequate=[])
check("INTERSECTION-UNION: dense clears but fused does not -> NOT a clear",
      v["lin"] == "NULL", v["lin"])

v = D.verdict_of(con_of(lin=(0.001, 0.01)), hp_of(), inadequate=[])
check("intersection-union holds the other way round too", v["lin"] == "NULL", v["lin"])

v = D.verdict_of(con_of(lin=(0.01, 0.01)), hp_of(lin=False), inadequate=[])
check("threshold clears but Holm does not -> flagged, not silently promoted",
      v["lin"] == "CLEARS-THRESHOLD-NOT-HOLM", v["lin"])

v = D.verdict_of(con_of(lin=(0.001, 0.001)), hp_of(), inadequate=["lin"])
check("a NULL under an inadequate step budget reports UNINFORMATIVE, not a method null",
      v["lin"] == "OPTIMIZATION-INADEQUATE", v["lin"])

v = D.verdict_of(con_of(lin=(0.01, 0.01)), hp_of(), inadequate=["lin"])
check("the adequacy gate can NEVER overturn a treatment that reached the bar",
      v["lin"] == "CLEARS", v["lin"])

v = D.verdict_of(con_of(lin=(0.001, 0.001)), hp_of(), inadequate=[])
check("a plain null with an adequate budget is a NULL", v["lin"] == "NULL", v["lin"])

v = D.verdict_of(con_of(lin=(0.01, 0.01), mlp=(0.001, 0.001)), hp_of(), inadequate=[])
check("the two treatments are judged independently",
      v["lin"] == "CLEARS" and v["mlp"] == "NULL", f"{v}")

# exactly AT the bar must clear: the registered rule is `>=`, and a rule that silently became `>`
# would move the bar without an amendment
v = D.verdict_of(con_of(lin=(0.0040, 0.0040)), hp_of(), inadequate=[])
check("a gain exactly AT the bar clears (the rule is >=)", v["lin"] == "CLEARS", v["lin"])

# ---- Holm ---------------------------------------------------------------------------------
h = D.holm({"lin": 0.001, "mlp": 0.002}, alpha=0.05)
check("Holm: the smaller p is tested at alpha/2", abs(h["lin"]["holm_threshold"] - 0.025) < 1e-12)
check("Holm: the larger p is tested at alpha", abs(h["mlp"]["holm_threshold"] - 0.05) < 1e-12)
check("Holm: both reject when both are small",
      h["lin"]["rejects_null"] and h["mlp"]["rejects_null"])

h = D.holm({"lin": 0.30, "mlp": 0.001}, alpha=0.05)
check("Holm sorts ASCENDING: the small p is step 1 and rejects, the large p is step 2 and does not",
      h["mlp"]["rejects_null"] and not h["lin"]["rejects_null"], f"{h}")

# The step-down property itself, which needs a case where the LATER test would pass on its own.
# 0.04 <= alpha=0.05, so an uncorrected reading would reject it; Holm must not, because the
# smaller p already failed its own stricter threshold.
h = D.holm({"lin": 0.03, "mlp": 0.04}, alpha=0.05)
check("Holm STEP-DOWN: failing at step 1 (0.03 > alpha/2) stops step 2, even though 0.04 <= alpha "
      "and would reject on its own",
      not h["lin"]["rejects_null"] and not h["mlp"]["rejects_null"], f"{h}")

h = D.holm({"lin": None, "mlp": 0.001}, alpha=0.05)
check("Holm: a missing p-value sorts last and never rejects", not h["lin"]["rejects_null"])

# ---- the p-value against the measured floor -------------------------------------------------
sig = 0.0010648942508266286      # the fitted A-leg sd for the dense endpoint
p_bar = D._p_one_sided(0.0040, sig, 3)
p_zero = D._p_one_sided(0.0, sig, 3)
p_half = D._p_one_sided(0.0020, sig, 3)
check("p at zero gain is 0.5", abs(p_zero - 0.5) < 1e-9, f"{p_zero}")
check("p is monotone in the gain", p_bar < p_half < p_zero, f"{p_bar} {p_half} {p_zero}")
check("at the registered bar the p-value is tiny, so Holm cannot be what decides",
      p_bar < 1e-4, f"p={p_bar:.3g}")
check("a zero sigma returns None rather than dividing by it", D._p_one_sided(0.01, 0.0, 3) is None)

# ---- contrasts pairs by seed and refuses a partial arm set ----------------------------------
vals = {}
for s in D.SEEDS:
    vals[f"m8e14-r0n-s{s}"] = 0.30
    vals[f"m8e14-lin-s{s}"] = 0.30 + 0.005 * (s + 1)
    vals[f"m8e14-mlp-s{s}"] = 0.30
sig_blob = {"dense": {"sigma_A": sig}, "fused": {"sigma_A": sig}}
c = D.contrasts({"dense": vals, "fused": vals}, 0.0040, sig_blob)
g = c["lin"]["dense"]["per_seed_gain"]
check("contrasts pairs each treatment arm with the R0N arm of the SAME seed",
      sorted(g) == [0, 1, 2] and all(abs(g[s] - 0.005 * (s + 1)) < 1e-12 for s in g), f"{g}")
check("mean over seeds is the mean of the paired gains",
      abs(c["lin"]["dense"]["mean_gain"] - 0.010) < 1e-12)
check("a zero-gain treatment does not meet the bar", not c["mlp"]["dense"]["meets_bar"])

partial = dict(vals)
del partial["m8e14-lin-s1"]
try:
    D.contrasts({"dense": partial, "fused": partial}, 0.0040, sig_blob)
    check("a missing arm raises rather than averaging over the survivors", False)
except SystemExit as e:
    check("a missing arm raises rather than averaging over the survivors",
          "m8e14-lin-s1" in str(e), str(e)[:70])

print(f"\n{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
sys.exit(1 if FAILS else 0)
