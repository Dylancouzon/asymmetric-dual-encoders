"""E14-HEAD's verdict, as code.

The registered rule, applied literally:

  * ENDPOINTS: DENSE = out-of-domain macro over cqadup-programmers and cqadup-physics; FUSED = the
    4-component fused macro under the frozen operator. Both at int8 / sqrt.
  * COMPARATOR: `R0N`, seed-paired -- the same patched path with the head frozen at identity. NOT
    the existing R0 arms, because at W = 0 the head emits `normalize(d)` while R0 scores the raw
    cached fp16 vectors, and those are not exactly unit-norm.
  * BAR: mean-over-seeds gain >= 0.0040 on BOTH scalars, per head. Intersection-union within a
    treatment (which only lowers type-I), Holm across the two treatments.
  * The STEP-ADEQUACY gate comes first and it can veto a null: if doubling the step budget on the
    tuning seed still buys more than a quarter of what the second half of the 2,500-step budget
    bought, the primary reports OPTIMIZATION-INADEQUATE and NOT a method null. It can only turn a
    null into UNINFORMATIVE; it can never overturn a treatment that reached the bar.

HOW "HOLM ACROSS THE TWO TREATMENTS" IS IMPLEMENTED, since the registered bar is a THRESHOLD and
Holm needs p-values. The threshold rule is primary and decides the verdict. Holm is computed
alongside it, on one-sided p-values obtained the way every other bar in this milestone is reasoned
about -- against the MEASURED noise floor for that endpoint -- at family alpha = 0.05:

    p = Phi( -mean_gain / (sqrt(2) * sigma_A / sqrt(n_seeds)) )

`sigma_A` is the A-leg variance component fitted on the crossed 3x3 (`m8_noise_floor_crossed.json`)
for the dense endpoint, and the K=3 range floor divided by 1.693 for the fused endpoint, which is
all that was measured there. Two things are stated rather than buried: the fused sigma comes from a
sample RANGE at K = 3 and so pins sigma only to about a 12x span (CODEMAP pitfall 18), and
sqrt(2) * sigma_A treats the two arms as INDEPENDENT A legs even though they share a seed, which
is the conservative direction. A treatment must clear BOTH scalars, so its p-value entering Holm
is the WORSE of its two. At the registered bar these p-values are ~1e-6, so Holm cannot change a
verdict the threshold rule reaches; it is reported because the registration asks for it, not
because it is doing work. WHERE THE TWO DISAGREE, THE STRICTER GOVERNS.

Everything else here is DESCRIPTIVE and gates nothing: the mechanism control, the R0N-vs-R0
end-to-end null on the patch stack, and the per-seed spread.
"""
import argparse
import gzip
import json
import math
import sys

import numpy as np

import m8base
import probe_guard

REPO = m8base.REPO
RESULTS = REPO / "results"
OUT = RESULTS / "m8_e14_head.json"

PROBE = "E14-HEAD"
PREC, MODE = "int8", "sqrt"
DENSE_ENDPOINT = "out_of_domain_macro"
SEEDS = (0, 1, 2)
TREATMENTS = ("lin", "mlp")     # `lin` PRIMARY, `mlp` its nonlinearity control
COMPARATOR = "r0n"
ALPHA = 0.05
K3_RANGE_TO_SIGMA = 1.693       # E[range] / sigma at K = 3


def _arm(tag, seed):
    return f"m8e14-{tag}-s{seed}"


def ood_macro(per_component):
    """The DENSE endpoint: equal-weight mean over the registered out-of-domain group.

    Computed from `m8base.DEV_GROUPS` rather than through `noise_floor._group_vector`, which is
    the same arithmetic on the same two components but additionally REQUIRES all six dev
    components to be present, because it also builds the groups no endpoint here reads. Reading
    the endpoint directly means this stays correct whatever else a dump happens to carry, and the
    group membership still comes from one place (CODEMAP pitfall 12: never restate a constant that
    lives in a shared definition).
    """
    members = m8base.DEV_GROUPS["out-of-domain"]
    missing = [c for c in members if c not in per_component]
    if missing:
        raise SystemExit(f"the dense endpoint's components are absent from the dump: {missing}")
    return float(np.mean([float(np.mean(list(per_component[c].values()))) for c in members]))


def _dense_scalars(dump_path):
    raw = json.loads(gzip.open(dump_path).read())
    pq = raw["per_query"] if "per_query" in raw else raw
    out = {}
    for key, comps in pq.items():
        parts = key.split("|")
        if len(parts) < 2 or parts[-1] != PREC:
            continue
        rid, _, mode = parts[0].partition(":")
        if (mode or "mean") != MODE:
            continue
        out[rid] = ood_macro(comps)
    return out


def _fused_scalars(path):
    d = json.loads(open(path).read())
    out = {}
    for key, v in d["arm_macros"].items():
        rid, _, rest = key.partition(":")
        mode, _, prec = rest.partition("|")
        if prec == PREC and mode == MODE:
            out[rid] = float(v)
    return out


def sigmas():
    """The measured per-arm noise for each endpoint, read from the artifacts, never restated."""
    cx = json.loads((RESULTS / "m8_noise_floor_crossed.json").read_text())
    dense = float(cx["components"][f"{PREC}.{MODE}.{DENSE_ENDPOINT}"]["sd"]["A"])
    fl = json.loads((RESULTS / "m8_noise_floor_fused.json").read_text())
    fused_range = float(fl["floor"][f"{PREC}.{MODE}.fused_macro"])
    return {"dense": {"sigma_A": dense, "source": "crossed 3x3 fitted A-leg variance component"},
            "fused": {"sigma_A": fused_range / K3_RANGE_TO_SIGMA,
                      "source": f"K=3 range floor {fused_range:.6f} / {K3_RANGE_TO_SIGMA} -- a "
                                f"sample range at K=3 pins sigma only to about a 12x span "
                                f"(CODEMAP pitfall 18); read this p-value as an order of "
                                f"magnitude, not a rate"}}


def _p_one_sided(mean_gain, sigma_A, n):
    sd = math.sqrt(2.0) * sigma_A / math.sqrt(n)
    if sd <= 0:
        return None
    z = mean_gain / sd
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def contrasts(scal, bar, sig):
    """Per treatment, per scalar: the seed-paired gains over R0N and whether they clear the bar."""
    out = {}
    for t in TREATMENTS:
        out[t] = {}
        for which, vals in scal.items():
            gains, missing = {}, []
            for s in SEEDS:
                hi, lo = _arm(t, s), _arm(COMPARATOR, s)
                if hi not in vals or lo not in vals:
                    missing.append(hi if hi not in vals else lo)
                    continue
                gains[s] = vals[hi] - vals[lo]
            if missing:
                raise SystemExit(f"arms absent from the scored artifacts: {sorted(set(missing))}")
            mean = float(np.mean(list(gains.values())))
            out[t][which] = {
                "mean_gain": mean, "per_seed_gain": gains,
                "seed_signs_agree": bool(len({np.sign(g) for g in gains.values()}) == 1),
                "meets_bar": mean >= bar,
                "p_one_sided_vs_measured_floor": _p_one_sided(mean, sig[which]["sigma_A"],
                                                              len(gains)),
            }
    return out


def holm(pvals, alpha=ALPHA):
    """Step-down over the treatments. Reported beside the threshold rule, never instead of it."""
    order = sorted(pvals, key=lambda k: (pvals[k] is None, pvals[k]))
    m, out, still = len(order), {}, True
    for i, k in enumerate(order):
        thr = alpha / (m - i)
        rej = still and pvals[k] is not None and pvals[k] <= thr
        out[k] = {"p": pvals[k], "holm_threshold": thr, "rejects_null": bool(rej)}
        still = rej
    return out


def verdict_of(con, hp, inadequate):
    """The registered rule, in one place so every branch can be exercised on synthetic input.

    Order matters and is registered: the step-adequacy gate can only convert a NULL into
    UNINFORMATIVE. It can never take a treatment that CLEARS and reject it -- an arm that reached
    the bar has, by demonstration, been trained enough to reach the bar.
    """
    out = {}
    for t in TREATMENTS:
        clears = all(con[t][w]["meets_bar"] for w in ("dense", "fused"))
        if clears:
            out[t] = "CLEARS" if hp[t]["rejects_null"] else "CLEARS-THRESHOLD-NOT-HOLM"
        elif t in inadequate:
            # A null at an inadequate step budget is evidence about the configuration, not the
            # method (CLAUDE.md standing directive #2). This is why the gate is pre-registered.
            out[t] = "OPTIMIZATION-INADEQUATE"
        else:
            out[t] = "NULL"
    return out


def r0_scalars():
    """R0's two scalars, from THEIR OWN canonical artifacts.

    The first version looked for `m8nf-seed*` inside the E14 inputs. They are not there and never
    will be: the E14 scorer enumerates the nine E14 arms, so the registered R0N-vs-R0 null would
    have quietly reported "R0 arms not present" instead of being measured.
    """
    dense = _dense_scalars(RESULTS / "m7_devperquery_m8noise.json.gz")
    fused = _fused_scalars(RESULTS / "m8_noise_floor_fused.json")
    return {"dense": dense, "fused": fused}


def end_to_end_null(scal, sig):
    """R0N against the existing R0 arms: is the patch stack itself visible? DESCRIPTIVE."""
    r0 = r0_scalars()
    out = {}
    for which, vals in scal.items():
        gains, missing = {}, []
        for s in SEEDS:
            a, b = _arm(COMPARATOR, s), f"m8nf-seed{s}"
            if a in vals and b in r0[which]:
                gains[s] = vals[a] - r0[which][b]
            else:
                missing.append(a if a not in vals else b)
        if missing:
            out[which] = {"note": f"cannot form the registered null; absent: {sorted(missing)}"}
            continue
        mean = float(np.mean(list(gains.values())))
        out[which] = {
            "mean_delta": mean, "per_seed_delta": gains,
            "max_abs_delta": float(max(abs(g) for g in gains.values())),
            "sigma_A": sig[which]["sigma_A"],
            "within_one_sigma": bool(abs(mean) <= sig[which]["sigma_A"]),
            "_what": ("R0N differs from R0 only by renormalization of the cached document "
                      "vectors and by the patch stack around it. A delta far above the floor "
                      "would mean the stack does something the head does not."),
        }
    return out


def mechanism(arms):
    """Bag gain MINUS teacher-query gain, per arm. DESCRIPTIVE, and it does not gate the bar."""
    out = {}
    for rid in arms:
        p = RESULTS / f"m8e14_mechanism_{rid}.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())["macros"]
        def macro(k):
            return float(np.mean(list(m[k].values())))
        bag = macro("bag|headed") - macro("bag|raw")
        tea = macro("teacher|headed") - macro("teacher|raw")
        out[rid] = {"bag_gain": bag, "teacher_query_gain": tea,
                    "bag_specific": bag - tea, "macros": {k: macro(k) for k in m}}
    if out:
        out["_what"] = ("the head is supervised document-side metric learning and can win by "
                        "fixing the teacher's relevance geometry or by separating training "
                        "sources, neither of which requires documents to have been "
                        "bag-unreachable. bag_gain minus teacher_query_gain is the part that is "
                        "specific to the bag.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="merged per-query dump over the E14 arms")
    ap.add_argument("--fused", required=True, help="merged fused artifact over the E14 arms")
    ap.add_argument("--adequacy", required=True, help="e14_run adequacy verdict json")
    a = ap.parse_args()

    reg = probe_guard.registry()["probes"][PROBE]
    bar = float(reg["bar_frozen"]["bar"])
    sig = sigmas()

    dense, fused = _dense_scalars(a.dump), _fused_scalars(a.fused)
    scal = {"dense": dense, "fused": fused}
    con = contrasts(scal, bar, sig)

    adeq = json.loads(open(a.adequacy).read())
    inadequate = [t for t in TREATMENTS
                  if adeq.get(t, {}).get("verdict") != "ADEQUATE"]

    # A treatment must clear BOTH scalars, so its p-value is the WORSE of the two before Holm
    # runs across treatments. The first version passed only the dense p, which is not the
    # intersection-union rule the registration states.
    hp = holm({t: max(con[t][w]["p_one_sided_vs_measured_floor"] for w in ("dense", "fused"))
               for t in TREATMENTS})
    verdicts = verdict_of(con, hp, inadequate)

    mech = mechanism([_arm(t, s) for t in TREATMENTS for s in SEEDS])
    mech_arms = [k for k in mech if k != "_what"]
    positive = [t for t in TREATMENTS if verdicts[t] == "CLEARS"]

    primary = verdicts["lin"]
    if positive and len(mech_arms) < len(TREATMENTS) * len(SEEDS):
        # The mechanism control is what separates "documents became bag-reachable" from "supervised
        # document-side adaptation helps". A positive without it is not interpretable, so it is not
        # reported as one.
        headline = (f"{positive} cleared the bar, but the mechanism control is incomplete "
                    f"({len(mech_arms)} of {len(TREATMENTS) * len(SEEDS)} arms). The gain cannot "
                    f"yet be attributed to bag reachability rather than to generic supervised "
                    f"document-side adaptation, so no positive interpretation is issued.")
    elif any(v == "CLEARS-THRESHOLD-NOT-HOLM" for v in verdicts.values()):
        headline = ("A treatment cleared the threshold on both scalars but did not survive Holm "
                    "across the two treatments. Reported as unresolved, not as a positive.")
    elif positive:
        headline = ("A cheap renormalized doc-side head clears the bar on both scalars. The "
                    "document space IS re-shapeable toward bag reachability at this scale, which "
                    "is STRONG evidence for buying E14-LORA -- read the mechanism control before "
                    "attributing the gain to bag reachability rather than to generic supervised "
                    "document-side adaptation, and note both dense components are CQADupStack "
                    "forums, so the claim is a CQADupStack-family claim.")
    elif any(v == "OPTIMIZATION-INADEQUATE" for v in verdicts.values()):
        headline = ("The winning arm had not plateaued by 2,500 steps, so this reports "
                    "UNINFORMATIVE rather than a null. A null at an inadequate step budget is "
                    "evidence about the configuration, not about the method.")
    else:
        headline = reg["no_survivor"]

    out = {
        "_note": __doc__.strip().splitlines()[0],
        "read_at": f"{PREC} / {MODE}",
        "bar": bar,
        "dense_endpoint": DENSE_ENDPOINT,
        "fused_endpoint": "4-component fused_macro under the frozen operator",
        "comparator": "R0N (same patched path, head frozen at identity), seed-paired",
        "seeds": list(SEEDS),
        "arm_scalars": {w: {r: v for r, v in sorted(vals.items())} for w, vals in scal.items()},
        "contrasts": con,
        "multiplicity": {"within_treatment": "intersection-union over the two scalars",
                         "across_treatments": hp, "alpha": ALPHA,
                         "sigma_sources": sig,
                         "_note": "the threshold rule is primary and decides the verdict; where "
                                  "the two disagree the stricter governs"},
        "step_adequacy": adeq,
        "verdicts": verdicts,
        "primary_verdict_lin": primary,
        "headline": headline,
        "descriptive": {
            "end_to_end_null_R0N_vs_R0": end_to_end_null(scal, sig),
            "mechanism_control": mech,
        },
        "scope": reg["scope_limit"],
    }
    probe_guard.write_result(OUT, out, PROBE)
    print(json.dumps({"verdicts": verdicts, "bar": bar,
                      "mean_gains": {t: {w: con[t][w]["mean_gain"] for w in ("dense", "fused")}
                                     for t in TREATMENTS},
                      "headline": headline}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
