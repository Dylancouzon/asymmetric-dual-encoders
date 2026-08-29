"""Codex B3 evidence: the sign-flip p-value controls type-I error at the actual data shapes.

Null construction: take real frozen per-query vectors (results/perquery.json) for the three
confirmatory-style comparator pairs, then make H0 true BY CONSTRUCTION with a paired label swap --
per query, swap (a_q, b_q) with probability 1/2. That preserves marginals, pairing, ties, zeros
and per-dataset n exactly; any rejection is then a false positive.

Verifies: (1) per-test P(p <= alpha) ~= alpha at 0.025 and 0.05; (2) FWER of Holm over the
3-test family <= 0.025 within binomial noise; (3) power sanity -- the real (un-swapped)
lr-dense-pertask vs bm25 delta must hit the p floor. Writes results/m7_signflip_calibration.json.
"""
import json
import sys
import time

import numpy as np

import boot
from _paths import REPO

PAIRS = [("lr-dense-pertask", "bm25"),
         ("opensearch-doc-v3-gte", "lr-dense-pertask"),
         ("bge-small-en-v1.5", "opensearch-doc-v3-gte")]
S = 1000          # simulations
R = 10_000        # flips per test; min p 1/(R+1) ~ 1e-4 << Holm's tightest 0.00833
ALPHA = 0.025


def swap_null(a, b, rng):
    A, B = {}, {}
    for ds in a:
        A[ds], B[ds] = {}, {}
        flip = rng.integers(0, 2, size=len(a[ds])).astype(bool)
        for f, q in zip(flip, sorted(a[ds])):
            A[ds][q], B[ds][q] = (b[ds][q], a[ds][q]) if f else (a[ds][q], b[ds][q])
    return A, B


def main():
    t0 = time.time()
    blob = json.loads((REPO / "results" / "perquery.json").read_text())
    data = [(boot.from_perquery_json(blob, x), boot.from_perquery_json(blob, y))
            for x, y in PAIRS]

    rng = np.random.default_rng(7)
    rej_any, per_test = 0, np.zeros(len(PAIRS))
    ps_all = []
    for s in range(S):
        pvals = {}
        for i, (a, b) in enumerate(data):
            A, B = swap_null(a, b, rng)
            p = boot.signflip(A, B, R=R, seed=int(rng.integers(2**31)),
                              alternative="greater")["p"]
            pvals[i] = p
            ps_all.append(p)
            per_test[i] += p <= ALPHA
        if any(v["reject"] for v in boot.holm(pvals, alpha=ALPHA).values()):
            rej_any += 1
        if (s + 1) % 100 == 0:
            print(f"  {s+1}/{S} sims, FWER so far {rej_any/(s+1):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    ps_all = np.array(ps_all)
    fwer = rej_any / S
    se = float(np.sqrt(ALPHA * (1 - ALPHA) / S))
    power = boot.signflip(*data[0], R=R, alternative="greater")
    out = {
        "S": S, "R": R, "alpha": ALPHA, "pairs": [f"{a} vs {b}" for a, b in PAIRS],
        "fwer_holm": fwer, "binomial_se_at_alpha": round(se, 4),
        "fwer_ok": bool(fwer <= ALPHA + 3 * se),
        "per_test_rate_at_alpha": [round(float(x / S), 4) for x in per_test],
        "p_deciles": [round(float(x), 3) for x in np.percentile(ps_all, range(10, 100, 10))],
        "power_sanity_lrdense_vs_bm25": {"p": power["p"], "p_str": power["p_str"],
                                         "delta": power["delta"],
                                         "at_floor": bool(power["p"] <= 1 / (1 + R))},
        "seconds": round(time.time() - t0, 1),
    }
    (REPO / "results" / "m7_signflip_calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    ok = out["fwer_ok"] and out["power_sanity_lrdense_vs_bm25"]["at_floor"]
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
