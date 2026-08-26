"""Review #2 MAJOR 8: sign-flip assumes per-query symmetry under H0; the confirmatory claim is
about the WEAK null (macro mean <= 0), and real nDCG differences are skewed. This measures the
damage empirically at our exact data shapes.

Null construction: take the real per-query differences of a confirmatory-style pair, center each
dataset's differences to mean zero (weak null true BY CONSTRUCTION, skew/boundedness preserved,
sharp exchangeability null false). Per simulation, resample queries with replacement within each
dataset from that centered pool and run signflip. Reports P(p <= alpha) per alpha.

The tier decision does not rest on signflip alone -- it is Holm(sign-flip) AND bootstrap CI > 0
(see final_run.py) -- so the number here sizes the residual risk of the conjunction's first leg.
Writes results/m7_signflip_weaknull.json.
"""
import json
import time

import numpy as np

import boot
from _paths import REPO

PAIRS = [("lr-dense-pertask", "bm25"), ("opensearch-doc-v3-gte", "lr-dense-pertask")]
S = 1000
R = 2000


def main():
    t0 = time.time()
    blob = json.loads((REPO / "results" / "perquery.json").read_text())
    out = {}
    rng = np.random.default_rng(11)
    for x, y in PAIRS:
        A = boot.from_perquery_json(blob, x)
        B = boot.from_perquery_json(blob, y)
        pairs = boot._align(A, B)
        centered = {ds: (da - db) - (da - db).mean() for ds, (da, db) in pairs.items()}
        skew = {ds: round(float(((v - v.mean()) ** 3).mean() / (v.std() ** 3 + 1e-12)), 2)
                for ds, v in centered.items()}
        ps = []
        for s in range(S):
            A2, B2 = {}, {}
            for ds, v in centered.items():
                draw = v[rng.integers(0, v.size, v.size)]
                qids = [f"q{i}" for i in range(v.size)]
                A2[ds] = dict(zip(qids, draw))
                B2[ds] = dict(zip(qids, np.zeros(v.size)))
            ps.append(boot.signflip(A2, B2, R=R, seed=int(rng.integers(2**31)),
                                    alternative="greater")["p"])
        ps = np.array(ps)
        out[f"{x} vs {y}"] = {
            "per_dataset_skew_of_centered_diffs": skew,
            "P(p<=0.025)": round(float((ps <= 0.025).mean()), 4),
            "P(p<=0.05)": round(float((ps <= 0.05).mean()), 4),
            "P(p<=0.00833)": round(float((ps <= 0.025 / 3).mean()), 4),
        }
        print(f"{x} vs {y}: {out[f'{x} vs {y}']}", flush=True)
    res = {"S": S, "R": R, "_note": "weak-null (centered, asymmetric) type-I of the sign-flip "
           "leg alone; the tier decision is the conjunction with the bootstrap CI",
           "pairs": out, "seconds": round(time.time() - t0, 1)}
    (REPO / "results" / "m7_signflip_weaknull.json").write_text(json.dumps(res, indent=1))
    print("done")


if __name__ == "__main__":
    main()
