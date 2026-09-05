"""Turn the three P arms into the two numbers M10.0-e is registered to produce, and nothing more.

Run when `work/m10calib/P{0,1,2}_cov.json` all exist:

    .venv/bin/python m10src/calib_report.py

It refuses a P arm whose record is not a full-dose run, because the 90-step smoke of
`calib.run_arm` writes to the SAME paths as the real arms — a dead arm would otherwise leave a
smoke record that reads like a result.

Registered scope (`m10/LEDGER.md` §M10.0-e): this brackets contrasts whose arms share backbone,
tokenizer, init, seed, data order and warm-start head — **B, D, G and C**. **F and E are read
against the 0.008619 unrelated-models number** and no LR pair speaks for them. It changes no
constant; the MDE decision is Dylan's, before family F, never after.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m10src"))
WORK = REPO / "work" / "m10calib"
OUT = REPO / "results" / "m10_calib_report.json"
EXPECTED_STEPS = 156_250

import numpy as np


def load(name):
    rec = json.loads((WORK / f"{name}.json").read_text())
    if rec.get("total_steps") != EXPECTED_STEPS:
        raise SystemExit(f"{name}: total_steps {rec.get('total_steps')} != {EXPECTED_STEPS} — "
                         f"this is a SMOKE record, not a full-dose arm")
    if rec.get("stopped"):
        raise SystemExit(f"{name}: stopped early ({rec['stopped']}) — not a usable arm")
    cov = json.loads((WORK / f"{name}_cov.json").read_text())
    return rec, cov


def main():
    import cov_macro
    import cov_probe
    uf = {u[0]: u[1] for u in cov_probe.units()}
    arms = {n: load(n) for n in ("P0", "P1", "P2")}
    per = {n: v[1]["per_unit_query"] for n, v in arms.items()}
    mac = {n: v[1]["macro"] for n, v in arms.items()}

    out = {"_what": "M10.0-e same-init calibration. Brackets B, D, G and C ONLY.",
           "_f_and_e": "read against the unrelated-models distance 0.008619, not this",
           "_changes": "no constant; MDE 0.0056 and alpha 0.025/13 untouched",
           "macro": mac,
           "seed_effect_point_estimate": abs(mac["P0"] - mac["P1"]),
           "_seed_note": "the SEED effect is a point estimate; the bootstrap resamples QUERIES "
                         "and cannot see it. A screen that resolves less than this has resolved "
                         "noise."}
    for a, b, label in (("P0", "P1", "seed_pair"), ("P0", "P2", "lr_pair")):
        al = cov_macro.align(per[a], per[b], uf)
        r = cov_macro.contrast(al, uf)
        out[label] = {k: r[k] for k in ("delta_raw", "lower_bound_raw", "distance_raw",
                                        "draws_sd", "B", "seed", "chunk", "quantile_method",
                                        "plan_sample_sha256", "draws_sha256")}
        out[label]["per_family_delta"] = {
            f: float(np.mean([r["per_unit_delta_raw"][u] for u in uf if uf[u] == f]))
            for f in sorted(set(uf.values()))}
    d_unrel = json.loads((REPO / "results" / "m10_cov_resolution.json").read_text())
    out["unrelated_models_distance"] = d_unrel["resolution_distance"]
    out["ratio_lr_pair_to_unrelated"] = (out["lr_pair"]["distance_raw"]
                                         / d_unrel["resolution_distance"])
    out["MDE"] = d_unrel["MDE_registered"]
    out["reading"] = (
        f"A same-init contrast's distance is {out['lr_pair']['distance_raw']:.5f} against "
        f"{d_unrel['resolution_distance']:.5f} between unrelated models "
        f"({out['ratio_lr_pair_to_unrelated']:.2f}x). The seed effect is "
        f"{out['seed_effect_point_estimate']:.5f}. Both are DISCLOSURES: B/D/G/C are read against "
        f"the first, F and E against the second, and the MDE decision is Dylan's before family F.")
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("macro", "seed_effect_point_estimate",
                                          "unrelated_models_distance",
                                          "ratio_lr_pair_to_unrelated", "reading")}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
