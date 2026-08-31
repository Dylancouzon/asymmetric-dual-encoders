"""Project the build's SCREEN-3 endpoint from its own dose-response curve.

Read-only, DEV-surface only. It informs nothing: the dose is fixed at `stable_token_cap` and no
extension mechanism is registered, so this cannot change the run. It exists so the report carries
a documented curve and a stated projection method instead of arithmetic done in someone's head.

Method: increments are fitted per DOUBLING of cumulative tokens, not per evaluation. Evaluations
are equally spaced in tokens, so per-eval increments shrink for two different reasons at once
(genuine saturation, and each eval covering a smaller relative dose increase) and extrapolating
them understates the tail. Per-doubling increments are then assumed to decay geometrically -- the
crudest defensible model, and stated as such.

Run: .venv/bin/python m9src/dose_curve.py
"""
import json
import math
from pathlib import Path

RUN = Path(__file__).resolve().parents[1] / "work" / "m9long"
CEILING = 0.68223          # stella-400M SCREEN-3 symmetric, registry.ceilings


def history():
    rows = []
    for line in (RUN / "history.jsonl").read_text().splitlines():
        r = json.loads(line)
        if not r.get("step0") and r.get("tokens"):
            rows.append((int(r["tokens"]), float(r["screen3"])))
    return sorted(set(rows))


def project(rows, cap):
    """-> dict of bracketing projections. Deliberately a RANGE, not a point.

    The first version fitted a decay ratio from the last two per-doubling increments and
    extrapolated geometrically. One noisy increment (+0.0171 -> +0.0256 at eval 6) moved the
    projected retention from 0.818 to 0.932, which is estimator noise masquerading as news. Any
    projection this sensitive to the newest point is not a forecast.

    Three bracketing readings instead, each stated for what it is:
      * FLOOR      -- training stops improving now. A genuine lower bound.
      * SATURATING -- least-squares power law s(t) = S_inf - B * t**(-c) over ALL points, the
                      only fit here that can saturate. This is the central estimate.
      * LOGLINEAR  -- s linear in log(t), fitted over all points. Cannot saturate, so it is an
                      OPTIMISTIC bound, not a candidate for the truth.
    """
    import numpy as np
    t = np.array([r[0] for r in rows], float)
    y = np.array([r[1] for r in rows], float)
    out = {"floor": float(y[-1]), "n_doublings_left": float(math.log2(cap / t[-1]))}

    # log-linear (optimistic bound): y = a + b*log2(t)
    b, a = np.polyfit(np.log2(t), y, 1)
    out["loglinear"] = float(a + b * math.log2(cap))

    # saturating power law, grid over the exponent then linear least squares for S_inf and B
    best = None
    if len(rows) >= 3:
        for c in np.linspace(0.05, 2.0, 196):
            X = np.column_stack([np.ones_like(t), -t ** (-c)])
            try:
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            resid = float(((X @ coef - y) ** 2).sum())
            if coef[1] <= 0:                      # B must be positive to approach from below
                continue
            if best is None or resid < best[0]:
                best = (resid, float(coef[0]), float(coef[1]), float(c))
    if best:
        _, S_inf, B, c = best
        out["saturating"] = float(S_inf - B * cap ** (-c))
        out["S_inf"] = S_inf
        out["exponent_c"] = c
    return out


def main():
    rows = history()
    cfg = json.loads((RUN / "config.json").read_text())
    cap = cfg["stable_token_cap"]
    print(f"{'tokens (B)':>12} {'SCREEN-3':>9} {'retention':>10} {'per-doubling':>13}")
    prev = None
    for t, s in rows:
        pd = ""
        if prev:
            d = math.log2(t / prev[0])
            pd = f"{(s - prev[1]) / d:+.5f}" if d > 0 else ""
        print(f"{t/1e9:12.3f} {s:9.5f} {s/CEILING:10.4f} {pd:>13}")
        prev = (t, s)
    pr = project(rows, cap)
    if "saturating" not in pr:
        print("\nneed at least 3 trained evaluations to project")
        return
    print(f"\nprojections at the {cap/1e9:.2f}B cap "
          f"({pr['n_doublings_left']:.2f} doublings away):")
    for k, label in (("floor", "FLOOR      (stops improving now; a true lower bound)"),
                     ("saturating", "SATURATING (power-law LSQ over all points; CENTRAL)"),
                     ("loglinear", "LOGLINEAR  (cannot saturate; OPTIMISTIC bound)")):
        v = pr[k]
        print(f"  {label:52s} SCREEN-3 {v:.4f}  retention {v/CEILING:.4f}")
    print(f"  fitted asymptote S_inf {pr['S_inf']:.4f} "
          f"(retention {pr['S_inf']/CEILING:.4f}), exponent c={pr['exponent_c']:.2f}")
    print("\nCAVEATS, which are the point:")
    print("  * SCREEN-3 is a DEV surface (NQ + two CQADupStack components). It is NOT avg-6, and")
    print("    no calibrated map between them exists -- the aim's 89.7% is an avg-6 figure.")
    print("  * three bracketing readings, not one number. The earlier single-point estimator")
    print("    fitted a decay from the last two increments and moved 0.818 -> 0.932 on one noisy")
    print("    eval; a projection that sensitive to the newest point is not a forecast.")
    print("  * LOGLINEAR cannot saturate and is an OPTIMISTIC BOUND, never a central estimate.")
    print("  * the fit uses STABLE-LR checkpoints only. The run ends with a cosine anneal to")
    print("    1e-5, which typically adds a step up that this curve structurally cannot see, so")
    print("    the central estimate is biased LOW by an unquantified amount.")
    print("  * this projection informs no decision: the dose is fixed and unextendable.")


if __name__ == "__main__":
    main()
