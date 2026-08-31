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
    """-> (projected endpoint, per-doubling decay used, n doublings remaining)."""
    if len(rows) < 3:
        return None, None, None
    # per-doubling increments from the last three points
    inc = []
    for (t0, s0), (t1, s1) in zip(rows, rows[1:]):
        d = math.log2(t1 / t0)
        if d > 0:
            inc.append((s1 - s0) / d)
    if len(inc) < 2:
        return None, None, None
    ratios = [b / a for a, b in zip(inc, inc[1:]) if a > 0]
    r = sum(ratios[-2:]) / len(ratios[-2:]) if ratios else 0.5
    r = min(max(r, 0.05), 0.98)                 # a decay >=1 would diverge
    last_inc, t_last, s_last = inc[-1], rows[-1][0], rows[-1][1]
    n = math.log2(cap / t_last)
    # sum of a geometric series over n doublings, with the fractional remainder
    total, cur, left = 0.0, last_inc * r, n
    while left > 0:
        total += cur * min(1.0, left)
        cur *= r
        left -= 1
        if cur < 1e-6:
            break
    return s_last + total, r, n


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
    proj, r, n = project(rows, cap)
    if proj is None:
        print("\nneed at least 3 trained evaluations to project")
        return
    print(f"\nprojected endpoint at the {cap/1e9:.2f}B cap ({n:.2f} doublings away, "
          f"decay {r:.3f}/doubling):")
    print(f"  SCREEN-3 ~ {proj:.4f}   retention ~ {proj/CEILING:.4f}")
    print("\nCAVEATS, which are the point:")
    print("  * SCREEN-3 is a DEV surface (NQ + two CQADupStack components). It is NOT avg-6, and")
    print("    no calibrated map between them exists -- the aim's 89.7% is an avg-6 figure.")
    print("  * a geometric decay of per-doubling increments is the crudest defensible model;")
    print("    with few points it is sensitive to the last increment.")
    print("  * this projection informs no decision: the dose is fixed and unextendable.")


if __name__ == "__main__":
    main()
