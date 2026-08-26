"""Independent pairing evidence for the frozen comparators beyond BM25 (review #2 MAJOR 9).

results/significance.json was produced on the M4 box from cached artifacts -- per-query runs
re-derived from vectors, never from perquery.json. A paired bootstrap CI depends on the per-query
PAIRING (the variance of within-query differences), not just the per-dataset means, so recomputing
each of its comparisons from today's perquery.json and matching delta AND CI bounds is evidence
the frozen per-qid pairing is the one the M4 caches produced. A permuted vector would leave means
intact and blow the CI comparison apart. RNG streams differ between the two implementations, so
bounds are compared to a Monte-Carlo tolerance at B=10k, not byte-exactly.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "m7src"))
import boot  # noqa: E402

TOL_DELTA = 2e-4       # deltas are MC-free; rounding only
TOL_CI = 3e-3          # two independent B=10k streams; ~2x the observed spread in practice

sig = json.load(open(REPO / "results" / "significance.json"))
pq = json.load(open(REPO / "results" / "perquery.json"))
systems = {name for d in pq["datasets"].values() for name in d["systems"]}

checked, skipped, failures = 0, [], []
for key, want in sig.items():
    a_name, b_name = [x.strip() for x in key.split(" vs ")]
    if a_name not in systems or b_name not in systems:
        skipped.append(key)
        continue
    A = boot.from_perquery_json(pq, a_name)
    B = boot.from_perquery_json(pq, b_name)
    r = boot.paired(A, B, alternative="two-sided")
    checked += 1
    d_dev = abs(r["delta"] - want["delta"])
    ci_dev = max(abs(r["ci95"][0] - want["ci95"][0]), abs(r["ci95"][1] - want["ci95"][1]))
    ok = d_dev <= TOL_DELTA and ci_dev <= TOL_CI
    print(f"  {'ok  ' if ok else 'FAIL'} {key}: delta {r['delta']:+.4f} vs {want['delta']:+.4f} "
          f"| CI {r['ci95']} vs {want['ci95']} (max dev {ci_dev:.4f})")
    if not ok:
        failures.append(key)

note = (f"{checked} comparisons cross-checked against the M4-cache-derived significance.json; "
        f"{len(skipped)} skipped (a side not in perquery.json)")
print(note)
out = {"checked": checked, "skipped": skipped, "failures": failures,
       "tol_delta": TOL_DELTA, "tol_ci": TOL_CI, "_note": __doc__.strip().split("\n")[0]}
(REPO / "results" / "m7_perquery_crosscheck.json").write_text(json.dumps(out, indent=1))
if failures:
    print("CROSSCHECK FAILED:", failures)
    sys.exit(1)
print("OK")
