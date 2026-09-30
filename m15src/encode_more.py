"""Encode Stella document vectors on the Mac for E9/E10 and run the reproduction gate on each.

Datasets: the three six-set members not yet local, and the two M7 dev forums. One MPS job.
"""
import json
import sys

from common import REPO, load_public
import encoders15 as E
import vectors15 as V

DATASETS = ("arguana", "scidocs", "cqadup-programmers", "cqadup-physics", "trec-covid")
OUT = REPO / "work" / "m15" / "gates.json"

if __name__ == "__main__":
    names = sys.argv[1:] or DATASETS
    gates = json.loads(OUT.read_text()) if OUT.exists() else {}
    enc = {n: E.make(n) for n in ("stella-query", "zero", "nano")}
    for ds in names:
        data = load_public(ds)
        dv, prov = V.doc_vectors(ds, data)
        ok, rows = V.gate(ds, data, dv, enc)
        gates[ds] = {"passed": ok, "rows": rows, "vectors": prov}
        OUT.write_text(json.dumps(gates, indent=2))
        print(ds, "PASS" if ok else "FAIL", flush=True)
