"""Build the judged-precision gate's samples — the decision that admits or refuses a seed store.

The gate that killed `ROUTE_WIDE` (28% / 38% on-topic against >= 80%) is re-run here on
`wikipedia-body`, with the two corrections a Fable pass required:

1. **200 per form, not 50.** 50 has SE ~5.7 points and cannot tell 80 from 72
   (`m10/HEADROOM.md`, registered).
2. **Sampled from the population the BUILD will use** — uniform-random within the drawn set,
   which is top-score-first over the screened store — not uniform over everything admitted.

And a **control**: 200 from the incumbent `ROUTE` intro store, judged in the same run. It is
REPORTED, never an escape clause: if the incumbent is itself under 80%, that is a finding for
Dylan, not a reason to pass the new store.

Samples are written blinded and interleaved — the judge sees `{"i": n, "text": ...}` and nothing
about which store a passage came from — so a judge cannot grade the two arms to different
standards. The key stays here.
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m10src"):
    sys.path.insert(0, str(REPO / p))
OUT = REPO / "work" / "m10gen" / "gate"
N_SAMPLE, SEED, GATE = 200, 11, 0.80

import numpy as np


def _pick(rows, n, rng):
    if len(rows) <= n:
        return list(range(len(rows)))
    return sorted(rng.choice(len(rows), n, replace=False).tolist())


def build(candidate_json=None, control_json=None):
    """-> per-form blinded sample files plus the key. Nothing is judged here."""
    import wikibody
    OUT.mkdir(parents=True, exist_ok=True)
    cand = json.loads(Path(candidate_json or (REPO / "work/m10gen/wikibody_gate_pool.json"))
                      .read_text())
    ctrl = json.loads(Path(control_json or (REPO / "work/m10gen/incumbent_control.json"))
                      .read_text())["seeds"]
    rng = np.random.default_rng(SEED)
    key, manifest = {}, {}
    for form in wikibody.FORMS:
        c_rows = cand[form]                       # [(passage_id, text)]
        k_rows = ctrl.get(form, [])
        ci, ki = _pick(c_rows, N_SAMPLE, rng), _pick(k_rows, N_SAMPLE, rng)
        items = ([("wikipedia-body", c_rows[i][0], c_rows[i][1]) for i in ci]
                 + [("incumbent", k_rows[i][0], k_rows[i][1]) for i in ki])
        order = rng.permutation(len(items))       # blinded AND interleaved
        blob, k = [], {}
        for n, j in enumerate(order):
            arm, pid, text = items[int(j)]
            blob.append({"i": n, "text": text})
            k[n] = {"arm": arm, "passage_id": pid}
        (OUT / f"sample-{form}.json").write_text(json.dumps(blob, indent=1))
        key[form] = k
        manifest[form] = {"candidate_pool": len(c_rows), "candidate_sampled": len(ci),
                          "control_pool": len(k_rows), "control_sampled": len(ki),
                          "total_items": len(blob)}
    (OUT / "key.json").write_text(json.dumps(key, indent=1))
    (OUT / "manifest.json").write_text(json.dumps(
        dict(n_sample=N_SAMPLE, seed=SEED, gate=GATE, blinded=True, forms=manifest), indent=1))
    print(json.dumps(manifest, indent=1))
    return manifest


def score(verdicts_json):
    """verdicts: {form: {index: true/false}} -> per-arm on-topic rate and the gate verdict."""
    key = json.loads((OUT / "key.json").read_text())
    v = json.loads(Path(verdicts_json).read_text())
    out = {}
    for form, ks in key.items():
        got = v.get(form, {})
        per = {}
        for i, meta in ks.items():
            if str(i) not in got and int(i) not in got:
                continue
            ok = got.get(str(i), got.get(int(i)))
            per.setdefault(meta["arm"], []).append(bool(ok))
        row = {arm: {"n": len(b), "on_topic": sum(b),
                     "rate": round(sum(b) / len(b), 4) if b else None}
               for arm, b in per.items()}
        cand = row.get("wikipedia-body", {})
        row["PASSES_GATE"] = bool(cand.get("rate") is not None and cand["rate"] >= GATE)
        row["gate"] = GATE
        out[form] = row
    p = REPO / "results" / "m10_wikibody_precision.json"
    p.write_text(json.dumps({"n_sample": N_SAMPLE, "seed": SEED, "gate": GATE,
                             "blinded": True, "forms": out}, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        score(sys.argv[2])
    else:
        build()
