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


def _filtered_incumbent(rows, form):
    """The incumbent store with T2-8 rung 1 applied. An intro passage IS its article's lead, so
    the subject patterns apply to the passage itself — which makes this the like-for-like arm
    that separates "the filter works" from "`wikipedia-body` is better". Reported, never gating;
    Fable's own note stands that filtering cannot rescue the incumbent's SUPPLY (8,663 x 0.59
    ~ 5.1K against a 33K need), so this arm is about precision only."""
    import wikibody
    return [r for r in rows if not wikibody.subject_reject(
        wikibody.lead_sentence(r[1]), form)]


def build(candidate_json=None, control_json=None, with_filtered_control=False, tag=""):
    """-> per-form blinded sample files plus the key. Nothing is judged here."""
    import wikibody
    OUT.mkdir(parents=True, exist_ok=True)
    cand = json.loads(Path(candidate_json or (REPO / "work/m10gen/wikibody_gate_pool.json"))
                      .read_text())
    ctrl = json.loads(Path(control_json or (REPO / "work/m10gen/incumbent_full.json"))
                      .read_text())["seeds"]
    rng = np.random.default_rng(SEED)
    key, manifest = {}, {}
    for form in wikibody.FORMS:
        c_rows = cand[form]                       # [(passage_id, text)]
        k_rows = ctrl.get(form, [])
        ci, ki = _pick(c_rows, N_SAMPLE, rng), _pick(k_rows, N_SAMPLE, rng)
        items = ([("wikipedia-body", c_rows[i][0], c_rows[i][1]) for i in ci]
                 + [("incumbent", k_rows[i][0], k_rows[i][1]) for i in ki])
        if with_filtered_control:
            f_rows = _filtered_incumbent(k_rows, form)
            fi = _pick(f_rows, N_SAMPLE, rng)
            items += [("incumbent-filtered", f_rows[i][0], f_rows[i][1]) for i in fi]
            manifest.setdefault(form, {})["filtered_control_pool"] = len(f_rows)
        order = rng.permutation(len(items))       # blinded AND interleaved
        blob, k = [], {}
        for n, j in enumerate(order):
            arm, pid, text = items[int(j)]
            blob.append({"i": n, "text": text})
            k[n] = {"arm": arm, "passage_id": pid}
        # Batched at 200 items: one judge reading 400 passages of up to 220 words is ~80K tokens
        # of careful per-item judgement, and batches also mean no single judge's drift decides a
        # form. Blinding and interleaving are preserved inside every batch.
        for b0 in range(0, len(blob), N_SAMPLE):
            (OUT / f"sample{tag}-{form}-{b0 // N_SAMPLE}.json").write_text(
                json.dumps(blob[b0:b0 + N_SAMPLE], indent=1))
        key[form] = k
        manifest[form] = {**manifest.get(form, {}),
                          "candidate_pool": len(c_rows), "candidate_sampled": len(ci),
                          "control_pool": len(k_rows), "control_sampled": len(ki),
                          "total_items": len(blob),
                          "batches": (len(blob) + N_SAMPLE - 1) // N_SAMPLE}
    (OUT / f"key{tag}.json").write_text(json.dumps(key, indent=1))
    (OUT / f"manifest{tag}.json").write_text(json.dumps(
        dict(n_sample=N_SAMPLE, seed=SEED, gate=GATE, blinded=True, forms=manifest), indent=1))
    print(json.dumps(manifest, indent=1))
    return manifest


def score(*verdict_files, tag=""):
    """verdicts: {form: {index: true/false}} per file -> per-arm on-topic rate and the verdict."""
    key = json.loads((OUT / f"key{tag}.json").read_text())
    v = {}
    for f in verdict_files:
        for form, d in json.loads(Path(f).read_text()).items():
            v.setdefault(form, {}).update({str(k): val for k, val in d.items()})
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
    p = REPO / "results" / f"m10_wikibody_precision{tag or ''}.json"
    p.write_text(json.dumps({"n_sample": N_SAMPLE, "seed": SEED, "gate": GATE,
                             "blinded": True, "forms": out}, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        score(*sys.argv[3:], tag=sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "regate":
        build(with_filtered_control=True, tag="-r1")
    else:
        build()
