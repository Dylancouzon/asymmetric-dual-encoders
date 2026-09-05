"""T2-7 — the build on-form diagnostic. Registered before its first generation call; ADMITS NOTHING.

T2-3 registered a "report-only build on-form diagnostic". This executes it early, on two forms and
two seed stores, to inform Dylan's W6 ruling. It is **not** an admission instrument: attaching
admission to it after the registered seed-precision gate returned FAIL would be a protocol change
after observation, which is the one change this project forbids (Fable, 2026-09-05).

The design is `m10/LEDGER.md` §3 T2-7, items 1-8, and the two that matter most:

- **Seeds are the exact 400 passages per form already judged by the precision gate**, so every
  generated query links to a seed-precision verdict and `P(on-form | the seed was judged
  off-subject)` is measurable per arm. A fresh draw could not separate "the generator absorbs seed
  noise" from "the on-subject half did all the work" — and that confusion is exactly what killed
  the 59%->84% argument this diagnostic replaces.
- **A8-style diversity metrics are reported beside the on-form rate.** The plausible harm from an
  off-subject seed is not an off-form query, it is a GENERIC one: a weak hook makes the generator
  fall back on templates, which shows up as near-duplicate rate and mean pairwise cosine, not as an
  on-form failure. A diagnostic without them measures the wrong failure.

Generator contract unchanged: `Qwen/Qwen3-8B-AWQ` at the pinned revision, the approved prompt
hashes, `n = 5`. `n` is NOT touched -- changing it moves the basis the prompts were approved on.
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m10src"):
    sys.path.insert(0, str(REPO / p))
GATE = REPO / "work" / "m10gen" / "gate"
OUT = GATE
BASE = "http://127.0.0.1:8001/v1"
FORMS = ("health", "finance")
N_PER_SEED, JUDGE_SEED = 5, 23

import numpy as np


def seeds_for(form):
    """-> [(passage_id, text, arm, seed_verdict)] over the 400 already-judged passages."""
    key = json.loads((GATE / "key.json").read_text())[form]
    verdicts = {}
    for b in (0, 1):
        p = GATE / f"verdicts-{form}-{b}.json"
        verdicts.update({str(k): v for k, v in json.loads(p.read_text())[form].items()})
    rows = []
    for b in (0, 1):
        for item in json.loads((GATE / f"sample-{form}-{b}.json").read_text()):
            i = str(item["i"])
            rows.append((key[i]["passage_id"], item["text"], key[i]["arm"], bool(verdicts[i])))
    return rows


def generate(form, limit=None, verbose=True):
    import gen
    rows = seeds_for(form)[:limit]
    seen = set()
    r = gen.generate(form, [(pid, txt) for pid, txt, _a, _v in rows], n=N_PER_SEED,
                     base=BASE, label=f"diag-{form}", seen=seen)
    meta = {pid: (arm, ver) for pid, _t, arm, ver in rows}
    for q in r["queries"]:
        q["arm"], q["seed_on_subject"] = meta[q["seed_id"]]
    (OUT / f"onform-raw-{form}.json").write_text(json.dumps(r))
    if verbose:
        print(f"{form}: {len(r['queries'])} unique queries from {len(rows)} seeds, "
              f"contract {r.get('contract_rate')}", flush=True)
    return r


def build_judge_files(form):
    """One query per seed, uniform-random among its five; blinded, interleaved, batched at 200."""
    r = json.loads((OUT / f"onform-raw-{form}.json").read_text())
    by_seed = {}
    for q in r["queries"]:
        by_seed.setdefault(q["seed_id"], []).append(q)
    rng = np.random.default_rng(JUDGE_SEED)
    picked = [qs[int(rng.integers(len(qs)))] for _sid, qs in sorted(by_seed.items())]
    order = rng.permutation(len(picked))
    blob, key = [], {}
    for n, j in enumerate(order):
        q = picked[int(j)]
        blob.append({"i": n, "text": q["query"]})
        key[n] = {"arm": q["arm"], "seed_id": q["seed_id"],
                  "seed_on_subject": q["seed_on_subject"]}
    for b0 in range(0, len(blob), 200):
        (OUT / f"onform-sample-{form}-{b0 // 200}.json").write_text(
            json.dumps(blob[b0:b0 + 200], indent=1))
    (OUT / f"onform-key-{form}.json").write_text(json.dumps(key, indent=1))
    print(f"{form}: {len(blob)} judged items in {(len(blob) + 199) // 200} batches", flush=True)
    return len(blob)


def score(form, *verdict_files):
    import decontam
    key = json.loads((OUT / f"onform-key-{form}.json").read_text())
    raw = json.loads((OUT / f"onform-raw-{form}.json").read_text())
    v = {}
    for f in verdict_files:
        for _form, d in json.loads(Path(f).read_text()).items():
            v.update({str(k): bool(val) for k, val in d.items()})
    per = {}
    for i, meta in key.items():
        if i not in v:
            continue
        cell = per.setdefault(meta["arm"], {"all": [], "on_subject": [], "off_subject": []})
        cell["all"].append(v[i])
        cell["on_subject" if meta["seed_on_subject"] else "off_subject"].append(v[i])
    def rate(b):
        if not b:
            return {"n": 0, "rate": None, "se": None}
        p = sum(b) / len(b)
        return {"n": len(b), "on_form": sum(b), "rate": round(p, 4),
                "se": round((p * (1 - p) / len(b)) ** 0.5, 4)}
    out = {arm: {k: rate(b) for k, b in cell.items()} for arm, cell in per.items()}
    # A8-style diversity, per arm, over EVERY generated query (not just the judged one)
    for arm in out:
        qs = [q["query"] for q in raw["queries"] if q["arm"] == arm]
        norm = [" ".join(q.split()).lower() for q in qs]
        grams = [decontam.query_grams(q) for q in qs]
        inv = decontam.Inverted(grams, [decontam.exact_u64(q) for q in qs])
        dup = 0
        for q in qs:
            ex, near = inv.match(q, 16)
            if len(set(ex.tolist()) | set(near.tolist())) > 1:
                dup += 1
        out[arm]["diversity"] = {
            "n_queries": len(qs), "exact_duplicates": len(norm) - len(set(norm)),
            "near_dup_rate_16of32": round(dup / max(len(qs), 1), 4),
            "_note": "A8's threshold is a 25% near-duplicate rate; underpowered at this n and "
                     "reported, not acted on. Mean pairwise stella cosine is NOT computed here "
                     "-- it needs the teacher and is the manifest-time gate."}
    p = REPO / "results" / f"m10_onform_diag_{form}.json"
    p.write_text(json.dumps({"form": form, "admits_nothing": True,
                             "registered": "m10/LEDGER.md §3 T2-7", "arms": out}, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "score":
        score(sys.argv[2], *sys.argv[3:])
    elif cmd == "judgefiles":
        for f in FORMS:
            build_judge_files(f)
    else:
        lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
        for f in FORMS:
            generate(f, limit=lim)
