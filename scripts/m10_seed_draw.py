"""Build-scale seed draw for the five generated forms served by `hotpotqa-corpus`.

`health` and `finance` come from `wikipedia-body` (`work/m10gen/wikibody_gate_pool.json`,
T2-5); the other five route through the registered `seeds.draw` -- `howto` topically at
`min_score >= 4`, and `argument`/`comparison`/`yesno`/`conversational` as `general`.

Nothing registered is changed here: this is the same `seeds.cached` call the 40-seed smoke
made, at build scale. `per_form` covers `build_form`'s 1.35x seed margin over a 143,000 quota
at 5 queries per seed (38,610) plus the 500-document FORMS-12 hold-out.
"""
import json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m10src"))

import protected10                    # BEFORE anything imports m9base (CODEMAP pitfall 14)
protected10.build(verbose=True)

import seeds as seedmod

FORMS = ["howto", "argument", "comparison", "yesno", "conversational"]
PER_FORM = 39_200                     # 38,610 build seeds + the 500-doc FORMS-12 hold-out
OUT = REPO / "work" / "m10gen" / "build_seeds.done"

if __name__ == "__main__":
    per_form = int(sys.argv[1]) if len(sys.argv) > 1 else PER_FORM
    pool = int(sys.argv[2]) if len(sys.argv) > 2 else 6_000_000
    t0 = time.time()
    kept, meta = seedmod.cached(FORMS, per_form=per_form, pool_size=pool, seed=0,
                                store="hotpotqa-corpus", min_score=4, verbose=True)
    meta["seconds"] = round(time.time() - t0, 1)
    meta["counts"] = {f: len(v) for f, v in kept.items()}
    meta["per_form_requested"] = per_form
    print(json.dumps(meta, indent=1), flush=True)
    if per_form == PER_FORM and pool == 6_000_000:
        OUT.write_text(json.dumps(meta, indent=1))
    print(f"DONE in {meta['seconds']}s", flush=True)
