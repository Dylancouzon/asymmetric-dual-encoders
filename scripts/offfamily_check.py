"""Exposure-free teacher ranking: rank the closed-form tables on held-out SQuAD and ESCI.

Why. Every component of the pinned dev suite carries stella exposure -- NQ and HotpotQA are on its
disclosed training list, and CQADupStack is StackExchange, of which stella discloses four datasets
(results/m7_teacher_contamination.json). bge-base discloses none of them. The pre-registered
family-exposure read in m7/RESULTS.md used nq-250k as the off-StackExchange control, which is
itself on stella's list. Codex finding 14 (2026-08-26, MAJOR, research/m7-codex-review-2026-08-26b.md)
named the fix and it was never run: SQuAD and ESCI are on NO candidate's disclosed list.

Design. Same closed-form ridge, same shared TRAIN bag matrix, each candidate's own λ*, each
candidate's own documents -- identical to scripts/teacher_learnability.py in every respect except
the component. Scored on `heldout-train` (dev, pinned, legal) against the FULL 6.17M-doc pool,
because heldout.py records that a 200K random-distractor store saturates at 0.84 and cannot
discriminate. Per-query scores are then stratified by source, and only the squad-train and esci-us
strata are read.

Falsification: if stella is not first on the exposure-free strata, the 2026-08-26 teacher selection
was family-specific and the swap rests on contaminated evidence.

    M7_ENCODER=<name> ../.venv/bin/python scripts/offfamily_check.py <lambda>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m7src"))

from _paths import REPO, WORK

COMPONENT = "heldout-train"
STRATA = ("squad-train", "esci-us")


def _bind_documents(name):
    """Verify the DOCUMENTS, then allow the encoder to differ -- narrow, deliberate bypass.

    heldout._verify_pool binds pool identity to the pinned build, which is stella's: it compares
    `dim`, `encoder`, `encoder_revision` and the vector bytes. That is right for every normal dev
    read and wrong for exactly this comparison, whose whole point is a second encoder over the SAME
    documents. So the check that matters is reproduced here and the encoder fields are the only
    ones dropped: stores, spans, counts, per-store id_sha256 and n must equal the pin, and the
    component JSON must still hash to the pinned value. If any document moved, this raises.
    stella's own arm passed the FULL pinned path -- no bypass was used for it."""
    import heldout
    import pool as poolmod
    man = json.loads((REPO / "results" / "m7_dev_manifest.json").read_text())
    pin = man.get("_pinned", {}).get("pool", {})
    _, _, meta = poolmod.build()
    for pin_key, meta_key in (("n", "n"), ("stores", "stores"), ("spans", "spans"),
                              ("counts", "counts"), ("store_id_sha256", "id_sha256")):
        if pin_key in pin and meta.get(meta_key) != pin[pin_key]:
            raise SystemExit(f"DOCUMENT identity differs from the pin: {meta_key}")
    import hashlib
    got = hashlib.sha256((WORK / "dev" / f"{COMPONENT}.json").read_bytes()).hexdigest()
    want = (man.get(COMPONENT) or {}).get("json_sha256")
    if want and got != want:
        raise SystemExit(f"PINNED dev component {COMPONENT} changed")
    print(f"  documents bound: {meta['n']:,} docs, layout and per-store ids match the pin; "
          f"encoder {meta.get('encoder')} dim {meta['dim']}", flush=True)
    heldout._VERIFIED = "cheap"


def main():
    import encoders
    lam = float(sys.argv[1])
    name = encoders.active().name

    import dev_eval
    import encode_trainq
    import stage0_ridge as sr
    from init_table import get_init
    from table import Preproc, QueryTable, get_tokenizer
    from teacher import QUERY_PREFIX, encode_cached

    _bind_documents(name)
    q_texts = encode_trainq.load_texts()
    pre, tok = Preproc(), get_tokenizer()
    print(f"{name}: bag matrix over {len(q_texts):,} TRAIN queries", flush=True)
    X = sr.bag_matrix(tok, q_texts, pre, tok.vocab_size)
    Y = np.asarray(encode_cached(f"trainq-{len(q_texts)}", q_texts, prefix=QUERY_PREFIX,
                                 dtype=torch.float16, verbose=False), dtype=np.float32)
    W = sr.solve_ridge(X, Y, get_init("teacher", pre), lam)
    from _paths import DEVICE
    model = QueryTable(W, weight_init=None, learned_weights=False).to(DEVICE)

    pq = dev_eval.eval_table(model, pre, components=[COMPONENT], tok=tok)[COMPONENT]
    by = {s: {q: v for q, v in pq.items() if q.startswith(s + ":")} for s in STRATA}
    out = {"encoder": name, "lambda": lam, "component": COMPONENT,
           "n_queries_total": len(pq),
           "strata": {s: {"n": len(d), "ndcg@10": round(float(np.mean(list(d.values()))), 4)}
                      for s, d in by.items()},
           "all_sources_macro": round(float(np.mean(list(pq.values()))), 4),
           "per_query": {s: {q: round(v, 6) for q, v in d.items()} for s, d in by.items()}}
    for s in STRATA:
        print(f"  {name} {s}: n={out['strata'][s]['n']} nDCG@10={out['strata'][s]['ndcg@10']}",
              flush=True)
    p = REPO / "results" / f"m7_offfamily_{name}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
