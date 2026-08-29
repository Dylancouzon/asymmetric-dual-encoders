"""Do the table's INIT rows come from the same read-out as the documents they will be scored against?

Codex gate 2026-08-26, BLOCKER 1: `init_table.teacher_rows` hardcoded `last_hidden_state[:, 0]`
and sized itself from `config.hidden_size`, so a mean-pooled teacher (or one with a post-pooling
Dense) would have had its table rows built with a different read-out from its document vectors.
Nothing downstream can catch that: the shapes agree and the numbers are plausible.

The check: for vocabulary tokens whose text round-trips to exactly [CLS] tok [SEP], the init row
must equal `teacher.encode([text])` -- the production document path -- to fp32 noise. Those are the
only tokens where the two paths are constructing the same sequence, which is what makes it a real
end-to-end comparison rather than a restatement of the code.

Run it for the ACTIVE encoder before any corpus encode with a new Spec:

    M7_ENCODER=arctic-embed-l ../.venv/bin/python test_init_rows.py
"""
import json
import sys

import numpy as np
import torch

import encoders
import teacher
from _paths import REPO
from init_table import get_init, spec_tag
from table import Preproc, get_tokenizer

N_SAMPLE = 256
COS_BAR = 0.999


def main():
    sp = encoders.active()
    pre = Preproc()                      # the one frozen query rule; prefix is empty by default
    tok = get_tokenizer()
    rows = get_init("teacher", pre)
    assert rows.shape == (tok.vocab_size, sp.dim), \
        f"init rows {rows.shape} != (vocab {tok.vocab_size}, Spec.dim {sp.dim})"

    cls, sep = tok.cls_token_id, tok.sep_token_id
    rng = np.random.default_rng(0)
    picked = []
    for tid in rng.permutation(tok.vocab_size):
        text = tok.convert_ids_to_tokens(int(tid))
        if not text.isalpha():                       # skip specials, subwords, punctuation
            continue
        if tok(text)["input_ids"] != [cls, int(tid), sep]:
            continue                                 # text does not round-trip to this single id
        picked.append((int(tid), text))
        if len(picked) == N_SAMPLE:
            break

    ids = [t for t, _ in picked]
    texts = [x for _, x in picked]
    # prefix=pre.prefix so both paths build the same sequence; fp32 to isolate the read-out.
    enc = teacher.encode(texts, prefix=pre.prefix, max_length=sp.max_length, model_id=sp.repo,
                         revision=sp.revision, dtype=torch.float32, device="cuda")
    cos = (rows[ids].astype(np.float64) * enc.astype(np.float64)).sum(1)
    ok = bool(cos.min() >= COS_BAR)
    out = {"_note": "Init rows vs the production encode path on tokens whose text round-trips to "
                    "[CLS] tok [SEP]. Guards the read-out (pooling + post-pooling Dense + width) "
                    "that init_table.teacher_rows used to hardcode as CLS.",
           "encoder": sp.name, "spec_tag": spec_tag(), "pooling": sp.pooling,
           "post_dense": sp.post_dense, "dim": sp.dim,
           "n_tokens_compared": len(ids), "min_cosine": float(cos.min()),
           "mean_cosine": float(cos.mean()), "bar": COS_BAR,
           "status": "pass" if ok else "fail"}
    path = REPO / "results" / f"m7_init_rows_{sp.name}.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
