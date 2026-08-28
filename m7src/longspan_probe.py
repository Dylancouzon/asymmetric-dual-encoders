"""Does the table's agreement with its teacher degrade with query LENGTH?

Pre-registered in m7/LEDGER.md 2026-08-28 as a diagnostic: no qrels, no six-set access, no
adoption attached. It exists because the project has been carrying an unmeasured extrapolation.
`pseudoq._span` caps pseudo-queries at the first sentence and 32 words, real TRAIN queries sit at
p50=13 WordPiece, and the only long-query dev slice is 55 queries that are 54/55 HotpotQA — while
ArguAna, one of the six confirmatory datasets, has ~250-word queries and has been the
architecture's predicted worst case since M1. EXPLORED.md recorded "dev cannot test long-query
behaviour"; that is false for teacher-agreement metrics, which need no relevance labels.

Method: take document text, cut spans at several word lengths, and for each span compare what the
TABLE retrieves from the frozen pool against what the TEACHER retrieves from the same pool.
Agreement is overlap@10 plus the cosine between the two query vectors. Both systems see the same
spans and the same corpus, so any trend across buckets is a property of length alone.

Reading it: a flat curve says there is no length gap to close and kills the long-span distillation
lever before it costs a training chain. A falling curve says the gap is real and sizes it.

Usage: longspan_probe.py [<run_id>] [--smoke]
"""
import json
import sys
import time

import numpy as np
import torch

import dev_audit
import devsuite
import encoders
import heldout
from _paths import REPO, WORK
from evalkit import topk_arrays
from table import Preproc, ensure_release, get_tokenizer, load_table, read_meta
from teacher import QUERY_PREFIX, encode_cached

# Word-count buckets. The top two are ArguAna's regime; the bottom two are TRAIN's.
BUCKETS = [8, 16, 32, 64, 128, 256]
PER_BUCKET = 300
SOURCE = "cqadup-physics"          # dev-side text, already cached; the corpus searched is the pool
SEED = 0


def spans(texts, n_words, count, rng):
    """First `n_words` words of documents long enough to supply them, sampled deterministically."""
    out, order = [], rng.permutation(len(texts))
    for i in order:
        w = texts[i].split()
        if len(w) >= n_words:
            out.append(" ".join(w[:n_words]))
        if len(out) == count:
            break
    return out


def main(run_id=None, smoke=False):
    t0 = time.time()
    tok = get_tokenizer()
    spec = encoders.active()
    rng = np.random.default_rng(SEED)
    if run_id is None:
        run_id = json.loads((REPO / "results" / "m7_dev_audit_full.json").read_text())[
            "surviving_candidate"]
    rel = ensure_release(WORK / "runs" / f"{run_id}.npz")
    pre = Preproc(**read_meta(rel)["preproc"])
    m = load_table(rel, variant="fp16")
    print(f"long-span probe: {run_id}, pool_mode={pre.pool_mode!r}, buckets {BUCKETS}", flush=True)

    _, doc_texts, _, _, _ = devsuite.load(SOURCE)
    per_bucket = PER_BUCKET if not smoke else 25
    sp = {n: spans(doc_texts, n, per_bucket, rng) for n in BUCKETS}
    del doc_texts
    n_tok = {n: float(np.mean([len(tok(s, truncation=True, max_length=512)["input_ids"])
                               for s in sp[n]])) for n in BUCKETS}

    pool = heldout.doc_vectors("heldout-train")
    doc_ids = heldout.pool_doc_ids(len(pool))
    if smoke:
        pool, doc_ids = pool[:200_000], doc_ids[:200_000]

    blocks, meta = [], []
    for n in BUCKETS:
        tv = np.asarray(encode_cached(f"longspan-{SOURCE}-{n}w-{per_bucket}", sp[n],
                                      prefix=QUERY_PREFIX, dtype=torch.float16, verbose=False),
                        dtype=np.float32)
        qv = m.encode(sp[n], pre, tok=tok)
        blocks += [tv, qv]
        meta += [("teacher", n), ("table", n)]
    Q = np.concatenate(blocks)
    del blocks
    print(f"  {len(Q):,} query rows over {len(doc_ids):,} pool docs", flush=True)
    bi, _ = topk_arrays(Q, pool, k=10, chunk=200_000)

    # Q and bi are laid out [teacher(b0), table(b0), teacher(b1), table(b1), ...]
    rows = {}
    for bi_idx, n in enumerate(BUCKETS):
        s0 = bi_idx * 2 * per_bucket
        a, b = bi[s0:s0 + per_bucket], bi[s0 + per_bucket:s0 + 2 * per_bucket]
        ov = np.array([len(set(a[i]) & set(b[i])) / 10.0 for i in range(len(a))])
        tv, qv = Q[s0:s0 + per_bucket], Q[s0 + per_bucket:s0 + 2 * per_bucket]
        cos = (tv * qv).sum(1) / (np.linalg.norm(tv, axis=1) * np.linalg.norm(qv, axis=1) + 1e-12)
        rows[n] = {"n_spans": int(len(ov)), "mean_wordpieces": round(n_tok[n], 1),
                   "overlap@10_mean": float(ov.mean()), "overlap@10_sd": float(ov.std()),
                   "cosine_to_teacher_mean": float(cos.mean())}
        print(f"  {n:>3d} words (~{n_tok[n]:.0f} wp): overlap@10 {ov.mean():.4f}  "
              f"cos {cos.mean():.4f}", flush=True)

    short, long_ = rows[BUCKETS[0]], rows[BUCKETS[-1]]
    out = {"run_id": run_id, "encoder": spec.name, "source": SOURCE, "seed": SEED,
           "pool_docs": int(len(doc_ids)), "preproc": {"pool_mode": pre.pool_mode},
           "per_bucket": {str(k): v for k, v in rows.items()},
           "gap_overlap@10_short_minus_long": short["overlap@10_mean"] - long_["overlap@10_mean"],
           "gap_cosine_short_minus_long": short["cosine_to_teacher_mean"]
                                          - long_["cosine_to_teacher_mean"],
           "code_identity": dev_audit.code_identity(),
           "_what": "teacher-agreement vs query length. Diagnostic only: no qrels are read, no "
                    "adoption is attached, and agreement is not quality — a table could disagree "
                    "with its teacher and be better. It sizes a gap; it does not price a fix.",
           "_protocol": "m7/LEDGER.md, pre-registered 2026-08-28 before any number",
           "seconds": round(time.time() - t0, 1)}
    name = f"m7_longspan_probe{'_smoke' if smoke else ''}.json"
    (REPO / "results" / name).write_text(json.dumps(out, indent=1))
    print(f"  gap short->long: overlap@10 {out['gap_overlap@10_short_minus_long']:+.4f}, "
          f"cosine {out['gap_cosine_short_minus_long']:+.4f}  ({out['seconds']:.0f}s)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else None, smoke="--smoke" in sys.argv)
