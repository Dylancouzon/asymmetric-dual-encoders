"""Does the table's agreement with its teacher degrade with query LENGTH?

Pre-registered in m7/LEDGER.md 2026-08-28 as a diagnostic: no qrels, no six-set access, no
adoption attached. It exists because the project has been carrying an unmeasured extrapolation.
`pseudoq._span` caps pseudo-queries at the first sentence and 32 words, real TRAIN queries sit at
p50=13 WordPiece, and the only long-query dev slice is 55 queries that are 54/55 HotpotQA — while
ArguAna, one of the six confirmatory datasets, has ~250-word queries and has been the
architecture's predicted worst case since M1. EXPLORED.md recorded "dev cannot test long-query
behaviour"; that is false for teacher-agreement metrics, which need no relevance labels.

Method: sample documents long enough for the LONGEST bucket, cut every bucket as a prefix of those
same documents, and for each span compare what the TABLE retrieves from the frozen pool against
what the TEACHER retrieves from the same pool. Agreement is overlap@10 plus the cosine between the
two query vectors.

The nesting is what makes the trend attributable to length. The first version of this probe drew a
fresh document sample per bucket, so its buckets differed in length AND in document population
while the docstring claimed "length alone" -- the 2026-08-28 numbers below are from that version
and do not identify a length effect. Cost of the fix: the population is now long documents only,
so this measures length sensitivity *within long documents*.

Reading it: a flat curve says there is no length gap to close and kills the long-span distillation
lever before it costs a training chain. A falling curve says the gap is real and sizes it.

RESULT (2026-08-28, CORRECTED probe): overlap@10 is 0.3337 at 8 words and 0.3057 / 0.3023 / 0.3043
/ 0.3067 / 0.3087 at 16 / 32 / 64 / 128 / 256 -- FLAT from 16 words up, and slightly rising. The
first version's 0.3443 -> 0.2997 "fall" was the document-population confound described above.
Capacity lever #7 was CLOSED on this, without training its arm: this probe is its pre-condition
and the pre-condition is not met.

SECOND ROLE, added with lever #7: with more than one run id this becomes the lever's PRIMARY
adjudication instrument, pre-registered in m7/LEDGER.md before any arm ran. Every table sees the
same spans and the same pool, so the comparison is paired per span; the bar is a dependence-
preserving signflip p<0.05 and a raw paired CI above zero on overlap@10 over the pooled 128- and
256-word buckets. The dev macro is only the guardrail for that lever, because four of six dev
components are short-query and would dilute exactly the effect being bought.

Dependence, and why it is not optional here: `spans` takes the first n words of any document long
enough, so ONE document supplies a span in several buckets. Those observations are not independent
and the pooled comparison uses the document as the resampling unit.

Usage: longspan_probe.py [<run_id> ...] [--smoke]
       longspan_probe.py p4n-teacher16-a p7-longspan-a      # candidate first: it is the baseline
"""
import json
import sys
import time

import numpy as np
import torch

import boot
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
# The buckets lever #7's primary bar reads. Fixed in m7/LEDGER.md before the lever ran, and they
# are the two that bracket ArguAna's ~250-word queries.
LEVER7_BUCKETS = (128, 256)


def nested_spans(texts, buckets, count, rng):
    """ONE document sample, and every bucket is a PREFIX of the same documents.

    The previous version drew a fresh permutation per bucket and kept documents long enough for
    THAT bucket, so the 8-word and 256-word buckets differed in length *and* in which documents
    they came from -- and the probe's docstring nonetheless claimed "any trend across buckets is a
    property of length alone". It was not: the endpoint difference it reported confounded length
    with document population, and that probe is what licensed capacity lever #7. Found by the
    pre-freeze Codex review.

    Sampling documents eligible for the LONGEST bucket once, and cutting every shorter bucket from
    those same documents, makes length the only thing that varies and makes the document the
    resampling unit in every bucket. It also narrows the population to long documents, which is a
    real limitation and is reported rather than hidden: this measures length sensitivity *within
    long documents*, not across the corpus.
    """
    longest = max(buckets)
    out, order = {n: [] for n in buckets}, rng.permutation(len(texts))
    picked = 0
    for i in order:
        w = texts[int(i)].split()
        if len(w) < longest:
            continue
        for n in buckets:
            out[n].append((int(i), " ".join(w[:n])))
        picked += 1
        if picked == count:
            break
    return out


def _span_unit(ds, qid):
    """One resampling unit per DOCUMENT, shared across every bucket that document appears in."""
    return ("span", qid)


def main(run_ids=None, smoke=False):
    t0 = time.time()
    tok = get_tokenizer()
    spec = encoders.active()
    rng = np.random.default_rng(SEED)
    if not run_ids:
        run_ids = [json.loads((REPO / "results" / "m7_dev_audit_full.json").read_text())[
            "surviving_candidate"]]
    tables, pres = {}, {}
    for rid in run_ids:
        rel = ensure_release(WORK / "runs" / f"{rid}.npz")
        pres[rid] = Preproc(**read_meta(rel)["preproc"])
        tables[rid] = load_table(rel, variant="fp16")
    print(f"long-span probe: {run_ids}, pool_mode="
          f"{ {r: pres[r].pool_mode for r in run_ids} }, buckets {BUCKETS}", flush=True)

    _, doc_texts, _, _, _ = devsuite.load(SOURCE)
    per_bucket = PER_BUCKET if not smoke else 25
    # The rng is consumed bucket by bucket in BUCKETS order, so the spans depend only on SEED and
    # PER_BUCKET -- never on how many run ids were passed. That is what makes a lever arm's probe
    # comparable to the candidate's, and what lets a rerun reuse the teacher encode cache.
    sp = nested_spans(doc_texts, BUCKETS, per_bucket, rng)
    del doc_texts
    # Every downstream slice uses FIXED offsets, so a short bucket would shift every later
    # bucket's slices and pair one bucket's teacher rows with another's table rows -- which would
    # depress agreement most in the long buckets and fabricate exactly the falling curve this
    # probe exists to detect. With nested spans every bucket is filled by construction or none is.
    underfilled = {n: len(sp[n]) for n in BUCKETS if len(sp[n]) != per_bucket}
    if underfilled:
        raise SystemExit(f"{SOURCE} cannot supply {per_bucket} documents of >= {max(BUCKETS)} "
                         f"words (buckets {underfilled}); lower PER_BUCKET or pick a "
                         "longer-document source")
    assert len({tuple(d for d, _ in sp[n]) for n in BUCKETS}) == 1, \
        "nested spans must come from the same documents in the same order in every bucket"
    texts = {n: [t for _, t in sp[n]] for n in BUCKETS}
    docidx = {n: [i for i, _ in sp[n]] for n in BUCKETS}
    n_tok = {n: float(np.mean([len(tok(s, truncation=True, max_length=512)["input_ids"])
                               for s in texts[n]])) for n in BUCKETS}

    pool = heldout.doc_vectors("heldout-train")
    doc_ids = heldout.pool_doc_ids(len(pool))
    if smoke:
        pool, doc_ids = pool[:200_000], doc_ids[:200_000]

    # Layout, explicit rather than positional arithmetic: `order` names every block in the order
    # it is concatenated, so a block's slice is looked up instead of re-derived at each use.
    blocks, order = [], []
    for n in BUCKETS:
        blocks.append(np.asarray(
            encode_cached(f"longspan-{SOURCE}-{n}w-{per_bucket}", texts[n], prefix=QUERY_PREFIX,
                          dtype=torch.float16, verbose=False), dtype=np.float32))
        order.append(("teacher", n))
        for rid in run_ids:
            blocks.append(tables[rid].encode(texts[n], pres[rid], tok=tok))
            order.append((rid, n))
    Q = np.concatenate(blocks)
    del blocks
    slice_of = {tag: slice(i * per_bucket, (i + 1) * per_bucket) for i, tag in enumerate(order)}
    print(f"  {len(Q):,} query rows over {len(doc_ids):,} pool docs", flush=True)
    bi, _ = topk_arrays(Q, pool, k=10, chunk=200_000)

    def overlap(rid, n):
        a, b = bi[slice_of[("teacher", n)]], bi[slice_of[(rid, n)]]
        return np.array([len(set(a[i]) & set(b[i])) / 10.0 for i in range(len(a))])

    per_run, per_span = {}, {}
    for rid in run_ids:
        rows = {}
        for n in BUCKETS:
            ov = overlap(rid, n)
            tv, qv = Q[slice_of[("teacher", n)]], Q[slice_of[(rid, n)]]
            cos = (tv * qv).sum(1) / (np.linalg.norm(tv, axis=1) * np.linalg.norm(qv, axis=1)
                                      + 1e-12)
            rows[n] = {"n_spans": int(len(ov)), "mean_wordpieces": round(n_tok[n], 1),
                       "overlap@10_mean": float(ov.mean()), "overlap@10_sd": float(ov.std()),
                       "cosine_to_teacher_mean": float(cos.mean())}
            # keyed by DOCUMENT index, which is the resampling unit the pooled test groups by
            per_span.setdefault(rid, {})[f"bucket-{n}"] = {
                str(d): float(v) for d, v in zip(docidx[n], ov)}
            print(f"  [{rid}] {n:>3d} words (~{n_tok[n]:.0f} wp): overlap@10 {ov.mean():.4f}  "
                  f"cos {cos.mean():.4f}", flush=True)
        short, long_ = rows[BUCKETS[0]], rows[BUCKETS[-1]]
        per_run[rid] = {
            "preproc": {"pool_mode": pres[rid].pool_mode},
            "per_bucket": {str(k): v for k, v in rows.items()},
            "gap_overlap@10_short_minus_long": short["overlap@10_mean"] - long_["overlap@10_mean"],
            "gap_cosine_short_minus_long": short["cosine_to_teacher_mean"]
                                           - long_["cosine_to_teacher_mean"]}

    # --- lever #7's primary bar, when there is something to compare against --------------------
    comparisons = {}
    base = run_ids[0]
    for rid in run_ids[1:]:
        a = {k: v for k, v in per_span[rid].items()
             if int(k.split("-")[1]) in LEVER7_BUCKETS}
        b = {k: per_span[base][k] for k in a}
        pd_ = boot.paired_dep(a, b, alternative="two-sided", unit_of=_span_unit)
        sf = boot.signflip_dep(a, b, alternative="greater", unit_of=_span_unit)
        passes = sf["p"] < 0.05 and pd_["ci95_raw"][0] > 0
        comparisons[f"{rid}_vs_{base}"] = {
            "buckets": list(LEVER7_BUCKETS), "paired_dep": pd_, "signflip_dep": sf,
            "primary_bar_passes": bool(passes),
            "_bar": "m7/LEDGER.md capacity lever #7, pre-registered before any arm ran: "
                    "dependence-preserving signflip p<0.05 AND raw paired CI > 0 on overlap@10 "
                    "over the pooled 128- and 256-word buckets, resampling by DOCUMENT because one "
                    "document supplies spans in several buckets. This is the lever's PRIMARY bar; "
                    "adoption additionally needs the dev-suite non-inferiority guardrail, which "
                    "this script does not measure and cannot waive."}
        print(f"\n  {rid} vs {base} on buckets {list(LEVER7_BUCKETS)}: "
              f"delta {pd_['delta_raw']:+.4f} ci {pd_['ci95_raw']} p={sf['p']:.4f} -> "
              f"primary bar {'PASSES' if passes else 'FAILS'}")

    out = {"run_ids": run_ids, "run_id": base, "encoder": spec.name, "source": SOURCE, "seed": SEED,
           "pool_docs": int(len(doc_ids)), "per_bucket_by_run": per_run,
           "comparisons": comparisons,
           "code_identity": dev_audit.code_identity(),
           "_what": "teacher-agreement vs query length. As a single-run PROBE it is diagnostic "
                    "only: no qrels are read and agreement is not quality — a table could disagree "
                    "with its teacher and be better. With two run ids it is capacity lever #7's "
                    "pre-registered primary instrument.",
           "_protocol": "m7/LEDGER.md, pre-registered 2026-08-28 before any number",
           "seconds": round(time.time() - t0, 1)}
    # single-run runs keep the original flat shape too, so the committed probe stays readable
    if len(run_ids) == 1:
        out.update({k: per_run[base][k] for k in
                    ("preproc", "per_bucket", "gap_overlap@10_short_minus_long",
                     "gap_cosine_short_minus_long")})
    tag = "_smoke" if smoke else ("" if len(run_ids) == 1 else "_lever7")
    (REPO / "results" / f"m7_longspan_probe{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"  gap short->long [{base}]: overlap@10 "
          f"{per_run[base]['gap_overlap@10_short_minus_long']:+.4f}, cosine "
          f"{per_run[base]['gap_cosine_short_minus_long']:+.4f}  ({out['seconds']:.0f}s)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args, smoke="--smoke" in sys.argv)
