"""Rank candidate teachers by MEASURING them, on the two dev components that are clean for all.

Why this exists. The M7 teacher projection maps MTEB v1 English Retrieval onto our six-set.
results/m7_calibration.json shows that map is loose (ratio 0.926-1.001 across the nine models we
measured; affine resid sd 0.0102), so a 1-point MTEB gap between two candidates is inside the
calibration noise and cannot rank them. A direct measurement can.

Which components. NOT the six -- those are the final eval and must not inform a selection.
NOT nq-250k or hotpotqa either: MTEB's registry records NQ, HotpotQA, FEVER, MSMARCO, ArguAna and
FiQA2018 as in-domain training data for stella_en_400M_v5, so those components would flatter it
against bge-base. CQADupStack is on no candidate's disclosed list, is real qrels, and the ledger
already flags it as the nearest dev analogue to FiQA -- the six-set row most at risk. So the two
CQADupStack components are the honest common ground.

What it reports: each candidate's own SYMMETRIC retrieval score, i.e. the teacher ceiling our
table would be distilling toward, in our units and our harness. Not a table, not a gate input.

Cost: ~70K docs + ~1.9K queries per candidate.
"""
import json
import sys

import numpy as np
import torch

import boot
import dev_eval
import encoders
from _paths import REPO, WORK
from evalkit import per_query_ndcg, topk_ids_scores

COMPONENTS = ("cqadup-programmers", "cqadup-physics")
OUT = WORK / "teacherprobe"
OUT.mkdir(parents=True, exist_ok=True)
# Candidates come from the shared registry in encoders.py -- pooling and prompt live in ONE place.
# An earlier version of this file kept its own copy of that table, which is how a comparison ends
# up silently running a mean-pooled model with CLS pooling.
CANDS = ("bge-base-en-v1.5", "bge-large-en-v1.5", "gte-large-en-v1.5", "stella-400M-v5",
         "arctic-embed-l")


# The loader, the pooling, the post-pooling Dense AND the batching all come from teacher.py, so
# the probe cannot drift from the encoder the rest of the pipeline uses.
import teacher
from teacher import release_teacher


def encode(spec, texts, prefix, tag):
    """fp16 encode via teacher.encode, cached per (candidate, tag) so reruns are free.

    This used to reimplement teacher.encode's length-bucketed batching, and got it wrong: it sized
    each batch from the SHORTEST sequence in it while the tokenizer pads to the LONGEST, so a batch
    starting at 96 tokens could take the 256 cap and pad to 512 -- 131K tokens against a 24,576
    budget. bge-base absorbed it; bge-large (1024-d) sat on the 10 GB ceiling and the allocator
    thrashed, turning a ~2-minute component into 50+ minutes at 100% GPU "utilization" and 1%
    memory bandwidth. Do not fork the harness's batching; call it.
    """
    p = OUT / f"{spec.name}-{tag}.npy"
    if p.exists():
        return np.load(p)
    v = teacher.encode(texts, prefix=prefix, max_length=spec.max_length, batch_tokens=16384,
                       model_id=spec.repo, revision=spec.revision, dtype=torch.float16,
                       device="cuda", verbose=True).astype(np.float16)
    np.save(p, v)
    return v


def main(names):
    res, per_all = {}, {}
    for name in names:
        spec = encoders.get(name)
        per_comp = {}
        for c in COMPONENTS:
            doc_ids, doc_texts, q_ids, q_texts, qrels, _ = dev_eval.doc_vecs(c)
            dv = encode(spec, doc_texts, spec.doc_prefix, f"{c}-docs")
            qv = encode(spec, q_texts, spec.query_prefix, f"{c}-q")
            run = topk_ids_scores(torch.from_numpy(qv.astype(np.float32)), dv, doc_ids,
                                 k=10, chunk=250_000, qids=q_ids)
            pq = per_query_ndcg(run, qrels)
            per_comp[c] = pq
            print(f"  {name:20s} {c:20s} nDCG@10 {np.mean(list(pq.values())):.4f}", flush=True)
        # Drop this candidate's weights before loading the next one. load_teacher memoizes, which
        # is right for a single-encoder pipeline and wrong for a loop over five 400M-1.3B models.
        release_teacher(spec.repo, spec.revision, torch.float16, "cuda")
        per_all[name] = per_comp
        macro = float(np.mean([np.mean(list(per_comp[c].values())) for c in COMPONENTS]))
        res[name] = {"repo": spec.repo, "pooling": spec.pooling,
                     "query_prefix": spec.query_prefix, "dim": int(dv.shape[1]),
                     "macro_cqadupstack": round(macro, 4),
                     "per_component": {c: round(float(np.mean(list(per_comp[c].values()))), 4)
                                       for c in COMPONENTS}}
        print(f"{name:20s} MACRO {macro:.4f}\n", flush=True)

    # Point macros over ~1.9K queries cannot rank candidates any better than the projection this
    # probe replaces, so every candidate is paired-bootstrapped against the current teacher. A
    # candidate that does not CI-resolve above bge-base has not earned a corpus re-encode.
    if "bge-base-en-v1.5" in per_all:
        ref = per_all["bge-base-en-v1.5"]
        for k in res:
            if k == "bge-base-en-v1.5":
                continue
            r = boot.paired(per_all[k], ref, alternative="greater")
            res[k]["vs_current_teacher_boot"] = r
            print(f"  {k:20s} vs bge-base: d={r['delta']:+.4f} CI={r['ci95']} p={r['p_str']} "
                  f"{'RESOLVED' if r['resolved'] else 'UNRESOLVED'}")

    # ALL PAIRS, not just vs the incumbent. The probe exists because the MTEB->six projection
    # cannot order the front-runners, and it turned out not to order them the way MTEB does either
    # (arctic-embed-l has the LOWEST MTEB v1 of the three 1024-d candidates and the highest measured
    # macro here), so "which candidate wins" needs its own interval rather than two deltas against
    # a third model differenced by eye.
    ordered = [k for k in names if k in per_all]
    res_pairs = {}
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            r = boot.paired(per_all[a], per_all[b], alternative="two-sided")
            res_pairs[f"{a}__vs__{b}"] = r
            print(f"  {a:20s} vs {b:20s} d={r['delta']:+.4f} CI={r['ci95']} p={r['p_str']} "
                  f"{'RESOLVED' if r['resolved'] else 'UNRESOLVED'}", flush=True)

    base = res.get("bge-base-en-v1.5", {}).get("macro_cqadupstack")
    if base:
        for k in res:
            res[k]["vs_current_teacher"] = round(res[k]["macro_cqadupstack"] - base, 4)
            res[k]["ratio_to_current_teacher"] = round(res[k]["macro_cqadupstack"] / base, 4)
    # MERGE, do not overwrite. Run for one candidate, this file used to be rewritten with that
    # candidate alone -- it silently replaced five measured ceilings with one, exactly the bug
    # validate_encoder.py had. Entries are keyed by candidate and the pairwise block is recomputed
    # from whatever survives, so a stale pairwise table cannot outlive the rows it compared.
    prior_p = REPO / "results" / "m7_teacher_probe.json"
    if prior_p.exists():
        prior = json.loads(prior_p.read_text()).get("candidates", {})
        for k, v in prior.items():
            res.setdefault(k, v)
    out = {"_note": "Candidate teacher ceilings measured on the two CQADupStack dev components "
                    "-- the only dev components on no candidate's disclosed training list, and "
                    "the nearest dev analogue to FiQA. Symmetric retrieval, each model's own "
                    "pooling and prompt, fp16, our harness. A SELECTION diagnostic on dev: not "
                    "a gate input, and the six-set is never read. Use this to rank teachers "
                    "instead of the loose MTEB->six map in results/m7_calibration.json.",
           "_pairwise_note": "All-pairs two-sided paired bootstrap over the same per-query scores. "
                             "A SELECTION diagnostic, so no Holm correction is applied and none is "
                             "claimed -- the mandate's family-wise alpha governs gate claims on "
                             "the six, not this. Read a pair as resolved only if its CI excludes "
                             "0.",
           "_pairwise_scope": "Pairwise rows exist only for candidates measured in the SAME "
                              "invocation -- per-query scores are not persisted, so a merged-in "
                              "candidate has a ceiling here but no pairwise row. Re-run the "
                              "candidates together to compare them.",
           "components": list(COMPONENTS), "candidates": res, "pairwise": res_pairs}
    (REPO / "results" / "m7_teacher_probe.json").write_text(json.dumps(out, indent=1))
    print("wrote results/m7_teacher_probe.json")


if __name__ == "__main__":
    main(sys.argv[1:] or list(CANDS))
