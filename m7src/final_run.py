"""THE final run. One shot, guarded.

Refuses to start unless:
  * the working tree is clean and HEAD equals the freeze commit PUSHED to origin (the remote
    timestamp is the external witness),
  * m7/LEDGER.md holds no prior final-run entry (unless --infra-retry, which is allowed once
    per crash with identical commit, config and inputs and is itself logged).

It is the sole reader of six-set and untouched-final qrels, reads them only from
results/frozen_eval/ after verifying eval_manifest.json corpus hashes against a fresh HF
download, scores the six first and the untouched-final sets after, and appends every access
to m7/LEDGER.md itself.

Confirmatory decisions are exactly three, one-sided, Holm step-down at family alpha = 0.025:
  C1  released int8 dense table   >  lr-dense-pertask (0.4583)   -- Tier 2, the release bar
  C2  released int8 dense table   >  bm25 (0.4174)               -- Tier 3
  C3  released system             >  opensearch-doc-v3-gte (0.4868) -- Tier 1, the aim
Everything else this script prints is exploratory and labeled as such.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

import boot
import fusion
from _paths import REPO
from core import DATASETS, load_beir
from evalkit import per_query_ndcg, topk_ids_scores
import freeze
from table import Preproc, load_table
from teacher import QUERY_PREFIX, encode_cached

LEDGER = REPO / "m7" / "LEDGER.md"
FROZEN = REPO / "results" / "frozen_eval"
MANIFEST = REPO / "results" / "eval_manifest.json"
UNTOUCHED = ["fever", "dbpedia-entity"]
CONFIRMATORY = {"C1_int8_table_gt_lr_dense_pertask": ("int8-table", "lr-dense-pertask"),
                "C2_int8_table_gt_bm25": ("int8-table", "bm25"),
                "C3_released_system_gt_opensearch": ("released-system", "opensearch-doc-v3-gte")}
ALPHA = 0.025


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, cwd=REPO).stdout.strip()


def ledger(line):
    with open(LEDGER, "a") as f:
        f.write(line.rstrip() + "\n")


SIX = ["scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid"]


def guard(freeze_hash, infra_retry, branch, fz):
    problems = []
    # BENCH_DATASETS is an env var with an M2-era five-dataset default; a stale export in the
    # same shell would silently redefine "the six" and the macro computed over it.
    if list(DATASETS) != SIX:
        problems.append(f"DATASETS is {list(DATASETS)}, not the six (check BENCH_DATASETS)")
    if sh("git", "status", "--porcelain"):
        problems.append("working tree is not clean")
    head = sh("git", "rev-parse", "HEAD")
    if head != freeze_hash:
        problems.append(f"HEAD {head[:12]} != freeze commit {freeze_hash[:12]}")
    remote = sh("git", "ls-remote", "origin", f"refs/heads/{branch}").split("\t")[0]
    if remote != freeze_hash:
        problems.append(f"freeze commit is not pushed: origin/{branch} is {remote[:12]}")
    text = LEDGER.read_text()
    prior_hashes = re.findall(r"FINAL-RUN-BEGIN freeze=([0-9a-f]{40}) table=([0-9a-f]{64})", text)
    if prior_hashes and not infra_retry:
        problems.append("m7/LEDGER.md already holds a final-run entry; a code fix requires a NEW "
                        "pushed freeze commit, and no later run may be relabeled as final")
    if infra_retry:
        if not prior_hashes:
            problems.append("--infra-retry with no prior FINAL-RUN entry in the ledger")
        else:
            pf, pt = prior_hashes[-1]
            if pt != fz["table_sha256"]:
                problems.append("--infra-retry with a different table than the aborted run")
            # only the ledger may have changed since the aborted run's freeze commit
            changed = [l for l in sh("git", "diff", "--name-only", f"{pf}..HEAD").splitlines() if l]
            stray = [c for c in changed if c != "m7/LEDGER.md"]
            if stray:
                problems.append("--infra-retry is for infrastructure only, but these files changed "
                                f"since the aborted run's freeze commit: {stray}. A code change "
                                "requires a new pushed freeze commit with the diff and its "
                                "classification.")
    if problems:
        print("FINAL RUN REFUSED:\n  - " + "\n  - ".join(problems))
        sys.exit(2)
    return head


def verify_and_load(ds, kind):
    """kind: 'six' | 'untouched'. Verifies corpus hashes, then reads labels from frozen_eval only."""
    key = {"six": "datasets", "untouched": "m7_untouched_final"}[kind]
    man = json.loads(MANIFEST.read_text())[key][ds]
    if kind == "six":
        doc_ids, doc_texts, *_ = load_beir(ds)
    else:
        from datasets import load_dataset
        from core import doc_text
        corpus = load_dataset(f"BeIR/{ds}", "corpus")["corpus"]
        doc_ids, doc_texts = [str(x) for x in corpus["_id"]], [doc_text(r) for r in corpus]
    from hashing import sha_stream_list
    for field, got in (("corpus_ids_sha256", sha_stream_list(doc_ids)),
                       ("corpus_text_sha256", sha_stream_list(doc_texts)),
                       ("n_docs", len(doc_ids))):
        if man[field] != got:
            print(f"FINAL RUN ABORTED: {ds}.{field} mismatch vs the frozen manifest")
            sys.exit(3)
    name = f"{ds}.json" if kind == "six" else f"untouched-{ds}.json"
    froz = json.loads((FROZEN / name).read_text())
    q_ids = sorted(froz["queries"])
    ledger(f"- {datetime.now(timezone.utc).isoformat()} — **FINAL-RUN access** ({kind}) `{ds}`: "
           f"{len(doc_ids):,} docs / {len(q_ids):,} queries, corpus hashes verified against a fresh "
           f"HF download, labels read from `results/frozen_eval/{name}`.")
    return doc_ids, doc_texts, q_ids, [froz["queries"][q] for q in q_ids], froz["qrels"]


def score_set(ds, kind, table_path, pre, fusion_spec, encode_dtype=torch.float32):
    doc_ids, doc_texts, q_ids, q_texts, qrels = verify_and_load(ds, kind)
    dv = encode_cached(f"final-{kind}-{ds}-docs", doc_texts, prefix="", dtype=encode_dtype)
    tag = "six" if kind == "six" else "unt"
    tqv = np.asarray(encode_cached(f"final-{tag}-{ds}-q-pfx", q_texts, prefix=QUERY_PREFIX,
                                   dtype=encode_dtype), dtype=np.float32)
    chunk = 200_000
    out, runs = {}, {}
    for variant in ("fp16", "int8"):
        m = load_table(table_path, variant=variant)
        runs[f"{variant}-table"] = topk_ids_scores(m.encode(q_texts, pre), dv, doc_ids,
                                                  k=fusion.DEPTH, chunk=chunk, qids=q_ids)
        del m
        torch.cuda.empty_cache()
    runs["bm25"] = fusion.bm25_run(doc_ids, doc_texts, q_ids, q_texts)  # the shared builder, B5
    runs["teacher-symmetric"] = topk_ids_scores(tqv, dv, doc_ids, k=fusion.DEPTH, chunk=chunk, qids=q_ids)
    if fusion_spec:
        runs["fusion"] = fusion.apply_frozen(fusion_spec, runs["int8-table"], runs["bm25"])
    for k, r in runs.items():
        out[k] = per_query_ndcg(r, qrels)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-hash", required=True)
    ap.add_argument("--branch", default="m7-query-encoder")
    ap.add_argument("--infra-retry", action="store_true")
    a = ap.parse_args()

    # everything decisive comes from the committed freeze manifest, not the command line
    fz = freeze.load_and_verify()
    spec = fz["fusion"]
    pre = Preproc(**fz["preproc"])
    if pre.fingerprint() != fz["preproc_fingerprint"]:
        raise SystemExit("FINAL RUN REFUSED: preproc fingerprint does not match its own fields")
    table_path = REPO / fz["table_relpath"]

    head = guard(a.freeze_hash, a.infra_retry, a.branch, fz)
    t0 = time.time()
    ledger(f"\n### FINAL-RUN {datetime.now(timezone.utc).isoformat()}\n"
           f"- FINAL-RUN-BEGIN freeze={head} table={fz['table_sha256']}\n"
           f"- pushed to origin/{a.branch}; table `{fz['table_relpath']}` "
           f"({fz['table_bytes']} bytes), preproc `{fz['preproc']}` "
           f"(fingerprint {fz['preproc_fingerprint']}), fusion `{json.dumps(spec)}`, "
           f"released system `{fz['released_system']}`"
           f"{', INFRA-RETRY' if a.infra_retry else ''}.")

    six = {ds: score_set(ds, "six", table_path, pre, spec) for ds in DATASETS}
    print("\n=== the six (KNOWN-TEST, development-informed) ===")
    systems = sorted({k for v in six.values() for k in v})
    by_sys = {s: {ds: six[ds][s] for ds in DATASETS if s in six[ds]} for s in systems}
    for s in systems:
        means = {ds: float(np.mean(list(v.values()))) for ds, v in by_sys[s].items()}
        print(f"  {s:20s} avg-6 {np.mean(list(means.values())):.4f}  " +
              " ".join(f"{d}={means[d]:.4f}" for d in DATASETS))

    pq = json.load(open(REPO / "results" / "perquery.json"))
    by_sys["released-system"] = by_sys["fusion" if fz["released_system"] == "fusion" else "int8-table"]
    conf, pvals = {}, {}
    for name, (a_name, b_name) in CONFIRMATORY.items():
        A = by_sys[a_name]
        # ALWAYS the frozen per-query comparator vectors, never a locally recomputed row: the
        # mandate says "never re-run a comparator system". A fresh BM25 run exists in by_sys for
        # the fusion input and as an exploratory row, and must not be used here.
        B = boot.from_perquery_json(pq, b_name, set(DATASETS))
        if not B:
            raise SystemExit(f"FINAL RUN ABORTED: no frozen per-query vectors for {b_name}")
        r = boot.paired(A, B, alternative="greater")            # intervals only (B3)
        t = boot.signflip(A, B, alternative="greater", strict=True)  # THE p-value (B3)
        r["signflip"] = t
        conf[name] = r
        pvals[name] = t["p"]
    decisions = boot.holm(pvals, alpha=ALPHA)

    print("\n=== confirmatory (one-sided, Holm, family alpha=0.025) ===")
    for name in CONFIRMATORY:
        d, h = conf[name], decisions[name]
        print(f"  {'REJECT H0' if h['reject'] else 'not resolved'}  {name}: d={d['delta']:+.4f} "
              f"CI={d['ci95']} p={d['signflip']['p_str']} (sign-flip) "
              f"thr={h['threshold']:.4f}")

    print("\n=== untouched-final (scored after the six; no recipe change after this point) ===")
    unt = {ds: score_set(ds, "untouched", table_path, pre, spec) for ds in UNTOUCHED}
    for ds in UNTOUCHED:
        print(f"  {ds}: " + " ".join(f"{s}={np.mean(list(v.values())):.4f}" for s, v in unt[ds].items()))

    blob = {"freeze_commit": head, "freeze": fz, "infra_retry": bool(a.infra_retry),
            "six": {s: {ds: {q: round(x, 6) for q, x in v.items()} for ds, v in by_sys[s].items()}
                    for s in systems},
            "untouched_final": {ds: {s: {q: round(x, 6) for q, x in v.items()}
                                     for s, v in unt[ds].items()} for ds in UNTOUCHED},
            "confirmatory": conf, "holm": decisions, "alpha": ALPHA,
            "seconds": round(time.time() - t0, 1)}
    (REPO / "results" / "m7_final_run.json").write_text(json.dumps(blob, indent=1))
    tiers = [k for k, v in decisions.items() if v["reject"]]
    ledger(f"- FINAL-RUN complete in {blob['seconds']:.0f}s. Confirmatory rejections: "
           f"{tiers or 'none'}. Results in `results/m7_final_run.json`.")
    print("\nwrote results/m7_final_run.json")


if __name__ == "__main__":
    main()
