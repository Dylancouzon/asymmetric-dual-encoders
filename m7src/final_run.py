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
from hashing import sha
from table import Preproc, load_table
from teacher import QUERY_PREFIX, encode_cached

LEDGER = REPO / "m7" / "LEDGER.md"
FROZEN = REPO / "results" / "frozen_eval"
MANIFEST = REPO / "results" / "eval_manifest.json"
UNTOUCHED = ["fever", "dbpedia-entity", "cqadup-android", "cqadup-english"]
CONFIRMATORY = {"C1_int8_table_gt_lr_dense_pertask": ("int8-table", "lr-dense-pertask"),
                "C2_int8_table_gt_bm25": ("int8-table", "bm25"),
                "C3_released_system_gt_opensearch": ("released-system", "opensearch-doc-v3-gte")}
ALPHA = 0.025


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, cwd=REPO).stdout.strip()


def sh_raw(*a):
    """`sh` without the strip. Column-oriented git output loses its first field to `.strip()`."""
    return subprocess.run(a, capture_output=True, text=True, cwd=REPO).stdout


def ledger(line):
    with open(LEDGER, "a") as f:
        f.write(line.rstrip() + "\n")


FREEZE_TAG = "m7-freeze"
# One aborted attempt may be retried; a second is an infrastructure problem, not a run.
MAX_INFRA_RETRIES = 2
SIX = ["scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid"]


def guard(freeze_hash, infra_retry, branch, fz):
    problems = []
    # BENCH_DATASETS is an env var with an M2-era five-dataset default; a stale export in the
    # same shell would silently redefine "the six" and the macro computed over it.
    if list(DATASETS) != SIX:
        problems.append(f"DATASETS is {list(DATASETS)}, not the six (check BENCH_DATASETS)")
    # The scorer itself appends to the ledger and the access trail, so a crashed attempt leaves
    # those two files changed. An infra retry must tolerate exactly that and nothing else
    # (review #2 MAJOR 20: the old guard could never pass after any post-BEGIN crash).
    ALLOWED_DRIFT = {"m7/LEDGER.md", "m7/SIX_ACCESS.log"}
    # `git status --porcelain` is "XY PATH" with XY exactly two columns, so the path starts at 3 --
    # but `sh` strips the whole output, which eats the leading space of the FIRST line when its
    # status is " M". That silently truncated the first dirty path ("m7/LEDGER.md" -> "7/LEDGER.md"),
    # so ALLOWED_DRIFT never matched it and a legitimate --infra-retry would have been refused for
    # the very file the retry is allowed to touch. Split without stripping the leading column.
    dirty = [l[3:].strip() for l in sh_raw("git", "status", "--porcelain").splitlines() if l.strip()]
    stray_dirty = [d for d in dirty if d not in ALLOWED_DRIFT]
    if stray_dirty:
        problems.append(f"working tree is not clean beyond the scorer's own files: {stray_dirty}")
    if dirty and not infra_retry:
        problems.append("working tree is not clean")
    head = sh("git", "rev-parse", "HEAD")
    if head != freeze_hash and not infra_retry:
        problems.append(f"HEAD {head[:12]} != freeze commit {freeze_hash[:12]}")
    remote = sh("git", "ls-remote", "origin", f"refs/heads/{branch}").split("\t")[0]
    if remote != freeze_hash:
        problems.append(f"freeze commit is not pushed: origin/{branch} is {remote[:12]}")
    # The freeze commit is resolved from an immutable pushed TAG, not taken on the caller's word.
    # `--freeze-hash` was the only identification of "the reviewed freeze commit", so any clean
    # pushed HEAD could be declared one (Codex one-shot-path review, BLOCKER 3).
    tagged = sh("git", "ls-remote", "origin", f"refs/tags/{FREEZE_TAG}").split("\t")[0]
    if not tagged:
        problems.append(f"no pushed tag `{FREEZE_TAG}`: the freeze commit must be marked by an "
                        f"immutable pushed tag, not asserted on the command line. "
                        f"`git tag -a {FREEZE_TAG} -m ... <commit> && git push origin {FREEZE_TAG}`")
    elif tagged != freeze_hash:
        problems.append(f"--freeze-hash {freeze_hash[:12]} != the pushed tag `{FREEZE_TAG}` "
                        f"({tagged[:12]}); the tag is authoritative")

    text = LEDGER.read_text()
    prior_hashes = re.findall(r"FINAL-RUN-BEGIN freeze=([0-9a-f]{40}) table=([0-9a-f]{64})", text)
    n_complete = len(re.findall(r"FINAL-RUN complete in", text))
    n_begin = len(prior_hashes)
    if n_complete:
        # A COMPLETE run is the one shot. Retry existed for an aborted run; it did not check
        # whether the prior run had finished, so a completed result could be re-rolled forever.
        problems.append(f"m7/LEDGER.md records {n_complete} COMPLETED final run(s). There is "
                        "exactly one confirmatory access and it has been spent; a further run is "
                        "a NEW milestone with its own pre-registration, never a retry.")
    if infra_retry and n_begin > MAX_INFRA_RETRIES:
        problems.append(f"--infra-retry already used {n_begin - 1} time(s); the cap is "
                        f"{MAX_INFRA_RETRIES - 1}. Repeated infrastructure failure is a reason to "
                        "fix the infrastructure, not to keep drawing from the six.")
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
            stray = [c for c in changed if c not in ALLOWED_DRIFT]
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
    elif ds.startswith("cqadup-"):
        import devsuite
        doc_ids, doc_texts, *_ = devsuite.load(ds)
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


def preflight():
    """Everything static that must hold, checked BEFORE the six are touched.

    Exists because the failure it prevents is the worst one available: `UNTOUCHED` named four
    datasets while the manifest and `frozen_eval/` held two, so this script would have scored the
    six -- consuming the ONE confirmatory access -- computed the tier decisions, scored two
    untouched sets, then died on a KeyError and written NOTHING. The access would have been spent
    and no result produced. Found by the pre-freeze Codex review, 2026-08-28.

    So: every manifest key, every frozen payload, every required field, for all ten datasets,
    before `FINAL-RUN-BEGIN`. A missing asset must cost a second, not the deliverable.
    """
    man = json.loads(MANIFEST.read_text())
    need = {"six": ("datasets", DATASETS, lambda d: f"{d}.json"),
            "untouched": ("m7_untouched_final", UNTOUCHED, lambda d: f"untouched-{d}.json")}
    fields = ("n_docs", "n_queries", "corpus_ids_sha256", "corpus_text_sha256",
              "qids_sha256", "qrels_sha256")
    problems = []
    for kind, (key, names, fname) in need.items():
        sect = man.get(key) or {}
        for ds in names:
            if ds not in sect:
                problems.append(f"{kind} `{ds}`: no `{key}` entry in eval_manifest.json")
                continue
            missing = [f for f in fields if f not in sect[ds]]
            if missing:
                problems.append(f"{kind} `{ds}`: manifest entry lacks {missing}")
            fp = FROZEN / fname(ds)
            if not fp.exists():
                problems.append(f"{kind} `{ds}`: {fp.relative_to(REPO)} missing")
                continue
            try:
                froz = json.loads(fp.read_text())
            except Exception as e:
                problems.append(f"{kind} `{ds}`: {fp.name} unreadable ({type(e).__name__})")
                continue
            if not isinstance(froz.get("queries"), dict) or not froz["queries"]:
                problems.append(f"{kind} `{ds}`: {fp.name} has no `queries` mapping")
            if not isinstance(froz.get("qrels"), dict) or not froz["qrels"]:
                problems.append(f"{kind} `{ds}`: {fp.name} has no `qrels` mapping")
            # the payload the scorer actually reads must match the hashes the manifest pins.
            # verify_and_load checks the CORPUS only; queries and qrels were never verified, so
            # changed query text with unchanged qids would have scored silently (Codex MAJOR 3).
            if isinstance(froz.get("queries"), dict) and isinstance(froz.get("qrels"), dict):
                for field, got in (("qids_sha256", sha(sorted(froz["queries"]))),
                                   ("qrels_sha256", sha(froz["qrels"]))):
                    if sect[ds].get(field) != got:
                        problems.append(f"{kind} `{ds}`: {fp.name} {field} mismatch vs the "
                                        f"frozen manifest")
    pq = REPO / "results" / "perquery.json"
    if not pq.exists():
        problems.append("results/perquery.json missing: the frozen comparator vectors")
    else:
        blob = json.loads(pq.read_text())
        for _, (a_name, b_name) in CONFIRMATORY.items():
            for sysname in (a_name, b_name):
                if sysname in ("int8-table", "released-system"):
                    continue      # produced by this run, not read from the frozen file
                for ds in DATASETS:
                    if sysname not in blob["datasets"].get(ds, {}).get("systems", {}):
                        problems.append(f"perquery.json: comparator `{sysname}` has no row for "
                                        f"`{ds}`")
                        break
    if problems:
        print("FINAL RUN REFUSED by preflight (the six were NOT touched):\n  - "
              + "\n  - ".join(problems))
        sys.exit(4)
    print(f"preflight OK: {len(DATASETS)} six + {len(UNTOUCHED)} untouched, manifest entries, "
          f"frozen payloads and comparator rows all present and hash-matched")


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

    preflight()                    # static checks first: a missing asset costs a second, not the run
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
        # strict=True: the registered rule is strict alignment on EVERY confirmatory path.
        # The sign-flip call below is strict and would abort first, so nothing could have
        # shrunk silently -- but relying on a neighbouring call for a guarantee is how the
        # guarantee gets lost when the calls are reordered.
        r = boot.paired(A, B, alternative="greater", strict=True)   # intervals only (B3)
        t = boot.signflip(A, B, alternative="greater", strict=True)  # THE p-value (B3)
        r["signflip"] = t
        conf[name] = r
        pvals[name] = t["p"]
    decisions = boot.holm(pvals, alpha=ALPHA)
    # Tier rule (pre-registered, review #2 BLOCKER 5 + MAJOR 8): a tier win requires BOTH the
    # Holm-corrected sign-flip rejection AND the paired-bootstrap CI resolved above zero. The
    # mandate's tier text is written in CIs; the sign-flip carries the multiplicity control; the
    # conjunction satisfies both and is conservative under either's failure mode.
    n_fam = len(CONFIRMATORY)
    for name in CONFIRMATORY:
        h = decisions[name]
        h["reject_holm_signflip"] = h["reject"]
        # ci95_raw, NEVER ci95: the display field is rounded to four decimals, so a true
        # lower endpoint of +4e-5 reads as 0.0000 and a resolved tier becomes "unresolved"
        # on the ONE-SHOT run. boot.py and m7/LEDGER.md both state the raw-endpoint rule;
        # this line broke it, in the single place where it is irreversible.
        h["ci_resolved"] = bool(conf[name]["ci95_raw"][0] > 0)
        # SIMULTANEOUS bound, added 2026-08-28 before any confirmatory number existed. Three
        # separate one-sided 2.5% intervals are not a family-wise 2.5%, and the sign-flip leg is
        # exact only under the SHARP null -- the repo's own weak-null simulation rejects at 0.038
        # for a nominal 0.025 on the worst pair. Bonferroni over the family puts each bound at
        # 0.025/3 = 0.8333%, where that same simulation measures 0.013 and 0.008. Strictly harder
        # than the previous rule, and fixed before the numbers, which is the only time a bar may
        # move.
        lb = (conf[name].get("one_sided_lower_raw") or {}).get("0.8333")
        h["bonferroni_level"] = round(ALPHA / n_fam, 6)
        h["one_sided_lower_bonferroni"] = lb
        h["ci_resolved_simultaneous"] = bool(lb is not None and lb > 0)
        h["reject"] = bool(h["reject_holm_signflip"] and h["ci_resolved"]
                           and h["ci_resolved_simultaneous"])

    print(f"\n=== confirmatory (one-sided; tier = Holm(sign-flip) AND CI>0 AND simultaneous "
          f"Bonferroni lower bound at {ALPHA}/{n_fam} > 0) ===")
    for name in CONFIRMATORY:
        d, h = conf[name], decisions[name]
        print(f"  {'REJECT H0' if h['reject'] else 'not resolved'}  {name}: d={d['delta']:+.4f} "
              f"CI={d['ci95']} p={d['signflip']['p_str']} (sign-flip) "
              f"thr={h['threshold']:.4f} ci_resolved={h['ci_resolved']}")

    # Pre-registered clean-4 robustness (teacher-exposure restriction): same comparisons on the
    # four datasets with no disclosed teacher benchmark overlap. Labeled, never a tier decision.
    CLEAN4 = {"scifact", "nfcorpus", "scidocs", "trec-covid"}
    robustness = {}
    for name, (a_name, b_name) in CONFIRMATORY.items():
        A4 = {ds: v for ds, v in by_sys[a_name].items() if ds in CLEAN4}
        B4 = boot.from_perquery_json(pq, b_name, CLEAN4)
        rr = boot.paired(A4, B4, alternative="greater", strict=True)
        rr["signflip"] = boot.signflip(A4, B4, alternative="greater", strict=True)
        robustness[name] = rr
        print(f"  [clean-4 robustness] {name}: d={rr['delta']:+.4f} CI={rr['ci95']} "
              f"p={rr['signflip']['p_str']}")

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
            "clean4_robustness": {"_note": "pre-registered exposure-restricted robustness "
                                  "(no disclosed teacher benchmark overlap); NOT a tier decision",
                                  **robustness},
            "seconds": round(time.time() - t0, 1)}
    (REPO / "results" / "m7_final_run.json").write_text(json.dumps(blob, indent=1))
    tiers = [k for k, v in decisions.items() if v["reject"]]
    ledger(f"- FINAL-RUN complete in {blob['seconds']:.0f}s. Confirmatory rejections: "
           f"{tiers or 'none'}. Results in `results/m7_final_run.json`.")
    print("\nwrote results/m7_final_run.json")


if __name__ == "__main__":
    main()
