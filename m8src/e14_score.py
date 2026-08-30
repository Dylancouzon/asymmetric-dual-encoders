"""E14-HEAD's scoring pass: one process per head arm, documents transformed on the way to the scorer.

WHY ONE PROCESS PER ARM, which is what makes this expensive. `multieval.eval_makers` stacks every
variant's queries into ONE matrix and scores them against ONE document array, because a corpus is
normally invariant across variants -- that is the optimization that let B3 score 24 variants in a
single ~20-minute pass. A doc-side head breaks that invariant: each arm has its OWN corpus. So the
arms cannot be stacked, and `dev_eval.doc_vecs` is shared process-wide, which is why the
registration requires each patched process to emit exactly one head arm. Registered as engineering
constraint (b); the cost is inherent to the treatment, not to this implementation.

WHY IT GOES THROUGH `compare_full` RATHER THAN A SCORER OF ITS OWN. R0N must be comparable to the
existing R0 arms, and the endpoint is B3's endpoint. A second implementation of the scoring maths
-- however careful -- would make every cross-arm comparison a comparison between two code paths as
well as between two treatments. So the arms run the same entry point B3's arms ran, and the only
difference in the process is the head.

The three passes:
  dense      -- `compare_full` per arm -> per-query dumps, merged into one B3-shaped dump.
  fused      -- `fused_floor` per arm under the frozen operator -> the FUSED scalar.
  mechanism  -- DESCRIPTIVE, and it is what makes a positive interpretable. The head is supervised
                document-side metric learning; it can win by fixing the teacher's relevance
                geometry or by separating training sources (HotpotQA is ~85% of the document pool
                but ~24% of positive pairs), neither of which requires documents to have been
                bag-unreachable. So score {raw, headed} documents against {bag, TEACHER} queries.
                The bag gain MINUS the teacher-query gain is the bag-specific evidence.
"""
import argparse
import gzip
import json
import subprocess
import sys
import time

import numpy as np

import m8base
import probe_guard

REPO = m8base.REPO
WORK = REPO / "work"
RUNS = WORK / "runs"
RESULTS = REPO / "results"

PROBE = "E14-HEAD"
MODE, PREC = "sqrt", "int8"     # the release format and R0's adopted pooling rule (b3_score.sh)
MERGED_TAG = "m8e14head"
# The DENSE endpoint's components. The mechanism control reads only these, because the gain the
# control is explaining is the gain on these.
OOD = ("cqadup-programmers", "cqadup-physics")


# The arms that are READ. Ladder (`m8e14-lad-*`) and step-adequacy (`m8e14-step-*`) arms also
# leave head artifacts on disk, and they must never be scored: they are tuning arms on the
# disjoint tuning seed, trained on the holdout-reduced pool. Globbing every head artifact would
# have put them in the merged dump -- and would also have spent hours scoring them.
REPORTED_TAGS = ("r0n", "lin", "mlp")
REPORTED_SEEDS = (0, 1, 2)


def reported_arms():
    return [f"m8e14-{t}-s{s}" for t in REPORTED_TAGS for s in REPORTED_SEEDS]


def arms_on_disk():
    want = reported_arms()
    have = [r for r in want if (RUNS / f"{r}.head.json").exists()]
    stray = sorted(p.stem[:-5] for p in RUNS.glob("m8e14-*.head.json")
                   if p.stem[:-5] not in want and not p.stem.endswith("-smoke.head"))
    if stray:
        print(f"not scoring {len(stray)} non-reported arm(s) (tuning seed / holdout-reduced "
              f"pool): {stray}", flush=True)
    return have


def _head_loader(rid):
    """Source for the subprocess: rebuild the arm's head and bind it to ITS table by sha256."""
    return (
        "import json, torch, hashlib\n"
        "import e14_head, e14_patch\n"
        "rec = json.loads(open(%r).read())\n"
        "blob = torch.load(%r, map_location='cuda')\n"
        "want = rec['sha256']['table']\n"
        "got = e14_patch._sha_file(%r)\n"
        "if want is not None and want != got:\n"
        "    raise SystemExit('the head at %s was trained against a DIFFERENT table than the one '\n"
        "                     'on disk (sha mismatch). A run_id does not stop a stale head being '\n"
        "                     'paired with a table; this is what does.')\n"
        "head = e14_head.build_head(blob['dim'], kind=blob['kind'], device='cuda')\n"
        "head.load_state_dict(blob['state_dict'])\n"
        "head.eval()\n"
        "[p.requires_grad_(False) for p in head.parameters()]\n"
        % (str(RUNS / f"{rid}.head.json"), str(RUNS / f"{rid}.head.pt"),
           str(RUNS / f"{rid}.npz"), rid))


def _preamble():
    return ("import os, sys\n"
            "os.environ.setdefault('M7_ENCODER', 'stella-400M-v5')\n"
            "sys.path.insert(0, %r); sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
            "import m8base\n"
            % (str(REPO / "m7src"), str(REPO / "bench"), str(REPO / "m8src")))


def dense_code(rid, smoke=False):
    # `compare_full`'s own --smoke truncates every corpus to 200K documents and writes to a
    # `_smoke`-suffixed tag, so it exercises the whole patched path -- head loading, the sha bind,
    # the proxy, multieval's identity check across the two pool-sharing components -- in a couple
    # of minutes instead of the full pass. This path has NO execution history, which is exactly
    # where the standing discipline says to smoke first.
    return (
        _preamble() + _head_loader(rid) +
        "import dev_eval\n"
        "dev_eval.doc_vecs = e14_patch.patch_doc_vecs(head)\n"
        "import compare_full\n"
        # baseline is the arm itself and there are no candidates: this pass exists to produce the
        # arm's per-query rows, not a comparison. Every comparison is made after the merge, across
        # arms, by the decision code.
        "compare_full.main(%r, %r, [], smoke=%r)\n"
        "print('DENSE OK', %r, flush=True)\n"
        % (f"{MERGED_TAG}-{rid}", f"{rid}:{MODE}", bool(smoke), rid))


def fused_code(rid, smoke=False):
    return (
        _preamble() + _head_loader(rid) +
        "import dev_eval\n"
        "dev_eval.doc_vecs = e14_patch.patch_doc_vecs(head)\n"
        "import fused_floor\n"
        "sys.argv = ['fused_floor.py', '--arms', %r, '--seed-arms', %r, '--modes', %r,\n"
        "            '--precisions', %r, '--out', %r] + (['--smoke'] if %r else [])\n"
        "fused_floor.main()\n"
        "print('FUSED OK', %r, flush=True)\n"
        % (rid, rid, MODE, PREC, f"m8e14_fused_{rid}.json", bool(smoke), rid))


def mechanism_code(rid, smoke=False):
    """{raw, headed} documents x {bag, teacher} queries, on the DENSE endpoint's components."""
    return (
        _preamble() + _head_loader(rid) +
        "import json, numpy as np, torch\n"
        "import dev_eval, compare_full, evalkit\n"
        "from teacher import encode_cached, QUERY_PREFIX\n"
        "rel, pre, models = compare_full.load(%r, %r)\n"
        "model = models[%r]\n"
        "out = {}\n"
        "for comp in %r:\n"
        "    doc_ids, _, q_ids, q_texts, qrels, dv_raw = dev_eval.doc_vecs(comp)\n"
        "    dv_head = e14_patch.HeadedVecs(dv_raw, head)\n"
        "    qv_bag = model.encode(q_texts, pre)\n"
        "    qv_teacher = np.asarray(encode_cached('dev-%%s-queries' %% comp, q_texts,\n"
        "                                          prefix=QUERY_PREFIX, verbose=False),\n"
        "                            dtype=np.float32)\n"
        "    for qname, qv in (('bag', qv_bag), ('teacher', qv_teacher)):\n"
        "        for dname, dvv in (('raw', dv_raw), ('headed', dv_head)):\n"
        "            pq = evalkit.score(qv, q_ids, dvv, doc_ids, qrels,\n"
        "                               chunk=dev_eval.CHUNK.get(comp, 200_000))\n"
        "            out.setdefault(f'{qname}|{dname}', {})[comp] = float(\n"
        "                np.mean(list(pq.values())))\n"
        "json.dump({'run_id': %r, 'components': list(%r), 'macros': out},\n"
        "          open(%r, 'w'), indent=1)\n"
        "print('MECHANISM OK', %r, flush=True)\n"
        % (rid, MODE, PREC, OOD, rid, OOD,
           str(RESULTS / f"m8e14_mechanism_{rid}.json"), rid))


def run(kind, arms, dry=False, smoke=False):
    maker = {"dense": dense_code, "fused": fused_code, "mechanism": mechanism_code}[kind]
    for rid in arms:
        code = maker(rid, smoke=smoke) if kind != "mechanism" else maker(rid)
        if dry:
            print(f"----- {kind} {rid} -----\n{code}")
            continue
        t0 = time.time()
        print(f"[{kind}:{rid}] launching", flush=True)
        r = subprocess.run([sys.executable, "-u", "-c", code], cwd=str(REPO))
        print(f"[{kind}:{rid}] exit {r.returncode} in {time.time()-t0:.0f}s", flush=True)
        if r.returncode != 0:
            raise SystemExit(f"{kind} scoring failed for {rid} (exit {r.returncode}); stopping "
                             f"rather than reading the probe from a partial arm set.")


def merge(arms):
    """One B3-shaped per-query dump from the per-arm dumps, so the decision code is unchanged.

    Refuses on a duplicate variant key: two arms writing the same key would mean one silently
    replaced the other, and the survivor would look like a complete result.
    """
    per, comps, enc = {}, None, None
    for rid in arms:
        p = RESULTS / f"m7_devperquery_{MERGED_TAG}-{rid}.json.gz"
        if not p.exists():
            raise SystemExit(f"{rid}: no dense dump at {p.name}; run `dense` first")
        d = json.loads(gzip.open(p).read())
        comps = comps or d["components"]
        enc = enc or d["encoder"]
        if sorted(d["components"]) != sorted(comps):
            raise SystemExit(f"{rid}: scored a different component set than the other arms")
        for k, v in d["per_query"].items():
            if k in per:
                raise SystemExit(f"duplicate variant key {k!r} across arm dumps")
            per[k] = v
    out = {"components": comps, "encoder": enc, "per_query": per,
           "_what": "E14-HEAD arms, each scored in its own process with its own headed corpus, "
                    "merged into one dump. Merging is a concatenation of disjoint variant keys; "
                    "no number is recomputed here."}
    dpath = RESULTS / f"m7_devperquery_{MERGED_TAG}.json.gz"
    raw = json.dumps(out, sort_keys=True).encode()
    with gzip.GzipFile(filename=str(dpath), mode="wb", mtime=0) as f:
        f.write(raw)
    print(f"merged {len(per)} variant keys from {len(arms)} arms -> {dpath.name}")
    return dpath


def fused_merge(arms):
    """The per-arm fused artifacts, collapsed to {run_id: fused_macro} in b3_decide's shape."""
    macros = {}
    for rid in arms:
        p = RESULTS / f"m8e14_fused_{rid}.json"
        if not p.exists():
            raise SystemExit(f"{rid}: no fused artifact at {p.name}; run `fused` first")
        d = json.loads(p.read_text())
        for k, v in d["arm_macros"].items():
            if k in macros:
                raise SystemExit(f"duplicate fused key {k!r}")
            macros[k] = v
    out = {"_what": "E14-HEAD fused scalars under the frozen operator, one process per arm",
           "arm_macros": macros, "mode": MODE, "precision": PREC}
    p = RESULTS / "m8e14_fused.json"
    probe_guard.write_result(p, out, PROBE)
    print(f"merged {len(macros)} fused keys -> {p.name}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dense", "fused", "mechanism", "merge", "fused-merge"])
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="truncate every corpus (compare_full/fused_floor --smoke). Exercises the "
                         "whole patched path in minutes; every number it produces is MEANINGLESS "
                         "and lands under a _smoke tag, so it cannot be mistaken for a result.")
    a = ap.parse_args()
    # `--arms` IS NOT A WAY ROUND THE ALLOWLIST. Making the DEFAULT discovery enumerate the
    # reported arms fixed the accident; it left the deliberate route open, and an operator could
    # still have endpoint-scored a tuning arm by naming it. Dev-blindness that holds inside the
    # training subprocess but not at the workflow level is not dev-blindness.
    arms = a.arms or arms_on_disk()
    illegal = [r for r in arms if r not in reported_arms()]
    if illegal:
        raise SystemExit(
            f"refusing to score {illegal}: not in the reported set. Ladder and step-adequacy arms "
            f"are trained on the tuning seed and a holdout-reduced pool, and scoring one against "
            f"the endpoint is exactly what the registration forbids.")
    if not arms:
        raise SystemExit("no E14 head arms on disk (looked for work/runs/m8e14-*.head.json)")
    print(f"{len(arms)} arms: {arms}", flush=True)
    if a.stage == "merge":
        merge(arms)
    elif a.stage == "fused-merge":
        fused_merge(arms)
    else:
        run(a.stage, arms, dry=a.dry, smoke=a.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
