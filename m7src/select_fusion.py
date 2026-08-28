"""Select and freeze the fusion parameter on dev, before any test access.

One family (RRF or convex, picked here), one parameter, no per-dataset weights, no routing.
BM25 runs at `fusion.DEPTH` -- the same depth the final run applies -- because both fusion
families are depth-sensitive and a parameter chosen at one depth is not valid at another.

Fusion selection runs on the TEXT-BACKED dev components only: the held-out slices' corpora are
pool row indices and carry no document text, so BM25 has no run on them. Same restriction the
gate's BM25 comparison carries, and disclosed the same way.

Output: work/runs/<run_id>.fusion.json, read back by freeze.write() -- which loads this file
itself rather than taking a spec from its caller, and refuses it unless every hash below still
describes the artifact being frozen.
"""
import json
import sys

import numpy as np

import dev_eval
import freeze
import fusion
from _paths import REPO, WORK
from evalkit import topk_ids_scores
from table import Preproc, ensure_release, load_table, read_meta

CACHE = WORK / "fusionruns"
CACHE.mkdir(parents=True, exist_ok=True)


def bm25_run_and_key(comp):
    """-> (run, cache_key). NAMED for its return shape on purpose: it used to be
    `bm25_run_cached` returning just the run, and adding the key turned every existing caller into
    a silent tuple bug that surfaced deep inside `fusion.convex` as
    `'tuple' object has no attribute 'items'`. A changed return shape should break at the name.

    Thin wrapper over fusion.bm25_run -- the one shared builder (Codex B5) -- adding the
    per-component raw-array cache. The cache is content-keyed on the ordered doc ids/texts, query
    ids/texts, depth, BM25 parameters and library versions; a cache that cannot be validated
    against those is rebuilt, not reused."""
    doc_ids, doc_texts, q_ids, q_texts, qrels, _ = dev_eval.doc_vecs(comp)
    path = CACHE / f"bm25-{comp}-d{fusion.DEPTH}.npz"
    run = fusion.bm25_run(doc_ids, doc_texts, q_ids, q_texts, cache_path=path)
    # Recomputed here for the provenance record rather than handed to `bm25_run`: the cache's
    # identity must be derived inside the builder from the arguments it actually used, never
    # supplied by the caller (Codex review #4).
    return run, fusion.cache_key(doc_ids, doc_texts, q_ids, q_texts)


def dense_run(comp, model, pre):
    doc_ids, _, q_ids, q_texts, _, dv = dev_eval.doc_vecs(comp)
    return topk_ids_scores(model.encode(q_texts, pre), dv, doc_ids, k=fusion.DEPTH,
                           chunk=dev_eval.CHUNK.get(comp, 250_000), qids=q_ids)


def main(run_id):
    # The RELEASE artifact, not the training checkpoint: `save_release` folds the learned token
    # weights into the rows BEFORE quantizing, so the training npz's int8 codes are quantized from
    # differently-scaled rows. Fitting a fusion parameter against those would fit it to a table
    # that does not ship (review #2 BLOCKER 2, missed here until 2026-08-28).
    npz = ensure_release(WORK / "runs" / f"{run_id}.npz")
    meta = read_meta(npz)
    assert meta.get("weights_folded"), f"{npz} is not a release-shape artifact"
    # The document vectors this parameter is fitted against are the AMBIENT encoder's, so a
    # selection run under the wrong M7_ENCODER fits the released table's query vectors against
    # another teacher's documents. Same guard the freeze and the final run carry.
    freeze.assert_encoder_matches_artifact(meta, "FUSION SELECTION")
    pre = Preproc(**meta["preproc"])
    comps = [c for c in dev_eval.dev_components() if not c.startswith("heldout-")]
    print(f"fusion selection on the text-backed dev components: {comps} at depth {fusion.DEPTH}")

    # the released artifact is the int8 table, so fusion is fitted against int8, not fp16
    model = load_table(npz, variant="int8")
    dense, bm25, qrels, bm25_keys = {}, {}, {}, {}
    for c in comps:
        dense[c] = dense_run(c, model, pre)
        bm25[c], bm25_keys[c] = bm25_run_and_key(c)
        qrels[c] = dev_eval.doc_vecs(c)[4]
        print(f"  runs built for {c}", flush=True)
    del model

    from evalkit import per_query_ndcg
    for label, runs in (("int8-table alone", dense), ("bm25 alone", bm25)):
        m = float(np.mean([np.mean(list(per_query_ndcg(runs[c], qrels[c]).values())) for c in comps]))
        print(f"  {label}: dev macro(text-backed) {m:.4f}")

    spec, _ = fusion.select_on_dev(dense, bm25, qrels)
    spec["depth"] = fusion.DEPTH
    spec["components"] = comps
    spec["fitted_against"] = "int8 table (the released artifact)"
    # PROVENANCE: which artifact this parameter was fitted on, and against which lexical runs.
    # Without it `freeze.write` could pair a spec fitted on artifact A with artifact B and nothing
    # would notice; the freeze now re-derives every one of these hashes and refuses on any
    # mismatch (Codex one-shot-path review 2026-08-28, MAJOR 1).
    meta_p = npz.parent / (npz.stem + ".meta.json")
    spec["selected_on"] = {
        "run_id": run_id,
        "table_relpath": f"work/runs/{npz.name}",
        "table_sha256": freeze.sha256_file(npz),
        "table_meta_sha256": freeze.sha256_file(meta_p),
        "preproc": meta["preproc"],
        "preproc_fingerprint": meta["preproc_fingerprint"],
        "encoder_spec": freeze.encoder_fingerprint(),
        "dev_manifest_sha256": freeze.sha256_file(REPO / "results" / "m7_dev_manifest.json"),
        "bm25_run_keys": bm25_keys,
    }
    # Derived, never asserted: CONVEX_W carries the dense-only endpoint w=1.0 so that "do not
    # fuse" wins or loses the same mechanical selection as the parameter (m7/LEDGER.md, Fusion).
    spec["released_system"] = "dense" if fusion.is_dense_only(spec) else "fusion"
    (WORK / "runs" / f"{run_id}.fusion.json").write_text(json.dumps(spec, indent=1))
    (REPO / "results" / f"m7_fusion_{run_id}.json").write_text(json.dumps(spec, indent=1))
    print(f"\nreleased system, derived from the selection: {spec['released_system']}")
    print(f"frozen fusion spec -> work/runs/{run_id}.fusion.json")
    return spec


if __name__ == "__main__":
    main(sys.argv[1])
