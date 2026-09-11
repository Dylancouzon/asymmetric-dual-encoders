"""The tiny synthetic end-to-end M17 rehearsal (pre-clock allocation, 0 hours).

Builds a disposable world under `work/m17/rehearsal` — a hand-written WordPiece tokenizer, a
seeded random unfolded "warm start", a few hundred bank documents, tens of queries with alias
pairs — and runs the PRODUCTION modules against it, unmodified:

    vocab discovery/ranking/selection -> count-weighted row init -> tokenizer extension
      -> cache.build (candidate cache + entropy diagnostic)
      -> train.run (tens of steps, snapshots, resume)
      -> export.build_bundle + the M17 gates (endpoint and mean_last_three)
      -> loader_np.parity (resident int8 vs eager fp32)
      -> evaluate.evaluate on SYNTHETIC fixtures only + the alias test

Nothing here reads a development component, the M17 panel, or any protected surface, and
nothing here is a quality observation: the fixture's qrels are constructed, the rows are random
and the vocabulary minima are scaled down so a few dozen queries can exercise the selection
code. Every scaled constant is recorded in the result under `scaled_constants`.

  .venv/bin/python m17src/train.py --rehearsal
  .venv/bin/python m17src/rehearse17.py --out work/m17/rehearsal --device cpu
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np

from common import RESULTS, WORK, registry, sha_text, write_json

DIM = 16
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
WORDS = ["storage", "bucket", "policy", "cluster", "ingress", "object", "the", "a", "query",
         "vector", "amazon", "kubernetes", "service", "network", "data", "index", "and", "for"]
PIECES = ["k", "##8", "##s", "s", "##3", "kube", "##ctl", "c", "+", "net", "x", "##y"]
# Multi-piece lexical units the discovery pass should find in the fixture queries.
TERMS = ["k8s", "s3", "kubectl"]
NEW_TERMS_EXPECTED = ("s3", "kubectl")      # k8s arrives as the owner-pinned row


def build_tokenizer():
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    vocab = {t: i for i, t in enumerate(SPECIALS + WORDS + PIECES)}
    tok = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]", max_input_chars_per_word=100))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    tok.no_padding()
    return tok, len(vocab)


def fixture_queries(n_general=60, n_coverage=24, n_pairs=8, n_heldout=12, seed=0):
    """QuerySpec records with alias pairs, families and a disjoint held-out slice."""
    import cache
    rng = np.random.default_rng(seed)
    qs, docs = [], [f"d{i:04d}" for i in range(240)]
    i = 0

    def q(text, bucket, domain, family, pair="", view="", pos=()):
        nonlocal i
        qs.append(cache.QuerySpec(qid=f"q{i:05d}", text=text, source="synthetic-" + domain,
                                  domain=domain, bucket=bucket, family=family,
                                  positive_ids=pos, alias_pair_id=pair, alias_view=view))
        i += 1

    for j in range(n_general):
        w = rng.choice(WORDS, size=4, replace=True)
        pos = (docs[int(rng.integers(0, len(docs)))],) if j % 3 else ()
        q(" ".join(w), "general", "general", f"g{j}", pos=pos)
    for j in range(n_coverage):
        term = TERMS[j % len(TERMS)]
        w = rng.choice(WORDS, size=2, replace=True)
        q(f"{term} {' '.join(w)}", "coverage",
          "cloud-software" if term != "s3" else "science-engineering", f"c{j}",
          pos=(docs[int(rng.integers(0, len(docs)))],))
    for j in range(n_pairs):
        fam = f"p{j}"
        q(f"k8s {WORDS[j % len(WORDS)]}", "coverage", "cloud-software", fam, f"pair{j}", "a")
        q(f"kubernetes {WORDS[j % len(WORDS)]}", "coverage", "cloud-software", fam,
          f"pair{j}", "b")
    for j in range(n_heldout):
        w = rng.choice(WORDS, size=3, replace=True)
        q(" ".join(w), "heldout", "general", f"h{j}",
          pos=(docs[int(rng.integers(0, len(docs)))],))
    return qs, docs


def unit(a):
    a = np.asarray(a, dtype=np.float32)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, 1e-9)


def rehearsal_registry():
    """The real registry with the fixture-scale constants substituted, recorded honestly."""
    reg = copy.deepcopy(registry())
    scaled = {
        "data.new_term_min_distinct_source_documents": (
            reg["data"]["new_term_min_distinct_source_documents"], 2),
        "data.new_term_min_distinct_training_contexts": (
            reg["data"]["new_term_min_distinct_training_contexts"], 3),
        "training.candidate_k": (reg["training"]["candidate_k"], 16),
        "training.candidate_mix_labeled": (reg["training"]["candidate_mix_labeled"],
                                           {"known_positive": 1, "teacher_top": 7,
                                            "zero_v1_top": 4, "uniform": 4}),
        "training.candidate_mix_query_only": (reg["training"]["candidate_mix_query_only"],
                                              {"known_positive": 0, "teacher_top": 8,
                                               "zero_v1_top": 4, "uniform": 4}),
        "training.batch": (reg["training"]["batch"], 16),
        "training.final_steps_per_fresh_run": (reg["training"]["final_steps_per_fresh_run"], 20),
        "training.warmup_steps": (reg["training"]["warmup_steps"], 4),
        "training.overfit_divergence_check.every_steps": (
            reg["training"]["overfit_divergence_check"]["every_steps"], 5),
        "checkpoint_averaging.checkpoint_steps": (
            reg["checkpoint_averaging"]["checkpoint_steps"], [15, 18, 20]),
    }
    reg["data"]["new_term_min_distinct_source_documents"] = 2
    reg["data"]["new_term_min_distinct_training_contexts"] = 3
    reg["training"]["candidate_k"] = 16
    reg["training"]["candidate_mix_labeled"] = scaled["training.candidate_mix_labeled"][1]
    reg["training"]["candidate_mix_query_only"] = scaled["training.candidate_mix_query_only"][1]
    reg["training"]["batch"] = 16
    reg["training"]["final_steps_per_fresh_run"] = 20
    reg["training"]["warmup_steps"] = 4
    reg["training"]["overfit_divergence_check"]["every_steps"] = 5
    reg["checkpoint_averaging"]["checkpoint_steps"] = [15, 18, 20]
    return reg, {k: {"registry": v[0], "rehearsal": v[1]} for k, v in scaled.items()}


MARKER = ".m17_rehearsal"


def _clear_rehearsal_dir(root):
    """Delete only a directory this script itself created (or an empty one).

    `--out` is a path a human typed. A recursive delete of an arbitrary directory is one
    typo away from a live M13 work tree, so the marker file is the only licence to remove one.
    """
    root = Path(root)
    if not root.exists():
        return
    if not root.is_dir():
        raise SystemExit(f"REFUSED: {root} is not a directory.")
    if any(root.iterdir()) and not (root / MARKER).exists():
        raise SystemExit(
            f"REFUSED: {root} is not empty and carries no {MARKER} marker, so it was not "
            "written by a previous rehearsal. Choose an empty or fresh --out directory; the "
            "rehearsal never deletes work it did not create.")
    shutil.rmtree(root)


def build(root, seed=0, device="cpu", log=print):
    """Build the world and run every stage. Returns the rehearsal record."""
    import cache
    import evaluate
    import export
    import loader_np
    import train as m17train
    import vocab as m17vocab
    from tokenizers import Tokenizer
    from table import Preproc, QueryTable, save_table

    root = Path(root)
    _clear_rehearsal_dir(root)
    (root / "data").mkdir(parents=True)
    (root / MARKER).write_text(
        "Disposable M17 rehearsal output. rehearse17.py may delete this directory because it "
        "wrote this marker; it deletes nothing else.\n")
    reg, scaled = rehearsal_registry()
    rng = np.random.default_rng(seed)
    t0 = time.time()
    stages = {}

    # --- tokenizer and the unfolded warm start -------------------------------------------
    tok, base_vocab = build_tokenizer()
    tok.save(str(root / "data" / "tokenizer_base.json"))
    rows = unit(rng.normal(size=(base_vocab, DIM))).astype(np.float32) * 0.4
    weights = (0.8 + 0.4 * rng.random(base_vocab)).astype(np.float32)
    pre = Preproc(prefix="", add_special_tokens=True, max_length=512, pool_mode="sqrt")
    warm = QueryTable(rows, weight_init=weights, learned_weights=True, fallback_id=2)
    save_table(root / "data" / "warm_start.npz", warm, pre,
               meta={"m17_fixture": True, "source": "rehearsal synthetic", "weights_folded": False})
    eff = m17vocab.effective_rows(rows, warm.token_weights().detach().numpy())
    # the "v1 release" this fixture stands in for: the same rows, folded once
    dev = m17vocab.verify_old_vocab_parity(eff, eff)
    stages["warm_start"] = {"vocab": base_vocab, "dim": DIM,
                            "old_vocab_parity_max_abs": dev}

    # --- queries, bank, teacher and v1 vectors -------------------------------------------
    queries, doc_ids = fixture_queries(seed=seed)
    bank_vecs = unit(rng.normal(size=(len(doc_ids), DIM))).astype(np.float32)
    bank = cache.Bank(doc_ids, bank_vecs, ["synthetic"] * len(doc_ids), seed=seed)
    v1_q = unit(np.stack([m17vocab._pool(np.asarray(tok.encode(q.text).ids), eff)
                          for q in queries]))
    teacher_q = unit(0.6 * v1_q + 0.4 * rng.normal(size=v1_q.shape))
    residual = 1.0 - (teacher_q * v1_q).sum(1)

    # --- vocabulary ------------------------------------------------------------------------
    stats, single = m17vocab.discover(
        [{"text": q.text, "qid": q.qid, "source_doc": f"doc-{q.family}", "domain": q.domain}
         for q in queries], residual, tok)
    sel = m17vocab.select(stats, reg, expansions={"k8s": "kubernetes"}, single_token=single)
    terms = [t["term"] for t in sel["terms"]]
    new_rows, pieces = m17vocab.init_new_rows(terms, tok, eff)
    tok_ext, tok_hash, n_added = m17vocab.extend_tokenizer(
        Tokenizer.from_file(str(root / "data" / "tokenizer_base.json")), terms)
    tok_ext.save(str(root / "data" / "tokenizer_ext.json"))
    new_ids = m17vocab.new_row_ids(tok_ext, terms)
    # The second and third probes deliberately reuse a CONSTITUENT piece elsewhere in the
    # query ("s" and "c"/"+"), which is exactly where sum-initialization stops being
    # output-preserving under sqrt-count pooling (P0b).
    drift = m17vocab.drift_report(tok, tok_ext, eff, new_rows, new_ids,
                                  ["s3 bucket policy", "s3 storage s cluster",
                                   "kubectl kube cluster"])
    stages["vocab"] = {"n_candidates": len(stats), "n_selected": len(terms), "terms": terms,
                       "per_domain": sel["per_domain"], "breadth": sel["breadth"],
                       "vocabulary_sha256": sel["vocabulary_sha256"],
                       "tokenizer_sha256": tok_hash, "n_added": n_added,
                       "already_single_token": sorted(single)[:10],
                       "p0b_drift": drift, "pieces_per_new_row": pieces[:3]}

    # --- candidate cache --------------------------------------------------------------------
    arrays, sidecar = cache.build(
        queries, bank, teacher_q, v1_q, reg, cache_seed=seed,
        manifests={"v1_artifact": {"fixture": "rehearsal synthetic v1 rows", "dim": DIM},
                   "teacher_query_preprocessing": "rehearsal fixture: raw text, no prefix"})
    cache.save(root / "data", arrays, sidecar)
    stages["cache"] = {"identity_sha256": sidecar["identity"]["sha256"],
                       "counts": sidecar["counts"],
                       "entropy_mean_nats": sidecar["entropy_diagnostic"]["entropy_nats"]["mean"],
                       "p_max_mean": sidecar["entropy_diagnostic"][
                           "p_max_of_teacher_distribution"]["mean"],
                       "provenance_totals": sidecar["provenance_counts"]["total"]}

    # --- training -----------------------------------------------------------------------------
    student_ids = [tok_ext.encode(q.text).ids for q in queries]
    heldout_idx = [i for i, q in enumerate(queries) if q.bucket == "heldout"]
    model = m17train.extend_model(
        m17train.load_warm_start(root / "data" / "warm_start.npz", expect_vocab=base_vocab,
                                 device=device, rehearsal=True)[0], new_rows, device)
    data = {"model": model, "ids": student_ids, "teacher_q": teacher_q, "bank": bank_vecs,
            "candidate_ids": arrays["candidate_ids"], "teacher_scores": arrays["teacher_scores"],
            "buckets": arrays["buckets"], "alias_pair_ids": arrays["alias_pair_ids"],
            "alias_views": arrays["alias_views"], "families": arrays["families"],
            "heldout_idx": heldout_idx, "preproc": {"prefix": "", "add_special_tokens": True,
                                                    "max_length": 512, "pool_mode": "sqrt"},
            "teacher": reg["teacher"], "teacher_revision": reg["teacher_revision"],
            "registry_status": reg["status"], "lineage": {"fixture": "rehearsal synthetic"}}
    cfg = m17train.RunCfg.from_registry(
        reg, "VL-A", seed=seed, device=device, rehearsal=True,
        tokenizer_sha256=tok_hash, vocabulary_sha256=sel["vocabulary_sha256"],
        cache_sha256=sidecar["identity"]["sha256"], heldout_queries=len(heldout_idx),
        run_id="m17-rehearsal-vla-s0", checkpoint_minutes_max=0.0)
    run_dir = root / "run"
    rec = m17train.run(cfg, data, run_dir, resume=False, log=log)
    stages["train"] = {"run_id": rec["run_id"], "arm": rec["arm"],
                       "batch_composition": rec["batch_composition"],
                       "bucket_passes": rec["bucket_passes"],
                       "first_history": rec["history"][0], "last_history": rec["history"][-1],
                       "divergence_flags": rec["divergence_flags"],
                       "snapshots": rec["snapshots"],
                       "wall_clock_seconds": rec["wall_clock_seconds"]}

    # --- resume: a second run over the same directory must restore and finish -------------
    rec2 = m17train.run(cfg, data, run_dir, resume=True, log=log)
    stages["resume"] = {"resumed_at_step": rec2["history"][-1]["step"] if rec2["history"] else None,
                        "steps_after_resume": rec2["config"]["steps"],
                        "note": "recovery checkpoint restored; the exhausted schedule adds no steps"}

    # --- export, gates ------------------------------------------------------------------------
    bundles = {}
    float_rows = None
    for form in ("endpoint", "mean_last_three"):
        if form == "endpoint":
            eff_out, diag = export.effective_rows(run_dir / "endpoint.npz")
            diag = {"snapshots": [{"step": diag["meta"].get("m17_step"), "rms": diag["rms"]}]}
        else:
            eff_out, diag = export.average_snapshots(
                [run_dir / f"snapshot_{s:06d}.npz" for s in cfg.snapshot_steps], reg)
        out = export.build_bundle(root / f"bundle-{form}", eff_out,
                                  Tokenizer.from_file(str(root / "data" / "tokenizer_ext.json")),
                                  {"bank": bank.identity(), "vocabulary_sha256":
                                   sel["vocabulary_sha256"], "averaging": diag,
                                   "rehearsal": True}, reg, form=form, fallback_id=2,
                                  fixture=True)
        gates = export.run_gates(out, log=log)
        bundles[form] = {"dir": str(out), "gates": {k: True for k in gates},
                         "rms": diag["snapshots"][-1]["rms"],
                         "model_npz_sha256": gates["gate_artifact"]["model_npz_sha256"]}
        if form == "endpoint":
            float_rows = eff_out
    stages["export"] = bundles

    # --- loader parity --------------------------------------------------------------------
    bundle = root / "bundle-endpoint"
    par = loader_np.parity(bundle, float_rows=float_rows)
    stages["loader"] = par

    # --- evaluation on synthetic fixtures only ---------------------------------------------
    enc = loader_np.M17QueryEncoder(bundle, variant="int8", mode="resident_int8")
    eval_q = [q for q in queries if q.bucket != "heldout"]
    qv = enc.encode([q.text for q in eval_q])
    qrels, domains, families = {}, {}, {}
    for i, q in enumerate(eval_q):
        domains[i] = q.domain
        families[i] = q.family
        qrels[i] = {d: 1 for d in q.positive_ids}
    # A stand-in BM25 prefetch: each query's own positive plus a deterministic slice of the
    # corpus, so the DBSF path fuses two runs that actually disagree.
    bm25 = {}
    for i, q in enumerate(eval_q):
        docs = {d: 10.0 for d in q.positive_ids}
        for j in range(100):
            docs.setdefault(doc_ids[(i * 7 + j) % len(doc_ids)], float(100 - j) / 100.0)
        bm25[i] = docs
    base = {i: 0.0 for i in range(len(eval_q))}
    rep = evaluate.evaluate(qv, bank_vecs, qrels, domains, families, doc_ids=doc_ids,
                            bm25_run=bm25, baseline_ndcg=base, reg=reg, seed=seed)
    rep.pop("per_query_ndcg@10", None)
    pair_a = [i for i, q in enumerate(eval_q) if q.alias_view == "a"]
    pair_b = [i for i, q in enumerate(eval_q) if q.alias_view == "b"]
    ra = evaluate.search(qv[pair_a], bank_vecs, k=10, doc_ids=doc_ids)
    rb = evaluate.search(qv[pair_b], bank_vecs, k=10, doc_ids=doc_ids)
    stages["evaluate"] = {"surface": "synthetic fixtures only",
                          "ndcg@10": rep["ndcg@10"], "recall@10": rep["recall@10"],
                          "fused_dbsf@100_macro": rep["fused_dbsf@100"]["macro"],
                          "vs_baseline": rep["vs_baseline"],
                          "alias_test": evaluate.alias_test(ra, rb)}

    return {
        "_schema": "m17-rehearsal-v1",
        "_note": "Pre-clock synthetic rehearsal of the M17 pipeline. Random rows, constructed "
                 "qrels, scaled constants: NOT a quality observation and not a forecast of any "
                 "rate. It certifies that the modules run end to end and refuse what they must.",
        "date": time.strftime("%Y-%m-%d"),
        "root": str(root), "device": device, "seed": seed,
        "registry_status_at_rehearsal": registry()["status"],
        "scaled_constants": scaled,
        "stages": stages,
        "wall_clock_seconds": round(time.time() - t0, 3),
        "source_sha256": sha_text(Path(__file__).read_text()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(WORK / "rehearsal"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--result", default=None,
                    help="default: results/m17_rehearsal.json for the canonical --out, "
                         "otherwise <out>/rehearsal.json")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing committed result (record why)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    # Only the CANONICAL rehearsal directory writes the committed record. A rehearsal pointed
    # somewhere else keeps its record beside its own outputs, so a scratch run cannot quietly
    # replace the observation the ledger cites.
    result = Path(args.result) if args.result else (
        RESULTS / "m17_rehearsal.json" if out.resolve() == (WORK / "rehearsal").resolve()
        else out / "rehearsal.json")
    if result.exists() and not args.force:
        raise SystemExit(f"REFUSED: {result} already records a rehearsal. Re-run with --force "
                         "only when you mean to replace it, and say why in the commit.")
    rec = build(out, seed=args.seed, device=args.device)
    write_json(result, rec)
    print(f"rehearsal OK -> {result} ({rec['wall_clock_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
