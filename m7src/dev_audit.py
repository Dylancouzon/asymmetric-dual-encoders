"""One full-suite dev pass that produces every number Codex review #3 asks for before the ablations.

Four questions, one corpus pass each, because they share the expensive part (reading the 6.17M-row
pool and the 5.23M-row HotpotQA corpus):

  B1  the three lever comparisons recomputed with DEPENDENCE-PRESERVING statistics, side by side
      with the ordinary ones. Rule pre-registered in m7/LEDGER.md before this ran: a decision
      whose dependence-preserving statistics miss its original bar reverts.
  M3  matrix-shortcut vs released `QueryTable` equivalence, per query, for every decision artifact
      -- max query-vector deviation, changed top-10 counts, max per-query nDCG deviation, macro
      delta. Every earlier lever number came from the matrix path; the gate and final run use
      QueryTable, so the two must be shown equal rather than assumed equal.
  M6  the provenance the comparison artifacts lacked: unrounded macros, per-component CIs,
      per-query values (dumped, hashed), encoder fingerprint, table hashes.
  L4  capacity lever #4 (count saturation), protocol pre-registered in m7/LEDGER.md.

Usage: dev_audit.py [--smoke] [--no-lever4]
"""
import gzip
import hashlib
import json
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

import boot
import dev_eval
import encoders
import multieval
from _paths import REPO, WORK
from hashing import sha
from stage0_ridge import bag_matrix
from table import POOL_MODES, Preproc, encode_pooled, ensure_release, get_tokenizer, \
    load_table, read_meta

# The chain of dev decisions, oldest first. Each adjacent pair is one lever comparison.
CHAIN = ["s2w-1e3-s1000", "p35w-500k-s1500", "p35a-2m-1e3", "p35w-2m-s2500"]
CANDIDATE = CHAIN[-1]
# The smoke MUST include a held-out component: those two share the 6.17M-row pool and are the
# only path where the shared-corpus pass, the memmap identity and the nesting all matter. A smoke
# over the two small text components missed exactly that and cost a 35-minute run (2026-08-27).
SMOKE_COMPS = ["cqadup-programmers", "heldout-longq", "heldout-train"]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_release(run_id, device="cuda"):
    rel = ensure_release(WORK / "runs" / f"{run_id}.npz", device=device)
    meta = read_meta(rel)
    assert meta["weights_folded"], f"{run_id}: not a release-shape artifact"
    # int8 comes from the artifact's OWN stored codes, not from re-quantizing the fp16 view --
    # that is the thing that ships. (compare_release.py re-quantized the fp16 rows; the two agree
    # to within one code, but only one of them is the released artifact.)
    return rel, meta, load_table(rel, variant="fp16", device=device), \
        load_table(rel, variant="int8", device=device)


EVALUATOR_SOURCES = ["boot.py", "multieval.py", "evalkit.py", "table.py", "dev_eval.py",
                     "heldout.py", "dev_audit.py", "devsuite.py", "pool.py", "encoders.py"]


def code_identity():
    """What produced these numbers. A committed revision is the real answer, but the files are
    often dirty mid-session, so hash them too (Codex review #3b MAJOR 5)."""
    import subprocess
    src = REPO / "m7src"
    rev = dirty = None
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                             text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "m7src"], cwd=REPO,
                                    capture_output=True, text=True).stdout.strip())
    except Exception:
        pass
    return {"git_head": rev, "m7src_dirty": dirty,
            "source_sha256": {f: sha_file(src / f) for f in EVALUATOR_SOURCES}}


def verify_pin(comps, pool_bytes=True):
    """The audit may not run on an unpinned suite (Codex review #3b BLOCKER 1)."""
    import heldout
    man_p = REPO / "results" / "m7_dev_manifest.json"
    man = json.loads(man_p.read_text())
    pinned = man.get("_pinned", {}).get("components")
    if not pinned:
        raise SystemExit("results/m7_dev_manifest.json carries no `_pinned.components`. Run "
                         "freeze_heldout.py before any authoritative dev number.")
    if list(comps) != list(pinned):
        raise SystemExit(f"components {list(comps)} != pinned {list(pinned)}")
    for c in comps:
        if c not in man:
            raise SystemExit(f"pinned component {c} has no manifest entry")
    heldout.verify_pinned(pool_bytes=pool_bytes)
    spec = encoders.active()
    pin_enc = man["_pinned"].get("active_encoder", {})
    if pin_enc.get("name") != spec.name or pin_enc.get("revision") != spec.revision:
        raise SystemExit(f"pinned encoder {pin_enc} != active {spec.name}@{spec.revision}")
    return {"dev_manifest_sha256": sha_file(man_p),
            "component_entries_sha256": {c: sha(man[c]) for c in comps},
            "pool_bytes_verified": bool(pool_bytes),
            "pinned_at": man["_pinned"].get("pinned_at")}


def main(smoke=False, lever4=True):
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = get_tokenizer()
    V = tok.vocab_size
    spec = encoders.active()
    comps = SMOKE_COMPS if smoke else dev_eval.dev_components()
    pin_evidence = verify_pin(dev_eval.dev_components(), pool_bytes=not smoke)
    print(f"dev audit: encoder={spec.name} components={comps}", flush=True)
    print(f"  pin verified: manifest {pin_evidence['dev_manifest_sha256'][:16]}, pool bytes "
          f"{pin_evidence['pool_bytes_verified']}", flush=True)

    models, metas, rels = {}, {}, {}
    pre = None
    for rid in CHAIN:
        rel, meta, m16, m8 = load_release(rid, device=dev)
        p = Preproc(**meta["preproc"])
        if pre is None:
            pre = p
        elif p != pre:
            raise AssertionError(f"{rid} preprocessing {p} != {pre}; not comparable")
        models[rid] = (m16, m8)
        metas[rid] = meta
        rels[rid] = rel

    # --- makers -------------------------------------------------------------------------
    # A maker is (comp, q_texts) -> query vectors. `_qt` is the RELEASED path (embedding_bag,
    # 1e-6 degeneracy threshold, CLS fallback); `_mx` is the matrix shortcut every earlier lever
    # number was computed with (count matrix @ rows, 1e-9 clip, no fallback).
    _bag_cache, _qtext_cache = {}, {}

    def qtexts(comp):
        """Memoized. `dev_eval.doc_vecs` re-parses the component's corpus cache on every call --
        HotpotQA's is 5.23M documents and peaks ~14 GB -- and the deviation check below needs only
        the query texts, so it must not trigger that parse once per (table, quantization)."""
        if comp not in _qtext_cache:
            _qtext_cache[comp] = dev_eval.doc_vecs(comp)[3]
        return _qtext_cache[comp]

    def bag(comp, q_texts):
        if comp not in _bag_cache:
            _bag_cache[comp] = bag_matrix(tok, q_texts, pre, V)
        return _bag_cache[comp]

    def qt_maker(model, mode=None):
        def f(comp, q_texts):
            if mode is None:
                return model.encode(q_texts, pre, tok=tok)
            return encode_pooled(model, q_texts, pre, mode=mode, tok=tok)
        return f

    def mx_maker(model):
        W = model.rows.detach().cpu().numpy().astype(np.float32)

        def f(comp, q_texts):
            qv = np.asarray(bag(comp, q_texts) @ W, dtype=np.float32)
            qv /= np.clip(np.linalg.norm(qv, axis=1, keepdims=True), 1e-9, None)
            return qv
        return f

    makers, pair_checks = {}, []
    for rid in CHAIN:
        m16, m8 = models[rid]
        makers[f"{rid}|fp16|table"] = qt_maker(m16)
        makers[f"{rid}|int8|table"] = qt_maker(m8)
        makers[f"{rid}|fp16|matrix"] = mx_maker(m16)
        makers[f"{rid}|int8|matrix"] = mx_maker(m8)
        for q in ("fp16", "int8"):
            pair_checks.append((f"{rid}|{q}|table", f"{rid}|{q}|matrix"))
        if lever4:
            # Pooling arms for EVERY artifact in the chain, not just CHAIN[-1]: which one is the
            # candidate is decided by the dependence recompute BELOW, and probing a table the
            # audit then rejects would answer the wrong question (Codex review #3b BLOCKER 2).
            # Only the surviving artifact's arms are adjudicated; the rest are descriptive.
            for mode in POOL_MODES:
                if mode == "mean":
                    continue                  # `<rid>|fp16|table` IS the mean-pooled baseline
                makers[f"{rid}|fp16|pool-{mode}"] = qt_maker(m16, mode=mode)
                makers[f"{rid}|int8|pool-{mode}"] = qt_maker(m8, mode=mode)

    print(f"  {len(makers)} variants, {len(pair_checks)} path-equivalence pairs", flush=True)
    per, ranks = multieval.eval_makers(makers, components=comps, pair_checks=pair_checks,
                                       max_docs=200_000 if smoke else None)
    print(f"  eval done in {time.time()-t0:.0f}s", flush=True)

    # --- B1: the three lever comparisons, both ways ---------------------------------------
    levers = {}
    for a, b in zip(CHAIN[1:], CHAIN[:-1]):
        for quant in ("fp16", "int8"):
            levers[f"{a}_vs_{b}|{quant}"] = boot.both_ways(per[f"{a}|{quant}|table"],
                                                           per[f"{b}|{quant}|table"])
    verdict = {}
    for a, b in zip(CHAIN[1:], CHAIN[:-1]):
        # ci95_raw, never the rounded display value: a true lower endpoint of +4e-5 rounds to
        # 0.0000 and would read as unresolved (Codex review #3b MAJOR 2).
        ok = all(levers[f"{a}_vs_{b}|{q}"]["dependence_preserving"]["signflip"]["p"] < 0.05
                 and levers[f"{a}_vs_{b}|{q}"]["dependence_preserving"]["paired"]["ci95_raw"][0] > 0
                 for q in ("fp16", "int8"))
        verdict[f"{a}_vs_{b}"] = "STANDS" if ok else "REVERTS"
    surviving = CHAIN[0]
    for a, b in zip(CHAIN[1:], CHAIN[:-1]):
        if verdict[f"{a}_vs_{b}"] != "STANDS":
            break
        surviving = a

    # --- M3: matrix vs released QueryTable, per query --------------------------------------
    equiv = {}
    for rid in CHAIN:
        for quant in ("fp16", "int8"):
            t_tab, t_mx = per[f"{rid}|{quant}|table"], per[f"{rid}|{quant}|matrix"]
            rows = {}
            for c in comps:
                if set(t_tab[c]) != set(t_mx[c]):
                    raise AssertionError(f"{rid}/{quant}/{c}: qid sets differ between paths")
                d = np.array([t_tab[c][q] - t_mx[c][q] for q in t_tab[c]])
                rows[c] = {"n": int(d.size), "max_abs_ndcg_dev": float(np.abs(d).max()),
                           "n_queries_ndcg_changed": int((d != 0).sum()),
                           "mean_dev": float(d.mean()),
                           # ranking equivalence, which equal nDCG does NOT imply: top-10
                           # membership can change entirely among non-relevant documents
                           **ranks[f"{rid}|{quant}|table|vs|{rid}|{quant}|matrix"][c]}
            equiv[f"{rid}|{quant}"] = {
                "per_component": rows,
                "macro_table": multieval.macro(t_tab), "macro_matrix": multieval.macro(t_mx),
                "macro_delta": multieval.macro(t_tab) - multieval.macro(t_mx)}
    # query-vector deviation is corpus-free, so it is measured directly rather than inferred
    vec_dev = {}
    for rid in CHAIN:
        m16, m8 = models[rid]
        for quant, m in (("fp16", m16), ("int8", m8)):
            worst, worst_c = 0.0, None
            W = m.rows.detach().cpu().numpy().astype(np.float32)
            for c in comps:
                q_texts = qtexts(c)
                a = m.encode(q_texts, pre, tok=tok)
                bmx = np.asarray(bag(c, q_texts) @ W, dtype=np.float32)
                bmx /= np.clip(np.linalg.norm(bmx, axis=1, keepdims=True), 1e-9, None)
                dv_ = float(np.abs(a - bmx).max())
                if dv_ > worst:
                    worst, worst_c = dv_, c
            vec_dev[f"{rid}|{quant}"] = {"max_abs_query_vector_dev": worst, "worst_component": worst_c}

    # --- L4: count saturation ---------------------------------------------------------------
    lever4_out = None
    if lever4:
        modes = [m for m in POOL_MODES if m != "mean"]

        def arms_for(rid):
            return {m: {q: boot.both_ways(per[f"{rid}|{q}|pool-{m}"], per[f"{rid}|{q}|table"])
                        for q in ("fp16", "int8")} for m in modes}

        def descriptive(rid):
            return {"baseline_macro_fp16": multieval.macro(per[f"{rid}|fp16|table"]),
                    "arms": {m: {"macro_fp16": multieval.macro(per[f"{rid}|fp16|pool-{m}"]),
                                 "macro_int8": multieval.macro(per[f"{rid}|int8|pool-{m}"])}
                             for m in modes}}

        arms = arms_for(surviving)
        # "int8 independently" implemented as a SECOND three-hypothesis Holm family, not as a bare
        # CI check: the selected arm must clear Holm and the raw CI in both precisions
        # (Codex review #3b BLOCKER 3).
        holm_by_q = {q: boot.holm({m: arms[m][q]["dependence_preserving"]["signflip"]["p"]
                                   for m in modes}, alpha=0.05) for q in ("fp16", "int8")}
        passing = [m for m in modes
                   if all(holm_by_q[q][m]["reject"]
                          and arms[m][q]["dependence_preserving"]["paired"]["ci95_raw"][0] > 0
                          for q in ("fp16", "int8"))]
        best = max(passing, key=lambda m: multieval.macro(per[f"{surviving}|fp16|pool-{m}"])) \
            if passing else None
        lever4_out = {
            "adjudicated_on": surviving, "components": comps,
            "baseline_macro_fp16": multieval.macro(per[f"{surviving}|fp16|table"]),
            "arms": {m: {"macro_fp16": multieval.macro(per[f"{surviving}|fp16|pool-{m}"]),
                         "macro_int8": multieval.macro(per[f"{surviving}|int8|pool-{m}"]),
                         "per_component": multieval.means(per[f"{surviving}|fp16|pool-{m}"]),
                         "stats": arms[m]} for m in modes},
            "holm_alpha0.05_per_precision": holm_by_q, "passing": passing, "adopted": best,
            "descriptive_other_artifacts": {rid: descriptive(rid) for rid in CHAIN
                                            if rid != surviving},
            "_protocol": "m7/LEDGER.md 'Capacity lever #4', pre-registered 2026-08-27 before any "
                         "number: adopt iff, under the dependence-preserving statistics, the arm "
                         "clears Holm at alpha=0.05 within its precision's three-arm family AND "
                         "its raw paired CI lower bound is > 0, in BOTH fp16 and int8; largest "
                         "fp16 dev macro among those passing. Pooling counts post-truncation "
                         "WordPiece occurrences INCLUDING specials.",
            "_scope": "arms were run for every chain artifact in the same corpus pass, but only "
                      "the artifact surviving the dependence recompute is adjudicated; the others "
                      "are descriptive.",
            "_status": "exploratory dev selection evidence (review #3 MAJOR 1)"}

    # --- per-query dump (M6) ------------------------------------------------------------------
    dump = {"encoder": asdict(spec), "components": comps, "chain": CHAIN,
            "per_query": {tag: {c: {q: float(v) for q, v in d.items()} for c, d in t.items()}
                          for tag, t in per.items()}}
    tag = "smoke" if smoke else "full"
    dpath = REPO / "results" / f"m7_devperquery_{tag}.json.gz"
    raw = json.dumps(dump, sort_keys=True).encode()
    with gzip.GzipFile(filename=str(dpath), mode="wb", mtime=0) as f:   # mtime=0: reproducible bytes
        f.write(raw)
    dump_shas = {"payload_sha256": hashlib.sha256(raw).hexdigest(),   # the JSON inside the gzip
                 "file_sha256": sha_file(dpath)}                      # the gzip file on disk

    out = {
        "_what": "dev audit answering Codex review #3 BLOCKER 1 / MAJOR 3 / MAJOR 6 in one pass",
        "_status": "ALL dev statistics here are exploratory SELECTION evidence (review #3 MAJOR 1); "
                   "the only confirmatory comparisons are the three frozen-test ones",
        "encoder": asdict(spec), "components": comps,
        "chain": CHAIN,
        "tables": {rid: {"release": rels[rid].name, "sha256": sha_file(rels[rid]),
                         "meta_sha256": sha_file(rels[rid].parent /
                                                 (rels[rid].stem + ".meta.json"))}
                   for rid in CHAIN},
        "preproc": asdict(pre), "preproc_fingerprint": pre.fingerprint(),
        "macros_unrounded": {t: multieval.macro(p) for t, p in per.items()},
        "per_component_unrounded": {t: multieval.means(p) for t, p in per.items()},
        "lever_comparisons": levers, "lever_verdicts": verdict,
        "surviving_candidate": surviving,
        "matrix_vs_querytable": equiv, "query_vector_deviation": vec_dev,
        "per_query_dump": {"path": dpath.name, **dump_shas},
        "pin_evidence": pin_evidence, "code_identity": code_identity(),
        "seconds": round(time.time() - t0, 1),
    }
    name = f"m7_dev_audit_{tag}.json"
    (REPO / "results" / name).write_text(json.dumps(out, indent=1))
    if lever4_out:
        (REPO / "results" / f"m7_lever4_pooling_{tag}.json").write_text(json.dumps(lever4_out, indent=1))
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("lever_verdicts", "surviving_candidate", "seconds")}, indent=1))
    for a, b in zip(CHAIN[1:], CHAIN[:-1]):
        for q in ("fp16",):
            L = levers[f"{a}_vs_{b}|{q}"]
            print(f"  {a} vs {b} [{q}]  ordinary {L['ordinary']['paired']['ci95']} "
                  f"p={L['ordinary']['signflip']['p']:.5f}   dep "
                  f"{L['dependence_preserving']['paired']['ci95']} "
                  f"p={L['dependence_preserving']['signflip']['p']:.5f}  -> {verdict[f'{a}_vs_{b}']}")
    rws = [r for e in equiv.values() for r in e["per_component"].values()]
    print(f"  matrix vs QueryTable: max |query-vector| dev "
          f"{max(v['max_abs_query_vector_dev'] for v in vec_dev.values()):.3e}, "
          f"max |per-query nDCG| dev {max(r['max_abs_ndcg_dev'] for r in rws):.3e}, "
          f"queries with a changed ordered top-10 "
          f"{sum(r['changed_ordered_top10'] for r in rws)}/{sum(r['n'] for r in rws)}, "
          f"changed top-100 set {sum(r['changed_topk_set'] for r in rws)}, "
          f"max matched-doc score dev {max(r['max_score_dev_matched_docs'] for r in rws):.3e}")
    if lever4_out:
        print(f"  lever4 on {lever4_out['adjudicated_on']}: baseline "
              f"{lever4_out['baseline_macro_fp16']:.4f}  " +
              "  ".join(f"{m}={lever4_out['arms'][m]['macro_fp16']:.4f}" for m in lever4_out["arms"]) +
              f"  -> adopted={lever4_out['adopted']}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, lever4="--no-lever4" not in sys.argv)
