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


def main(smoke=False, lever4=True):
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = get_tokenizer()
    V = tok.vocab_size
    spec = encoders.active()
    comps = SMOKE_COMPS if smoke else dev_eval.dev_components()
    print(f"dev audit: encoder={spec.name} components={comps}", flush=True)

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

    makers = {}
    for rid in CHAIN:
        m16, m8 = models[rid]
        makers[f"{rid}|fp16|table"] = qt_maker(m16)
        makers[f"{rid}|int8|table"] = qt_maker(m8)
        makers[f"{rid}|fp16|matrix"] = mx_maker(m16)
        makers[f"{rid}|int8|matrix"] = mx_maker(m8)
    if lever4:
        m16, m8 = models[CANDIDATE]
        for mode in POOL_MODES:
            if mode == "mean":
                continue                      # `CANDIDATE|fp16|table` IS the mean-pooled baseline
            makers[f"{CANDIDATE}|fp16|pool-{mode}"] = qt_maker(m16, mode=mode)
            makers[f"{CANDIDATE}|int8|pool-{mode}"] = qt_maker(m8, mode=mode)

    print(f"  {len(makers)} variants", flush=True)
    per = multieval.eval_makers(makers, components=comps,
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
        ok = all(levers[f"{a}_vs_{b}|{q}"]["dependence_preserving"]["signflip"]["p"] < 0.05
                 and levers[f"{a}_vs_{b}|{q}"]["dependence_preserving"]["paired"]["ci95"][0] > 0
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
                           "n_queries_changed": int((d != 0).sum()),
                           "mean_dev": float(d.mean())}
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
        arms = {}
        for mode in POOL_MODES:
            if mode == "mean":
                continue
            arms[mode] = {q: boot.both_ways(per[f"{CANDIDATE}|{q}|pool-{mode}"],
                                            per[f"{CANDIDATE}|{q}|table"])
                          for q in ("fp16", "int8")}
        pv = {m: arms[m]["fp16"]["dependence_preserving"]["signflip"]["p"] for m in arms}
        hol = boot.holm(pv, alpha=0.05)
        passing = [m for m in arms
                   if hol[m]["reject"]
                   and all(arms[m][q]["dependence_preserving"]["paired"]["ci95"][0] > 0
                           for q in ("fp16", "int8"))]
        best = max(passing, key=lambda m: multieval.macro(per[f"{CANDIDATE}|fp16|pool-{m}"])) \
            if passing else None
        lever4_out = {
            "candidate": CANDIDATE, "components": comps,
            "baseline_macro_fp16": multieval.macro(per[f"{CANDIDATE}|fp16|table"]),
            "arms": {m: {"macro_fp16": multieval.macro(per[f"{CANDIDATE}|fp16|pool-{m}"]),
                         "macro_int8": multieval.macro(per[f"{CANDIDATE}|int8|pool-{m}"]),
                         "per_component": multieval.means(per[f"{CANDIDATE}|fp16|pool-{m}"]),
                         "stats": arms[m]} for m in arms},
            "holm_alpha0.05_over_arms": hol, "passing": passing, "adopted": best,
            "_protocol": "m7/LEDGER.md 'Capacity lever #4', pre-registered 2026-08-27 before any "
                         "number: signflip p<0.05 AND paired CI>0 vs the same table under `mean`, "
                         "int8 independently, dependence-preserving stats, Holm over the 3 arms",
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
    dump_sha = hashlib.sha256(raw).hexdigest()

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
        "per_query_dump": {"path": dpath.name, "sha256": dump_sha},
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
    worst_vec = max(v["max_abs_query_vector_dev"] for v in vec_dev.values())
    worst_nd = max(r["max_abs_ndcg_dev"] for e in equiv.values() for r in e["per_component"].values())
    print(f"  matrix vs QueryTable: max |query-vector| dev {worst_vec:.3e}, "
          f"max |per-query nDCG| dev {worst_nd:.3e}")
    if lever4_out:
        print(f"  lever4: baseline {lever4_out['baseline_macro_fp16']:.4f} " +
              "  ".join(f"{m}={lever4_out['arms'][m]['macro_fp16']:.4f}" for m in lever4_out["arms"]) +
              f"  -> adopted={lever4_out['adopted']}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, lever4="--no-lever4" not in sys.argv)
