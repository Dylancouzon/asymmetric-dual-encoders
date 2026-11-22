"""E23: corpus-size deconfound (m15/MEASUREMENTS.md, E23). Runpod.

    e23_fiqa25k.py sweep <name>   one space: FiQA subsampled to SCIDOCS size, two builds, ef grid
    e23_fiqa25k.py assemble       results/m15_e23_fiqa25k.json
    e23_fiqa25k.py all            every space, then assemble
    e23_fiqa25k.py selftest       synthetic check of the subsample rule

Reuses E20's cached FiQA document and query vectors and its sweep code with a document subsample:
25,657 documents drawn uniformly with seed 20260930, keeping every document judged relevant to a
test query. Each path's exact top-10 is recomputed over the subsample.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in ("m15src", "m8src", "m7src"):
    if str(REPO / p) not in sys.path:
        sys.path.insert(0, str(REPO / p))
import e20_ann_spaces as E20          # noqa: E402
from common import load_public, receipt, sha_file, utc_now, write_result  # noqa: E402

N_SUB = 25657
SEED = 20260930
OUT_DIR = REPO / "work" / "m15" / "e23"
RESULT = REPO / "results" / "m15_e23_fiqa25k.json"
PATHS = ("teacher", "table")


def subsample(doc_ids, qrels, n_sub, seed):
    """Indices of a seeded uniform subsample that keeps every judged-relevant document."""
    keep = {d for q in qrels.values() for d, s in q.items() if s > 0}
    must = [i for i, d in enumerate(doc_ids) if d in keep]
    rest = [i for i, d in enumerate(doc_ids) if d not in keep]
    rng = np.random.default_rng(seed)
    extra = rng.choice(rest, size=n_sub - len(must), replace=False)
    return np.sort(np.concatenate([must, extra]).astype(np.int64))


def sweep(name):
    from qdrant_client import QdrantClient, models as m
    import vectors15 as V
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    done = out / "sweep.json"
    if done.exists():
        print(f"{name}: already done", flush=True)
        return
    data = load_public("fiqa", with_corpus=False)
    src = E20.vec_dir(name)
    ids_path = src / "fiqa-doc-ids.json"
    if not ids_path.exists():
        raise SystemExit(f"E23 STOP: {ids_path} missing (written by the E20 sweep); refusing to guess")
    doc_ids_full = json.loads(ids_path.read_text())
    dv_full = np.load(src / "fiqa-docs.npy")
    if len(doc_ids_full) != len(dv_full):
        raise SystemExit(f"E23 STOP: {name} has {len(doc_ids_full)} ids for {len(dv_full)} vectors")
    idx = subsample(doc_ids_full, data["qrels"], N_SUB, SEED)
    doc_ids = [doc_ids_full[i] for i in idx]
    dv = dv_full[idx]
    qv = {p: np.load(src / f"fiqa-{p}-q.npy") for p in PATHS}
    from evalkit import topk_arrays
    exact, exact_ndcg = {}, {}
    for p in PATHS:
        ids, scores = topk_arrays(qv[p], dv, k=100, chunk=250_000, device="cpu")
        exact[p] = E20.top10([list(map(int, r)) for r in np.asarray(ids)], doc_ids, data["q_ids"])
        exact_ndcg[p] = float(np.mean(list(V.ndcg10(
            {q: {d: float(10 - r) for r, d in enumerate(l)} for q, l in exact[p].items()},
            data["qrels"], data["q_ids"]).values())))
    doc_index = {d: i for i, d in enumerate(doc_ids)}
    tie = {p: E20.tie_thresholds(qv[p], dv, exact[p], doc_index, data["q_ids"]) for p in PATHS}
    proc = E20.start_qdrant(f"e23-{name}")
    rows, builds, parity = [], {}, {}
    try:
        client = QdrantClient(host="127.0.0.1", port=E20.PORT, grpc_port=E20.GRPC, prefer_grpc=True,
                              timeout=600)
        for b in range(E20.BUILDS):
            order = np.arange(len(dv)) if b == 0 else np.random.default_rng(SEED + b).permutation(len(dv))
            cname = f"e23-fiqa25k-b{b}"
            builds[f"b{b}"] = E20.build(client, cname, dv, order)
            for p in PATHS:
                qc = data["q_ids"][:E20.EXACT_CHECK]
                ids, _ = E20.search_all(client, cname, qv[p][:E20.EXACT_CHECK],
                                        m.SearchParams(exact=True), limit=11)
                got = E20.top10(ids, doc_ids, qc)
                parity[f"b{b}-{p}"] = E20.recovery_tied(got, tie[p][0], tie[p][1], doc_index, qc)
                if parity[f"b{b}-{p}"] < 0.995:
                    client.delete_collection(cname)
                    E20.E8.atomic_json(done, {"name": name, "status": "EXCLUDED", "exact_parity": parity,
                                              "reason": f"parity {parity[f'b{b}-{p}']:.4f} < 0.995 b{b} {p}"})
                    return
            for ef in E20.EFS:
                for p in PATHS:
                    ids, lat = E20.search_all(client, cname, qv[p], m.SearchParams(hnsw_ef=ef))
                    run = E20.top10(ids, doc_ids, data["q_ids"])
                    ndcg = float(np.mean(list(V.ndcg10(
                        {q: {d: float(10 - r) for r, d in enumerate(l)} for q, l in run.items()},
                        data["qrels"], data["q_ids"]).values())))
                    rows.append({"workload": "fiqa25k", "build": b, "path": p, "hnsw_ef": ef,
                                 "ann_ndcg10": ndcg, "ann_ndcg_loss_rel": 1 - ndcg / exact_ndcg[p],
                                 "ann_recovery_at10": E20.recovery_tied(run, tie[p][0], tie[p][1],
                                                                        doc_index, data["q_ids"]),
                                 "search_p50_ms": float(np.median(lat))})
                    print(f"  {name} b{b} ef={ef} {p}: loss {rows[-1]['ann_ndcg_loss_rel']:.3f} "
                          f"rec {rows[-1]['ann_recovery_at10']:.3f}", flush=True)
            client.delete_collection(cname)
    finally:
        proc.terminate()
        proc.wait(timeout=60)
    E20.E8.atomic_json(done, {"name": name, "status": "COMPLETE", "n_docs": len(dv),
                              "n_relevant_kept": int(len({d for q in data["qrels"].values() for d, s in q.items() if s > 0})),
                              "exact_ndcg10": exact_ndcg, "exact_parity": parity, "builds": builds,
                              "rows": rows, "subsample_first_last_idx": [int(idx[0]), int(idx[-1])]})


def included():
    """The E20 included roster (the committed result's spaces), not the E8/E8x eligibility list."""
    e20 = json.loads(E20.RESULT.read_text())
    return [n for n in e20["spaces"] if n != "arctic-embed-l-mean"]


def assemble():
    e20 = json.loads(E20.RESULT.read_text())
    out = {}
    for name in included():
        f = OUT_DIR / name / "sweep.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        if s.get("status") != "COMPLETE":
            out[name] = {"status": s.get("status"), "reason": s.get("reason")}
            continue
        gaps = []
        for b in range(E20.BUILDS):
            pick = lambda p: next(r for r in s["rows"] if r["build"] == b and r["path"] == p and r["hnsw_ef"] == E20.REF_EF)
            gaps.append(pick("table")["ann_recovery_at10"] - pick("teacher")["ann_recovery_at10"])
        mult = [E20.multiplier(s["rows"], "fiqa25k", b, "table", "ann_recovery_at10") for b in range(E20.BUILDS)]
        w = e20["spaces"].get(name, {}).get("workloads", {})
        out[name] = {"status": "COMPLETE", "gap_fiqa25k": 100 * float(np.mean(gaps)),
                     "gap_fiqa57k": 100 * float(np.mean(w["fiqa"]["table_minus_teacher_recovery_at_ref_ef"])) if "fiqa" in w else None,
                     "gap_scidocs": 100 * float(np.mean(w["scidocs"]["table_minus_teacher_recovery_at_ref_ef"])) if "scidocs" in w else None,
                     "recovery_multiplier_builds": mult, "exact_ndcg10": s["exact_ndcg10"],
                     "exact_parity": s["exact_parity"]}
    done = [v for v in out.values() if v.get("status") == "COMPLETE"]
    d25 = np.array([v["gap_fiqa25k"] for v in done]); d57 = np.array([v["gap_fiqa57k"] for v in done])
    dsc = np.array([v["gap_scidocs"] for v in done])
    rng = np.random.default_rng(SEED)
    def boot(x):
        vals = [float(np.mean(x[rng.integers(0, len(x), len(x))])) for _ in range(10_000)]
        return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]
    summary = {"n_spaces": len(done),
               "mean_gap_fiqa25k": float(d25.mean()), "mean_gap_fiqa57k": float(d57.mean()),
               "mean_gap_scidocs": float(dsc.mean()),
               "paired_25k_minus_57k": float((d25 - d57).mean()), "paired_25k_minus_57k_bootstrap95": boot(d25 - d57),
               "paired_25k_minus_scidocs": float((d25 - dsc).mean()), "paired_25k_minus_scidocs_bootstrap95": boot(d25 - dsc),
               "spaces_gap_negative_25k": int((d25 < 0).sum())}
    write_result(RESULT, {"status": "COMPLETE", "measurement": "E23 (exploratory, pre-specified)",
                          "scope": "FiQA subsampled to 25,657 documents keeping judged-relevant ones; same 648 queries; "
                                   "size versus domain is a comparison across two corpora, not an identified cause",
                          "n_sub": N_SUB, "seed": SEED, "summary": summary, "spaces": out,
                          "receipt": receipt(__file__, [{"e20_result": str(E20.RESULT.relative_to(REPO))}], utc_now(),
                                             ("qdrant-client", "scipy"))})


def all_steps():
    py = sys.executable
    env = {**os.environ, "PYTHONPATH": f"{REPO / 'm15src'}:{REPO / 'm8src'}:{REPO / 'm7src'}",
           "M7_ENCODER": "stella-400M-v5"}
    for name in included():
        if not (E20.vec_dir(name) / "fiqa-doc-ids.json").exists():
            raise SystemExit(f"E23 STOP: inputs missing for {name}")
        subprocess.run([py, __file__, "sweep", name], check=True, cwd=REPO, env=env)
    subprocess.run([py, __file__, "assemble"], check=True, env=env, cwd=REPO)


def selftest():
    doc_ids = [f"d{i}" for i in range(100)]
    qrels = {"q1": {"d5": 1, "d7": 0}, "q2": {"d99": 2}}
    idx = subsample(doc_ids, qrels, 10, 0)
    assert len(idx) == 10 and len(set(idx)) == 10 and 5 in idx and 99 in idx, idx
    idx2 = subsample(doc_ids, qrels, 10, 0)
    assert (idx == idx2).all()
    print("selftest ok: subsample keeps relevant docs, is seeded, has no duplicates")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["all"]
    {"assemble": assemble, "all": all_steps, "selftest": selftest, "sweep": lambda: sweep(rest[0])}[cmd]()
