"""E20: ANN effort for table queries across teacher spaces (m15/MEASUREMENTS.md, E20). Runpod.

    e20_ann_spaces.py qdrant           fetch the Linux Qdrant v1.19.1 binary into work/m15/qdrant-linux
    e20_ann_spaces.py vectors <name>   one teacher: doc/teacher/table(/head) query vectors, exact top-100
    e20_ann_spaces.py sweep <name>     one teacher: two HNSW builds per workload, ef grid, every path
    e20_ann_spaces.py assemble         results/m15_e20_ann_spaces.json
    e20_ann_spaces.py all              every step, one subprocess per teacher, resumable
    e20_ann_spaces.py selftest         synthetic check of recovery/multiplier logic (no data)

`vectors` runs with M7_ENCODER set before any import (m7src resolves the tower at import time) and
reuses E8's Tower: cached document and teacher query vectors, and the table re-solved at the frozen
E8/E8x lambda. `sweep` needs no GPU and no tower import.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in ("m15src", "m8src", "m7src"):
    if str(REPO / p) not in sys.path:
        sys.path.insert(0, str(REPO / p))
import e8_towers as E8          # noqa: E402
import e8x_towers as E8X        # noqa: E402
from e19_head_screen import E8_CONFIGS, family  # noqa: E402  the registered roster, copied verbatim
from common import load_public, receipt, sha_file, utc_now, write_result  # noqa: E402

WORKLOADS = ("fiqa", "scidocs", "trec-covid")
EFS = (16, 32, 64, 128, 256, 512)
REF_EF = 64
BUILDS = 2
PATHS = ("teacher", "table", "head")      # head only when E19 has scored that teacher
OUT_DIR = REPO / "work" / "m15" / "e20"
QDRANT_DIR = REPO / "work" / "m15" / "qdrant-linux"
RESULT = REPO / "results" / "m15_e20_ann_spaces.json"
E8_RESULT = REPO / "results" / "m15_e8_towers.json"
E8X_RESULT = REPO / "results" / "m15_e8x_towers.json"
E19_DIR = REPO / "work" / "m15" / "e19"
SEED = 20260930
WARMUP = 100
EXACT_CHECK = 200
PORT = int(os.environ.get("E20_PORT", "6433"))   # dedicated ports: never the E2/E16 instance;
GRPC = PORT + 1                                  # parallel sweep loops set E20_PORT apart


def recipe1():
    rows = {}
    rows.update(json.loads(E8_RESULT.read_text())["configs"])
    rows.update(json.loads(E8X_RESULT.read_text())["new_configs"])
    return rows


def eligible():
    r1 = recipe1()
    return [n for n in E8_CONFIGS + tuple(E8X.NEW) if n in r1]


# ---------------------------------------------------------------- qdrant binary

def fetch_qdrant():
    """Linux x86_64 Qdrant v1.19.1 from the GitHub release; hash recorded in the result."""
    binary = QDRANT_DIR / "qdrant"
    if binary.exists():
        print("qdrant binary present", flush=True)
        return
    QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    rel = json.load(urllib.request.urlopen(
        "https://api.github.com/repos/qdrant/qdrant/releases/tags/v1.19.1"))
    # The static musl build: the pod image's glibc is older than the gnu build requires.
    assets = [a for a in rel["assets"] if "x86_64-unknown-linux-musl" in a["name"]
              and a["name"].endswith(".tar.gz")]
    if not assets:
        raise SystemExit("E20 STOP: no x86_64 linux-musl asset for v1.19.1")
    tar = QDRANT_DIR / assets[0]["name"]
    urllib.request.urlretrieve(assets[0]["browser_download_url"], tar)
    # tar exits 2 on the pod's network filesystem (ownership/utime warnings) after a complete
    # extraction, so the binary's presence is the check, not tar's exit code.
    subprocess.run(["tar", "-xzf", str(tar), "-C", str(QDRANT_DIR)])
    if not binary.exists():
        found = [p for p in QDRANT_DIR.rglob("qdrant") if p.is_file()]
        if not found:
            raise SystemExit("E20 STOP: qdrant binary not found after extraction")
        found[0].rename(binary)
    binary.chmod(0o755)
    print("fetched", assets[0]["name"], sha_file(binary)[:16], flush=True)


def start_qdrant(tag):
    storage = QDRANT_DIR / f"storage-{tag}"
    log = open(QDRANT_DIR / f"qdrant-{tag}.log", "w")
    proc = subprocess.Popen([str(QDRANT_DIR / "qdrant")], cwd=QDRANT_DIR, stdout=log, stderr=log,
                            env={"QDRANT__STORAGE__STORAGE_PATH": str(storage),
                                 "QDRANT__SERVICE__HTTP_PORT": str(PORT),
                                 "QDRANT__SERVICE__GRPC_PORT": str(GRPC),
                                 "QDRANT__TELEMETRY_DISABLED": "true",
                                 "PATH": "/usr/bin:/bin"})
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/readyz", timeout=1)
            return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise SystemExit("Qdrant did not start")


# ---------------------------------------------------------------- vectors (GPU, per teacher)

def vec_dir(name):
    return OUT_DIR / name


def top10(id_lists, doc_ids, q_ids):
    """Rank lists -> top-10 doc ids with the registered self-hit drop (as E2)."""
    return {q: [doc_ids[i] for i in lst if doc_ids[i] != q][:10] for q, lst in zip(q_ids, id_lists)}


def vectors(name):
    out = vec_dir(name)
    done = out / "vectors.json"
    if done.exists():
        print(f"{name}: vectors already done", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    r1 = recipe1()[name]
    t = E8.Tower(name)
    W, diag = t.solve(r1["lambda"], f"{name} E20 re-solve")
    head = None
    e19_six = E19_DIR / f"six-{name}.json"
    if e19_six.exists():
        import e19_head_screen as E19
        h = E19.Head(name)
        head = h.solve(json.loads(e19_six.read_text())["lambda"])
    from evalkit import topk_arrays
    meta = {"name": name, "lambda": r1["lambda"], "solve": diag, "head": head is not None,
            "workloads": {}}
    for ds in WORKLOADS:
        data = load_public(ds)                                   # M20 pins, SIX_ACCESS.log
        dv = t.docs(f"e8-six-{ds}-docs", data["doc_texts"])      # cached fp16, teacher doc path
        np.save(out / f"{ds}-docs.npy", np.asarray(dv, dtype=np.float16))
        paths = {"teacher": np.asarray(t.ceiling_queries(f"e8-six-{ds}-q", data["q_texts"]),
                                       dtype=np.float32),
                 "table": np.asarray(t.table_queries(W, data["q_texts"]), dtype=np.float32)}
        if head is not None:
            import e19_head_screen as E19
            paths["head"] = E19.apply(head, E19.features_for(f"six-{ds}-q", data["q_texts"]))
        wl = {"n_docs": len(data["doc_ids"]), "n_queries": len(data["q_ids"]),
              "pins": data["pins"], "paths": {}}
        for pname, qv in paths.items():
            qv = qv / np.maximum(np.linalg.norm(qv, axis=1, keepdims=True), 1e-12)
            np.save(out / f"{ds}-{pname}-q.npy", qv.astype(np.float32))
            ids, scores = topk_arrays(qv, dv, k=100, chunk=250_000, device="cuda")
            ids, scores = np.asarray(ids, dtype=np.int32), np.asarray(scores, dtype=np.float32)
            np.save(out / f"{ds}-{pname}-exact-ids.npy", ids)
            import vectors15 as V
            run = top10([list(map(int, r)) for r in ids], data["doc_ids"], data["q_ids"])
            ndcg = float(np.mean(list(V.ndcg10({q: {d: float(10 - r) for r, d in enumerate(l)}
                                                for q, l in run.items()},
                                               data["qrels"], data["q_ids"]).values())))
            wl["paths"][pname] = {
                "exact_ndcg10": ndcg,
                "geometry": {"top1_cos_quantiles": np.quantile(scores[:, 0], [.1, .5, .9]).tolist(),
                             "top1_minus_top10_quantiles":
                                 np.quantile(scores[:, 0] - scores[:, 9], [.1, .5, .9]).tolist()}}
            print(f"  {name} {ds} {pname}: exact nDCG@10 {ndcg:.4f}, median top-1 cos "
                  f"{np.median(scores[:, 0]):.3f}", flush=True)
        meta["workloads"][ds] = wl
        del dv
        t.m8base.empty_cache()
    E8.atomic_json(done, meta)


# ---------------------------------------------------------------- sweep (CPU, per teacher)

def build(client, cname, dv, order):
    from qdrant_client import models as m
    if client.collection_exists(cname):
        client.delete_collection(cname)
    client.create_collection(
        cname, vectors_config={"dense": m.VectorParams(size=dv.shape[1],
                                                       distance=m.Distance.COSINE)},
        hnsw_config=m.HnswConfigDiff(m=16, ef_construct=100),
        optimizers_config=m.OptimizersConfigDiff(indexing_threshold=1))
    t0 = time.time()
    for lo in range(0, len(order), 1024):
        idx = order[lo:lo + 1024]
        vecs = dv[idx].astype(np.float32)
        client.upsert(cname, points=[m.PointStruct(id=int(i), vector={"dense": v.tolist()})
                                     for i, v in zip(idx, vecs)], wait=False)
    while True:
        info = client.get_collection(cname)
        if (info.status == m.CollectionStatus.GREEN and info.points_count == len(dv) and
                (info.indexed_vectors_count or 0) >= len(dv)):
            break
        time.sleep(3)
    return {"seconds_to_green": round(time.time() - t0, 1), "points": info.points_count,
            "indexed_vectors": info.indexed_vectors_count}


def search_all(client, cname, qvecs, params, limit=11):
    for v in qvecs[:WARMUP]:
        client.query_points(cname, query=v.tolist(), using="dense", limit=limit, search_params=params)
    ids, lat = [], []
    for v in qvecs:
        s = time.perf_counter()
        r = client.query_points(cname, query=v.tolist(), using="dense", limit=limit,
                                search_params=params)
        lat.append((time.perf_counter() - s) * 1000)
        ids.append([p.id for p in r.points])
    return ids, np.array(lat)


def recovery(ann_run, exact_run):
    return float(np.mean([len(set(ann_run[q]) & set(exact_run[q])) / max(1, len(exact_run[q]))
                          for q in exact_run]))


def sweep(name):
    from qdrant_client import QdrantClient, models as m
    import vectors15 as V
    out = vec_dir(name)
    done = out / "sweep.json"
    if done.exists():
        print(f"{name}: sweep already done", flush=True)
        return
    meta = json.loads((out / "vectors.json").read_text())
    proc = start_qdrant(name)
    rows, builds, parity = [], {}, {}
    try:
        client = QdrantClient(host="127.0.0.1", port=PORT, grpc_port=GRPC, prefer_grpc=True,
                              timeout=600)
        for ds in WORKLOADS:
            data = load_public(ds, with_corpus=False)
            doc_ids = json.loads((out / f"{ds}-doc-ids.json").read_text()) \
                if (out / f"{ds}-doc-ids.json").exists() else load_public(ds)["doc_ids"]
            (out / f"{ds}-doc-ids.json").write_text(json.dumps(doc_ids))
            dv = np.load(out / f"{ds}-docs.npy")
            paths = [p for p in PATHS if p in meta["workloads"][ds]["paths"]]
            qv = {p: np.load(out / f"{ds}-{p}-q.npy") for p in paths}
            exact = {p: top10([list(map(int, r)) for r in np.load(out / f"{ds}-{p}-exact-ids.npy")],
                              doc_ids, data["q_ids"]) for p in paths}
            for b in range(BUILDS):
                order = np.arange(len(dv)) if b == 0 else \
                    np.random.default_rng(SEED + b).permutation(len(dv))
                cname = f"e20-{ds}-b{b}"
                builds[f"{ds}-b{b}"] = build(client, cname, dv, order)
                print(f"built {cname}: {builds[f'{ds}-b{b}']}", flush=True)
                for p in paths:                       # collection agrees with numpy exact search
                    ids, _ = search_all(client, cname, qv[p][:EXACT_CHECK],
                                        m.SearchParams(exact=True), limit=11)
                    got = top10(ids, doc_ids, data["q_ids"][:EXACT_CHECK])
                    parity[f"{ds}-b{b}-{p}"] = recovery(got, {q: exact[p][q] for q in got})
                    if parity[f"{ds}-b{b}-{p}"] < 0.999:
                        raise SystemExit(f"E20 STOP: exact parity {parity[f'{ds}-b{b}-{p}']} "
                                         f"for {name} {ds} {p}")
                for ef in EFS:
                    for p in paths:
                        ids, lat = search_all(client, cname, qv[p], m.SearchParams(hnsw_ef=ef))
                        run = top10(ids, doc_ids, data["q_ids"])
                        ndcg = float(np.mean(list(V.ndcg10(
                            {q: {d: float(10 - r) for r, d in enumerate(l)} for q, l in run.items()},
                            data["qrels"], data["q_ids"]).values())))
                        ex = meta["workloads"][ds]["paths"][p]["exact_ndcg10"]
                        rows.append({"workload": ds, "build": b, "path": p, "hnsw_ef": ef,
                                     "ann_ndcg10": ndcg, "ann_ndcg_loss_rel": 1 - ndcg / ex,
                                     "ann_recovery_at10": recovery(run, exact[p]),
                                     "search_p50_ms": float(np.median(lat)),
                                     "search_p95_ms": float(np.quantile(lat, .95))})
                        print(f"  {name} {ds} b{b} ef={ef} {p}: loss {rows[-1]['ann_ndcg_loss_rel']:.3f} "
                              f"rec {rows[-1]['ann_recovery_at10']:.3f}", flush=True)
                client.delete_collection(cname)
    finally:
        proc.terminate()
        proc.wait(timeout=60)
    E8.atomic_json(done, {"name": name, "rows": rows, "builds": builds, "exact_parity": parity,
                          "qdrant_binary_sha256": sha_file(QDRANT_DIR / "qdrant")})


# ---------------------------------------------------------------- analysis

def multiplier(rows, ds, b, path, metric):
    """Smallest grid ef at which `path` matches the teacher's ef=64 value of `metric`; None = censored."""
    pick = lambda p, ef: next(r for r in rows if r["workload"] == ds and r["build"] == b
                              and r["path"] == p and r["hnsw_ef"] == ef)
    ref = pick("teacher", REF_EF)[metric]
    better = (lambda v: v <= ref) if metric == "ann_ndcg_loss_rel" else (lambda v: v >= ref)
    for ef in EFS:
        if better(pick(path, ef)[metric]):
            return ef / REF_EF
    return None


def assemble():
    from scipy.stats import spearmanr
    rows_by, spaces = {}, {}
    for name in eligible():
        meta = json.loads((vec_dir(name) / "vectors.json").read_text())
        sw = json.loads((vec_dir(name) / "sweep.json").read_text())
        rows_by[name] = sw["rows"]
        sp = {"family": family(name), "lambda": meta["lambda"], "head": meta["head"],
              "builds": sw["builds"], "exact_parity": sw["exact_parity"], "workloads": {}}
        for ds in WORKLOADS:
            w = meta["workloads"][ds]
            paths = [p for p in PATHS if p in w["paths"]]
            d = {"n_docs": w["n_docs"], "n_queries": w["n_queries"], "exact": w["paths"]}
            for p in paths:
                if p == "teacher":
                    continue
                d[f"{p}_multiplier_loss"] = [multiplier(sw["rows"], ds, b, p, "ann_ndcg_loss_rel")
                                             for b in range(BUILDS)]
                d[f"{p}_multiplier_recovery"] = [multiplier(sw["rows"], ds, b, p, "ann_recovery_at10")
                                                 for b in range(BUILDS)]
                gap = [next(r for r in sw["rows"] if r["workload"] == ds and r["build"] == b and
                            r["path"] == p and r["hnsw_ef"] == REF_EF)["ann_recovery_at10"] -
                       next(r for r in sw["rows"] if r["workload"] == ds and r["build"] == b and
                            r["path"] == "teacher" and r["hnsw_ef"] == REF_EF)["ann_recovery_at10"]
                       for b in range(BUILDS)]
                d[f"{p}_minus_teacher_recovery_at_ref_ef"] = gap
                g = w["paths"][p]["geometry"]
                gt = w["paths"]["teacher"]["geometry"]
                d[f"{p}_geometry_delta"] = {
                    "top1_cos_median": g["top1_cos_quantiles"][1] - gt["top1_cos_quantiles"][1],
                    "margin_median": g["top1_minus_top10_quantiles"][1]
                                     - gt["top1_minus_top10_quantiles"][1]}
            sp["workloads"][ds] = d
        spaces[name] = sp
    summary = {}
    for ds in WORKLOADS:
        names = [n for n in spaces if n != E8.CONTROL]
        mult = {n: spaces[n]["workloads"][ds]["table_multiplier_loss"] for n in names}
        # Builds are repetitions: one multiplier per space (mean over builds), censored if either
        # build is censored (Astra pre-spend review, finding 1).
        per_space = {n: (None if any(v is None for v in mult[n]) else float(np.mean(mult[n])))
                     for n in names}
        gaps = [float(np.mean(spaces[n]["workloads"][ds]["table_minus_teacher_recovery_at_ref_ef"]))
                for n in names]
        dcos = [spaces[n]["workloads"][ds]["table_geometry_delta"]["top1_cos_median"] for n in names]
        dmar = [spaces[n]["workloads"][ds]["table_geometry_delta"]["margin_median"] for n in names]
        reached = [v for v in per_space.values() if v is not None]
        summary[ds] = {
            "n_spaces": len(names), "censored_spaces": sum(v is None for v in per_space.values()),
            "multiplier_per_space": per_space,
            "multiplier_quantiles_over_reached_spaces": np.quantile(reached, [.1, .5, .9]).tolist()
            if reached else None,
            "spaces_with_multiplier_above_1": sum(v is None or v > 1 for v in per_space.values()),
            "spaces_with_multiplier_1_or_below": sum(v is not None and v <= 1
                                                     for v in per_space.values()),
            "mean_table_minus_teacher_recovery_at_ref_ef": float(np.mean(gaps)),
            "exploratory_spearman_gap_vs_top1_cos_delta": float(spearmanr(gaps, dcos)[0]),
            "exploratory_spearman_gap_vs_margin_delta": float(spearmanr(gaps, dmar)[0]),
            "exploratory_bootstrap95_gap_vs_top1_cos_delta": E8X.boot_spearman(gaps, dcos),
            "by_family": {f: [n for n in names if spaces[n]["family"] == f]
                          for f in sorted({spaces[n]["family"] for n in names})}}
    # Non-gating consistency check against E2's unquantized Stella query rows on FiQA
    # (Astra pre-spend review, finding 3). E2's `zero` is the trained Zero, not the E8 table, so
    # only the teacher path is comparable.
    e2_path = REPO / "results" / "m15_e2_ann_fiqa.json"
    e2_check = None
    if e2_path.exists() and "stella-400M-v5" in rows_by:
        e2 = json.loads(e2_path.read_text())
        e2_rows = {r["hnsw_ef"]: r for r in e2["rows"]
                   if r["encoder"] == "stella-query" and r["quant"] == "none"}
        mine = [r for r in rows_by["stella-400M-v5"]
                if r["workload"] == "fiqa" and r["path"] == "teacher"]
        e2_check = {"comparable": "E2 stella-query, quant none versus E20 stella-400M-v5 teacher path",
                    "per_ef": {}}
        for ef in EFS:
            b = [r["ann_recovery_at10"] for r in mine if r["hnsw_ef"] == ef]
            e2_check["per_ef"][ef] = {
                "e2_recovery": e2_rows[ef]["ann_recovery_at10"] if ef in e2_rows else None,
                "e20_recovery_builds": b,
                "e2_within_build_spread": (ef in e2_rows and len(b) == 2 and
                                           min(b) - 0.01 <= e2_rows[ef]["ann_recovery_at10"]
                                           <= max(b) + 0.01)}
    write_result(RESULT, {
        "status": "COMPLETE", "measurement": "E20 (exploratory, pre-specified before scoring)",
        "e2_consistency_check_not_a_gate": e2_check,
        "scope": "closed-form tables over their own teachers' indexes; one engine; HNSW builds are "
                 "repetitions; related checkpoints are not independent; latency is not a claim",
        "qdrant": {"version": "1.19.1", "binary_sha256": sha_file(QDRANT_DIR / "qdrant"),
                   "hnsw": {"m": 16, "ef_construct": 100}, "indexing_threshold_kb": 1},
        "ef_grid": EFS, "reference_ef": REF_EF, "builds_per_workload": BUILDS,
        "summary": summary, "spaces": spaces, "rows": rows_by,
        "receipt": receipt(__file__, [{"recipe1": [str(E8_RESULT.relative_to(REPO)),
                                                   str(E8X_RESULT.relative_to(REPO))]}],
                           utc_now(), ("qdrant-client", "torch", "scipy"))})


def all_steps():
    py = sys.executable
    env = {**os.environ, "PYTHONPATH": f"{REPO / 'm15src'}:{REPO / 'm8src'}:{REPO / 'm7src'}"}
    subprocess.run([py, __file__, "qdrant"], check=True, env=env, cwd=REPO)
    for name in eligible():
        subprocess.run([py, __file__, "vectors", name], check=True, cwd=REPO,
                       env={**env, "M7_ENCODER": name})
    for name in eligible():
        subprocess.run([py, __file__, "sweep", name], check=True, cwd=REPO, env=env)
    subprocess.run([py, __file__, "assemble"], check=True, env=env, cwd=REPO)


def selftest():
    rows = []
    for b in range(BUILDS):
        for ef in EFS:
            rows.append({"workload": "x", "build": b, "path": "teacher", "hnsw_ef": ef,
                         "ann_ndcg_loss_rel": 0.5 / ef, "ann_recovery_at10": 1 - 0.5 / ef})
            rows.append({"workload": "x", "build": b, "path": "table", "hnsw_ef": ef,
                         "ann_ndcg_loss_rel": 1.0 / ef, "ann_recovery_at10": 1 - 1.0 / ef})
            rows.append({"workload": "x", "build": b, "path": "head", "hnsw_ef": ef,
                         "ann_ndcg_loss_rel": 9.0 / ef, "ann_recovery_at10": 0.1})
    assert multiplier(rows, "x", 0, "table", "ann_ndcg_loss_rel") == 2.0
    assert multiplier(rows, "x", 1, "table", "ann_recovery_at10") == 2.0
    assert multiplier(rows, "x", 0, "head", "ann_ndcg_loss_rel") is None      # censored
    ex = {"q1": ["a", "b"], "q2": ["c"]}
    assert recovery({"q1": ["a", "x"], "q2": ["c"]}, ex) == 0.75
    assert top10([[0, 1, 2]], ["q1", "d1", "d2"], ["q1"]) == {"q1": ["d1", "d2"]}   # self-hit drop
    print("selftest ok: multiplier 2.0 / censored, recovery 0.75, self-hit drop")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["all"]
    {"qdrant": fetch_qdrant, "assemble": assemble, "all": all_steps, "selftest": selftest,
     "vectors": lambda: vectors(rest[0]), "sweep": lambda: sweep(rest[0])}[cmd]()
