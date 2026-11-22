"""E25: LightRetriever's own lookup path under graph search (m15/MEASUREMENTS.md, E25). Runpod.

    e25_lightretriever.py vectors <fiqa|scidocs>   doc vectors, lookup and full query vectors, exact top-100
    e25_lightretriever.py sweep <fiqa|scidocs>     two HNSW builds, ef grid, both query paths (E20 protocol)
    e25_lightretriever.py assemble                 results/m15_e25_lightretriever.json
    e25_lightretriever.py all
    e25_lightretriever.py selftest                 synthetic check of the two query constructions

Model handling follows the project's earlier faithful reproduction (bench/run_lightretriever.py):
Qwen2.5-1.5B base plus the released LoRA adapter, bfloat16, last-token pooling, L2 normalize;
tables built from [bos] + prompt + [token] + [eos]; lookup queries are the normalized mean of
token rows (add_special_tokens=False). The full query path encodes [bos] + prompt + query + [eos]
through the same model and takes the EOS state.
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
import e20_ann_spaces as E20          # noqa: E402  Qdrant protocol
from common import load_public, receipt, sha_file, utc_now, write_result  # noqa: E402

ADAPTER = "lightretriever/lightretriever-qwen2.5-1.5b"
ADAPTER_REV = "59776f218c4d13a1fbb075409514a82c452d27ae"
BASE = "Qwen/Qwen2.5-1.5B"
WORKLOADS = ("fiqa", "scidocs")
INSTRUCTIONS = {"websearch": "Given a web search query, retrieve relevant passages that answer the query",
                "fiqa": "Given a financial question, retrieve user replies that best answer the question",
                "scidocs": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper"}
PATHS = ("full_websearch", "lookup_websearch", "full_task", "lookup_task")
OUT_DIR = REPO / "work" / "m15" / "e25"
RESULT = REPO / "results" / "m15_e25_lightretriever.json"
SEED = 20260930
DEVICE = "cuda"


# ---------------------------------------------------------------- model

def load_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ADAPTER, revision=ADAPTER_REV)
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, ADAPTER, revision=ADAPTER_REV).merge_and_unload()
    model.eval().to(DEVICE)
    return tok, model


def encode_last_token(tok, model, id_lists, bs_tokens=8192, max_bs=64):
    """Normalized EOS/last-token states for pre-tokenized sequences, length-sorted batches."""
    import torch
    order = np.argsort([-len(e) for e in id_lists])
    out = np.zeros((len(id_lists), model.config.hidden_size), dtype=np.float32)
    i, t0, done = 0, time.time(), 0
    with torch.no_grad():
        while i < len(order):
            L = len(id_lists[order[i]])
            bs = min(max_bs, max(1, bs_tokens // max(L, 1)))
            idx = order[i:i + bs]
            batch = tok.pad({"input_ids": [id_lists[j] for j in idx]}, return_tensors="pt").to(DEVICE)
            hidden = model.model(**batch).last_hidden_state
            last = batch["attention_mask"].sum(1) - 1
            v = hidden[torch.arange(hidden.shape[0]), last]
            out[idx] = torch.nn.functional.normalize(v.float(), p=2, dim=-1).cpu().numpy()
            i += bs
            done += len(idx)
            if done % 5000 < bs:
                print(f"  {done}/{len(id_lists)} at {done / (time.time() - t0):.1f}/s", flush=True)
    return out


def doc_ids_for(tok, texts):
    return tok(texts, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]


def prompt_head(tok, instruction):
    return [tok.bos_token_id] + tok(f"Instruct: {instruction}\nQuery: ", add_special_tokens=False)["input_ids"]


def full_query_ids(tok, instruction, texts):
    head, eos = prompt_head(tok, instruction), tok.eos_token_id
    return [head + tok(t, add_special_tokens=False, truncation=True, max_length=512)["input_ids"] + [eos]
            for t in texts]


def build_table(tok, model, instruction, bs=1000):
    import torch
    head, eos = prompt_head(tok, instruction), tok.eos_token_id
    V, H = len(tok), model.config.hidden_size
    table = np.zeros((V, H), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, V, bs):
            ids = list(range(start, min(start + bs, V)))
            batch = torch.tensor([head + [i, eos] for i in ids], device=DEVICE)
            table[ids] = model.model(input_ids=batch).last_hidden_state[:, -1].float().cpu().numpy()
    return table, time.time() - t0


def lookup_queries(table, tok, texts):
    ids = [tok(t, add_special_tokens=False, truncation=True, max_length=512)["input_ids"] for t in texts]
    out = np.zeros((len(ids), table.shape[1]), dtype=np.float32)
    for i, row in enumerate(ids):
        v = table[row].mean(0) if row else np.zeros(table.shape[1], dtype=np.float32)
        out[i] = v / (np.linalg.norm(v) + 1e-12)
    return out


# ---------------------------------------------------------------- vectors

def vectors(ds):
    out = OUT_DIR / ds
    done = out / "vectors.json"
    if done.exists():
        print(f"{ds}: vectors already done", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    data = load_public(ds)
    tok, model = load_model()
    meta = {"workload": ds, "n_docs": len(data["doc_ids"]), "n_queries": len(data["q_ids"]),
            "pins": data["pins"], "adapter": ADAPTER, "adapter_revision": ADAPTER_REV, "base": BASE,
            "paths": {}, "timing": {}}
    t0 = time.time()
    dv = encode_last_token(tok, model, doc_ids_for(tok, data["doc_texts"]))
    meta["timing"]["docs_seconds"] = time.time() - t0
    np.save(out / f"{ds}-docs.npy", dv.astype(np.float16))
    # Exact references use the stored fp16 representation, the same vectors the sweep indexes.
    dv = np.load(out / f"{ds}-docs.npy").astype(np.float32)
    (out / f"{ds}-doc-ids.json").write_text(json.dumps(data["doc_ids"]))
    tables = {}
    for name, instr in (("websearch", INSTRUCTIONS["websearch"]), ("task", INSTRUCTIONS[ds])):
        tables[name], secs = build_table(tok, model, instr)
        meta["timing"][f"table_{name}_seconds"] = secs
    qv = {}
    for name, instr in (("websearch", INSTRUCTIONS["websearch"]), ("task", INSTRUCTIONS[ds])):
        t0 = time.time()
        qv[f"full_{name}"] = encode_last_token(tok, model, full_query_ids(tok, instr, data["q_texts"]))
        meta["timing"][f"full_{name}_query_seconds_per_query"] = (time.time() - t0) / len(data["q_ids"])
        t0 = time.time()
        qv[f"lookup_{name}"] = lookup_queries(tables[name], tok, data["q_texts"])
        meta["timing"][f"lookup_{name}_query_seconds_per_query"] = (time.time() - t0) / len(data["q_ids"])
    del model
    from evalkit import topk_arrays
    import vectors15 as V
    for p, q in qv.items():
        np.save(out / f"{ds}-{p}-q.npy", q)
        ids, scores = topk_arrays(q, dv, k=100, chunk=250_000, device=DEVICE)
        ids = np.asarray(ids, dtype=np.int32)
        np.save(out / f"{ds}-{p}-exact-ids.npy", ids)
        run = E20.top10([list(map(int, r)) for r in ids], data["doc_ids"], data["q_ids"])
        ndcg = float(np.mean(list(V.ndcg10({qq: {d: float(10 - r) for r, d in enumerate(l)}
                                            for qq, l in run.items()}, data["qrels"], data["q_ids"]).values())))
        meta["paths"][p] = {"exact_ndcg10": ndcg,
                            "geometry": {"top1_cos_quantiles": np.quantile(np.asarray(scores)[:, 0], [.1, .5, .9]).tolist()}}
        print(f"  {ds} {p}: exact nDCG@10 {ndcg:.4f}", flush=True)
    E20.E8.atomic_json(done, meta)


# ---------------------------------------------------------------- sweep

def sweep(ds):
    from qdrant_client import QdrantClient, models as m
    import vectors15 as V
    out = OUT_DIR / ds
    done = out / "sweep.json"
    if done.exists():
        print(f"{ds}: sweep already done", flush=True)
        return
    meta = json.loads((out / "vectors.json").read_text())
    data = load_public(ds, with_corpus=False)
    doc_ids = json.loads((out / f"{ds}-doc-ids.json").read_text())
    dv = np.load(out / f"{ds}-docs.npy")
    qv = {p: np.load(out / f"{ds}-{p}-q.npy") for p in PATHS}
    exact = {p: E20.top10([list(map(int, r)) for r in np.load(out / f"{ds}-{p}-exact-ids.npy")],
                          doc_ids, data["q_ids"]) for p in PATHS}
    doc_index = {d: i for i, d in enumerate(doc_ids)}
    tie = {p: E20.tie_thresholds(qv[p], dv, exact[p], doc_index, data["q_ids"]) for p in PATHS}
    proc = E20.start_qdrant(f"e25-{ds}")
    rows, builds, parity = [], {}, {}
    try:
        client = QdrantClient(host="127.0.0.1", port=E20.PORT, grpc_port=E20.GRPC, prefer_grpc=True, timeout=600)
        for b in range(E20.BUILDS):
            order = np.arange(len(dv)) if b == 0 else np.random.default_rng(SEED + b).permutation(len(dv))
            cname = f"e25-{ds}-b{b}"
            builds[f"b{b}"] = E20.build(client, cname, dv, order)
            print(f"built {cname}: {builds[f'b{b}']}", flush=True)
            for p in PATHS:
                qc = data["q_ids"][:E20.EXACT_CHECK]
                ids, _ = E20.search_all(client, cname, qv[p][:E20.EXACT_CHECK], m.SearchParams(exact=True), limit=11)
                parity[f"b{b}-{p}"] = E20.recovery_tied(E20.top10(ids, doc_ids, qc), tie[p][0], tie[p][1], doc_index, qc)
                if parity[f"b{b}-{p}"] < 0.995:
                    raise SystemExit(f"E25 STOP: exact parity {parity[f'b{b}-{p}']:.4f} for {ds} {p}")
            for ef in E20.EFS:
                for p in PATHS:
                    ids, lat = E20.search_all(client, cname, qv[p], m.SearchParams(hnsw_ef=ef))
                    run = E20.top10(ids, doc_ids, data["q_ids"])
                    ndcg = float(np.mean(list(V.ndcg10({qq: {d: float(10 - r) for r, d in enumerate(l)}
                                                        for qq, l in run.items()}, data["qrels"], data["q_ids"]).values())))
                    ex = meta["paths"][p]["exact_ndcg10"]
                    rows.append({"workload": ds, "build": b, "path": p, "hnsw_ef": ef, "ann_ndcg10": ndcg,
                                 "ann_ndcg_loss_rel": 1 - ndcg / ex,
                                 "ann_recovery_at10": E20.recovery_tied(run, tie[p][0], tie[p][1], doc_index, data["q_ids"]),
                                 "per_query_recovery": [float((tie[p][0][i, [doc_index[d] for d in run[q]]] >= tie[p][1][q]).sum()) / 10
                                                        if run[q] else 0.0 for i, q in enumerate(data["q_ids"])],
                                 "search_p50_ms": float(np.median(lat))})
                    print(f"  {ds} b{b} ef={ef} {p}: loss {rows[-1]['ann_ndcg_loss_rel']:.3f} rec {rows[-1]['ann_recovery_at10']:.3f}", flush=True)
            client.delete_collection(cname)
    finally:
        proc.terminate()
        proc.wait(timeout=60)
    E20.E8.atomic_json(done, {"workload": ds, "rows": rows, "builds": builds, "exact_parity": parity,
                              "qdrant_binary_sha256": sha_file(E20.QDRANT_DIR / "qdrant")})


# ---------------------------------------------------------------- assemble

def multiplier(rows, ds, b, path, ref_path, metric):
    pick = lambda p, ef: next(r for r in rows if r["workload"] == ds and r["build"] == b and r["path"] == p and r["hnsw_ef"] == ef)
    ref = pick(ref_path, E20.REF_EF)[metric]
    better = (lambda v: v <= ref) if metric == "ann_ndcg_loss_rel" else (lambda v: v >= ref)
    for ef in E20.EFS:
        if better(pick(path, ef)[metric]):
            return ef / E20.REF_EF
    return None


def assemble():
    from common import bootstrap_mean
    out = {"workloads": {}}
    for ds in WORKLOADS:
        meta = json.loads((OUT_DIR / ds / "vectors.json").read_text())
        sw = json.loads((OUT_DIR / ds / "sweep.json").read_text())
        w = {"n_docs": meta["n_docs"], "n_queries": meta["n_queries"], "exact": meta["paths"],
             "timing": meta["timing"], "exact_parity": sw["exact_parity"], "builds": sw["builds"], "contrasts": {}}
        for instr in ("websearch", "task"):
            lk, fl = f"lookup_{instr}", f"full_{instr}"
            lm = [multiplier(sw["rows"], ds, b, lk, fl, "ann_ndcg_loss_rel") for b in range(E20.BUILDS)]
            rm = [multiplier(sw["rows"], ds, b, lk, fl, "ann_recovery_at10") for b in range(E20.BUILDS)]
            agg = lambda v: None if any(x is None for x in v) else float(np.mean(v))   # E20 rule: censored if any build is
            c = {"loss_multiplier_builds": lm, "loss_multiplier": agg(lm),
                 "recovery_multiplier_builds": rm, "recovery_multiplier": agg(rm)}
            gaps, pq = [], []
            for b in range(E20.BUILDS):
                rl = next(r for r in sw["rows"] if r["build"] == b and r["path"] == lk and r["hnsw_ef"] == E20.REF_EF)
                rf = next(r for r in sw["rows"] if r["build"] == b and r["path"] == fl and r["hnsw_ef"] == E20.REF_EF)
                gaps.append(rl["ann_recovery_at10"] - rf["ann_recovery_at10"])
                pq.append(np.array(rl["per_query_recovery"]) - np.array(rf["per_query_recovery"]))
            d = np.mean(pq, axis=0)
            c["recovery_gap_at_ref_ef_pp"] = 100 * float(np.mean(gaps))
            c["recovery_gap_query_bootstrap95_pp"] = [100 * v for v in bootstrap_mean(d)]
            c["loss_at_ref_ef"] = {p: float(np.mean([r["ann_ndcg_loss_rel"] for r in sw["rows"] if r["path"] == p and r["hnsw_ef"] == E20.REF_EF])) for p in (lk, fl)}
            w["contrasts"][instr] = c
        out["workloads"][ds] = w
        out[f"rows_{ds}"] = [{k: v for k, v in r.items() if k != "per_query_recovery"} for r in sw["rows"]]
    write_result(RESULT, {"status": "COMPLETE", "measurement": "E25 (exploratory, pre-specified before encoding)",
                          "scope": "LightRetriever qwen2.5-1.5b dense paths over its own FiQA and SCIDOCS indexes; "
                                   "one engine, E20 protocol; websearch instruction primary; no retention or speedup claim",
                          "model": {"adapter": ADAPTER, "adapter_revision": ADAPTER_REV, "base": BASE},
                          "qdrant": {"version": "1.19.1", "binary_sha256": sha_file(E20.QDRANT_DIR / "qdrant")},
                          **out, "receipt": receipt(__file__, [], utc_now(), ("torch", "transformers", "peft", "qdrant-client"))})


def all_steps():
    py = sys.executable
    env = {**os.environ, "PYTHONPATH": f"{REPO / 'm15src'}:{REPO / 'm8src'}:{REPO / 'm7src'}", "M7_ENCODER": "stella-400M-v5"}
    subprocess.run([py, str(REPO / "m15src" / "e20_ann_spaces.py"), "qdrant"], check=True, env=env, cwd=REPO)
    for ds in WORKLOADS:
        subprocess.run([py, __file__, "vectors", ds], check=True, env=env, cwd=REPO)
    for ds in WORKLOADS:
        subprocess.run([py, __file__, "sweep", ds], check=True, env=env, cwd=REPO)
    subprocess.run([py, __file__, "assemble"], check=True, env=env, cwd=REPO)


def selftest():
    class Tok:
        bos_token_id, eos_token_id = 1, 2
        def __call__(self, t, add_special_tokens=False, truncation=True, max_length=512):
            ids = [10 + (ord(c) % 50) for c in t if c != " "]
            return {"input_ids": ([1] + ids + [2]) if add_special_tokens else ids}
        def __len__(self): return 100
    tok = Tok()
    table = np.random.default_rng(0).normal(size=(100, 8)).astype(np.float32)
    q = lookup_queries(table, tok, ["ab", "ba", "a a"])
    assert np.allclose(q[0], q[1]) and np.allclose(np.linalg.norm(q, axis=1), 1)      # order-blind, normalized
    assert np.allclose(q[2], table[[10 + ord("a") % 50]].mean(0) / np.linalg.norm(table[10 + ord("a") % 50]))
    f = full_query_ids(tok, "x", ["ab"])
    assert f[0][0] == 1 and f[0][-1] == 2 and f[0][1:-1] != []
    rows = [{"workload": "w", "build": 0, "path": p, "hnsw_ef": ef, "ann_ndcg_loss_rel": (k / ef), "ann_recovery_at10": 1 - k / ef}
            for p, k in (("lookup_websearch", 1.0), ("full_websearch", 0.5)) for ef in E20.EFS]
    assert multiplier(rows, "w", 0, "lookup_websearch", "full_websearch", "ann_ndcg_loss_rel") == 2.0
    print("selftest ok: lookup is order-blind and normalized, full query wraps bos/eos, multiplier 2.0")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["all"]
    {"assemble": assemble, "all": all_steps, "selftest": selftest,
     "vectors": lambda: vectors(rest[0]), "sweep": lambda: sweep(rest[0])}[cmd]()
