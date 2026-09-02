"""M10.0 diagnostic (Mac, MPS): is a rank-k linear query head a structural ceiling for nano?

Teacher query vectors (stella-400M, s2p prompt) are projected onto their top-k principal
components (basis fit on an INDEPENDENT query set), renormalized, and retrieved against the
unmodified teacher document vectors of two dev components. A student with hidden width h and a
linear head can only emit vectors in an h-dimensional affine subspace, so the k=h row is an
UPPER BOUND on what any such student can retain -- whatever its backbone learns. Also repeated
in stella's 768-d and 256-d MRL heads, to price "regress to a smaller teacher target".

Reads DEV components only (cqadup-programmers, cqadup-physics); never the six or reserved sets.
Writes results/m10_rank_probe_mac.json. Diagnostic: read by no rule.
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m7src"))
os.environ.setdefault("M7_ENCODER", "stella-400M-v5")
import devsuite, encoders, evalkit, teacher  # noqa: E402
from _paths import DEVICE, REPO, WORK  # noqa: E402

COMPONENTS = ["cqadup-programmers", "cqadup-physics"]
HEADS = {1024: "2_Dense_1024", 768: "2_Dense_768", 256: "2_Dense_256"}
KS = [64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 1024]
OUT = REPO / "results" / "m10_rank_probe_mac.json"
CACHE = WORK / "m10_rank_probe"
CACHE.mkdir(parents=True, exist_ok=True)
SMOKE = int(os.environ.get("SMOKE", "0"))


def verify_manifest(name, doc_ids, doc_texts, q_ids, qrels):
    m = json.loads((REPO / "results" / "m7_dev_manifest.json").read_text())[name]
    got = {"corpus_ids_sha256": devsuite.sha(doc_ids), "corpus_text_sha256": devsuite.sha(doc_texts),
           "qids_sha256": devsuite.sha(sorted(q_ids)), "qrels_sha256": devsuite.sha(qrels)}
    bad = {k: (got[k], m[k]) for k in got if got[k] != m[k]}
    if bad:
        raise SystemExit(f"{name}: dev manifest mismatch {bad}")


@torch.no_grad()
def pooled_states(texts, prefix, tag, max_length=512, batch_tokens=16384):
    """Mean-pooled backbone states BEFORE any Dense head, fp32, cached. One forward, many heads."""
    p = CACHE / f"{tag}.npy"
    if p.exists():
        return np.load(p)
    tok, model = teacher.load_teacher()
    order, n_tok = teacher._order_by_length(tok, [prefix + t for t in texts], max_length)
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    i, t0, done = 0, time.time(), 0
    while i < len(order):
        longest, j = 0, i
        while j < len(order):
            L = max(longest, n_tok[order[j]])
            if (j - i + 1) * L > batch_tokens and j > i:
                break
            longest, j = L, j + 1
        idx = order[i:j]
        b = tok([prefix + texts[k] for k in idx], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt").to(DEVICE)
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        out[idx] = ((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy()
        done += len(idx)
        if done % 5000 < len(idx) or j >= len(order):
            print(f"    {tag}: {done}/{len(texts)} @ {done/(time.time()-t0):.1f} texts/s", flush=True)
        i = j
    np.save(p, out)
    return out


def load_head(dim):
    from dataclasses import replace
    spec = replace(encoders.by_repo(teacher.TEACHER), post_dense=HEADS[dim], dim=dim)
    return teacher.load_post_dense(spec)


def apply_head(pooled, dense):
    v = torch.from_numpy(pooled).to(DEVICE)
    W, bias = dense
    v = v @ W.T + (0.0 if bias is None else bias)
    return torch.nn.functional.normalize(v, dim=-1).cpu().numpy()


def pca_basis(X):
    mu = X.mean(0)
    _, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, vt, (s ** 2) / (s ** 2).sum()


def project(Q, mu, vt, k):
    Z = (Q - mu) @ vt[:k].T
    R = mu + Z @ vt[:k]
    return R / np.linalg.norm(R, axis=1, keepdims=True)


def main():
    out = {"_what": __doc__.strip(), "device": DEVICE, "teacher": teacher.TEACHER,
           "teacher_revision": teacher.TEACHER_REV, "smoke": SMOKE, "components": {}}
    # Independent PCA-fit queries: NQ-open train questions (never a dev/test qrel set here).
    from datasets import load_dataset
    fit_q = list(load_dataset("google-research-datasets/nq_open", split="train")["question"])
    rng = np.random.default_rng(0)
    fit_q = [fit_q[i] for i in rng.choice(len(fit_q), size=min(20000 if not SMOKE else 512, len(fit_q)), replace=False)]
    fit_pooled = pooled_states(fit_q, teacher.QUERY_PREFIX, f"fitq_{len(fit_q)}", max_length=128)
    heads = {d: load_head(d) for d in HEADS}
    fit_vec = {d: apply_head(fit_pooled, heads[d]) for d in HEADS}
    bases = {d: pca_basis(fit_vec[d]) for d in HEADS}
    out["pca_fit_set"] = {"source": "google-research-datasets/nq_open train", "n": len(fit_q), "seed": 0}
    out["explained_variance_cum"] = {str(d): {str(k): float(bases[d][2][:k].sum()) for k in KS if k <= d}
                                     for d in HEADS}
    ceiling = json.loads((REPO / "results" / "m9_dev_symmetric_stella-400M-v5.json").read_text())
    for name in COMPONENTS:
        doc_ids, doc_texts, q_ids, q_texts, qrels = devsuite.load(name)
        verify_manifest(name, doc_ids, doc_texts, q_ids, qrels)
        if SMOKE:
            doc_ids, doc_texts = doc_ids[:SMOKE], doc_texts[:SMOKE]
            q_ids, q_texts = q_ids[:64], q_texts[:64]
        dp = pooled_states(doc_texts, "", f"{name}_docs{'_smoke' if SMOKE else ''}")
        qp = pooled_states(q_texts, teacher.QUERY_PREFIX, f"{name}_queries{'_smoke' if SMOKE else ''}", max_length=512)
        comp = {"n_docs": len(doc_ids), "n_queries": len(q_ids), "spaces": {}}
        ref = ceiling["per_component"].get(name, {})
        comp["box_ceiling_1024"] = ref.get("mean", ref) if isinstance(ref, dict) else ref
        for d in HEADS:
            D = apply_head(dp, heads[d]).astype(np.float16)
            Q = apply_head(qp, heads[d])
            full = evalkit.score(Q, q_ids, D, doc_ids, qrels)
            full_mean = float(np.mean(list(full.values())))
            mu, vt, _ = bases[d]
            rows = {}
            for k in [k for k in KS if k < d]:
                Qk = project(Q, mu, vt, k)
                s = evalkit.score(Qk, q_ids, D, doc_ids, qrels)
                m = float(np.mean(list(s.values())))
                rows[str(k)] = {"ndcg10": m, "retention": m / full_mean,
                                "mean_cos_to_full": float(np.mean((Qk * Q).sum(1)))}
                print(f"  {name} {d}d k={k}: {m:.4f} ({m/full_mean:.3f})", flush=True)
            # oracle basis: fit on the dev queries themselves (best case for ANY rank-k subspace)
            mu_o, vt_o, _ = pca_basis(Q)
            Qo = project(Q, mu_o, vt_o, 384)
            so = evalkit.score(Qo, q_ids, D, doc_ids, qrels)
            comp["spaces"][str(d)] = {"full_ndcg10": full_mean, "rank_k": rows,
                                     "oracle_k384_ndcg10": float(np.mean(list(so.values())))}
            print(f"  {name} {d}d FULL {full_mean:.4f}  oracle384 {np.mean(list(so.values())):.4f}", flush=True)
        out["components"][name] = comp
        OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
