"""M10.0 diagnostic (Mac, MPS): does a wider pooled FEATURE help a frozen bge-small backbone?

Closed-form ridge head from frozen bge-small features to stella-400M query targets (the M9
head probe's design), with three feature sets: mean-pooled last layer (384, M9's head), last two
of layers {12, 8, 4} (768), and all three (1152). Lambda is selected on a training-only holdout
(warmfit.select's rule). Retrieval is scored on the two CQA dev components against the cached
stella document vectors of rank_probe.py. Reads the same 20,000 NQ-open questions as the fit set.
Diagnostic: a frozen backbone is a floor for a trained student, not a forecast; read by no rule.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m7src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m10src"))
os.environ.setdefault("M7_ENCODER", "stella-400M-v5")
import devsuite, evalkit  # noqa: E402
from _paths import DEVICE, REPO, WORK  # noqa: E402
from rank_probe import load_head, apply_head, verify_manifest  # noqa: E402

CACHE = WORK / "m10_rank_probe"
OUT = REPO / "results" / "m10_head_width_probe_mac.json"
LAYERS = (12, 8, 4)
FEATS = {"L12 (384)": (12,), "L12+L8 (768)": (12, 8), "L12+L8+L4 (1152)": (12, 8, 4)}
GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
STUDENT = "BAAI/bge-small-en-v1.5"


@torch.inference_mode()
def layer_pooled(tok, model, texts, batch=128, max_length=512):
    """{layer: (N, 384) masked-mean-pooled hidden states} for the frozen backbone."""
    out = {l: np.empty((len(texts), model.config.hidden_size), dtype=np.float32) for l in LAYERS}
    order = np.argsort([len(t) for t in texts], kind="stable")
    for i in range(0, len(order), batch):
        sel = order[i:i + batch]
        b = tok([texts[j] for j in sel], padding=True, truncation=True, max_length=max_length,
                return_tensors="pt").to(DEVICE)
        hs = model(**b, output_hidden_states=True).hidden_states
        m = b["attention_mask"].unsqueeze(-1).to(hs[0].dtype)
        for l in LAYERS:
            out[l][sel] = ((hs[l] * m).sum(1) / m.sum(1).clamp(min=1e-9)).float().cpu().numpy()
    return out


def solve(Xc, Y, lam):
    G = Xc.T @ Xc
    scale = float(np.trace(G) / G.shape[0])
    return np.linalg.solve(G + lam * scale * np.eye(G.shape[0], dtype=np.float32), Xc.T @ Y)


def objective(Xc, Y, A):
    P = Xc @ A
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def select(Xc, Y, n_fit=16000, seed=0):
    perm = np.random.default_rng(seed).permutation(Xc.shape[0])
    fit, val = perm[:n_fit], perm[n_fit:]
    rows = [{"lambda": lam, "val_objective": objective(Xc[val], Y[val], solve(Xc[fit], Y[fit], lam))}
            for lam in GRID]
    best = min(rows, key=lambda r: (r["val_objective"], -r["lambda"]))
    return best["lambda"], rows


def feats(pooled, layers):
    X = np.hstack([pooled[l] for l in layers])
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])


def main():
    from transformers import AutoModel, AutoTokenizer
    from datasets import load_dataset
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(STUDENT)
    model = AutoModel.from_pretrained(STUDENT, dtype=torch.float32).to(DEVICE).eval()
    fit_q = list(load_dataset("google-research-datasets/nq_open", split="train")["question"])
    rng = np.random.default_rng(0)
    fit_q = [fit_q[i] for i in rng.choice(len(fit_q), size=20000, replace=False)]
    head = load_head(1024)
    Y = apply_head(np.load(CACHE / "fitq_20000.npy"), head)           # stella query targets, unit norm
    Xp = layer_pooled(tok, model, fit_q, max_length=128)
    print(f"fit features in {time.time()-t0:.0f}s", flush=True)
    ref = json.loads((REPO / "results" / "m10_rank_probe_mac.json").read_text())
    comps = {}
    for name in ["cqadup-programmers", "cqadup-physics"]:
        doc_ids, doc_texts, q_ids, q_texts, qrels = devsuite.load(name)
        verify_manifest(name, doc_ids, doc_texts, q_ids, qrels)
        D = apply_head(np.load(CACHE / f"{name}_docs.npy"), head).astype(np.float16)
        comps[name] = (doc_ids, q_ids, qrels, D, layer_pooled(tok, model, q_texts),
                       ref["components"][name]["spaces"]["1024"]["full_ndcg10"])
    out = {"_what": __doc__.strip(), "student": STUDENT, "fit_set": "20,000 nq_open train questions (seed 0)",
           "lambda_rule": "training-only holdout 16,000/4,000, normalized objective, ties to larger lambda",
           "features": {}}
    for fname, layers in FEATS.items():
        Xc = feats(Xp, layers)
        lam, rows = select(Xc, Y)
        A = solve(Xc, Y, lam)
        row = {"dim": int(Xc.shape[1] - 1), "lambda": lam, "grid": rows, "components": {}}
        for name, (doc_ids, q_ids, qrels, D, Qp, full) in comps.items():
            P = feats(Qp, layers) @ A
            P = (P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)).astype(np.float32)
            s = evalkit.score(P, q_ids, D, doc_ids, qrels)
            m = float(np.mean(list(s.values())))
            row["components"][name] = {"ndcg10": m, "retention_of_stella": m / full}
            print(f"{fname:18s} lambda={lam:g} {name}: {m:.4f} ({m/full:.3f} of stella)", flush=True)
        out["features"][fname] = row
    out["m9_reference"] = "M9's head probe: frozen bge-small + ridge head (384) scored 0.3463 on SCREEN-3 = 50.8% of the ceiling (results/m9_head_probe.json, -diag)"
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT, f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
