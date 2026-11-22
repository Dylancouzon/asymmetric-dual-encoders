"""E19: the teacher screen under a second student recipe (m15/MEASUREMENTS.md, E19). Runpod.

    e19_head_screen.py features       bge-small 12/8/4 features for the fit list and dev queries
    e19_head_screen.py dev <name>     lambda grid on the two dev forums for one teacher
    e19_head_screen.py freeze         write work/m15/e19/frozen_lambdas.json from every dev file
    e19_head_screen.py six <name>     re-solve at the frozen lambda, score the six BEIR sets once
    e19_head_screen.py assemble       results/m15_e19_head_screen.json
    e19_head_screen.py all            every step in order, one subprocess per teacher, resumable
    e19_head_screen.py selftest       synthetic check of the solve and scoring path (no data)

Same teachers, fit list, cached targets, document vectors, selection order and scoring as E8/E8x
(`e8_towers.py`); only the student changes: a frozen bge-small backbone with a closed-form ridge
head instead of a token table. `dev`/`six` run one process per teacher with M7_ENCODER set before
any import, because m7src resolves the active tower at import time.
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
import e8_towers as E8          # noqa: E402  fit_list, DEV, SIX, CLEAN4, GRID, atomic_json, sha
import e8x_towers as E8X        # noqa: E402  NEW roster, boot_spearman; rewrites E8.CONFIGS/OUT_DIR
# E8's registered roster, copied verbatim: e8x_towers replaces E8.CONFIGS at import time.
E8_CONFIGS = ("stella-400M-v5", "arctic-embed-l", "arctic-embed-l-mean", "arctic-embed-m-v1.5",
              "bge-base-en-v1.5", "bge-large-en-v1.5", "e5-base-v2", "e5-large-v2",
              "gte-base-en-v1.5", "gte-large-en-v1.5", "mxbai-embed-large-v1")

TEACHERS = E8_CONFIGS + tuple(E8X.NEW)   # 11 + 17; eligibility from the committed E8/E8x results
REGISTERED = tuple(n for n in E8_CONFIGS if n != E8.CONTROL)
BACKBONE = "bge-small-en-v1.5"                            # also a teacher: reported with/without
LAYERS = (12, 8, 4)
OUT_DIR = REPO / "work" / "m15" / "e19"
FEATS = OUT_DIR / "feats"
FROZEN = OUT_DIR / "frozen_lambdas.json"
RESULT = REPO / "results" / "m15_e19_head_screen.json"
E8_RESULT = REPO / "results" / "m15_e8_towers.json"
E8X_RESULT = REPO / "results" / "m15_e8x_towers.json"
FAMILY = {"minilm-l6": "minilm", "minilm-l12": "minilm", "multi-qa-minilm-l6": "minilm",
          "msmarco-minilm-l6": "minilm", "contriever-msmarco": "contriever", "tas-b": "tas-b"}


def family(name):
    return FAMILY.get(name, name.split("-")[0])


def eligible():
    """Teachers with a complete recipe-1 row in E8 or E8x (the pooled 26 plus the E8 control)."""
    rows = {}
    rows.update(json.loads(E8_RESULT.read_text())["configs"])
    rows.update(json.loads(E8X_RESULT.read_text())["new_configs"])
    return [n for n in TEACHERS if n in rows]


# ---------------------------------------------------------------- features (backbone, once)

def backbone_spec():
    os.environ.setdefault("M7_ENCODER", "stella-400M-v5")
    import encoders
    return encoders.get(BACKBONE)


def load_backbone():
    import torch
    from transformers import AutoModel, AutoTokenizer
    spec = backbone_spec()
    tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)
    model = AutoModel.from_pretrained(spec.repo, revision=spec.revision,
                                      dtype=torch.float16).cuda().eval()
    return tok, model


def layer_pooled(tok, model, texts, batch=256, max_length=512):
    """(N, 1153) fp16: masked-mean-pooled hidden states of layers 12/8/4, then a bias column."""
    import torch
    out = np.empty((len(texts), 3 * model.config.hidden_size + 1), dtype=np.float16)
    out[:, -1] = 1.0
    order = np.argsort([len(t) for t in texts], kind="stable")
    with torch.inference_mode():
        for i in range(0, len(order), batch):
            sel = order[i:i + batch]
            b = tok([texts[j] for j in sel], padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt").to("cuda")
            hs = model(**b, output_hidden_states=True).hidden_states
            m = b["attention_mask"].unsqueeze(-1).to(hs[0].dtype)
            cols = [((hs[l] * m).sum(1) / m.sum(1).clamp(min=1e-9)) for l in LAYERS]
            out[sel, :-1] = torch.cat(cols, dim=1).cpu().numpy().astype(np.float16)
    return out


def feat_path(key):
    return FEATS / f"{key}.npy"


def features_for(key, texts, loader=[None]):
    """Cached backbone features for one named text list; computed once, shared by every teacher."""
    p = feat_path(key)
    if p.exists():
        return np.load(p, mmap_mode="r")
    if loader[0] is None:
        loader[0] = load_backbone()
    FEATS.mkdir(parents=True, exist_ok=True)
    f = layer_pooled(*loader[0], texts)
    np.save(p.with_suffix(".tmp.npy"), f)
    p.with_suffix(".tmp.npy").replace(p)
    return np.load(p, mmap_mode="r")


def features():
    import devsuite
    texts = E8.fit_list()
    spec = backbone_spec()
    t0 = time.time()
    features_for(f"fit-{len(texts)}", texts)
    for c in E8.DEV:
        features_for(f"dev-{c}-q", devsuite.load(c)[3])
    E8.atomic_json(OUT_DIR / "features.json", {
        "backbone": {"repo": spec.repo, "revision": spec.revision, "layers": LAYERS,
                     "pooling": "masked mean per layer, concatenated, bias column"},
        "fit_list_sha256": E8.FIT_SHA, "n_fit": len(texts),
        "fit_features_sha256": E8.sha(feat_path(f"fit-{len(texts)}")),
        "seconds": round(time.time() - t0, 1)})


# ---------------------------------------------------------------- ridge head

class Head:
    """Closed-form ridge from frozen features to one teacher's cached query targets."""

    def __init__(self, name):
        if os.environ.get("M7_ENCODER") != name:
            raise SystemExit(f"M7_ENCODER must be {name!r} before import")
        import m8base  # noqa: F401  installs the protected-path guard
        import encoders
        import torch
        from teacher import QUERY_PREFIX, encode_cached
        self.torch, self.encode_cached, self.qprefix = torch, encode_cached, QUERY_PREFIX
        self.spec = encoders.active()
        assert self.spec.name == name
        texts = E8.fit_list()
        X = features_for(f"fit-{len(texts)}", texts)
        # Targets are E8's cached fp16 fit-query vectors under the teacher's own query prefix.
        Y = encode_cached(f"trainq-{len(texts)}", texts, prefix=QUERY_PREFIX,
                          dtype=torch.float16, verbose=True)
        self.G, self.XtY, self.scale = gram_and_cross(X, Y)

    def solve(self, lam):
        return ridge_solve(self.G, self.XtY, self.scale, lam)

    def docs(self, key, texts):
        return np.asarray(self.encode_cached(key, texts, prefix=self.spec.doc_prefix,
                                             dtype=self.torch.float16, verbose=True))

    @staticmethod
    def score(qv, q_ids, dv, doc_ids, qrels):
        from evalkit import score
        pq = score(np.asarray(qv, dtype=np.float32), q_ids, dv, doc_ids, qrels, k=100)
        return {q: float(pq.get(q, 0.0)) for q in q_ids}


def gram_and_cross(X, Y, chunk=32768):
    """G = X'X and X'Y in fp32 on the GPU, streamed over rows; scale = trace(G)/d."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d, k = X.shape[1], Y.shape[1]
    G = torch.zeros((d, d), dtype=torch.float32, device=dev)
    XtY = torch.zeros((d, k), dtype=torch.float32, device=dev)
    for lo in range(0, X.shape[0], chunk):
        x = torch.as_tensor(np.asarray(X[lo:lo + chunk], dtype=np.float32), device=dev)
        y = torch.as_tensor(np.asarray(Y[lo:lo + chunk], dtype=np.float32), device=dev)
        G += x.T @ x
        XtY += x.T @ y
    return G, XtY, float(torch.trace(G) / d)


def ridge_solve(G, XtY, scale, lam):
    import torch
    d = G.shape[0]
    A = torch.linalg.solve(G + lam * scale * torch.eye(d, device=G.device), XtY)
    return A.cpu().numpy().astype(np.float32)


def apply(A, F):
    P = np.asarray(F, dtype=np.float32) @ A
    return P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)


def _mean(d):
    return float(np.mean(list(d.values())))


# ---------------------------------------------------------------- per-teacher steps

def dev(name):
    out_path = OUT_DIR / f"dev-{name}.json"
    if out_path.exists():
        print(f"{name}: dev already done", flush=True)
        return
    if FROZEN.exists():
        raise SystemExit("E19 STOP: lambdas are frozen; dev files are never rewritten")
    if name not in eligible():
        raise SystemExit(f"E19 STOP: {name} has no recipe-1 row")
    import devsuite
    h = Head(name)
    comps = {c: devsuite.load(c) for c in E8.DEV}
    dvecs = {c: h.docs(f"e8-dev-{c}-docs", comps[c][1]) for c in E8.DEV}
    qfeat = {c: features_for(f"dev-{c}-q", comps[c][3]) for c in E8.DEV}
    lambdas, grid = {}, list(E8.GRID)

    def run(lam):
        A = h.solve(lam)
        per = {c: h.score(apply(A, qfeat[c]), comps[c][2], dvecs[c], comps[c][0], comps[c][4])
               for c in E8.DEV}
        lambdas[repr(lam)] = {"dev_macro_2": float(sum(_mean(per[c]) for c in E8.DEV) / 2),
                              "per_forum": {c: _mean(per[c]) for c in E8.DEV}}
        print(f"  {name} lambda={lam:g}: {lambdas[repr(lam)]['dev_macro_2']:.4f}", flush=True)

    extended = None
    for lam in grid:
        run(lam)
    best = max(lambdas, key=lambda k: lambdas[k]["dev_macro_2"])
    if best in (repr(E8.GRID[0]), repr(E8.GRID[-1])):   # one decade further, once, as E8
        extended = E8.GRID[0] / 10 if best == repr(E8.GRID[0]) else E8.GRID[-1] * 10
        run(extended)
        best = max(lambdas, key=lambda k: lambdas[k]["dev_macro_2"])
    E8.atomic_json(out_path, {
        "name": name, "status": "COMPLETE", "repo": h.spec.repo, "revision": h.spec.revision,
        "dim": h.spec.dim, "pooling": h.spec.pooling, "doc_prefix": h.spec.doc_prefix,
        "query_prefix": h.qprefix, "lambdas": lambdas, "extended_with": extended,
        "best_lambda": float(best), "best_dev_macro_2": lambdas[best]["dev_macro_2"],
        "best_at_grid_edge_after_extension": float(best) in (min(map(float, lambdas)),
                                                             max(map(float, lambdas))),
        "ridge_scale": h.scale})


def freeze():
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists; lambdas are frozen once")
    blob = {}
    for name in eligible():
        d = json.loads((OUT_DIR / f"dev-{name}.json").read_text())
        blob[name] = {"lambda": d["best_lambda"], "dev_macro_2": d["best_dev_macro_2"]}
    FROZEN.write_text(json.dumps({"frozen_utc": E8.time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                 E8.time.gmtime()),
                                  "configs": blob}, indent=2))
    print(f"FROZEN lambdas sha256 {E8.sha(FROZEN)}", flush=True)


def six(name):
    from common import load_public
    out_path = OUT_DIR / f"six-{name}.json"
    if out_path.exists():
        print(f"{name}: six already done", flush=True)
        return
    if not FROZEN.exists():
        raise SystemExit("E19 STOP: lambdas are not frozen")
    frozen = json.loads(FROZEN.read_text())
    lam = frozen["configs"][name]["lambda"]
    h = Head(name)
    A = h.solve(lam)
    partial = OUT_DIR / f"six-{name}.partial.json"
    done = json.loads(partial.read_text()) if partial.exists() else {}
    if done and done["frozen_sha256"] != E8.sha(FROZEN):
        raise SystemExit(f"E19 STOP: {partial} was written under a different freeze")
    table, pins = done.get("table", {}), done.get("pins", {})
    for ds in E8.SIX:
        if ds in table:                   # scored once; a resume never re-scores a set
            continue
        data = load_public(ds)            # M20 pins, appends m7/SIX_ACCESS.log
        qf = features_for(f"six-{ds}-q", data["q_texts"])   # first six step computes, later reuse
        dv = h.docs(f"e8-six-{ds}-docs", data["doc_texts"])
        table[ds] = h.score(apply(A, qf), data["q_ids"], dv, data["doc_ids"], data["qrels"])
        pins[ds] = data["pins"]
        E8.atomic_json(partial, {"table": table, "pins": pins, "frozen_sha256": E8.sha(FROZEN)})
        print(f"  {name} {ds}: head {_mean(table[ds]):.4f}", flush=True)
        del dv
    E8.atomic_json(out_path, {
        "name": name, "status": "COMPLETE", "lambda": lam, "pins": pins,
        "frozen_sha256": E8.sha(FROZEN), "table": {ds: _mean(v) for ds, v in table.items()},
        "per_query": table})


# ---------------------------------------------------------------- assemble

def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0])


def assemble():
    from common import receipt, write_result
    r1 = {}
    r1.update(json.loads(E8_RESULT.read_text())["configs"])
    r1.update(json.loads(E8X_RESULT.read_text())["new_configs"])
    frozen = json.loads(FROZEN.read_text())
    feats = json.loads((OUT_DIR / "features.json").read_text())
    mac = lambda v, keys: float(np.mean([v[k] for k in keys]))
    rows = {}
    for name in eligible():
        d = json.loads((OUT_DIR / f"dev-{name}.json").read_text())
        s = json.loads((OUT_DIR / f"six-{name}.json").read_text())
        if not (frozen["configs"][name]["lambda"] == d["best_lambda"] == s["lambda"] and
                s["frozen_sha256"] == E8.sha(FROZEN)):
            raise SystemExit(f"E19 STOP: {name} dev, frozen and six files disagree")
        rows[name] = {
            "family": family(name), "repo": d["repo"], "revision": d["revision"], "dim": d["dim"],
            "lambda": s["lambda"], "lambda_curve": d["lambdas"], "extended_with": d["extended_with"],
            "dev_head": d["best_dev_macro_2"], "six_head": s["table"],
            "six_head_macro_all6": mac(s["table"], E8.SIX),
            "six_head_macro_clean4": mac(s["table"], E8.CLEAN4),
            "recipe1_dev_table": r1[name]["dev_table"],
            "recipe1_six_table_macro_all6": r1[name]["six_table_macro_all6"],
            "teacher_six_macro_all6": r1[name]["six_ceiling_macro_all6"],
            "recipe1_six_table_macro_clean4": mac(r1[name]["six_table"], E8.CLEAN4),
            "teacher_six_macro_clean4": mac(r1[name]["six_ceiling"], E8.CLEAN4),
            "pins": s["pins"]}
        rows[name]["retention_all6"] = rows[name]["six_head_macro_all6"] / rows[name]["teacher_six_macro_all6"]

    def stats(names):
        g = lambda k: [rows[n][k] for n in names]
        out = {"n": len(names), "names": list(names)}
        for label, x, y in (
                ("recipe2_vs_recipe1_all6", "six_head_macro_all6", "recipe1_six_table_macro_all6"),
                ("recipe2_vs_teacher_all6", "six_head_macro_all6", "teacher_six_macro_all6"),
                ("recipe2_dev_vs_six_all6", "dev_head", "six_head_macro_all6"),
                ("recipe2_vs_recipe1_clean4", "six_head_macro_clean4", "recipe1_six_table_macro_clean4"),
                ("recipe2_vs_teacher_clean4", "six_head_macro_clean4", "teacher_six_macro_clean4"),
                ("recipe2_dev_vs_six_clean4", "dev_head", "six_head_macro_clean4")):
            out[label] = {"spearman": spearman(g(x), g(y)),
                          "bootstrap95_over_checkpoints": E8X.boot_spearman(g(x), g(y))}
        fams = sorted({rows[n]["family"] for n in names})
        pairs = (("recipe2_vs_recipe1_all6", "six_head_macro_all6", "recipe1_six_table_macro_all6"),
                 ("recipe2_vs_teacher_all6", "six_head_macro_all6", "teacher_six_macro_all6"),
                 ("recipe2_vs_recipe1_clean4", "six_head_macro_clean4", "recipe1_six_table_macro_clean4"),
                 ("recipe2_vs_teacher_clean4", "six_head_macro_clean4", "teacher_six_macro_clean4"))
        out["leave_one_family_out"] = {
            f: {label: spearman([rows[n][x] for n in names if rows[n]["family"] != f],
                                [rows[n][y] for n in names if rows[n]["family"] != f])
                for label, x, y in pairs}
            for f in fams if sum(rows[n]["family"] != f for n in names) > 3}
        return out

    pooled = [n for n in rows if n != E8.CONTROL]
    registered = [n for n in REGISTERED if n in rows]
    rosters = {"registered_ten": registered, "pooled": pooled,
               "registered_ten_without_backbone": [n for n in registered if n != BACKBONE],
               "pooled_without_backbone": [n for n in pooled if n != BACKBONE]}
    strongest = max(registered, key=lambda n: rows[n]["teacher_six_macro_all6"])
    choice = max(registered, key=lambda n: rows[n]["dev_head"])
    regret = {"strongest_registered_teacher": strongest, "recipe2_screen_choice": choice,
              "recipe2_six_strongest": rows[strongest]["six_head_macro_all6"],
              "recipe2_six_choice": rows[choice]["six_head_macro_all6"],
              "recipe1_screen_choice": max(registered, key=lambda n: rows[n]["recipe1_dev_table"])}
    write_result(RESULT, {
        "status": "COMPLETE", "measurement": "E19 (exploratory, pre-specified before scoring)",
        "scope": "second student recipe over the E8/E8x teachers; descriptive, no p-values; "
                 "frozen-backbone head is a floor for a trained student, not Nano",
        "backbone": feats["backbone"], "features": feats,
        "rosters": {k: {"n": len(v), "names": v} for k, v in rosters.items()},
        "spearman": {k: stats(v) for k, v in rosters.items()},
        "strongest_teacher_comparison": regret, "configs": rows, "frozen_lambdas": frozen,
        "receipt": receipt(__file__, [{"fit_list_sha256": E8.FIT_SHA},
                                      {"frozen_lambdas": str(FROZEN.relative_to(REPO)),
                                       "sha256": E8.sha(FROZEN)},
                                      {"recipe1": [str(E8_RESULT.relative_to(REPO)),
                                                   str(E8X_RESULT.relative_to(REPO))]}],
                           feats.get("started_utc", frozen["frozen_utc"]),
                           ("torch", "transformers", "scipy"))})


def all_steps():
    py = sys.executable
    env = {**os.environ, "PYTHONPATH": f"{REPO / 'm15src'}:{REPO / 'm8src'}:{REPO / 'm7src'}"}
    subprocess.run([py, __file__, "features"], check=True, env=env, cwd=REPO)
    for name in eligible():
        subprocess.run([py, __file__, "dev", name], check=True, cwd=REPO,
                       env={**env, "M7_ENCODER": name})
    if not FROZEN.exists():
        subprocess.run([py, __file__, "freeze"], check=True, env=env, cwd=REPO)
    for name in eligible():
        subprocess.run([py, __file__, "six", name], check=True, cwd=REPO,
                       env={**env, "M7_ENCODER": name})
    subprocess.run([py, __file__, "assemble"], check=True, env=env, cwd=REPO)


def selftest():
    """Synthetic: a head fitted to Y = normalize(X A*) recovers A* up to scale; scoring runs."""
    rng = np.random.default_rng(0)
    n, d, k = 5000, 40, 16
    X = rng.normal(size=(n, d)).astype(np.float16)
    X[:, -1] = 1.0
    A_true = rng.normal(size=(d, k)).astype(np.float32)
    Y = apply(A_true, X).astype(np.float16)
    G, XtY, scale = gram_and_cross(X, Y, chunk=1024)
    A = ridge_solve(G, XtY, scale, 1e-6)
    P = apply(A, X[:200])
    cos = float(np.mean(np.sum(P * apply(A_true, X[:200]), axis=1)))
    assert cos > 0.99, cos
    # a heavy penalty must move the map away from the truth, and the scoring path must run
    heavy = apply(ridge_solve(G, XtY, scale, 1e3), X[:200])
    cos_heavy = float(np.mean(np.sum(heavy * apply(A_true, X[:200]), axis=1)))
    assert cos_heavy < cos, (cos_heavy, cos)
    from evalkit import score
    doc_ids = [f"d{i}" for i in range(200)]
    qrels = {f"q{i}": {f"d{i}": 1} for i in range(200)}
    nd = np.mean(list(score(P, list(qrels), np.asarray(Y[:200], np.float32), doc_ids, qrels).values()))
    assert nd > 0.99, nd
    print(f"selftest ok: cosine to true map {cos:.4f} (heavy ridge {cos_heavy:.4f}), nDCG {nd:.3f}")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["all"]
    {"features": features, "freeze": freeze, "assemble": assemble, "all": all_steps,
     "selftest": selftest, "dev": lambda: dev(rest[0]), "six": lambda: six(rest[0])}[cmd]()
