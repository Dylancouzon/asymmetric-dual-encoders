"""E8: tower generality under the closed-form table recipe (m15/MEASUREMENTS.md, E8). Runpod.

    e8_towers.py preflight        fit-list hash, GPU, tokenizer check for all 11 configurations
    e8_towers.py dev <name>       lambda grid on the two dev forums + dev ceiling (one tower)
    e8_towers.py freeze           write work/m15/e8/frozen_lambdas.json from every dev file
    e8_towers.py six <name>       re-solve at the frozen lambda, score the six BEIR sets once
    e8_towers.py assemble         results/m15_e8_towers.json
    e8_towers.py all              every step in order, one subprocess per tower, resumable

Every `dev`/`six` step runs in its own process with M7_ENCODER set before any import, because
m7src resolves the active tower at import time.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m15src", "m8src"):
    if str(REPO / p) not in sys.path:
        sys.path.insert(0, str(REPO / p))

CONFIGS = ("stella-400M-v5", "arctic-embed-l", "arctic-embed-l-mean", "arctic-embed-m-v1.5",
           "bge-base-en-v1.5", "bge-large-en-v1.5", "e5-base-v2", "e5-large-v2",
           "gte-base-en-v1.5", "gte-large-en-v1.5", "mxbai-embed-large-v1")
CONTROL = "arctic-embed-l-mean"          # same checkpoint as arctic-embed-l, off-spec readout
DEV = ("cqadup-programmers", "cqadup-physics")
SIX = ("scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid")
CLEAN4 = ("nfcorpus", "scidocs", "scifact", "trec-covid")
GRID = (1e-4, 1e-3, 1e-2, 1e-1)
FIT_SHA = "da0f208ef29bae3833ffead070eb34b61f390f9d72ed96d448d45ae6b52072c2"
FIT_N = 337_981
OUT_DIR = REPO / "work" / "m15" / "e8"
FROZEN = OUT_DIR / "frozen_lambdas.json"
RESULT = REPO / "results" / "m15_e8_towers.json"
EXPOSURE = REPO / "m15" / "e8_exposure.json"


def sha(path):
    from common import sha_file
    return sha_file(path)


def fit_list():
    path = REPO / "work" / "m8_trainq_texts.json"
    if sha(path) != FIT_SHA:
        raise SystemExit(f"E8 STOP: {path} does not have the registered sha256")
    texts = json.loads(path.read_text())
    if len(texts) != FIT_N:
        raise SystemExit(f"E8 STOP: fit list has {len(texts)} queries, expected {FIT_N}")
    return texts


# ---------------------------------------------------------------- preflight

def preflight():
    import torch
    from transformers import AutoTokenizer
    fit_list()
    if not torch.cuda.is_available():
        raise SystemExit("E8 STOP: no CUDA device")
    os.environ.setdefault("M7_ENCODER", "stella-400M-v5")
    import m8base  # noqa: F401  installs the protected-path guard
    import encoders
    import hashlib
    report = {"gpu": torch.cuda.get_device_name(0), "tokenizers": {}}
    for name in CONFIGS:
        spec = encoders.get(name)
        kw = {"trust_remote_code": True} if spec.trust_remote_code else {}
        tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision, **kw)
        vocab = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])
        norm = json.loads(tok.backend_tokenizer.to_str()).get("normalizer") or {}
        row = {"vocab_size": len(tok), "cls_id": tok.cls_token_id,
               "ordered_vocab_sha256": hashlib.sha256(
                   "\n".join(t for t, _ in vocab).encode()).hexdigest(),
               "lowercase": norm.get("lowercase"), "strip_accents": norm.get("strip_accents"),
               "normalizer_type": norm.get("type")}
        row["compatible"] = row["vocab_size"] == 30522 and row["cls_id"] == 101
        report["tokenizers"][name] = row
        print(name, row, flush=True)
    hashes = {r["ordered_vocab_sha256"] for r in report["tokenizers"].values()}
    report["distinct_vocab_hashes"] = len(hashes)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "preflight.json").write_text(json.dumps(report, indent=2))


# ---------------------------------------------------------------- one tower

def _gate(info, where):
    """Convergence gate (MEASUREMENTS.md, E8): a solve counts only if block CG converged."""
    ok = bool(info["converged"]) and info["worst_rel_residual"] <= info["tol"]
    if not ok:
        raise SystemExit(f"E8 STOP {where}: block CG did not converge "
                         f"(worst residual {info['worst_rel_residual']:.3g}, tol {info['tol']})")
    return {k: info[k] for k in ("iterations", "worst_rel_residual", "converged", "tol",
                                 "seconds", "preconditioner") if k in info}


class Tower:
    """Everything that depends on the active tower; built after M7_ENCODER is set."""

    def __init__(self, name):
        if os.environ.get("M7_ENCODER") != name:
            raise SystemExit(f"M7_ENCODER must be {name!r} before import")
        import m8base
        import encoders
        import init_m8
        import stage0_ridge as sr
        import torch
        from table import Preproc, get_tokenizer
        from teacher import QUERY_PREFIX, encode_cached
        self.m8base, self.torch = m8base, torch
        self.spec = encoders.active()
        assert self.spec.name == name
        self.device = m8base.device()
        self.pre, self.tok = Preproc(), get_tokenizer()
        self.qprefix, self.encode_cached = QUERY_PREFIX, encode_cached
        texts = fit_list()
        self.X = sr.bag_matrix(self.tok, texts, self.pre, self.spec.vocab)
        # fp16 compute and storage, as the M7/M8 screen did for every tower.
        self.Y = self.torch_free(encode_cached(f"trainq-{len(texts)}", texts, prefix=QUERY_PREFIX,
                                               dtype=torch.float16, verbose=True))
        self.W0 = init_m8.get_init("teacher", self.pre)
        assert self.W0.shape[0] == self.spec.vocab

    @staticmethod
    def torch_free(a):
        import numpy as np
        return np.asarray(a, dtype=np.float32)

    def solve(self, lam, where):
        import blockcg
        W, info = blockcg.block_cg_ridge(self.X, self.Y, self.W0, lam, device=self.device)
        return W, _gate(info, where)

    def docs(self, key, texts):
        import numpy as np
        # The tower's registered document path: its own doc prefix (MEASUREMENTS.md, E8).
        return np.asarray(self.encode_cached(key, texts, prefix=self.spec.doc_prefix,
                                             dtype=self.torch.float16, verbose=True))

    def score(self, qv, q_ids, dv, doc_ids, qrels):
        import numpy as np
        from evalkit import score
        pq = score(np.asarray(qv, dtype=np.float32), q_ids, dv, doc_ids, qrels, k=100)
        return {q: float(pq.get(q, 0.0)) for q in q_ids}

    def table_queries(self, W, texts):
        from table import QueryTable
        model = QueryTable(W, weight_init=None, learned_weights=False,
                           fallback_id=self.spec.cls_id).to(self.device)
        out = model.encode(texts, self.pre, tok=self.tok)
        del model
        return out

    def ceiling_queries(self, key, texts):
        return self.encode_cached(key, texts, prefix=self.qprefix,
                                  dtype=self.torch.float16, verbose=False)


def _mean(d):
    import numpy as np
    return float(np.mean(list(d.values())))


def dev(name):
    import devsuite
    out_path = OUT_DIR / f"dev-{name}.json"
    if out_path.exists():
        print(f"{name}: dev already done", flush=True)
        return
    t = Tower(name)
    comps = {c: devsuite.load(c) for c in DEV}
    dvecs = {c: t.docs(f"e8-dev-{c}-docs", comps[c][1]) for c in DEV}
    lambdas, grid = {}, list(GRID)

    def run(lam):
        W, diag = t.solve(lam, f"{name} lambda={lam:g}")
        per = {c: t.score(t.table_queries(W, comps[c][3]), comps[c][2], dvecs[c], comps[c][0],
                          comps[c][4]) for c in DEV}
        lambdas[repr(lam)] = {"dev_macro_2": float(sum(_mean(per[c]) for c in DEV) / 2),
                              "per_forum": {c: _mean(per[c]) for c in DEV}, "solve": diag}
        print(f"  {name} lambda={lam:g}: {lambdas[repr(lam)]['dev_macro_2']:.4f}", flush=True)
        t.m8base.empty_cache()

    for lam in grid:
        run(lam)
    best = max(lambdas, key=lambda k: lambdas[k]["dev_macro_2"])
    extended = None
    if best in (repr(GRID[0]), repr(GRID[-1])):   # one decade further, once
        extended = GRID[0] / 10 if best == repr(GRID[0]) else GRID[-1] * 10
        run(extended)
        best = max(lambdas, key=lambda k: lambdas[k]["dev_macro_2"])
    ceiling = {c: t.score(t.ceiling_queries(f"e8-dev-{c}-q", comps[c][3]), comps[c][2], dvecs[c],
                          comps[c][0], comps[c][4]) for c in DEV}
    blob = {"name": name, "repo": t.spec.repo, "revision": t.spec.revision, "dim": t.spec.dim,
            "pooling": t.spec.pooling, "doc_prefix": t.spec.doc_prefix,
            "query_prefix": t.qprefix, "lambdas": lambdas, "extended_with": extended,
            "best_lambda": float(best), "best_dev_macro_2": lambdas[best]["dev_macro_2"],
            "best_at_grid_edge_after_extension": float(best) in (min(map(float, lambdas)),
                                                                 max(map(float, lambdas))),
            "dev_ceiling_macro_2": float(sum(_mean(ceiling[c]) for c in DEV) / 2),
            "dev_ceiling_per_forum": {c: _mean(ceiling[c]) for c in DEV}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(blob, indent=2))


def freeze():
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists; lambdas are frozen once")
    blob = {}
    for name in CONFIGS:
        d = json.loads((OUT_DIR / f"dev-{name}.json").read_text())
        blob[name] = {"lambda": d["best_lambda"], "dev_macro_2": d["best_dev_macro_2"],
                      "dev_ceiling_macro_2": d["dev_ceiling_macro_2"]}
    FROZEN.write_text(json.dumps({"frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                             time.gmtime()),
                                  "configs": blob}, indent=2))
    print(f"FROZEN lambdas sha256 {sha(FROZEN)}", flush=True)


def six(name):
    from common import load_public
    out_path = OUT_DIR / f"six-{name}.json"
    if out_path.exists():
        print(f"{name}: six already done", flush=True)
        return
    if not FROZEN.exists():
        raise SystemExit("E8 STOP: lambdas are not frozen")
    lam = json.loads(FROZEN.read_text())["configs"][name]["lambda"]
    t = Tower(name)
    W, diag = t.solve(lam, f"{name} final re-solve")
    table, ceiling, pins = {}, {}, {}
    for ds in SIX:
        data = load_public(ds)            # M20 pins, appends m7/SIX_ACCESS.log
        dv = t.docs(f"e8-six-{ds}-docs", data["doc_texts"])
        table[ds] = t.score(t.table_queries(W, data["q_texts"]), data["q_ids"], dv,
                            data["doc_ids"], data["qrels"])
        ceiling[ds] = t.score(t.ceiling_queries(f"e8-six-{ds}-q", data["q_texts"]),
                              data["q_ids"], dv, data["doc_ids"], data["qrels"])
        pins[ds] = data["pins"]
        print(f"  {name} {ds}: table {_mean(table[ds]):.4f} ceiling {_mean(ceiling[ds]):.4f}",
              flush=True)
        del dv
        t.m8base.empty_cache()
    out_path.write_text(json.dumps({
        "name": name, "lambda": lam, "solve": diag, "pins": pins,
        "table": {ds: _mean(v) for ds, v in table.items()},
        "ceiling": {ds: _mean(v) for ds, v in ceiling.items()},
        "per_query": {"table": table, "ceiling": ceiling}}, indent=2))


# ---------------------------------------------------------------- assemble

def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def assemble():
    from common import receipt, utc_now, write_result
    import numpy as np
    started = utc_now()
    frozen = json.loads(FROZEN.read_text())
    rows = {}
    for name in CONFIGS:
        d = json.loads((OUT_DIR / f"dev-{name}.json").read_text())
        s = json.loads((OUT_DIR / f"six-{name}.json").read_text())
        mac = lambda v, keys: float(np.mean([v[k] for k in keys]))
        rows[name] = {
            "lambda": s["lambda"], "dev_table": d["best_dev_macro_2"],
            "dev_ceiling": d["dev_ceiling_macro_2"], "lambda_curve": d["lambdas"],
            "extended_with": d["extended_with"],
            "best_at_grid_edge_after_extension": d["best_at_grid_edge_after_extension"],
            "six_table": s["table"], "six_ceiling": s["ceiling"],
            "six_table_macro_all6": mac(s["table"], SIX),
            "six_ceiling_macro_all6": mac(s["ceiling"], SIX),
            "six_table_macro_clean4": mac(s["table"], CLEAN4),
            "six_ceiling_macro_clean4": mac(s["ceiling"], CLEAN4),
            "final_solve": s["solve"], "pins": s["pins"]}
        r = rows[name]
        r["retention_all6"] = r["six_table_macro_all6"] / r["six_ceiling_macro_all6"]
        r["retention_clean4"] = r["six_table_macro_clean4"] / r["six_ceiling_macro_clean4"]
        r["retention_dev"] = r["dev_table"] / r["dev_ceiling"]
    rosters = {"checkpoints_10": [n for n in CONFIGS if n != CONTROL],
               "configurations_11": list(CONFIGS)}
    rho = {}
    for roster, names in rosters.items():
        g = lambda k: [rows[n][k] for n in names]
        rho[roster] = {
            "primary_dev_table_vs_six_table_all6": spearman(g("dev_table"), g("six_table_macro_all6")),
            "dev_table_vs_six_table_clean4": spearman(g("dev_table"), g("six_table_macro_clean4")),
            "six_ceiling_vs_six_table_all6": spearman(g("six_ceiling_macro_all6"),
                                                      g("six_table_macro_all6")),
            "six_ceiling_vs_six_table_clean4": spearman(g("six_ceiling_macro_clean4"),
                                                        g("six_table_macro_clean4")),
            "dev_ceiling_vs_dev_table": spearman(g("dev_ceiling"), g("dev_table"))}
    inputs = [{"fit_list_sha256": FIT_SHA}, {"frozen_lambdas": str(FROZEN.relative_to(REPO)),
                                             "sha256": sha(FROZEN)},
              {"preflight": json.loads((OUT_DIR / "preflight.json").read_text())}]
    write_result(RESULT, {
        "status": "COMPLETE", "measurement": "E8",
        "scope": "descriptive; one closed-form recipe; n = 10 checkpoints, no p-values",
        "rosters": rosters, "spearman": rho, "configs": rows,
        "exposure": json.loads(EXPOSURE.read_text()) if EXPOSURE.exists() else None,
        "receipt": receipt(__file__, inputs, started, ("torch", "transformers", "scipy"))})


def all_steps():
    py = sys.executable
    env = {**os.environ, "PYTHONPATH": f"{REPO / 'm15src'}:{REPO / 'm8src'}:{REPO / 'm7src'}"}
    subprocess.run([py, __file__, "preflight"], check=True, env=env, cwd=REPO)
    for name in CONFIGS:
        subprocess.run([py, __file__, "dev", name], check=True, cwd=REPO,
                       env={**env, "M7_ENCODER": name})
    if not FROZEN.exists():
        subprocess.run([py, __file__, "freeze"], check=True, env=env, cwd=REPO)
    for name in CONFIGS:
        subprocess.run([py, __file__, "six", name], check=True, cwd=REPO,
                       env={**env, "M7_ENCODER": name})
    subprocess.run([py, __file__, "assemble"], check=True, env=env, cwd=REPO)


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["all"]
    {"preflight": preflight, "freeze": freeze, "assemble": assemble, "all": all_steps,
     "dev": lambda: dev(rest[0]), "six": lambda: six(rest[0])}[cmd]()
