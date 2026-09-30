"""E11 (exploratory): why do some towers distill into a table and others not? Runpod, after E8.

Per configuration, on the test queries of the six E8 sets and the two dev forums:
  additivity   mean cosine between the closed-form table's query vector (frozen E8 lambda) and
               the tower's own query vector: how much of the tower is a bag of tokens;
  order_cos    mean cosine between the tower's vector for a query and for the same words shuffled
               (seed 20260930): how much the tower reads word order.
Then Spearman of each against the E8 six-set table score and retention over the 10 checkpoints.

    e11_mechanism.py tower <name>    one configuration, in its own process (M7_ENCODER set)
    e11_mechanism.py all             every configuration, then results/m15_e11_mechanism.json
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in ("m15src", "m8src", "m7src"):
    if str(REPO / p) not in sys.path:
        sys.path.insert(0, str(REPO / p))

import e8_towers as E8

SETS = E8.SIX + E8.DEV
OUT_DIR = E8.OUT_DIR
RESULT = REPO / "results" / "m15_e11_mechanism.json"


def shuffled(texts, seed=20260930):
    rng = np.random.default_rng(seed)
    out = []
    for t in texts:
        w = t.split()
        out.append(" ".join(w[i] for i in rng.permutation(len(w))) if len(w) > 1 else t)
    return out


def tower(name):
    from common import load_public
    out_path = OUT_DIR / f"e11-{name}.json"
    if out_path.exists():
        return
    lam = json.loads(E8.FROZEN.read_text())["configs"][name]["lambda"]
    t = E8.Tower(name)
    W, diag = t.solve(lam, f"{name} e11 re-solve")
    rows = {}
    for ds in SETS:
        texts = load_public(ds, with_corpus=False)["q_texts"]
        tab = np.asarray(t.table_queries(W, texts), dtype=np.float32)
        own = np.asarray(t.ceiling_queries(f"e8-six-{ds}-q" if ds in E8.SIX else f"e8-dev-{ds}-q",
                                           texts), dtype=np.float32)
        shuf = np.asarray(t.ceiling_queries(f"e11-{ds}-q-shuffled", shuffled(texts)),
                          dtype=np.float32)
        norm = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
        tab, own, shuf = norm(tab), norm(own), norm(shuf)
        multi = np.array([len(x.split()) > 1 for x in texts])
        rows[ds] = {"additivity": float((tab * own).sum(1).mean()),
                    "order_cos": float((own * shuf).sum(1)[multi].mean()),
                    "n_queries": len(texts), "n_multiword": int(multi.sum())}
        print(f"  {name} {ds}: additivity {rows[ds]['additivity']:.4f} "
              f"order_cos {rows[ds]['order_cos']:.4f}", flush=True)
    E8.atomic_json(out_path, {"name": name, "lambda": lam, "solve": diag, "sets": rows})


def assemble():
    from common import receipt, utc_now, write_result
    from scipy.stats import spearmanr
    e8 = json.loads((REPO / "results" / "m15_e8_towers.json").read_text())["configs"]
    rows = {}
    for name in e8:
        d = json.loads((OUT_DIR / f"e11-{name}.json").read_text())["sets"]
        rows[name] = {
            "additivity_six": float(np.mean([d[s]["additivity"] for s in E8.SIX])),
            "additivity_dev": float(np.mean([d[s]["additivity"] for s in E8.DEV])),
            "order_cos_six": float(np.mean([d[s]["order_cos"] for s in E8.SIX])),
            "six_table": e8[name]["six_table_macro_all6"],
            "retention_all6": e8[name]["retention_all6"], "per_set": d}
    ckpt = [n for n in rows if n != E8.CONTROL]
    rho = {}
    for x in ("additivity_six", "additivity_dev", "order_cos_six"):
        for y in ("six_table", "retention_all6"):
            rho[f"{x}_vs_{y}"] = float(spearmanr([rows[n][x] for n in ckpt],
                                                 [rows[n][y] for n in ckpt])[0])
    write_result(RESULT, {"status": "COMPLETE", "measurement": "E11 (exploratory)",
                          "scope": "exploratory mechanism probe; n = 10 checkpoints; no p-values",
                          "checkpoints": ckpt, "spearman": rho, "configs": rows,
                          "receipt": receipt(__file__, [], utc_now(), ("torch", "scipy"))})


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["all"]
    if cmd == "tower":
        tower(rest[0])
    elif cmd == "assemble":
        assemble()
    else:
        env = {**os.environ, "PYTHONPATH": f"{REPO / 'm15src'}:{REPO / 'm8src'}:{REPO / 'm7src'}"}
        for name in E8.CONFIGS:
            subprocess.run([sys.executable, __file__, "tower", name], check=True, cwd=REPO,
                           env={**env, "M7_ENCODER": name})
        assemble()
