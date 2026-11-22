"""E8x (exploratory, 2026-09-30): E8's recipe on 17 more BERT-WordPiece towers (Runpod).

Same fit list, recipe, convergence gate, dev-then-freeze-then-six order and document paths as E8
(`m15src/e8_towers.py`, whose functions this reuses with a separate output directory). The
assembly pools E8's 10 checkpoints with these and adds 10,000-draw tower bootstrap intervals for
both Spearman correlations. Exploratory: added after E8's result was known.

    e8x_towers.py all  |  preflight  |  dev <name>  |  freeze  |  six <name>  |  assemble
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
import e8_towers as E8

NEW = ("minilm-l6", "minilm-l12", "multi-qa-minilm-l6", "msmarco-minilm-l6", "bge-small-en-v1.5",
       "bge-base-en-v1", "e5-small-v2", "e5-base-v1", "gte-small", "gte-base", "gte-large",
       "arctic-embed-xs", "arctic-embed-s", "arctic-embed-m-v1", "contriever", "contriever-msmarco",
       "tas-b")
STELLA_VOCAB = "0f052197aac305fe42f378b5c61554b7bacb34b8be4af83fec2cfdcdffbc9973"  # E8 preflight
E8.CONFIGS = NEW
E8.OUT_DIR = REPO / "work" / "m15" / "e8x"
E8.FROZEN = E8.OUT_DIR / "frozen_lambdas.json"
E8.PREFLIGHT = E8.OUT_DIR / "preflight.json"
RESULT = REPO / "results" / "m15_e8x_towers.json"


def preflight():
    import hashlib
    import torch
    from transformers import AutoTokenizer
    if E8.PREFLIGHT.exists():
        return
    E8.fit_list()
    os.environ.setdefault("M7_ENCODER", "stella-400M-v5")
    import m8base  # noqa: F401
    import encoders
    rep = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "gpu": torch.cuda.get_device_name(0), "tokenizers": {}}
    for name in NEW:
        spec = encoders.get(name)
        tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)
        vocab = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])
        h = hashlib.sha256("\n".join(t for t, _ in vocab).encode()).hexdigest()
        norm = json.loads(tok.backend_tokenizer.to_str()).get("normalizer") or {}
        rep["tokenizers"][name] = {"vocab_size": len(tok), "cls_id": tok.cls_token_id,
                                   "ordered_vocab_sha256": h, "lowercase": norm.get("lowercase"),
                                   "compatible": len(tok) == 30522 and tok.cls_token_id == 101
                                   and h == STELLA_VOCAB}
        print(name, rep["tokenizers"][name], flush=True)
    rep["eligible"] = [n for n in NEW if rep["tokenizers"][n]["compatible"]]
    rep["excluded"] = [n for n in NEW if n not in rep["eligible"]]
    E8.OUT_DIR.mkdir(parents=True, exist_ok=True)
    E8.atomic_json(E8.PREFLIGHT, rep)


def boot_spearman(x, y, draws=10_000, seed=20260930):
    from scipy.stats import spearmanr
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        i = rng.integers(0, len(x), len(x))
        if len(set(i)) > 2:
            vals.append(spearmanr(x[i], y[i])[0])
    vals = np.array([v for v in vals if np.isfinite(v)])
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def assemble():
    from scipy.stats import spearmanr
    from common import receipt, write_result
    old = json.loads((REPO / "results" / "m15_e8_towers.json").read_text())["configs"]
    pre = json.loads(E8.PREFLIGHT.read_text())
    rows, stopped = {}, {}
    for name in pre["eligible"]:
        d = json.loads((E8.OUT_DIR / f"dev-{name}.json").read_text())
        s = json.loads((E8.OUT_DIR / f"six-{name}.json").read_text()) \
            if (E8.OUT_DIR / f"six-{name}.json").exists() else {"status": "missing"}
        if d["status"] != "COMPLETE" or s["status"] != "COMPLETE":
            stopped[name] = {"dev": d.get("status"), "six": s.get("status")}
            continue
        six = lambda k: float(np.mean([s[k][ds] for ds in E8.SIX]))
        rows[name] = {"repo": d["repo"], "revision": d["revision"], "dim": d["dim"],
                      "pooling": d["pooling"], "lambda": s["lambda"],
                      "dev_table": d["best_dev_macro_2"], "dev_ceiling": d["dev_ceiling_macro_2"],
                      "six_table_macro_all6": six("table"), "six_ceiling_macro_all6": six("ceiling"),
                      "six_table": s["table"], "six_ceiling": s["ceiling"], "final_solve": s["solve"]}
    pooled = {n: {k: old[n][k] for k in ("dev_table", "six_table_macro_all6",
                                         "six_ceiling_macro_all6")}
              for n in old if n != E8.CONTROL}
    pooled.update({n: {k: r[k] for k in ("dev_table", "six_table_macro_all6",
                                         "six_ceiling_macro_all6")} for n, r in rows.items()})
    names = list(pooled)
    g = lambda k: [pooled[n][k] for n in names]
    stats = {}
    for label, xk in (("screen", "dev_table"), ("tower", "six_ceiling_macro_all6")):
        stats[label] = {"spearman": float(spearmanr(g(xk), g("six_table_macro_all6"))[0]),
                        "bootstrap95_over_towers": boot_spearman(g(xk), g("six_table_macro_all6"))}
    write_result(RESULT, {"status": "COMPLETE", "measurement": "E8x (exploratory)",
                          "scope": "E8's recipe on more towers, added after E8; pooled with E8's 10",
                          "n_checkpoints_pooled": len(names), "pooled_names": names,
                          "stats": stats, "new_configs": rows, "stopped": stopped,
                          "excluded_by_tokenizer_check": pre["excluded"],
                          "frozen_lambdas": json.loads(E8.FROZEN.read_text()),
                          "receipt": receipt(__file__, [{"preflight": pre}], pre["started_utc"],
                                             ("torch", "transformers", "scipy"))})


def all_steps():
    py = sys.executable
    env = {**os.environ, "PYTHONPATH": f"{REPO / 'm15src'}:{REPO / 'm8src'}:{REPO / 'm7src'}"}
    subprocess.run([py, __file__, "preflight"], check=True, env=env, cwd=REPO)
    for name in E8.eligible():
        subprocess.run([py, __file__, "dev", name], check=True, cwd=REPO,
                       env={**env, "M7_ENCODER": name})
    if not E8.FROZEN.exists():
        subprocess.run([py, __file__, "freeze"], check=True, env=env, cwd=REPO)
    for name in E8.eligible():
        subprocess.run([py, __file__, "six", name], check=True, cwd=REPO,
                       env={**env, "M7_ENCODER": name})
    subprocess.run([py, __file__, "assemble"], check=True, env=env, cwd=REPO)


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["all"]
    {"preflight": preflight, "freeze": E8.freeze, "assemble": assemble, "all": all_steps,
     "dev": lambda: E8.dev(rest[0]), "six": lambda: E8.six(rest[0])}[cmd]()
