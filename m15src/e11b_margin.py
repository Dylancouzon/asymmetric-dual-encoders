"""E11b (exploratory): does the table's error relative to the tower's own ranking margin explain
retention? Per configuration and six-set dataset: the tower's own exact top-10 scores (cached tower
query and document encodes from E8), margin = median(s1 - s10); error = sqrt(2 - 2 * additivity)
from E11; ratio = error / margin. Spearman against E8 retention over the 10 checkpoints.

    e11b_margin.py tower <name>  |  e11b_margin.py all
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

RESULT = REPO / "results" / "m15_e11b_margin.json"


def tower(name):
    from common import load_public
    out = E8.OUT_DIR / f"e11b-{name}.json"
    if out.exists():
        return
    import m8base  # noqa: F401
    import torch
    from evalkit import topk_arrays
    from teacher import QUERY_PREFIX, encode_cached
    import encoders
    spec = encoders.active()
    rows = {}
    for ds in E8.SIX:
        data = load_public(ds)
        dv = np.asarray(encode_cached(f"e8-six-{ds}-docs", data["doc_texts"],
                                      prefix=spec.doc_prefix, dtype=torch.float16, verbose=False))
        qv = np.asarray(encode_cached(f"e8-six-{ds}-q", data["q_texts"], prefix=QUERY_PREFIX,
                                      dtype=torch.float16, verbose=False), dtype=np.float32)
        _, bs = topk_arrays(qv, dv, k=11, chunk=250_000)
        s = np.sort(bs, axis=1)[:, ::-1]
        rows[ds] = {"median_gap_1_10": float(np.median(s[:, 0] - s[:, 9])),
                    "median_top1": float(np.median(s[:, 0]))}
        print(f"  {name} {ds}: gap {rows[ds]['median_gap_1_10']:.4f}", flush=True)
    E8.atomic_json(out, {"name": name, "sets": rows})


def assemble():
    from common import receipt, utc_now, write_result
    from scipy.stats import spearmanr
    e11 = json.loads((REPO / "results" / "m15_e11_mechanism.json").read_text())["configs"]
    rows = {}
    for name, c in e11.items():
        g = json.loads((E8.OUT_DIR / f"e11b-{name}.json").read_text())["sets"]
        ratio = [np.sqrt(2 - 2 * c["per_set"][ds]["additivity"]) / g[ds]["median_gap_1_10"]
                 for ds in E8.SIX]
        rows[name] = {"median_gap_six": float(np.mean([g[d]["median_gap_1_10"] for d in E8.SIX])),
                      "error_to_margin_six": float(np.mean(ratio)),
                      "retention_all6": c["retention_all6"], "six_table": c["six_table"],
                      "per_set": g}
    ckpt = [n for n in rows if n != E8.CONTROL]
    rho = {f"{x}_vs_{y}": float(spearmanr([rows[n][x] for n in ckpt],
                                          [rows[n][y] for n in ckpt])[0])
           for x in ("median_gap_six", "error_to_margin_six")
           for y in ("retention_all6", "six_table")}
    write_result(RESULT, {"status": "COMPLETE", "measurement": "E11b (exploratory)",
                          "scope": "exploratory; n = 10 checkpoints; no p-values",
                          "spearman": rho, "configs": rows,
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
