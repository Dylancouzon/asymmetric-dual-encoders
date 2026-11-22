"""E14 (exploratory): two checks on the tower screen, from committed E8/E8x results.

1. Single-forum screens (registered ten, whose lambda curves are in E8's result): Spearman between
   each forum's table score at the frozen lambda and the six-set table score.
2. Family-cluster bootstrap over the 26 pooled checkpoints: resample model families with
   replacement (all checkpoints of a drawn family enter together), 10,000 draws.
"""
import json

import numpy as np
from scipy.stats import spearmanr

from common import REPO, SEED, receipt, utc_now, write_result

OUT = REPO / "results" / "m15_e14_screen_checks.json"
FAMILY = {"stella-400M-v5": "stella", "arctic-embed-l": "arctic", "arctic-embed-m-v1.5": "arctic",
          "arctic-embed-xs": "arctic", "arctic-embed-s": "arctic", "bge-base-en-v1.5": "bge",
          "bge-large-en-v1.5": "bge", "bge-small-en-v1.5": "bge", "bge-base-en-v1": "bge",
          "e5-base-v2": "e5", "e5-large-v2": "e5", "e5-small-v2": "e5", "e5-base-v1": "e5",
          "gte-base-en-v1.5": "gte-v1.5", "gte-large-en-v1.5": "gte-v1.5", "gte-small": "gte",
          "gte-base": "gte", "gte-large": "gte", "mxbai-embed-large-v1": "mxbai",
          "minilm-l6": "minilm", "minilm-l12": "minilm", "multi-qa-minilm-l6": "minilm",
          "msmarco-minilm-l6": "minilm", "contriever": "contriever",
          "contriever-msmarco": "contriever", "tas-b": "tas-b"}


def main():
    started = utc_now()
    e8 = json.loads((REPO / "results" / "m15_e8_towers.json").read_text())["configs"]
    x = json.loads((REPO / "results" / "m15_e8x_towers.json").read_text())
    ten = [n for n in e8 if n != "arctic-embed-l-mean"]
    single = {}
    for forum in ("cqadup-physics", "cqadup-programmers"):
        dev = [e8[n]["lambda_curve"][repr(e8[n]["lambda"])]["per_forum"][forum] for n in ten]
        single[forum] = float(spearmanr(dev, [e8[n]["six_table_macro_all6"] for n in ten])[0])
    pooled = {n: e8[n] for n in ten}
    pooled.update(x["new_configs"])
    names = list(pooled)
    fams = sorted({FAMILY[n] for n in names})
    rng = np.random.default_rng(SEED)
    boot = {"screen": [], "tower": []}
    for _ in range(10_000):
        draw = rng.choice(fams, len(fams), replace=True)
        pick = [n for f in draw for n in names if FAMILY[n] == f]
        y = [pooled[n]["six_table_macro_all6"] for n in pick]
        if len(set(y)) < 3:
            continue
        boot["screen"].append(spearmanr([pooled[n]["dev_table"] for n in pick], y)[0])
        boot["tower"].append(spearmanr([pooled[n]["six_ceiling_macro_all6"] for n in pick], y)[0])
    ci = {k: [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))] for k, v in boot.items()}
    write_result(OUT, {"status": "COMPLETE", "measurement": "E14 (exploratory)",
                       "single_forum_screen_registered_ten": single,
                       "family_cluster_bootstrap95": ci, "n_families": len(fams),
                       "families": {f: [n for n in names if FAMILY[n] == f] for f in fams},
                       "receipt": receipt(__file__, [], started, ("scipy",))})
    print(single, ci, len(fams))


if __name__ == "__main__":
    main()
