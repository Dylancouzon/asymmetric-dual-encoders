"""E13 (exploratory): robustness of E8 and worked word-order examples for the paper.

1. Leave-one-family-out Spearman for E8's screen (dev table vs six-set table) and tower quality
   (six-set tower vs six-set table), over the 10 checkpoints.
2. E8 screen wall time per tower from the dev-file timestamps recorded on the pod (preflight
   02:46 UTC; dev files at the times listed; each includes encoding 337,981 fit queries and two
   forums, four or five ridge solves and dev scoring on one A100).
3. Worked examples: the queries of the six sets with the largest order dependence and a Stella-Zero
   gap of at least 0.5, with the first-relevant rank for Stella (full and shuffled), Nano and Zero.
"""
import json

import numpy as np
from scipy.stats import spearmanr

from common import REPO, SEED, load_public, m20_rows, receipt, utc_now, write_result
import encoders15 as E
import vectors15 as V
from e4_prefix import conditions

OUT = REPO / "results" / "m15_e13_robustness.json"
FAMILY = {"stella-400M-v5": "stella", "arctic-embed-l": "arctic", "arctic-embed-m-v1.5": "arctic",
          "bge-base-en-v1.5": "bge", "bge-large-en-v1.5": "bge", "e5-base-v2": "e5",
          "e5-large-v2": "e5", "gte-base-en-v1.5": "gte", "gte-large-en-v1.5": "gte",
          "mxbai-embed-large-v1": "mxbai"}
# From `ls -l work/m15/e8/` on the pod after E8 (UTC, minutes); preflight.json at 02:46.
DEV_DONE = {"stella-400M-v5": "02:53", "arctic-embed-l": "02:59", "arctic-embed-l-mean": "03:06",
            "arctic-embed-m-v1.5": "03:10", "bge-base-en-v1.5": "03:15",
            "bge-large-en-v1.5": "03:22", "e5-base-v2": "03:26", "e5-large-v2": "03:32",
            "gte-base-en-v1.5": "03:37", "gte-large-en-v1.5": "03:44",
            "mxbai-embed-large-v1": "03:50"}


def leave_one_family_out():
    cf = json.loads((REPO / "results" / "m15_e8_towers.json").read_text())["configs"]
    out = {}
    for drop in sorted(set(FAMILY.values())):
        keep = [n for n in FAMILY if FAMILY[n] != drop]
        out[drop] = {"n": len(keep),
                     "screen": float(spearmanr([cf[n]["dev_table"] for n in keep],
                                               [cf[n]["six_table_macro_all6"] for n in keep])[0]),
                     "tower": float(spearmanr([cf[n]["six_ceiling_macro_all6"] for n in keep],
                                              [cf[n]["six_table_macro_all6"] for n in keep])[0])}
    return out


def screen_minutes():
    t = lambda s: int(s[:2]) * 60 + int(s[3:])
    prev, out = t("02:46"), {}
    for name, done in DEV_DONE.items():
        out[name] = t(done) - prev
        prev = t(done)
    return out


def rank(run, q, qrels):
    order = sorted(run[q], key=run[q].get, reverse=True)
    hits = [i for i, d in enumerate(order) if qrels[q].get(d, 0) > 0]
    return hits[0] + 1 if hits else None


def examples(k=4):
    d = np.load(REPO / "work" / "m15" / "e12_per_query.npz")
    pick = np.where((d["gap_stella"] >= 0.5) & (d["stella_full"] > 0))[0]
    pick = pick[np.argsort(-d["order"][pick])][:k]
    enc = {n: E.make(n) for n in ("stella-query", "nano", "zero")}
    out = []
    for ds in sorted(set(d["ds"][pick])):
        data = load_public(ds)
        dv, _ = V.doc_vectors(ds, data)
        offset = int(np.where(d["ds"] == ds)[0][0])
        for i in pick[d["ds"][pick] == ds]:
            j = int(i) - offset
            q, text = data["q_ids"][j], data["q_texts"][j]
            if abs(m20_rows(ds, "stella-query")[0][q] - float(d["stella_full"][i])) > 1e-9:
                raise SystemExit(f"{ds}: E12 array does not align with query {q}")
            shuf = conditions([text], np.random.default_rng(SEED))["shuffle"][0]
            row = {"dataset": ds, "qid": q, "text": text}
            for name, t in (("stella", text), ("stella_shuffled", shuf), ("nano", text),
                            ("zero", text)):
                e = enc["stella-query" if name.startswith("stella") else name]
                run = V.exact_run(e.encode([t]), dv, data["doc_ids"], [q])
                row[f"{name}_first_relevant_rank"] = rank(run, q, data["qrels"])
            row["shuffled_text"] = shuf
            out.append(row)
    return out


def main():
    started = utc_now()
    write_result(OUT, {"status": "COMPLETE", "measurement": "E13 (exploratory)",
                       "leave_one_family_out": leave_one_family_out(),
                       "screen_minutes_per_tower_a100": screen_minutes(),
                       "order_examples": examples(),
                       "note": "E12 per-query arrays follow E12's dataset order and load_public's "
                               "query order; rank None = not in the top 100",
                       "receipt": receipt(__file__, [], started, ("onnxruntime", "scipy"))})


if __name__ == "__main__":
    main()
