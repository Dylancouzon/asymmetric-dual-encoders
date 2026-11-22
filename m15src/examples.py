"""Worked examples for the paper (exploratory): SciFact queries, first-relevant rank per tier.

Writes results/m15_examples.json with, per named query, its text, Zero tokens and fertility, and the
rank of the first relevant document under Zero and Nano (exact search, top 100, M20-gated vectors).
"""
import numpy as np

from common import REPO, load_public, receipt, utc_now, write_result
import e6_router as R6
import encoders15 as E
import vectors15 as V

QIDS = ("49", "94", "659", "660", "300")
OUT = REPO / "results" / "m15_examples.json"


def main():
    started = utc_now()
    data = load_public("scifact")
    dv, prov = V.doc_vectors("scifact", data)
    tok, _ = R6.zero_tokenizer()
    idx = [data["q_ids"].index(q) for q in QIDS]
    texts = [data["q_texts"][i] for i in idx]
    out = {}
    runs = {n: V.exact_run(E.make(n).encode(texts), dv, data["doc_ids"], list(QIDS))
            for n in ("zero", "nano")}
    for q, t in zip(QIDS, texts):
        row = {"text": t, "zero_tokens": tok.encode(t, add_special_tokens=False).tokens,
               "fertility": float(R6.fertility(tok, [t])[0])}
        for n, run in runs.items():
            order = sorted(run[q], key=run[q].get, reverse=True)
            rel = [i for i, d in enumerate(order) if data["qrels"][q].get(d, 0) > 0]
            row[f"{n}_first_relevant_rank"] = rel[0] + 1 if rel else None
        out[q] = row
    write_result(OUT, {"status": "COMPLETE", "measurement": "worked examples (exploratory)",
                       "dataset": "scifact", "note": "rank None = not in the top 100",
                       "examples": out, "receipt": receipt(__file__, [prov], started)})


if __name__ == "__main__":
    main()
