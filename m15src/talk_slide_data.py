"""Slide data for the 2026-11-23 talk (exploratory).

(a) Word order: cosine between the query vectors of each frozen reversed pair
    (m15/talk/word_order_pairs.json), per encoder. Zero pools a bag of tokens, so reversals give
    cosine 1 exactly.
(b) Zero lookup example: the dequantized table rows Zero gathers for one query (first 48
    dimensions, for drawing), Zero's pooled vector, and its cosine with Stella's query vector.

Run: .venv/bin/python m15src/talk_slide_data.py   -> results/m15_talk_slide_data.json
"""
import json
import statistics

import numpy as np

import encoders15
from common import REPO, receipt, sha_file, utc_now, write_result

PAIRS = REPO / "m15/talk/word_order_pairs.json"
OUT = REPO / "results/m15_talk_slide_data.json"
EXAMPLE = "cheap flights paris to rome"
DIMS = 48


def main():
    started = utc_now()
    pairs = json.loads(PAIRS.read_text())["pairs"]
    texts = [t for p in pairs for t in (p["q_ab"], p["q_ba"])]
    order, vecs = {}, {}
    for name in ("stella-query", "nano", "zero"):
        enc = encoders15.make(name)
        q = enc.encode(texts).reshape(len(pairs), 2, -1).astype(np.float64)
        cos = [float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) for a, b in q]
        order[name] = {"median_cos": statistics.median(cos), "min_cos": min(cos), "max_cos": max(cos),
                       "per_pair": {p["id"]: round(c, 6) for p, c in zip(pairs, cos)}}
        vecs[name] = enc.encode([EXAMPLE])[0].astype(np.float64)
        print(name, {k: v for k, v in order[name].items() if k != "per_pair"}, flush=True)

    zero = encoders15.ZeroEncoder().model
    enc_ids = zero.tokenizer.encode(EXAMPLE)
    rows = [{"token": tok, "id": int(i), "row": [round(float(x), 4) for x in zero.rows[i][:DIMS]]}
            for tok, i in zip(enc_ids.tokens, enc_ids.ids)]
    z, s, n = vecs["zero"], vecs["stella-query"], vecs["nano"]
    example = {"query": EXAMPLE, "dims_drawn": DIMS, "rows": rows,
               "zero_vector": [round(float(x), 4) for x in z[:DIMS]],
               "stella_vector": [round(float(x), 4) for x in s[:DIMS]],
               "cos_zero_stella": float(z @ s / (np.linalg.norm(z) * np.linalg.norm(s))),
               "cos_nano_stella": float(n @ s / (np.linalg.norm(n) * np.linalg.norm(s)))}
    print({k: example[k] for k in ("cos_zero_stella", "cos_nano_stella")}, [r["token"] for r in rows])
    write_result(OUT, {"status": "COMPLETE", "analysis": "exploratory", "order_cosine": order,
                       "zero_example": example,
                       "receipt": receipt(__file__, {"pairs": {"path": str(PAIRS.relative_to(REPO)),
                                                               "sha256": sha_file(PAIRS)}},
                                          started, ("onnxruntime", "tokenizers"))})


if __name__ == "__main__":
    main()
