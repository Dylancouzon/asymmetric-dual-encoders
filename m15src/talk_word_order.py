"""Exploratory word-order probe for the 2026-11-23 talk.

Pairs were frozen in git (m15/talk/word_order_pairs.json, commit ba452d9) before this ran.
Run: .venv/bin/python m15src/talk_word_order.py
"""
import json

import numpy as np

import encoders15
from common import REPO, receipt, sha_file, utc_now, write_result

PAIRS = REPO / "m15/talk/word_order_pairs.json"
OUT = REPO / "results/m15_talk_word_order.json"


def main():
    started = utc_now()
    pairs = json.loads(PAIRS.read_text())["pairs"]
    stella = encoders15.stella_query()
    # Documents use Stella's registered document path: same graph, no prompt.
    doc_enc = encoders15.OnnxEncoder("stella-doc", _stella_dir(), "mean", prefix="")
    del stella
    docs = doc_enc.encode([t for p in pairs for t in (p["p_ab"], p["p_ba"])]).reshape(len(pairs), 2, -1)
    out = {}
    for name in ("stella-query", "nano", "zero"):
        enc = encoders15.make(name)
        q = enc.encode([t for p in pairs for t in (p["q_ab"], p["q_ba"])]).reshape(len(pairs), 2, -1)
        rows, both, one = [], 0, 0
        for i, p in enumerate(pairs):
            s = np.einsum("qd,pd->qp", q[i], docs[i])  # rows: q_ab, q_ba; cols: p_ab, p_ba
            ab_ok, ba_ok = bool(s[0, 0] > s[0, 1]), bool(s[1, 1] > s[1, 0])
            same_vec = bool(np.allclose(q[i, 0], q[i, 1], atol=1e-6))
            both += ab_ok and ba_ok
            one += ab_ok + ba_ok
            rows.append({"id": p["id"], "ab_correct": ab_ok, "ba_correct": ba_ok,
                         "identical_query_vectors": same_vec,
                         "scores": [[round(float(x), 5) for x in r] for r in s]})
        out[name] = {"pairs_both_directions_correct": both, "directions_correct": one,
                     "n_pairs": len(pairs), "rows": rows}
        print(name, both, "/", len(pairs), "pairs;", one, "/", 2 * len(pairs), "directions")
    write_result(OUT, {"status": "COMPLETE", "analysis": "exploratory",
                       "question": "Does each encoder rank the intended passage first in both word orders?",
                       "results": out,
                       "receipt": receipt(__file__, {"pairs": {"path": str(PAIRS.relative_to(REPO)),
                                                               "sha256": sha_file(PAIRS)}},
                                          started, ("onnxruntime", "tokenizers"))})


def _stella_dir():
    from huggingface_hub import snapshot_download
    return snapshot_download(encoders15.STELLA_ONNX_REPO, revision=encoders15.STELLA_ONNX_REVISION)


if __name__ == "__main__":
    main()
