"""Analytic per-query FLOPs for the three query encoders (exploratory estimate for the talk).

FLOPs = 2 * P_matmul * T + 4 * L * T^2 * d per transformer forward pass, where P_matmul counts the
weights used in matrix multiplies. Stella's feed-forward block is gated (up+gate, then down), so it
holds 3 * d * ff weights per layer. Nano's linear head runs per token before pooling. Zero costs
2 FLOPs per gathered value plus normalization. T is measured with each served tokenizer on the
100 medium (5-12 word) queries of the registered E1 latency protocol.

Run: .venv/bin/python m15src/talk_flops.py
"""
import glob
import json
import statistics
from pathlib import Path

from common import REPO, receipt, sha_file, utc_now, write_result

E1 = REPO / "results/m15_e1_latency.json"
OUT = REPO / "results/m15_talk_flops.json"
STELLA_PROMPT = ("Instruct: Given a web search query, retrieve relevant passages that answer the "
                 "query.\nQuery: ")


def stella(T, L=24, d=1024, ff=4096):
    return 2 * L * (4 * d * d + 3 * d * ff) * T + 4 * L * T * T * d + 2 * d * d


def nano(T, L=12, d=384, ff=1536, feat=1152, out=1024):
    return 2 * L * (4 * d * d + 2 * d * ff) * T + 4 * L * T * T * d + 2 * feat * out * T


def zero(ids, d=1024):
    return 2 * len(set(ids)) * d + 3 * d


def tok(pattern):
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(glob.glob(pattern)[0])
    t.no_padding()
    return t


def main():
    started = utc_now()
    hub = str(Path.home() / ".cache/huggingface/hub")
    queries = [q["text"] for q in json.loads(E1.read_text())["queries"]["medium"]]
    st = tok(f"{hub}/models--Qdrant--stella-en-400M-v5-doc-onnx/snapshots/*/tokenizer.json")
    na = tok(f"{hub}/models--Qdrant--constella-nano/snapshots/*/tokenizer.json")

    rows = {"stella": [], "nano": [], "zero": []}
    toks = {"stella": [], "nano": [], "zero": []}
    for q in queries:
        ts, tn, tz = (len(st.encode(STELLA_PROMPT + q).ids), len(na.encode(q).ids),
                      st.encode(q).ids)
        toks["stella"].append(ts)
        toks["nano"].append(tn)
        toks["zero"].append(len(tz))
        rows["stella"].append(stella(ts))
        rows["nano"].append(nano(tn))
        rows["zero"].append(zero(tz))
    med = {k: statistics.median(v) for k, v in rows.items()}
    out = {k: {"median_tokens": statistics.median(toks[k]), "median_flops": med[k],
               "ratio_to_zero": med[k] / med["zero"]} for k in rows}
    for k, v in out.items():
        print(k, v)
    write_result(OUT, {"status": "COMPLETE", "analysis": "exploratory (analytic estimate)",
                       "results": out, "n_queries": len(queries),
                       "formula": "2*P_matmul*T + 4*L*T^2*d; Stella gated FFN 3*d*ff; Nano head per token",
                       "receipt": receipt(__file__, {"e1_latency": {"path": str(E1.relative_to(REPO)),
                                                                    "sha256": sha_file(E1)}},
                                          started, ("tokenizers",))})


if __name__ == "__main__":
    main()
