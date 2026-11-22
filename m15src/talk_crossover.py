"""When does query encoding overtake the one-time corpus encode? (exploratory analytic estimate for the talk)

Same FLOPs formulas as m15src/talk_flops.py, on BEIR NQ: the corpus as the registered Stella document path
encodes it (title + " " + text, no prompt, truncated at 512 tokens) against the 3,452 NQ test queries (Stella
with its query prompt; Nano and Zero with their served tokenizers). The crossover is the number of queries
whose summed encoding FLOPs equal the whole corpus encode, using each encoder's mean query.

Run: .venv/bin/python m15src/talk_crossover.py
"""
import statistics
from pathlib import Path

from common import REPO, load_public, receipt, utc_now, write_result
from talk_flops import STELLA_PROMPT, nano, stella, tok, zero

OUT = REPO / "results/m15_talk_crossover.json"


def main():
    started = utc_now()
    hub = str(Path.home() / ".cache/huggingface/hub")
    st = tok(f"{hub}/models--Qdrant--stella-en-400M-v5-doc-onnx/snapshots/*/tokenizer.json")
    na = tok(f"{hub}/models--Qdrant--constella-nano/snapshots/*/tokenizer.json")
    data = load_public("nq")
    doc_t = [min(512, len(e.ids)) for e in st.encode_batch(data["doc_texts"])]
    corpus = sum(stella(T) for T in doc_t)
    q = data["q_texts"]
    per = {"stella": [stella(len(e.ids)) for e in st.encode_batch([STELLA_PROMPT + x for x in q])],
           "nano": [nano(len(e.ids)) for e in na.encode_batch(q)],
           "zero": [zero(e.ids) for e in st.encode_batch(q)]}
    out = {"n_docs": len(doc_t), "doc_tokens_mean": statistics.mean(doc_t), "doc_tokens_median": statistics.median(doc_t),
           "corpus_flops": corpus, "n_queries": len(q), "encoders": {}}
    for k, v in per.items():
        m = statistics.mean(v)
        out["encoders"][k] = {"query_flops_mean": m, "query_flops_median": statistics.median(v),
                              "crossover_queries": corpus / m}
        print(k, f"{m:.3g}", f"crossover {corpus / m:.3g} queries", flush=True)
    print("corpus", f"{corpus:.3g}", "mean doc tokens", out["doc_tokens_mean"])
    write_result(OUT, {"status": "COMPLETE", "analysis": "exploratory (analytic estimate)", "dataset": "nq",
                       "results": out, "pins": data["pins"],
                       "receipt": receipt(__file__, {}, started, ("tokenizers",))})


if __name__ == "__main__":
    main()
