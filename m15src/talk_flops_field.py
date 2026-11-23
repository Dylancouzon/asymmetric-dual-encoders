"""Analytic per-query FLOPs for the 25 E8/E8x retrievers beside Constella (exploratory estimate for the talk).

Same formula and queries as m15src/talk_flops.py: FLOPs = 2 * P_matmul * T + 4 * L * T^2 * d, gated feed-forward
blocks (gte-*-en-v1.5) hold 3 * d * ff weights per layer, others 2 * d * ff. T is the median token count of each
model's query prefix plus the 100 medium E1 queries. All 26 towers passed the E8/E8x preflight with Stella's
uncased 30,522-token WordPiece vocabulary, so one tokenizer counts tokens for all. Layer sizes come from each
model's config.json at its pinned revision. Quality is each model's own six-BEIR nDCG@10 from E8/E8x.

Run: .venv/bin/python m15src/talk_flops_field.py
"""
import glob
import json
import statistics
import sys
from pathlib import Path

from common import REPO, receipt, sha_file, utc_now, write_result

sys.path.insert(0, str(REPO / "m7src"))
import encoders  # noqa: E402

E1 = REPO / "results/m15_e1_latency.json"
E8 = REPO / "results/m15_e8_towers.json"
E8X = REPO / "results/m15_e8x_towers.json"
OUT = REPO / "results/m15_talk_flops_field.json"


def layers(cfg):
    if "n_layers" in cfg:                                    # DistilBERT
        return cfg["n_layers"], cfg["dim"], cfg["hidden_dim"], False
    gated = any("New" in a for a in cfg.get("architectures", []))   # gte-*-en-v1.5 GLU blocks
    return cfg["num_hidden_layers"], cfg["hidden_size"], cfg["intermediate_size"], gated


def main():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    started = utc_now()
    hub = str(Path.home() / ".cache/huggingface/hub")
    tok = Tokenizer.from_file(glob.glob(f"{hub}/models--Qdrant--stella-en-400M-v5-doc-onnx/snapshots/*/tokenizer.json")[0])
    tok.no_padding()
    queries = [q["text"] for q in json.loads(E1.read_text())["queries"]["medium"]]
    six = {n: c["six_ceiling_macro_all6"] for n, c in json.loads(E8.read_text())["configs"].items()}
    six |= {n: c["six_ceiling_macro_all6"] for n, c in json.loads(E8X.read_text())["new_configs"].items()}
    six.pop("arctic-embed-l-mean", None)                     # a pooling variant of arctic-embed-l, not a model
    six.pop("stella-400M-v5")                                # Stella comes from m15_talk_flops.json
    out = {}
    for name in sorted(six):
        spec = encoders.get(name)
        cfg = json.loads(Path(hf_hub_download(spec.repo, "config.json", revision=spec.revision)).read_text())
        L, d, ff, gated = layers(cfg)
        ts = [len(tok.encode(spec.query_prefix + q).ids) for q in queries]
        fl = [2 * L * (4 * d * d + (3 if gated else 2) * d * ff) * T + 4 * L * T * T * d for T in ts]
        out[name] = {"repo": spec.repo, "revision": spec.revision, "layers": L, "hidden": d, "ff": ff,
                     "gated_ffn": gated, "median_tokens": statistics.median(ts),
                     "median_flops": statistics.median(fl), "six_ndcg10": six[name]}
        print(name, L, d, ff, gated, out[name]["median_tokens"], f"{out[name]['median_flops']:.3g}", round(six[name], 4))
    inputs = {k: {"path": str(p.relative_to(REPO)), "sha256": sha_file(p)} for k, p in
              (("e1_latency", E1), ("e8_towers", E8), ("e8x_towers", E8X))}
    write_result(OUT, {"status": "COMPLETE", "analysis": "exploratory (analytic estimate)", "results": out,
                       "n_queries": len(queries),
                       "formula": "2*P_matmul*T + 4*L*T^2*d; gated FFN 3*d*ff, else 2*d*ff; query prefix included",
                       "receipt": receipt(__file__, inputs, started, ("tokenizers", "huggingface_hub"))})


if __name__ == "__main__":
    main()
