"""M10.0-c (box, CPU): the serving-parity gate every family-F student must pass before it runs.

§Recipe: "Before F can select MiniLM, its three-layer head passes the same parity check", and
"L12's three- and four-layer heads pass the parity check first"; §Screen: a failing head
disqualifies that arm. The check is the one bge-small already passed
(`results/m10_head_width_parity_mac.json`, `m10_head_mlp_parity_box.json`): export the per-token
head, let fastembed do the masked mean and the normalize, and require it to reproduce the
training-time `normalize(mean_t(W x_t))` — because the head precedes the pool, the two are the
same linear map and the parity is exact up to float error, not approximate.

Feature layers per student (§Recipe): bge-small and MiniLM-L12 concatenate layers 12, 8, 4
(1152-d), MiniLM-L6 layers 6, 4, 2; family G's four-layer arm adds layer 2 (MiniLM-L6: layer 1),
1536-d. Random head weights: parity is a property of the export path, not of the weights.

Diagnostic; read by one rule only — the F-arm disqualification.
Usage: student_parity.py [student-key ...]     (default: all three)
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "m10_student_parity_box.json"
WORK = REPO / "work" / "m10onnx"
NAME = "qdrant/nano-student-parity-check"

STUDENTS = {
    "bge-small": {"repo": "BAAI/bge-small-en-v1.5", "layers3": (12, 8, 4), "layers4": (12, 8, 4, 2)},
    "MiniLM-L6": {"repo": "sentence-transformers/all-MiniLM-L6-v2", "layers3": (6, 4, 2),
                  "layers4": (6, 4, 2, 1)},
    "MiniLM-L12": {"repo": "sentence-transformers/all-MiniLM-L12-v2", "layers3": (12, 8, 4),
                   "layers4": (12, 8, 4, 2)},
}
CAP = 35_000_000
MAX_LEN = 512          # the served sequence length, written into the exported tokenizer


class PerTokenLinear(torch.nn.Module):
    """The M10 head: concat the chosen hidden states, one Linear, PER TOKEN (before the pool)."""
    def __init__(self, backbone, layers, d_in, d_out=1024):
        super().__init__()
        self.backbone, self.layers = backbone, tuple(layers)
        self.head = torch.nn.Linear(d_in, d_out)

    def forward(self, input_ids, attention_mask):
        hs = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True).hidden_states
        return self.head(torch.cat([hs[l] for l in self.layers], dim=-1))


def reference(model, tok, texts):
    b = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.inference_mode():
        t = model(b["input_ids"], b["attention_mask"])
        m = b["attention_mask"].unsqueeze(-1).to(t.dtype)
        v = (t * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(v, dim=-1).numpy(), b


def texts_for_parity():
    """Real queries plus the shapes an export trace gets wrong: a 1-token input, a single
    punctuation mark, a text longer than max_length, and one that pads the batch hardest."""
    from datasets import load_dataset
    qs = list(load_dataset("google-research-datasets/nq_open", split="train")["question"])[:56]
    return qs + [" ".join(qs[i:i + 12]) for i in range(0, 48, 12)] + \
        ["Instruct: retrieve", "?", "a " * 300, "The quick brown fox jumps over the lazy dog."]


def check(key, n_layers, texts):
    from transformers import AutoModel, AutoTokenizer
    spec = STUDENTS[key]
    layers = spec[f"layers{n_layers}"]
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(spec["repo"])
    backbone = AutoModel.from_pretrained(spec["repo"], dtype=torch.float32).eval()
    if max(layers) > backbone.config.num_hidden_layers:
        return {"key": key, "n_layers": n_layers, "PASS": False,
                "error": f"layer {max(layers)} requested, backbone has "
                         f"{backbone.config.num_hidden_layers}"}
    d_in = backbone.config.hidden_size * len(layers)
    model = PerTokenLinear(backbone, layers, d_in).eval()
    n_params = sum(p.numel() for p in model.parameters())

    d = WORK / f"parity-{key}-{n_layers}L"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "model.onnx"
    ex = tok(["a short query", "a somewhat longer example sentence for the export trace"],
             padding=True, truncation=True, max_length=64, return_tensors="pt")
    torch.onnx.export(model, (ex["input_ids"], ex["attention_mask"]), str(p),
                      input_names=["input_ids", "attention_mask"],
                      output_names=["token_embeddings"],
                      dynamic_axes={"input_ids": {0: "b", 1: "s"},
                                    "attention_mask": {0: "b", 1: "s"},
                                    "token_embeddings": {0: "b", 1: "s"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    # The served max sequence length is OURS to set, and it is what the first run of this check
    # got wrong: `all-MiniLM-*-v2`'s own `tokenizer.json` carries truncation at 128, so fastembed
    # truncated at 128 while the torch reference ran at 512 and the check read 0.93-0.95 min-cos.
    # That was a mis-specified check, not a serving defect -- every text under the limit was
    # bit-exact (median cos 0.99999998, 1.000000 on real queries). The export writes the
    # tokenizer we intend to ship, at MAX_LEN, so the two sides compare the same model.
    tok.backend_tokenizer.enable_truncation(max_length=MAX_LEN)
    tok.model_max_length = MAX_LEN
    tok.save_pretrained(d)
    # THE mechanism, read out of `fastembed.common.preprocessor_utils.load_tokenizer`: it takes
    # `min(model_max_length, max_length)` from `tokenizer_config.json`. `all-MiniLM-*-v2` ships
    # BOTH -- model_max_length 512 and max_length 128 -- so fastembed serves at 128 while torch
    # runs at 512, and `save_pretrained` faithfully re-writes the 128. bge-small ships no
    # `max_length` key at all, which is the whole reason it read exactly 1.0 and MiniLM did not.
    # The exported model is ours, so the key is set to what we serve.
    tc = d / "tokenizer_config.json"
    cfg = json.loads(tc.read_text())
    cfg["max_length"] = MAX_LEN
    cfg["model_max_length"] = MAX_LEN
    tc.write_text(json.dumps(cfg, indent=1))
    backbone.config.save_pretrained(d)
    import onnx
    g = onnx.load(str(p))
    custom = sorted({n.domain for n in g.graph.node} - {"", "ai.onnx", "ai.onnx.ml"})

    ref, b = reference(model, tok, texts)
    served_trunc = {"tokenizer.json": json.loads((d / "tokenizer.json").read_text())
                    .get("truncation", {}).get("max_length"),
                    "fastembed_effective": min(
                        cfg.get("model_max_length", 10**9), cfg.get("max_length", 10**9))}
    import onnxruntime as ort
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    toks = sess.run(None, {"input_ids": b["input_ids"].numpy(),
                           "attention_mask": b["attention_mask"].numpy()})[0]
    m = b["attention_mask"].numpy()[..., None].astype(np.float32)
    man = (toks * m).sum(1) / np.maximum(m.sum(1), 1e-9)
    man = man / np.linalg.norm(man, axis=1, keepdims=True)

    res = {"key": key, "repo": spec["repo"], "n_layers": n_layers, "layers": list(layers),
           "d_in": d_in, "params_total": int(n_params), "under_35M_cap": bool(n_params <= CAP),
           "custom_domain_ops": custom, "n_nodes": len(g.graph.node),
           "onnx_bytes": p.stat().st_size, "n_texts": len(texts),
           "served_truncation": served_trunc,
           "ort_then_manual_pool": {"min_cos": float((ref * man).sum(1).min()),
                                    "max_abs": float(np.abs(ref - man).max())}}
    try:
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType
        TextEmbedding.add_custom_model(model=f"{NAME}-{key}-{n_layers}", pooling=PoolingType.MEAN,
                                       normalization=True, sources=ModelSource(hf=NAME), dim=1024,
                                       model_file="model.onnx",
                                       description="M10 family-F head parity check", license="mit",
                                       size_in_gb=round(p.stat().st_size / 1e9, 4))
        t0 = time.time()
        fe = TextEmbedding(model_name=f"{NAME}-{key}-{n_layers}", specific_model_path=str(d),
                           threads=4)
        got = np.stack(list(fe.embed(texts, batch_size=32)))
        cos = (ref * got).sum(1)
        res["fastembed"] = {"version": __import__("fastembed").__version__, "served": True,
                            "min_cos": float(cos.min()), "mean_cos": float(cos.mean()),
                            "max_abs": float(np.abs(ref - got).max()),
                            "seconds": round(time.time() - t0, 1)}
    except Exception as e:
        res["fastembed"] = {"served": False, "error": repr(e)[:400]}
    fe = res["fastembed"]
    res["PASS"] = bool(res["ort_then_manual_pool"]["min_cos"] >= 1 - 1e-4
                       and not custom and res["under_35M_cap"]
                       and fe.get("served") and fe["min_cos"] >= 1 - 1e-4)
    return res


def main(keys=None):
    texts = texts_for_parity()
    rows = []
    for key in (keys or list(STUDENTS)):
        for n in (3, 4):
            r = check(key, n, texts)
            rows.append(r)
            print(f"{key:11s} {n}L  PASS={r['PASS']}  params={r.get('params_total')}  "
                  f"min_cos(ort)={r.get('ort_then_manual_pool', {}).get('min_cos')}  "
                  f"fastembed={r['fastembed'].get('min_cos', r['fastembed'].get('error'))}",
                  flush=True)
    rec = {"_what": __doc__.strip(), "cap": CAP, "rows": rows,
           "disqualified": [f"{r['key']}/{r['n_layers']}L" for r in rows if not r["PASS"]]}
    OUT.write_text(json.dumps(rec, indent=1))
    print("\ndisqualified:", rec["disqualified"] or "none")
    return rec


if __name__ == "__main__":
    main(sys.argv[1:] or None)
